#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_pick_v3.py — categorical + win-rate 版（防 GPU split crash）
"""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import top_k_accuracy_score

# ---------------- CLI ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--n_trials",  type=int, default=10)
ap.add_argument("--num_boost", type=int, default=400)
ap.add_argument("--timeout",   type=int, default=1800)
ap.add_argument("--cpu",      action="store_true")
args = ap.parse_args()

ROOT = Path(__file__).resolve().parents[1]
PICK = pd.read_parquet(ROOT/"output/pick_dataset.parquet")
WRMAP = pd.read_parquet(ROOT/"output/champ_wr.parquet")

# ------------ feature engineering ------------ #
pick_arr = np.zeros((len(PICK), 5), dtype=np.int16)
ban_arr = np.zeros_like(pick_arr)
for i, col in enumerate(["teampicks", "teambans"]):
    tgt = pick_arr if i == 0 else ban_arr
    for r, s in enumerate(PICK[col].fillna("")):
        toks = [int(x) for x in s.split(",") if x.isdigit()][:5]
        tgt[r, :len(toks)] = toks

feat = {}
for i in range(5):
    feat[f"pick_{i+1}"] = pick_arr[:, i]
    feat[f"ban_{i+1}"] = ban_arr[:, i]
    feat[f"pick_wr{i+1}"] = [WRMAP.loc[c, "wr_blue"]
                             if c in WRMAP.index else 0.5 for c in pick_arr[:, i]]
    feat[f"ban_wr{i+1}"] = 0.5  # ban win-rate 不明 → 給 0.5

for col in ["pick_order", "role_code", "side_code"]:
    feat[col] = PICK[col].values.astype(np.int16)

X = pd.DataFrame(feat)
cat_cols = [c for c in X.columns if c.startswith(("pick_", "ban_"))]

# ------------ label / groups ------------ #
le = LabelEncoder().fit(PICK["label"])
y = le.transform(PICK["label"])
groups = PICK["matchid"]

# ------------ Optuna ------------ #


def objective(trial):
    print(f"🔍 Trial {trial.number} ...")
    params = dict(
        objective="multiclass",
        num_class=len(le.classes_),
        metric="multi_logloss",
        device_type="cpu" if args.cpu else "gpu",
        force_row_wise=True,
        learning_rate=trial.suggest_float("lr", 0.02, 0.1, log=True),
        num_leaves=trial.suggest_int("leaves", 31, 127),
        max_depth=trial.suggest_int("depth", 4, 8),
        min_data_in_leaf=trial.suggest_int("min_leaf", 10, 120),
        feature_fraction=trial.suggest_float("ff", 0.7, 1.0),
        bagging_fraction=0.8, bagging_freq=1,
        seed=42, verbosity=-1
    )
    gkf, scores = GroupKFold(3), []
    for tr, va in gkf.split(X, y, groups):
        dtr = lgb.Dataset(
            X.iloc[tr], y[tr], categorical_feature=cat_cols, free_raw_data=False)
        dva = lgb.Dataset(
            X.iloc[va], y[va], categorical_feature=cat_cols, free_raw_data=False)
        mdl = lgb.train(params, dtr,
                        num_boost_round=args.num_boost,
                        valid_sets=[dva],
                        callbacks=[lgb.early_stopping(30, verbose=False)])
        scores.append(top_k_accuracy_score(
            y[va], mdl.predict(X.iloc[va]), k=5))
    return 1 - float(np.mean(scores))


print("🚀 Optuna search …")
study = optuna.create_study(direction="minimize")
study.optimize(objective,
               n_trials=args.n_trials,
               timeout=args.timeout)

# ------------ final train ------------ #
best = study.best_params | dict(objective="multiclass",
                                num_class=len(le.classes_),
                                metric="multi_logloss",
                                device_type="cpu" if args.cpu else "gpu",
                                force_row_wise=True,
                                min_data_in_leaf=max(10, study.best_params.get("min_leaf", 10)))

try:
    final = lgb.train(best,
                      lgb.Dataset(X, y, categorical_feature=cat_cols),
                      num_boost_round=args.num_boost)
except lgb.basic.LightGBMError:
    print("⚠️  GPU crash → fallback to CPU")
    best["device_type"] = "cpu"
    final = lgb.train(best,
                      lgb.Dataset(X, y, categorical_feature=cat_cols),
                      num_boost_round=args.num_boost)

# ------------ save ------------ #
MODEL, REPORT = ROOT/"models", ROOT/"reports"
MODEL.mkdir(exist_ok=True)
REPORT.mkdir(exist_ok=True)
final.save_model(MODEL/"pick_lgb_v3.txt")
joblib.dump(le, MODEL/"pick_encoder_v3.pkl")

top5 = top_k_accuracy_score(y, final.predict(X))
with open(REPORT/"pick_cv_v3.txt", "w") as f:
    json.dump(dict(top5=top5, params=study.best_params), f, indent=2)

print(f"🎯 Done | Top-5 = {top5:.4f}")
