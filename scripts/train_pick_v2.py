#!/usr/bin/env python
# train_pick_v2.py — Optuna + LightGBM (GPU)  v2025-05-24

import time
import argparse
import json
import joblib
import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder
from optuna.integration import LightGBMPruningCallback

# ---------------- CLI 參數 ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--n_trials", type=int, default=15)
parser.add_argument("--num_boost", type=int, default=200)
parser.add_argument("--timeout",   type=int, default=1800)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT/"output/pick_dataset.parquet")

# ---------------- 特徵編碼 ----------------


def splt(s): return [int(x) for x in s.split(',')] if s else []


for i in range(5):
    df[f"pick_{i+1}"] = df["teampicks"].apply(lambda l: (splt(l)+[0]*5)[i])
    df[f"ban_{i+1}"] = df["teambans"].apply(lambda l: (splt(l)+[0]*5)[i])

feat_cols = [f"pick_{i}" for i in range(1, 6)] + \
            [f"ban_{i}" for i in range(1, 6)] + \
            ["pick_order", "role_code", "side_code"]

le = LabelEncoder().fit(df["label"])
y = le.transform(df["label"])
X = df[feat_cols]
groups = df["matchid"]

# ---------------- Optuna 目標函式 ----------------


def objective(trial):
    params = dict(
        objective="multiclass",
        num_class=len(le.classes_),
        metric="multi_logloss",
        device_type="gpu",
        learning_rate=trial.suggest_float("lr", 0.02, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 31, 127),
        max_depth=trial.suggest_int("max_depth", 4, 8),
        min_data_in_leaf=trial.suggest_int("min_leaf", 20, 100),
        feature_fraction=trial.suggest_float("ff", 0.7, 1.0),
        bagging_fraction=0.8, bagging_freq=1,
        seed=42, verbosity=-1
    )

    gkf = GroupKFold(n_splits=3)
    scores = []
    for tr_idx, va_idx in gkf.split(X, y, groups):
        dtr = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx])
        dva = lgb.Dataset(X.iloc[va_idx], label=y[va_idx])

        mdl = lgb.train(
            params, dtr,
            num_boost_round=args.num_boost,
            valid_sets=[dva],
            callbacks=[
                LightGBMPruningCallback(trial, "multi_logloss"),
                lgb.early_stopping(20, verbose=True),        # <- 會每 10 iter 顯示
                lgb.log_evaluation(period=10)                # <- log 頻率
            ]
        )
        pred = mdl.predict(X.iloc[va_idx], num_iteration=mdl.best_iteration)
        scores.append(top_k_accuracy_score(y[va_idx], pred, k=5))

    return 1 - float(np.mean(scores))   # minimize


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

# ---------------- 全資料再訓練 ----------------
best = study.best_params
best.update(objective="multiclass", num_class=len(le.classes_),
            metric="multi_logloss", device_type="gpu")

final = lgb.train(best, lgb.Dataset(
    X, label=y), num_boost_round=study.best_trial.params.get("num_boost", args.num_boost))

Path(ROOT/"models").mkdir(exist_ok=True)
final.save_model(ROOT/"models/pick_lgb_v2.txt")
joblib.dump(le, ROOT/"models/pick_encoder_v2.pkl")

# ---------------- CV 報告 ----------------
pred_all = final.predict(X)
top5 = top_k_accuracy_score(y, pred_all, k=5)
Path(ROOT/"reports").mkdir(exist_ok=True)
with open(ROOT/"reports/pick_cv_v2.txt", "w") as f:
    f.write(f"Top-5 Accuracy: {top5:.4f}\n")
print(f"🎯 Finished — Top-5 = {top5:.4f}")
