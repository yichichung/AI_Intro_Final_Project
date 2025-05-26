#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_wr_v4.py — Win-Rate LightGBM(GPU) with Pair-Synergy
  • 基礎 WR + patch/duration/obj
  • pair-synergy (syn_blue / syn_red)
  • champion id 用 categorical 索引
  ▶ 目標 AUC ≈ 0.64–0.66
"""
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from itertools import combinations
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MDF = pd.read_parquet(ROOT/"output/match_dataset.parquet")
WR = pd.read_parquet(ROOT/"output/champ_wr.parquet")        # wr_blue / wr_red

# ---------- 1. pick / WR ----------


def top5(s):
    a = [int(x) for x in s.split(",")[:5] if x.isdigit()]
    return a+[0]*(5-len(a))


def wr_of(cid, side): return WR.loc[cid,
                                    f"wr_{side}"] if cid in WR.index else 0.5


for i in range(5):
    MDF[f"bp{i}"] = MDF["bluepicks"].fillna("").apply(lambda s: top5(s)[i])
    MDF[f"rp{i}"] = MDF["redpicks"].fillna("").apply(lambda s: top5(s)[i])
    MDF[f"bp_wr{i}"] = MDF[f"bp{i}"].apply(lambda x: wr_of(x, "blue"))
    MDF[f"rp_wr{i}"] = MDF[f"rp{i}"].apply(lambda x: wr_of(x, "red"))

# ---------- 2. patch / obj ----------
MDF["patch"] = (MDF["version"].astype(str)
                .str.extract(r"(\d+\.\d+)").astype(float).fillna(0.0))
for col in ["baronkills", "dragonkills", "towerkills"]:
    if col not in MDF.columns:
        MDF[col] = 0

# ---------- 3. pair-synergy ----------
print("⏳  computing pair-synergy …")
pair_cnt, pair_win = {}, {}
for p_b, p_r, w in tqdm(MDF[["bluepicks", "redpicks", "win"]].itertuples(index=False),
                        total=len(MDF)):
    for side, picks in (("blue", p_b), ("red", p_r)):
        arr = [int(x) for x in picks.split(",")[:5] if x.isdigit()]
        for a, b in combinations(arr, 2):
            k = (min(a, b), max(a, b), side)
            pair_cnt[k] = pair_cnt.get(k, 0)+1
            pair_win[k] = pair_win.get(k, 0)+w
pair_wr = {k: pair_win[k]/pair_cnt[k] for k in pair_cnt}


def mean_syn(row, col):
    arr = [int(x) for x in row[col].split(",")[:5] if x.isdigit()]
    side = "blue" if "blue" in col else "red"
    vals = [pair_wr.get((min(a, b), max(a, b), side), 0.5)
            for a, b in combinations(arr, 2)]
    return np.mean(vals) if vals else 0.5


MDF["syn_blue"] = MDF.apply(lambda r: mean_syn(r, "bluepicks"), axis=1)
MDF["syn_red"] = MDF.apply(lambda r: mean_syn(r, "redpicks"), axis=1)
del pair_cnt, pair_win
gc.collect()

# ---------- 4. features ----------
cat_cols = [f"bp{i}" for i in range(5)]+[f"rp{i}" for i in range(5)]
num_cols = [f"bp_wr{i}" for i in range(5)]+[f"rp_wr{i}" for i in range(5)]+[
    "patch", "duration", "baronkills", "dragonkills", "towerkills",
    "syn_blue", "syn_red"]
X = pd.concat([MDF[cat_cols].astype("int16"),
               MDF[num_cols].astype("float32")], axis=1)
y = MDF["win"].astype(int)
cat_idx = [X.columns.get_loc(c) for c in cat_cols]   # <<< 關鍵

# ---------- 5. K-Fold ----------
kf, aucs = KFold(3, shuffle=True, random_state=42), []
print("🚀  training …")
for tr, va in tqdm(kf.split(X), total=3, desc="CV"):
    mdl = lgb.LGBMClassifier(
        objective="binary", learning_rate=0.05,
        num_leaves=128, n_estimators=500,
        feature_fraction=0.9, min_data_in_leaf=200,
        device_type="gpu", seed=42,
        categorical_feature=cat_idx            # 用索引！
    )
    mdl.fit(
        X.iloc[tr], y.iloc[tr],
        eval_set=[(X.iloc[va], y.iloc[va])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    proba = mdl.predict_proba(
        X.iloc[va], num_iteration=mdl.best_iteration_)[:, 1]
    aucs.append(roc_auc_score(y.iloc[va], proba))

auc = float(np.mean(aucs))
print(f"✅  WR v4 AUC = {auc:.4f}")

# ---------- 6. save ----------
MODEL = ROOT/"models"
MODEL.mkdir(exist_ok=True)
REPORT = ROOT/"reports"
REPORT.mkdir(exist_ok=True)
mdl.booster_.save_model(str(MODEL/"wr_lgb_v4.txt"))
with open(REPORT/"wr_cv_v4.txt", "w") as f:
    f.write(f"AUC: {auc:.4f}\n")
