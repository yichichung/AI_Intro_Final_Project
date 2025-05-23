#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_wr.py —— 預測藍方勝率的分類模型
輸出：
  models/wr_lgb.txt
  reports/wr_cv.txt
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "output/match_dataset.parquet")
# 將每場比賽的 5v5 pick → 特徵向量（0/1 是否有選過）
CHAMPIONS = sorted(set(sum(df["bluepicks"].str.split(","), []) +
                       sum(df["redpicks"].str.split(","), [])))
CHAMPIONS = [int(c) for c in CHAMPIONS if c.isdigit()]
N = max(CHAMPIONS) + 1  # 總共有幾種 champion


def to_vector(picks):
    vec = np.zeros(N)
    for pid in picks:
        if pid.isdigit():
            vec[int(pid)] = 1
    return vec


X = []
for _, row in df.iterrows():
    blue = to_vector(row["bluepicks"].split(","))
    red = to_vector(row["redpicks"].split(","))
    X.append(np.concatenate([blue, red]))

X = np.array(X)
y = df["bluewin"].astype(int).values
# ---------- Optuna ----------


def objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("leaves", 64, 256, step=32),
        "feature_fraction": trial.suggest_float("ff", 0.6, 1.0),
        "min_data_in_leaf": trial.suggest_int("min", 10, 100),
        "verbosity": -1,
        "seed": 42
    }
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in kf.split(X):
        train_data = lgb.Dataset(X[train_idx], label=y[train_idx])
        val_data = lgb.Dataset(X[val_idx],   label=y[val_idx])
        model = lgb.train(params, train_data, num_boost_round=100)
        pred = model.predict(X[val_idx])
        aucs.append(roc_auc_score(y[val_idx], pred))
    return 1 - np.mean(aucs)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15, timeout=900)

# ---------- Final Training ----------
best = study.best_params
best.update({"objective": "binary", "metric": "auc",
            "verbosity": -1, "seed": 42})
final_model = lgb.train(best, lgb.Dataset(X, label=y), num_boost_round=120)

# ---------- Save Output ----------
MODEL = ROOT / "models"
MODEL.mkdir(exist_ok=True)
REPORT = ROOT / "reports"
REPORT.mkdir(exist_ok=True)

final_model.save_model(str(MODEL / "wr_lgb.txt"))
with open(REPORT / "wr_cv.txt", "w") as f:
    f.write(f"AUC: {1 - study.best_value:.4f}\n")
    f.write(f"Params: {best}\n")

print("✅ 勝率模型訓練完成！AUC =", round(1 - study.best_value, 4))
