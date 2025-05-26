#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_pick_v2.py —— 多類別 LightGBM（GPU 強化修正版）
------------------------------------------------
* GPU 加速 + use_zero_as_missing
* 強化錯誤保護：避免 left_count=0 split crash
* 實時 log、最佳 checkpoint、Optuna 超參數搜尋
"""

import time
import json
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

# 路徑設置
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output/pick_dataset_v2.parquet"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# 載入資料
print("[INFO] loading dataset …")
df = pd.read_parquet(DATA)


def to_intlist(s):
    return [int(t) for t in s.split(',') if t.isdigit()]


df["teampicks"] = df["teampicks"].apply(to_intlist)
df["teambans"] = df["teambans"].apply(to_intlist)

le = LabelEncoder().fit(df["label"])
df["label_enc"] = le.transform(df["label"])
NUM_CLASS = len(le.classes_)

MAX = 5
feat_cols = []
for i in range(MAX):
    df[f"pick_{i+1}"] = df["teampicks"].apply(lambda l: (l + [0]*MAX)[i])
    df[f"ban_{i+1}"] = df["teambans"].apply(lambda l: (l + [0]*MAX)[i])
    feat_cols.extend([f"pick_{i+1}", f"ban_{i+1}"])
feat_cols += ["pick_order", "role_code", "side_code",
              "patch_code", "pick_rate", "win_rate"]

X = df[feat_cols]
y = df["label_enc"]
groups = df["matchid"]

# log 檔與 checkpoint
LOG_FILE = REPORT_DIR / "train_log.txt"
CV_FILE = REPORT_DIR / "pick_cv_v2.txt"
MODEL_PATH = MODEL_DIR / "pick_lgb_v2.txt"
ENC_PATH = MODEL_DIR / "pick_encoder_v2.pkl"

LOG = open(LOG_FILE, "w", encoding="utf-8")
BEST_ACC = 0.0


def ts():
    return datetime.now().strftime("[%H:%M:%S]")


def objective(trial):
    global BEST_ACC
    t0 = time.time()
    try:
        params = {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "num_class": NUM_CLASS,
            "learning_rate": trial.suggest_float("lr", 0.02, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "feature_fraction": trial.suggest_float("ff", 0.7, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "min_gain_to_split": 0.01,
            "device_type": "gpu",
            "use_zero_as_missing": True,
            "verbosity": -1,
            "seed": 42
        }
        gkf = GroupKFold(n_splits=3)
        scores = []
        for tr, va in gkf.split(X, y, groups):
            lgb_train = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
            lgb_val = lgb.Dataset(X.iloc[va], label=y.iloc[va])
            md = lgb.train(params, lgb_train, num_boost_round=120)
            pred = md.predict(X.iloc[va])
            scores.append(top_k_accuracy_score(y.iloc[va], pred, k=5))
        acc = float(np.mean(scores))
        LOG.write(f"{ts()} Trial {trial.number} ✅ Top5={acc:.4f} params=" +
                  json.dumps(trial.params)+"\n")
        LOG.flush()
        if acc > BEST_ACC:
            BEST_ACC = acc
            final = lgb.train(params, lgb.Dataset(X, label=y),
                              num_boost_round=int(md.best_iteration or 150))
            final.save_model(str(MODEL_PATH))
            joblib.dump(le, ENC_PATH)
        print(
            f"Trial {trial.number} done in {time.time()-t0:.1f}s  acc={acc:.4f}")
        return 1 - acc
    except Exception as e:
        LOG.write(f"{ts()} Trial {trial.number} ❌ Error: {e}\n")
        LOG.flush()
        return 1.0


# 執行 optuna 搜尋
print("[INFO] start training with Optuna …")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15, timeout=1200)

with open(CV_FILE, "w", encoding="utf-8") as f:
    f.write(f"Top-5 Accuracy : {BEST_ACC:.4f}\n")
    f.write("Best Params    : "+json.dumps(study.best_params)+"\n")

LOG.close()
print(f"🎯 Done. Best Top-5 Accuracy = {BEST_ACC:.4f}")
