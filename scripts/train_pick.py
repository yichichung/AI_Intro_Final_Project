#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_pick.py
-------------
訓練多類別 LightGBM，輸出：
  models/pick_lgb.txt         ← LightGBM 原生模型
  models/pick_encoder.pkl     ← champ LabelEncoder
  reports/pick_cv.txt         ← 交叉驗證結果
"""

import pandas as pd, numpy as np, lightgbm as lgb, joblib, optuna
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, top_k_accuracy_score

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "output"  / "pick_dataset.parquet"
MODEL  = ROOT / "models";  MODEL.mkdir(exist_ok=True)
REPORT = ROOT / "reports"; REPORT.mkdir(exist_ok=True)

##############################################
# 1. 載入資料
##############################################
df = pd.read_parquet(DATA)

# 解析字串 → list[int]
def to_int_list(s):
    """把 '1,2,3' → [1,2,3]；把非整數 token (如 'ban1') 直接過濾掉"""
    if not isinstance(s, str) or s.strip() == "":
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.append(int(tok))
    return out

df["teampicks"] = df["teampicks"].apply(to_int_list)
df["teambans"]  = df["teambans"]. apply(to_int_list)

# champion label 編碼
le = LabelEncoder().fit(df["label"])
df["label_enc"] = le.transform(df["label"])
NUM_CLASS = len(le.classes_)

##############################################
# 2. 特徵工程：將 5 picks + 5 bans padding 成長度 5
##############################################
MAX_PICK = 5
feat_cols = []

def pad(lst, max_len=MAX_PICK):
    return lst + [0]*(max_len-len(lst))

for i in range(MAX_PICK):
    df[f"pick_{i+1}"] = df["teampicks"].apply(lambda l: pad(l)[i])
    df[f"ban_{i+1}"]  = df["teambans"]. apply(lambda l: pad(l)[i])
    feat_cols.extend([f"pick_{i+1}", f"ban_{i+1}"])

df["pick_order"] = df["pick_order"].astype(np.int8)
feat_cols.append("pick_order")

X = df[feat_cols]
y = df["label_enc"]
groups = df["matchid"]        # 防洩漏：同場資料同 fold

##############################################
# 3. Optuna 超參搜尋 + 5 fold GroupKFold
##############################################
def objective(trial):
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": NUM_CLASS,
        "learning_rate": trial.suggest_float("lr", 0.02, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255, step=16),
        "feature_fraction": trial.suggest_float("ff", 0.6, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data", 20, 200),
        "verbosity": -1,
        "seed": 42
    }

    cv_acc, cv_top5 = [], []
    gkf = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf.split(X, y, groups):
        lgb_train = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        lgb_val   = lgb.Dataset(X.iloc[val_idx],   label=y.iloc[val_idx])

        model = lgb.train(
    params,
    lgb_train,
    num_boost_round=100,
    valid_sets=[lgb_val],
    callbacks=[
        lgb.early_stopping(stopping_rounds=20),
        lgb.log_evaluation(period=10)   # 相當於 verbose_eval=False
    ]
)

        pred = model.predict(X.iloc[val_idx])
        cv_acc.append(accuracy_score(y.iloc[val_idx], np.argmax(pred, 1)))
        cv_top5.append(top_k_accuracy_score(y.iloc[val_idx], pred, k=5))

    trial.set_user_attr("acc",   np.mean(cv_acc))
    trial.set_user_attr("top5",  np.mean(cv_top5))
    return 1 - np.mean(cv_top5)      # 以 top-5 Accuracy 做優化

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30, timeout=1800)   # 30 次 or 30 分鐘

best_params = study.best_trial.params
best_params.update({"objective":"multiclass","metric":"multi_logloss",
                    "num_class":NUM_CLASS,"verbosity":-1,"seed":42})

##############################################
# 4. 用最佳參數全資料再訓練一次
##############################################
train_set = lgb.Dataset(X, label=y)
final_model = lgb.train(best_params, train_set, num_boost_round=study.best_trial.user_attrs.get("best_iter", 150))

##############################################
# 5. 儲存模型 & 報告
##############################################
final_model.save_model(str(MODEL / "pick_lgb.txt"))
joblib.dump(le, MODEL / "pick_encoder.pkl")

with open(REPORT / "pick_cv.txt", "w", encoding="utf-8") as f:
    t = study.best_trial
    f.write(f"Best top-1 ACC  : {t.user_attrs['acc']:.4f}\n")
    f.write(f"Best top-5 ACC : {t.user_attrs['top5']:.4f}\n")
    f.write(f"Params         : {best_params}\n")

print("🎉 模型訓練完成！")
print("  ➜ 模型檔  :", MODEL / 'pick_lgb.txt')
print("  ➜ Encoder :", MODEL / 'pick_encoder.pkl')
print("  ➜ 報告     :", REPORT / 'pick_cv.txt')
