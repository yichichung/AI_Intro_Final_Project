import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# 路徑設定
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
# 含 bluepicks, redpicks, win, version, duration, objectives
MDF = pd.read_parquet(OUT / "match_dataset.parquet",
                      engine="pyarrow", use_nullable_dtypes=True)
# champ_wr.parquet 中索引為 champion id
WR = pd.read_parquet(OUT / "champ_wr.parquet")

# helpers


def top5(s):
    """將 'a,b,c,…' → 前 5 個 int list + 0 padding"""
    arr = [int(x) for x in str(s).split(",") if x.isdigit()][:5]
    return arr + [0] * (5 - len(arr))


def wr_of(cid, side):
    """回傳 champion 在指定 side 的平均勝率"""
    return WR.loc[cid, f"wr_{side}"] if cid in WR.index else 0.5


# 1) 展開 blue/red picks 及其 avg-winrate
for i in range(5):
    MDF[f"bp{i}"] = MDF["bluepicks"].apply(lambda s: top5(s)[i])
    MDF[f"rp{i}"] = MDF["redpicks"].apply(lambda s: top5(s)[i])
    MDF[f"bp_wr{i}"] = MDF[f"bp{i}"].apply(lambda c: wr_of(c, "blue"))
    MDF[f"rp_wr{i}"] = MDF[f"rp{i}"].apply(lambda c: wr_of(c, "red"))
# 2) patch & duration & objectives
MDF["patch"] = MDF["version"].astype(str).str.extract(
    r"(\d+\.\d+)").astype(float).fillna(0.0)
for col in ["baronkills", "dragonkills", "towerkills"]:
    if col not in MDF.columns:
        MDF[col] = 0

# 3) 計算 pair-synergy 特徵
pair_cnt, pair_win = {}, {}
for bstr, rstr, win in MDF[["bluepicks", "redpicks", "win"]].itertuples(index=False):
    for side, s in (("blue", bstr), ("red", rstr)):
        arr = [int(x) for x in str(s).split(",") if x.isdigit()][:5]
        for a, b in combinations(arr, 2):
            k = (min(a, b), max(a, b), side)
            pair_cnt[k] = pair_cnt.get(k, 0) + 1
            pair_win[k] = pair_win.get(k, 0) + win
pair_wr = {k: pair_win[k]/pair_cnt[k] for k in pair_cnt}


def mean_syn(row, col):
    arr = [int(x) for x in str(row[col]).split(",") if x.isdigit()][:5]
    side = "blue" if "blue" in col else "red"
    vals = [pair_wr.get((min(a, b), max(a, b), side), 0.5)
            for a, b in combinations(arr, 2)]
    return float(np.mean(vals)) if vals else 0.5


MDF["syn_blue"] = MDF.apply(lambda r: mean_syn(r, "bluepicks"), axis=1)
MDF["syn_red"] = MDF.apply(lambda r: mean_syn(r, "redpicks"),  axis=1)
# 4) 建立特徵矩陣 X, y
cat_cols = [f"bp{i}" for i in range(5)] + [f"rp{i}" for i in range(5)]
num_cols = [f"bp_wr{i}" for i in range(5)] + [f"rp_wr{i}" for i in range(5)] + \
           ["patch", "duration", "baronkills", "dragonkills",
               "towerkills", "syn_blue", "syn_red"]

X = pd.concat([
    MDF[cat_cols].astype("int16"),
    MDF[num_cols].astype("float32")
], axis=1)
y = MDF["win"].astype(int)

# 5) 3-fold CV 訓練 & 評估
kf, aucs = KFold(3, shuffle=True, random_state=42), []
for tr_idx, va_idx in kf.split(X):
    mdl = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        num_leaves=128,
        n_estimators=500,
        feature_fraction=0.9,
        min_data_in_leaf=200,
        device_type="cpu",
        seed=42,
        categorical_feature=[X.columns.get_loc(c) for c in cat_cols]
    )
    mdl.fit(
        X.iloc[tr_idx], y.iloc[tr_idx],
        eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    proba = mdl.predict_proba(
        X.iloc[va_idx], num_iteration=mdl.best_iteration_)[:, 1]
    aucs.append(roc_auc_score(y.iloc[va_idx], proba))

mean_auc = float(np.mean(aucs))
print(f"✅  WR v5 AUC = {mean_auc:.4f}")
# 6) 儲存模型與報告
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# 取最後一折的 booster
mdl.booster_.save_model(str(MODEL_DIR/"wr_lgb_v5.txt"))
with open(REPORT_DIR/"wr_cv_v5.txt", "w") as f:
    f.write(f"AUC: {mean_auc:.4f}\n")
