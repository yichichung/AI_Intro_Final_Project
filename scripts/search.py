#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
search.py - 查詢某場比賽的隊伍陣容與推薦下一隻 pick
"""
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models"
OUTPUT = ROOT / "output"

# 載入模型與編碼器
model = lgb.Booster(model_file=str(MODEL / "pick_lgb_v2.txt"))
encoder = joblib.load(MODEL / "pick_encoder_v2.pkl")

# 載入資料集
df = pd.read_parquet(OUTPUT / "pick_dataset_v2.parquet")

# 預測函式


def predict_pick(picks, bans, pick_order, role_code, side_code, patch_code, pick_rate=0.03, win_rate=0.5):
    def pad(lst, l=5): return lst + [0]*(l-len(lst))
    x = {}
    for i in range(5):
        x[f"pick_{i+1}"] = pad(picks)[i]
        x[f"ban_{i+1}"] = pad(bans)[i]
    x.update({
        "pick_order": pick_order,
        "role_code": role_code,
        "side_code": side_code,
        "patch_code": patch_code,
        "pick_rate": pick_rate,
        "win_rate": win_rate
    })
    x = pd.DataFrame([x])
    prob = model.predict(x)[0]
    top5 = np.argsort(prob)[::-1][:5]
    return encoder.inverse_transform(top5), prob[top5]


# 示範：查詢某場比賽、某支隊伍
target_match = 123456  # <--- 請改成實際 matchid
target_team = 100     # 藍方是 100，紅方是 200

row = df[(df["matchid"] == target_match) & (
    df["teamid"] == target_team)].iloc[0]
reco, prob = predict_pick(
    picks=row["teampicks"],
    bans=row["teambans"],
    pick_order=row["pick_order"],
    role_code=row["role_code"],
    side_code=row["side_code"],
    patch_code=row["patch_code"],
    pick_rate=row["pick_rate"],
    win_rate=row["win_rate"]
)

print(f"🏷️  該局已選：{row['teampicks']}")
print(f"🚫 已禁用：{row['teambans']}")
print("🎯 推薦 Top-5 Picks：")
for champ, p in zip(reco, prob):
    print(f"  → {champ:15} | 機率：{p:.4f}")
