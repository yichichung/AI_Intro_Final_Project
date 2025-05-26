#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
search.py - 查詢某場比賽的隊伍陣容與推薦下一隻 pick（v3）
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
model = lgb.Booster(model_file=str(MODEL / "pick_lgb_v3.txt"))
encoder = joblib.load(MODEL / "pick_encoder_v3.pkl")

# 載入資料集
df = pd.read_parquet(OUTPUT / "pick_dataset.parquet")


def predict_pick(row):
    def pad(lst, l=5): return lst + [0]*(l - len(lst))
    picks = [int(x) for x in row["teampicks"].split(",") if x.isdigit()]
    bans = [int(x) for x in row["teambans"].split(",") if x.isdigit()]
    x = {}
    for i in range(5):
        x[f"pick_{i+1}"] = pad(picks)[i]
        x[f"ban_{i+1}"] = pad(bans)[i]
        x[f"pick_wr{i+1}"] = 0.5
        x[f"ban_wr{i+1}"] = 0.5
    x.update({
        "pick_order": row["pick_order"],
        "role_code": row["role_code"],
        "side_code": row["side_code"]
    })
    x = pd.DataFrame([x])
    prob = model.predict(x)[0]
    top5 = np.argsort(prob)[::-1][:5]
    return encoder.inverse_transform(top5), prob[top5]


# 測試用
target_match, target_team = 123456, 100
row = df[(df["matchid"] == target_match) & (
    df["teamid"] == target_team)].iloc[0]
reco, prob = predict_pick(row)

print(f"🏷️ 該局已選：{row['teampicks']}")
print(f"🚫 已禁用：{row['teambans']}")
print("🎯 推薦 Top-5 Picks：")
for champ, p in zip(reco, prob):
    print(f"  → {champ:15} | 機率：{p:.4f}")
