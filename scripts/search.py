#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
search.py — 指定當前 team picks + bans，列出推薦 top-k champion picks
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from lightgbm import Booster

# ---------------- CLI ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--team", type=str, required=True,
                help="目前已選擇的 champion ids，用逗號隔開（例如 1,11,64）")
ap.add_argument("--ban", type=str, default="",
                help="目前 ban 的 champion ids（例如 117,103,157）")
ap.add_argument("--topk", type=int, default=5,
                help="要顯示的推薦數量")
args = ap.parse_args()

# ---------------- Load ----------------
ROOT = Path(__file__).resolve().parents[1]
model = Booster(model_file=ROOT / "models/pick_lgb_v3.txt")
le = joblib.load(ROOT / "models/pick_encoder_v3.pkl")
wr_tbl = pd.read_parquet(ROOT / "output/champ_wr.parquet")
champs = pd.read_csv(
    ROOT / "data/champs.csv").set_index("id")["name"].to_dict()

# ---------------- Prepare Input Row ----------------
picks = [int(x) for x in args.team.split(",") if x.strip().isdigit()]
bans = [int(x) for x in args.ban.split(",") if x.strip().isdigit()]
picks = picks[:5] + [0] * (5 - len(picks))
bans = bans[:5] + [0] * (5 - len(bans))

row = {}
for i in range(5):
    row[f"pick_{i+1}"] = picks[i]
    row[f"ban_{i+1}"] = bans[i]
    row[f"pick_wr{i+1}"] = wr_tbl.loc[picks[i],
                                      "wr_blue"] if picks[i] in wr_tbl.index else 0.5
    row[f"ban_wr{i+1}"] = 0.5
row["pick_order"] = len([x for x in picks if x > 0]) + 1
row["role_code"] = -1
row["side_code"] = 0

X = pd.DataFrame([row])

# ---------------- Predict & Rank ----------------
pred = model.predict(X)[0]
topk = np.argsort(pred)[::-1]

print("🏷️ 該局已選：")
for pid in picks:
    if pid != 0:
        print(f"  → {pid:>5} | {champs.get(pid, '未知英雄'):10}")

print("🚫 已禁用：" + ",".join(map(str, bans)))
print(f"🎯 推薦 Top-{args.topk} Picks：")
cnt = 0
for i in topk:
    cid = le.inverse_transform([i])[0]
    if cid in picks or cid in bans:
        continue
    name = champs.get(cid, "未知英雄")
    prob = f"{pred[i]*100:.2f}%"
    print(f"  → {cid:>5} | {name:10} | 機率：{prob}")
    cnt += 1
    if cnt >= args.topk:
        break
