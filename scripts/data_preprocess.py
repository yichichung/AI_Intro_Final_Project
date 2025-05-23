#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清理原始資料 → 生成 pick_dataset.parquet & match_dataset.parquet
假設欄位名稱可能為 team_id / teamId → 統一 rename 成 teamid
"""

import pandas as pd, numpy as np
from pathlib import Path
from tqdm import tqdm

RAW = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parents[1] / "output"
OUT.mkdir(exist_ok=True)

# ---------- 1. 讀檔 ----------
bans  = pd.read_csv(RAW / "teambans.csv")
stats = pd.read_csv(RAW / "teamstats.csv")
parts = pd.read_csv(RAW / "participants.csv")
matches = pd.read_csv(RAW / "matches.csv")

# ---------- 2. 欄位名稱標準化 ----------
def unify_columns(df):
    rename_map = {
        "match_id": "matchid",
        "matchId":  "matchid",
        "team_id":  "teamid",
        "teamId":   "teamid",
        "champion_id": "championid",
        "championId":  "championid"
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

bans    = unify_columns(bans)
stats   = unify_columns(stats)
parts   = unify_columns(parts)
matches = unify_columns(matches)

# ---------- 3. bans 處理 → ban1~ban5 ----------
bans = (bans.drop_duplicates()
            .fillna({"championid": 0})
            .astype({"matchid": int, "teamid": int,
                     "championid": int, "banturn": int}))
bans = (bans
        .pivot(index=["matchid", "teamid"],
               columns="banturn", values="championid")
        .rename(columns=lambda x: f"ban{x}")
        .reset_index()
        .fillna(0).astype(int))

# ---------- 4. stats 處理 ----------
for col in ["goldat10", "xpat10", "csat10"]:
    if col in stats.columns:
        q1, q3 = stats[col].quantile([.25, .75])
        iqr = q3 - q1
        stats[col] = stats[col].where(stats[col].between(q1-1.5*iqr, q3+1.5*iqr),
                                      stats[col].median())

# ---------- 5. participants 處理 ----------
parts["role"] = (parts["role"].str.upper()
                 .replace({"DUO_SUPPORT": "SUPPORT", "DUO_CARRY": "ADC"}))

# 推斷 pick_order 和 side（若沒有 player/id 欄位可用 participant_id）
if "player" in parts.columns:
    parts["side"] = np.where(parts["player"] <= 5, "BLUE", "RED")
    parts["teamid"] = np.where(parts["player"] <= 5, 100, 200)
elif "participant_id" in parts.columns:
    parts["side"] = np.where(parts["participant_id"] <= 5, "BLUE", "RED")
    parts["teamid"] = np.where(parts["participant_id"] <= 5, 100, 200)
else:
    parts["side"] = "UNKNOWN"
    parts["teamid"] = 999

# pick_order 若沒有 id 欄位就用 cumcount
if "id" in parts.columns:
    parts["pick_order"] = parts.groupby("matchid")["id"].rank("dense").astype(int)
else:
    parts["pick_order"] = parts.groupby("matchid").cumcount() + 1

# ---------- 6. matches 處理 ----------
matches["patch"] = matches["version"].astype(str).str.extract(r"(\d+\.\d+)").astype(float)
matches = matches[matches["patch"] >= 10.24]

# ---------- 7. merge 成 match_dataset ----------
match_ds = (stats
            .merge(bans, on=["matchid", "teamid"])
            .merge(matches[["id", "patch", "duration"]],
                   left_on="matchid", right_on="id", how="left")
            .drop(columns=["id"]))
match_ds.to_parquet(OUT / "match_dataset.parquet", index=False)

# ---------- 8. 展開成 pick_dataset ----------
rows = []
ban_map = (bans.set_index(["matchid", "teamid"])
                [[f"ban{i}" for i in range(1, 6)]]
                .to_dict("index"))

for (mid, tid), grp in tqdm(parts.groupby(["matchid", "teamid"]),
                            desc="build picks"):
    grp = grp.sort_values("pick_order")
    picked = []
    bans_now = tuple(ban_map.get((mid, tid), [0,0,0,0,0]))
    for _, r in grp.iterrows():
        rows.append({
            "matchid": mid,
            "teamid": tid,
            "pick_order": r["pick_order"],
            "teampicks": tuple(picked),
            "teambans": bans_now,
            "label": r["championid"]
        })
        picked.append(r["championid"])

pick_df = pd.DataFrame(rows)

# ➜ 把 tuple 轉成逗號字串，Parquet 才吃得下
pick_df["teampicks"] = pick_df["teampicks"].apply(lambda t: ",".join(map(str, t)))
pick_df["teambans"]  = pick_df["teambans"].apply(lambda t: ",".join(map(str, t)))

pick_df.to_parquet(OUT / "pick_dataset.parquet", index=False)
print("✅ 輸出完成 → output/")

