#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_preprocess_v2.py — 產生
  • output/pick_dataset.parquet
  • output/match_dataset.parquet（含 win）
  • output/champ_wr.parquet      （英雄在藍/紅基礎勝率）
"""
import pandas as pd
import numpy as np
import gc
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data", ROOT / "output"
OUT.mkdir(exist_ok=True)

# ---------- 工具 ----------
REN = {"match_id": "matchid", "matchId": "matchid",
       "team_id": "teamid", "teamId": "teamid",
       "champion_id": "championid", "championId": "championid"}


def uni(df): return df.rename(
    columns={k: v for k, v in REN.items() if k in df.columns})


def to_parq(df, p): df.to_parquet(p, index=False)


# ---------- 載入 ----------
bans = uni(pd.read_csv(RAW/"teambans.csv"))
teamstats = uni(pd.read_csv(RAW/"teamstats.csv"))
parts = uni(pd.read_csv(RAW/"participants.csv"))
matches = uni(pd.read_csv(RAW/"matches.csv"))
if "matchid" not in matches and "id" in matches:
    matches = matches.rename(columns={"id": "matchid"})

# ---------- participants 基礎欄 ----------
parts["side_code"] = np.where(parts.get("player", parts.index) % 10 < 5, 0, 1)
parts["teamid"] = np.where(parts.side_code == 0, 100, 200)
parts["pick_order"] = parts.groupby("matchid").cumcount()+1
parts["role"] = parts["role"].str.upper().replace(
    {"DUO_SUPPORT": "SUPPORT", "DUO_CARRY": "ADC"})
ROLE = {"TOP": 0, "JUNGLE": 1, "MID": 2, "ADC": 3, "SUPPORT": 4}
parts["role_code"] = parts["role"].map(ROLE).fillna(-1).astype("int8")

# ---------- 1. 從 stats1 / stats2 取 win ----------
win_src = []
for fn in ["stats1.csv", "stats2.csv"]:
    fp = RAW/fn
    if fp.exists():
        win_src.append(pd.read_csv(fp, usecols=["id", "win"]))
if not win_src:
    raise RuntimeError("❌ 無 stats1/2.csv → 無 win 資料")

win_raw = pd.concat(win_src, ignore_index=True)
parts_sub = parts[["id", "matchid", "teamid"]]
win_df = win_raw.merge(parts_sub, on="id", how="left")[
    ["matchid", "teamid", "win"]].dropna()
win_df["win"] = win_df["win"].astype(int)
print("✅ win rows =", len(win_df))

# ---------- 2. bans pivot ----------
bans = (bans.drop_duplicates(["matchid", "teamid", "banturn"])
            .pivot(index=["matchid", "teamid"], columns="banturn", values="championid")
            .rename(columns=lambda x: f"ban{x}")
            .reset_index().fillna(0).astype(int))
BAN = bans.set_index(["matchid", "teamid"]).to_dict("index")

# ---------- 3. match_dataset ----------
match_ds = (teamstats
            .merge(bans, on=["matchid", "teamid"], how="left")
            .merge(matches[["matchid", "version", "duration"]], on="matchid", how="left")
            .merge(win_df, on=["matchid", "teamid"], how="left"))
match_ds["win"] = match_ds["win"].fillna(0).astype(int)

# blue / red picks & roles
crew_c = parts.groupby(["matchid", "side_code"])[
    "championid"].apply(lambda s: ",".join(map(str, s)))
crew_r = parts.groupby(["matchid", "side_code"])[
    "role_code"].apply(lambda s: ",".join(map(str, s)))
crew = pd.concat([crew_c, crew_r], axis=1).reset_index()
bl, rd = crew[crew.side_code == 0].set_index(
    "matchid"), crew[crew.side_code == 1].set_index("matchid")
match_ds["bluepicks"] = match_ds["matchid"].map(bl["championid"])
match_ds["blueroles"] = match_ds["matchid"].map(bl["role_code"])
match_ds["redpicks"] = match_ds["matchid"].map(rd["championid"])
match_ds["redroles"] = match_ds["matchid"].map(rd["role_code"])
to_parq(match_ds.fillna({"bluepicks": "", "blueroles": "", "redpicks": "", "redroles": ""}),
        OUT/"match_dataset.parquet")

# ---------- 4. pick_dataset ----------
rows = []
for (mid, tid), g in tqdm(parts.groupby(["matchid", "teamid"]), desc="build pick"):
    g = g.sort_values("pick_order")
    picked = []
    for _, r in g.iterrows():
        rows.append({"matchid": mid, "teamid": tid, "pick_order": r.pick_order,
                     "teampicks": ",".join(map(str, picked)),
                     "teambans": ",".join(map(str, BAN.get((mid, tid), {}).values())),
                     "role_code": r.role_code, "side_code": r.side_code, "label": r.championid})
        picked.append(r.championid)
pd.DataFrame(rows).to_parquet(OUT/"pick_dataset.parquet", index=False)

# ---------- 5. 英雄基礎勝率 (分批避免 OOM/segfault) ----------
chunk = 200_000
cols = ["cid", "side", "win"]
accum = []


def append_chunk(df_chunk):
    df_chunk["cid"] = df_chunk["cid"].fillna("").astype(str)
    df_chunk = df_chunk[df_chunk["cid"].str.isdigit()]
    df_chunk["cid"] = df_chunk["cid"].astype(int)
    accum.append(df_chunk[cols])


tmp = match_ds[["matchid", "teamid", "win", "bluepicks", "redpicks"]].copy()
tmp["bluepicks"] = tmp["bluepicks"].str.split(",")
tmp["redpicks"] = tmp["redpicks"].str.split(",")

# 分批 explode
for side in ["blue", "red"]:
    col = "bluepicks" if side == "blue" else "redpicks"
    base = tmp[[col, "win"]].rename(columns={col: "cid"}).assign(side=side)
    for start in range(0, len(base), chunk):
        append_chunk(base.iloc[start:start+chunk].explode("cid"))

cat = pd.concat(accum, ignore_index=True)
del accum, tmp
gc.collect()
wr_tbl = (cat.pivot_table(index="cid", columns="side", values="win", aggfunc="mean")
          .fillna(0.5)
          .rename(columns={"blue": "wr_blue", "red": "wr_red"}))
to_parq(wr_tbl, OUT/"champ_wr.parquet")

print("🎯 Preprocess 完成，win 分布：")
print(match_ds["win"].value_counts())
