#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_preprocess_v3.py  ‒  FINAL ROBUST VERSION
------------------------------------------------
* 完全避免 pivot duplicate 與 parquet Segfault
* 產生：
    output/pick_dataset.parquet
    output/match_dataset.parquet  (含 blue/red picks & roles)
* 若 pyarrow 寫檔失敗，自動 fallback → fastparquet / CSV
* 允許 `SKIP_MATCH=1` 環境變數僅建 pick 資料集
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

# ---------- helper ----------
RENAME = {"match_id": "matchid", "matchId": "matchid", "team_id": "teamid", "teamId": "teamid",
          "champion_id": "championid", "championId": "championid"}


def unify(df: pd.DataFrame):
    return df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})


def safe_to_parquet(df: pd.DataFrame, path: Path):
    """pyarrow → fastparquet → csv fallback"""
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        try:
            df.to_parquet(path, index=False, engine="fastparquet")
        except Exception:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            print(
                f"⚠️  pyarrow & fastparquet failed → wrote CSV: {csv_path.name}")
        else:
            print(f"⚠️  pyarrow failed, fastparquet ok → {path.name}")
    else:
        print(f"✅ {path.name} written (pyarrow)")


# ---------- load ----------
bans = unify(pd.read_csv(RAW/"teambans.csv"))
stats = unify(pd.read_csv(RAW/"teamstats.csv"))
parts = unify(pd.read_csv(RAW/"participants.csv"))
matches = unify(pd.read_csv(RAW/"matches.csv"))
if "matchid" not in matches.columns and "id" in matches.columns:
    matches = matches.rename(columns={"id": "matchid"})

# ---------- bans pivot ----------
bans = (bans.sort_values("championid")
            .drop_duplicates(subset=["matchid", "teamid", "banturn"], keep="last")
            .pivot(index=["matchid", "teamid"], columns="banturn", values="championid")
            .rename(columns=lambda x: f"ban{x}")
            .reset_index().fillna(0).astype(int))
BAN_MAP = bans.set_index(["matchid", "teamid"])[
    [f"ban{i}" for i in range(1, 6)]].to_dict("index")

# ---------- participants ----------
parts["role"] = parts["role"].str.upper().replace(
    {"DUO_SUPPORT": "SUPPORT", "DUO_CARRY": "ADC"})
ROLE_MAP = {"TOP": 0, "JUNGLE": 1, "MID": 2, "ADC": 3, "SUPPORT": 4}
parts["role_code"] = parts["role"].map(ROLE_MAP).fillna(-1).astype("int8")
parts["side_code"] = np.where(parts.get("player", parts.index) % 10 < 5, 0, 1)
parts["teamid"] = np.where(parts["side_code"] == 0, 100, 200)
parts["pick_order"] = parts.groupby("matchid").cumcount()+1

# ---------- matches ----------
matches["patch"] = matches["version"].astype(
    str).str.extract(r"(\d+\.\d+)").astype(float)
matches = matches[matches["patch"] >= 10.24]

# ---------- build match_dataset ----------
SKIP_MATCH = bool(int(os.getenv("SKIP_MATCH", "0")))
if not SKIP_MATCH:
    match_ds = stats.merge(bans, on=["matchid", "teamid"], how="left")
    match_ds = match_ds.merge(
        matches[["matchid", "patch", "duration"]], on="matchid", how="left")

    # add blue/red picks & roles
    crew = parts.sort_values("pick_order").groupby(["matchid", "side_code"]).agg({
        "championid": lambda s: ",".join(map(str, s)),
        "role_code": lambda s: ",".join(map(str, s))
    }).reset_index()
    blues = crew[crew["side_code"] == 0].set_index("matchid")
    reds = crew[crew["side_code"] == 1].set_index("matchid")
    match_ds["bluepicks"] = match_ds["matchid"].map(blues["championid"])
    match_ds["blueroles"] = match_ds["matchid"].map(blues["role_code"])
    match_ds["redpicks"] = match_ds["matchid"].map(reds["championid"])
    match_ds["redroles"] = match_ds["matchid"].map(reds["role_code"])

    safe_to_parquet(match_ds, OUT/"match_dataset.parquet")
else:
    print("⚠️  SKIP_MATCH=1 → match_dataset skipped")

# ---------- build pick_dataset ----------
rows = []
for (mid, tid), grp in tqdm(parts.groupby(["matchid", "teamid"]), desc="build pick_df"):
    grp = grp.sort_values("pick_order")
    picked = []
    bans_now = BAN_MAP.get((mid, tid), {f"ban{i}": 0 for i in range(1, 6)})
    bans_tuple = tuple(bans_now.values())
    for _, r in grp.iterrows():
        rows.append({
            "matchid": mid,
            "teamid": tid,
            "pick_order": r["pick_order"],
            "teampicks": tuple(picked),
            "teambans": bans_tuple,
            "role_code": r["role_code"],
            "side_code": r["side_code"],
            "label": r["championid"]
        })
        picked.append(r["championid"])

pick_df = pd.DataFrame(rows)
pick_df["teampicks"] = pick_df["teampicks"].apply(
    lambda t: ",".join(map(str, t)))
pick_df["teambans"] = pick_df["teambans"].apply(
    lambda t: ",".join(map(str, t)))

safe_to_parquet(pick_df, OUT/"pick_dataset.parquet")

print("\n🎯 Preprocess finished without crash")
