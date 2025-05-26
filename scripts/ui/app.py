#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App — BanPick 推薦 + 勝率預測（v3 格式 + 英雄名稱顯示）
"""

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path

st.set_page_config(layout="wide", page_title="LoL BanPick 推薦器")

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models"
OUTPUT = ROOT / "output"
DATA = ROOT / "data"

# 載入模型與資料
model = lgb.Booster(model_file=str(MODEL / "pick_lgb_v3.txt"))
encoder = joblib.load(MODEL / "pick_encoder_v3.pkl")
df = pd.read_parquet(OUTPUT / "pick_dataset.parquet")

# 載入英雄名稱對應表（改用 id 對應 name）
champs_df = pd.read_csv(DATA / "champs.csv")  # 須有欄位：name, id
champ_name_map = champs_df.set_index("id")["name"].to_dict()

# 所有 champion 選項


def extract_champs(col):
    return [int(x) for s in df[col].fillna("") for x in s.split(",") if x.isdigit()]


all_champs = sorted(set(extract_champs("teampicks")))
champ_options = [str(c) for c in all_champs]

# 側邊選單輸入
st.sidebar.header("🟦 藍方 Picks / Bans")
blue_picks = st.sidebar.multiselect("藍方已選英雄", champ_options, key="bp")
blue_bans = st.sidebar.multiselect("藍方已禁英雄", champ_options, key="bb")

st.sidebar.header("🟥 紅方 Picks / Bans")
red_picks = st.sidebar.multiselect("紅方已選英雄", champ_options, key="rp")
red_bans = st.sidebar.multiselect("紅方已禁英雄", champ_options, key="rb")

st.sidebar.header("⚙️ 額外設定")
pick_order = st.sidebar.number_input("目前 Pick 順序", 0, 9, 0)
role_code = st.sidebar.selectbox("角色代碼", [0, 1, 2, 3, 4], index=0)
side_code = st.sidebar.selectbox(
    "陣營", [("藍方", 0), ("紅方", 1)], format_func=lambda x: x[0])[1]

# ---- 組成輸入向量 ----


def pad(lst, l=5):
    return [int(x) for x in lst] + [0] * (l - len(lst))


x = {}
for i in range(5):
    x[f"pick_{i+1}"] = pad(blue_picks, 5)[i]
    x[f"ban_{i+1}"] = pad(blue_bans + red_bans, 5)[i]
    x[f"pick_wr{i+1}"] = 0.5
    x[f"ban_wr{i+1}"] = 0.5

x["pick_order"] = pick_order
x["role_code"] = role_code
x["side_code"] = side_code

X = pd.DataFrame([x])
probs = model.predict(X)[0]
topk = np.argsort(probs)[::-1][:5]
top_champs = encoder.inverse_transform(topk)

# 顯示結果
st.header("🎯 推薦英雄 Top-5")
results = []
for cid, prob in zip(top_champs, probs[topk]):
    name = champ_name_map.get(cid, f"ID {cid}")
    results.append({"ID": cid, "英雄": name, "預測機率 (%)": f"{prob*100:.2f}%"})

st.table(pd.DataFrame(results))
