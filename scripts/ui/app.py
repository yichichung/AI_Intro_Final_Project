#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Web App for Ban/Pick + Winrate Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from pathlib import Path

st.set_page_config(layout="wide", page_title="LoL BanPick 推薦器")

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models"
OUTPUT = ROOT / "output"

# 載入模型
pick_model = lgb.Booster(model_file=str(MODEL / "pick_lgb_v2.txt"))
pick_encoder = joblib.load(MODEL / "pick_encoder_v2.pkl")
wr_model = lgb.Booster(model_file=str(MODEL / "wr_lgb.txt"))

# 載入資料
pick_df = pd.read_parquet(OUTPUT / "pick_dataset_v2.parquet")
all_champs = sorted({c for picks in pick_df["teampicks"] for c in picks.split(
    ",") if c.isdigit()}, key=int)
champ_options = [str(c) for c in all_champs]
st.sidebar.header("🟦 藍方 Picks / Bans")
blue_picks = st.sidebar.multiselect("藍方已選英雄", champ_options, key="bp")
blue_bans = st.sidebar.multiselect("藍方已禁英雄", champ_options, key="bb")

st.sidebar.header("🟥 紅方 Picks / Bans")
red_picks = st.sidebar.multiselect("紅方已選英雄", champ_options, key="rp")
red_bans = st.sidebar.multiselect("紅方已禁英雄", champ_options, key="rb")

st.sidebar.header("⚙️ 額外設定")
pick_order = st.sidebar.number_input("目前 Pick 順序（0～9）", 0, 9, 0)
role_code = st.sidebar.selectbox("角色代碼", [0, 1, 2, 3, 4], index=0)
side_code = st.sidebar.selectbox(
    "陣營", [("藍方", 0), ("紅方", 1)], format_func=lambda x: x[0])[1]
patch_code = st.sidebar.slider("Patch 版本", 0.0, 20.0, 13.1, step=0.1)
# --- 功能：推薦下一隻 Pick ---
def pad(lst, l=5): return lst + [0] * (l - len(lst))


input_row = {
    **{f"pick_{i+1}": int(pad(blue_picks, 5)[i]) for i in range(5)},
    **{f"ban_{i+1}":  int(pad(blue_bans,  5)[i]) for i in range(5)},
    "pick_order": pick_order,
    "role_code":  role_code,
    "side_code":  side_code,
    "patch_code": patch_code,
    "pick_rate":  0.03,
    "win_rate":   0.5
}
X_pick = pd.DataFrame([input_row])
pred = pick_model.predict(X_pick)[0]
top5_idx = np.argsort(pred)[::-1][:5]
top5_champs = pick_encoder.inverse_transform(top5_idx)

# --- 功能：預測勝率 ---


def to_vector(picks):
    vec = np.zeros(len(champ_options))
    for pid in picks:
        if pid.isdigit():
            vec[int(pid)] = 1
    return vec


blue_vec = to_vector(blue_picks)
red_vec = to_vector(red_picks)
X_wr = np.concatenate([blue_vec, red_vec])[np.newaxis, :]
wr_prob = wr_model.predict(X_wr)[0]

# --- 顯示推薦與勝率 ---
st.header("🎯 推薦英雄 Top-5")
st.write(pd.DataFrame({
    "英雄": top5_champs,
    "預測機率": pred[top5_idx]
}))

st.header("🔵 藍方勝率預測")
st.metric("藍方勝率機率", f"{wr_prob:.2%}")
