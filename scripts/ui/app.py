# working version
"""
Streamlit App — BanPick 推薦 UI（包含 AI 推薦 + 勝率預測 + Red 方輸入加權）
"""

import streamlit as st
st.set_page_config(layout="wide", page_title="LoL BanPick 控制介面")
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
import asyncio
import aiohttp
import requests
import time
from collections import Counter


# ---------- 使用者輸入 Riot ID ----------
st.markdown("### 🔍 輸入 Riot ID 以進行個人化推薦")
riot_id = st.text_input("Riot ID", value="Hikaru", key="riot_input")
tag_line = st.text_input("Tag Line", value="0706", key="tag_input")
API_KEY = "RGAPI-c4e80e58-0ae6-4a6d-96f8-6999b2976c9b"
headers = {"X-Riot-Token": API_KEY}

@st.cache_data
def get_champion_id_name_map():
    url = "https://ddragon.leagueoflegends.com/cdn/14.9.1/data/en_US/champion.json"
    res = requests.get(url)
    if res.status_code != 200:
        st.error("❌ Failed to load champion list")
        return {}
    data = res.json()["data"]
    return {int(info["key"]): name for name, info in data.items()}

@st.cache_data
def get_puuid(riot_id, tag_line):
    url = f"https://asia.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{riot_id}/{tag_line}"
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        return res.json()["puuid"]
    return None

def get_match_ids(puuid, total=50):
    all_ids = []
    for start in range(0, total, 100):
        url = f"https://sea.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start={start}&count=100"
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            all_ids.extend(res.json())
        else:
            break
        time.sleep(1)
    return all_ids

async def fetch_match(session, match_id, sem):
    url = f"https://sea.api.riotgames.com/lol/match/v5/matches/{match_id}"
    async with sem:
        async with session.get(url, headers=headers) as res:
            if res.status == 200:
                return await res.json()
            return None

async def fetch_all_matches(match_ids):
    sem = asyncio.Semaphore(10)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_match(session, mid, sem) for mid in match_ids]
        return await asyncio.gather(*tasks)

def collect_player_stats(puuid, match_data):
    records = []
    for match in match_data:
        if not match:
            continue
        for p in match["info"]["participants"]:
            if p["puuid"] == puuid:
                records.append({
                    "champion": p["championName"],
                    "win": p["win"]
                })
                break
    return pd.DataFrame(records)

@st.cache_data
def get_personal_winrates(riot_id, tag_line):
    puuid = get_puuid(riot_id, tag_line)
    if not puuid:
        return {}
    match_ids = get_match_ids(puuid, total=50)
    match_data = asyncio.run(fetch_all_matches(match_ids))
    df = collect_player_stats(puuid, match_data)
    if df.empty:
        return {}
    win_df = df.groupby("champion").agg(count=("win", "count"), winrate=("win", "mean")).reset_index()
    boost_map = {}
    for _, row in win_df.iterrows():
        if row["winrate"] > 0.6:
            boost_map[row["champion"]] = 0.1
        elif row["winrate"] < 0.4:
            boost_map[row["champion"]] = -0.1
    return boost_map

# 🔧 取得 boost map（在推薦前呼叫）
champion_map = get_champion_id_name_map()
personal_boost = get_personal_winrates(riot_id, tag_line)


# ---------- 路徑與資料載入 ----------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODEL = ROOT / "models"
OUTPUT = ROOT / "output"

# 讀取英雄名稱對應表
try:
    champs_df = pd.read_csv(DATA / "champs.csv")
    champ_name_map = champs_df.set_index("id")["name"].to_dict()
except Exception as e:
    st.error(f"❌ 無法載入英雄名稱對應表：{DATA / 'champs.csv'}\n錯誤訊息：{e}")
    st.stop()

# 讀取推薦模型（Pick 模型）與編碼器
try:
    pick_model = lgb.Booster(model_file=str(MODEL / "pick_lgb_v3.txt"))
    encoder = joblib.load(MODEL / "pick_encoder_v3.pkl")
except Exception as e:
    st.error(f"❌ 推薦模型或編碼器載入錯誤：{e}")
    st.stop()

# 讀取勝率預測模型（僅使用 CPU-only 版，避免 GPU 版在此環境下崩潰）
try:
    wr_model = lgb.Booster(model_file=str(MODEL / "wr_lgb_v5_cpu.txt"))
    wr_features = wr_model.feature_name()
except Exception as e:
    st.error(f"❌ 勝率預測模型載入錯誤：{e}")
    st.stop()
# ---------- 讀取 merged_output.csv（用於 bp_wr* / rp_wr*） ----------
WR_CSV = ROOT / "scripts" / "merged_output.csv"        # ← 與 app.py 同資料夾
try:
    wr_df = pd.read_csv(WR_CSV)
    # 例：Win % 欄是 "58.11%" → 0.5811
    wr_df["win_rate"] = wr_df["Win %"].str.rstrip("%").astype(float) / 100.0
    WINRATE_LOOKUP = wr_df.set_index("Champion ID")["win_rate"].to_dict()
except Exception as e:
    st.warning(f"⚠️ 無法載入 {WR_CSV.name}：{e}（將使用 0.5 作為預設）")
    WINRATE_LOOKUP = {}

# ---------- 功能函式 ----------


def id_to_name(cid):
    try:
        return champ_name_map.get(int(cid), f"ID {cid}")
    except:
        return f"ID {cid}"


def clean_ids(raw_list):
    cleaned = []
    for item in raw_list:
        try:
            cleaned.append(int(str(item).strip()))
        except:
            continue
    return cleaned


def pad(lst, l=5):
    # 把已有的 IDs 取前 l 個，若不足就補 0
    return (clean_ids(lst) + [0] * l)[:l]


# ---------- 所有英雄選項 ----------
champ_options = sorted([str(cid) for cid in champ_name_map.keys()])

# ---------- multiselect 包裝函式 ----------


def safe_multiselect(label, options, key, filter_ids=None):
    filter_ids = filter_ids or []
    options_cleaned = [cid for cid in options if int(cid) not in filter_ids]
    default_values = [cid for cid in st.session_state.get(
        key, []) if cid in options_cleaned]
    return st.sidebar.multiselect(
        label,
        options=options_cleaned,
        default=default_values,
        key=key,
        format_func=id_to_name
    )


# ---------- 側邊輸入選單 ----------
st.sidebar.header("🔵 藍方 Picks / Bans")
blue_bans = safe_multiselect(
    "藍方 Ban", champ_options, key="bb",
    filter_ids=clean_ids(st.session_state.get("rb", []))
)
blue_picks = safe_multiselect(
    "藍方 Pick", champ_options, key="bp",
    filter_ids=clean_ids(st.session_state.get(
        "rb", []) + st.session_state.get("bb", []))
)

st.sidebar.header("🔴 紅方 Picks / Bans")
red_bans = safe_multiselect(
    "紅方 Ban", champ_options, key="rb",
    filter_ids=clean_ids(st.session_state.get("bb", []))
)
red_picks = safe_multiselect(
    "紅方 Pick", champ_options, key="rp",
    filter_ids=clean_ids(st.session_state.get(
        "bb", []) + st.session_state.get("rb", []))
)

# ---------- 顯示 Ban / Pick 現況 ----------
st.markdown("## 📊 Ban / Pick 現況")


def ids_to_names(id_list):
    return [id_to_name(cid) for cid in clean_ids(id_list)]


ban_blue_list = ids_to_names(st.session_state.get("bb", []))
ban_red_list = ids_to_names(st.session_state.get("rb", []))
pick_blue_list = ids_to_names(st.session_state.get("bp", []))
pick_red_list = ids_to_names(st.session_state.get("rp", []))

max_len = max(len(ban_blue_list), len(ban_red_list),
              len(pick_blue_list), len(pick_red_list))

ban_data = {
    "藍方 Ban": ban_blue_list + [""] * (max_len - len(ban_blue_list)),
    "紅方 Ban": ban_red_list + [""] * (max_len - len(ban_red_list))
}
pick_data_blue = {
    "藍方 Pick": pick_blue_list + [""] * (max_len - len(pick_blue_list))
}
pick_data_red = {
    "紅方 Pick": pick_red_list + [""] * (max_len - len(pick_red_list))
}

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 🚫 Ban 狀況")
    st.table(pd.DataFrame(ban_data))
with col2:
    st.markdown("### 🔵 藍方 Pick 狀況")
    st.table(pd.DataFrame(pick_data_blue))
with col3:
    st.markdown("### 🔴 紅方 Pick 狀況")
    st.table(pd.DataFrame(pick_data_red))

# ---------- 🎯 推薦英雄 Top-5 ----------
blue_pick_ids = clean_ids(st.session_state.get("bp", []))
red_pick_ids = clean_ids(st.session_state.get("rp", []))
ban_ids = clean_ids(st.session_state.get("bb", []) +
                    st.session_state.get("rb", []))
pick_features = pick_model.feature_name()

# 建立輸入給推薦模型的特徵表
x_rec = {}
for i in range(5):
    x_rec[f"pick_{i+1}"] = pad(blue_pick_ids)[i]
    x_rec[f"ban_{i+1}"] = pad(ban_ids)[i]
    # 推薦模型的 wr 欄位沒在用，設為 0
    x_rec[f"pick_wr{i+1}"] = 0
    x_rec[f"ban_wr{i+1}"] = 0
x_rec["pick_order"] = 0
x_rec["role_code"] = 0
x_rec["side_code"] = 0

X_rec = pd.DataFrame([x_rec])
for f in pick_features:
    if f not in X_rec.columns:
        X_rec[f] = 0
X_rec = X_rec[pick_features].astype(np.int32)

# ---------- 🎯 推薦英雄 Top-5 ----------
blue_pick_ids = clean_ids(st.session_state.get("bp", []))
red_pick_ids = clean_ids(st.session_state.get("rp", []))
ban_ids = clean_ids(st.session_state.get("bb", []) +
                    st.session_state.get("rb", []))
pick_features = pick_model.feature_name()

if len(blue_pick_ids) >= 1:
    x_rec = {}
    for i in range(5):
        x_rec[f"pick_{i+1}"] = pad(blue_pick_ids)[i]
        x_rec[f"ban_{i+1}"] = pad(ban_ids)[i]
        x_rec[f"pick_wr{i+1}"] = 0
        x_rec[f"ban_wr{i+1}"] = 0
    x_rec["pick_order"] = 0
    x_rec["role_code"] = 0
    x_rec["side_code"] = 0

    X_rec = pd.DataFrame([x_rec])
    for f in pick_features:
        if f not in X_rec.columns:
            X_rec[f] = 0
    X_rec = X_rec[pick_features].astype(np.int32)

    try:
        probs = pick_model.predict(X_rec)[0]
        for i, cid in enumerate(encoder.classes_):
            cname = champ_name_map.get(cid, "")
            if cname in personal_boost:
                probs[i] += personal_boost[cname]


        topk = np.argsort(probs)[::-1][:5]

        # 🛡️ 防止 topk 超出 encoder 可用範圍
        if np.max(topk) < len(encoder.classes_):
            top_champs = encoder.inverse_transform(topk)

            st.markdown("## 🎯 推薦英雄 Top-5")
            results = []
            for cid, prob in zip(top_champs, probs[topk]):
                name = champ_name_map.get(cid, f"ID {cid}")
                results.append(
                    {"ID": cid, "英雄": name, "預測機率 (%)": f"{prob * 100:.2f}%"})
            st.table(pd.DataFrame(results))
        else:
            st.warning("⚠️ 模型預測結果超出 encoder 範圍，請檢查模型與編碼器是否匹配。")

    except Exception as e:
        st.warning(f"⚠️ 藍方推薦預測錯誤：{e}")
else:
    st.info("請至少選擇一位藍方英雄以啟用推薦功能。")


# 先設定預設：若尚未滿足「雙方至少各有一個 Pick」，就先顯示 50% : 50%
blue_strength = 0.5
red_strength = 0.5
show_wr = False  # 是否要顯示真正從模型計算的勝率

if len(blue_pick_ids) >= 1 and len(red_pick_ids) >= 1:
    # 滿足條件，才真正 build 特徵並呼叫模型
    show_wr = True

        # ---------- 構造 wr 特徵 ----------
    x_wr = {feat: 0 for feat in wr_features}

    for i in range(5):
        cid_b = pad(blue_pick_ids)[i]
        cid_r = pad(red_pick_ids)[i]

        x_wr[f"bp{i}"] = cid_b
        x_wr[f"rp{i}"] = cid_r

        # 從 merged_output.csv 取 win-rate；若查不到 → 0.5
        x_wr[f"bp_wr{i}"] = WINRATE_LOOKUP.get(cid_b, 0.5)
        x_wr[f"rp_wr{i}"] = WINRATE_LOOKUP.get(cid_r, 0.5)

    # 其餘數值特徵 (patch, duration, objectives, synergy) 仍保持 0
    X_wr = pd.DataFrame([x_wr])[wr_features].astype(np.float32)

    st.write("📤 輸入特徵：", X_wr)

    try:
        # model.predict 回傳藍隊勝率（單一 float）
        blue_strength = wr_model.predict(X_wr)[0]
        red_strength = 1.0 - blue_strength
    except Exception as e:
        st.warning(f"⚠️ 勝率預測錯誤：{e}")
        # 若失敗，就保留預設 50%：50%

# 顯示胜率區塊
st.markdown("## ⚔️ 雙方隊伍預估勝率（動態）")
if show_wr:
    st.markdown(f"🔵 藍隊預估勝率：**{blue_strength * 100:.2f}%**")
    st.markdown(f"🔴 紅隊預估勝率：**{red_strength * 100:.2f}%**")
else:
    st.markdown("🔵 藍隊預估勝率：**50.00%**")
    st.markdown("🔴 紅隊預估勝率：**50.00%**")
    st.info("請先讓雙方各自選至少一名英雄 (Pick) 後，才會顯示動態勝率。")

# 用表格顯示原始數值（如果有真實計算出的勝率，就顯示該數值；否則保持 50%）
winrate_data = pd.DataFrame({
    "隊伍": ["藍隊", "紅隊"],
    "預估勝率 (%)": [
        f"{(blue_strength * 100):.2f}%" if show_wr else "50.00%",
        f"{(red_strength * 100):.2f}%" if show_wr else "50.00%"
    ]
})
st.table(winrate_data)