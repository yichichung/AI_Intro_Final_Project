import requests
import asyncio
import aiohttp
import time
import pandas as pd
from collections import Counter


API_KEY = ""
riot_id = "Hikaru"
tag_line = "0706"

headers = {
    "X-Riot-Token": API_KEY
}

def get_champion_id_name_map():
    url = "https://ddragon.leagueoflegends.com/cdn/14.9.1/data/en_US/champion.json"
    res = requests.get(url)
    if res.status_code != 200:
        print("❌ Failed to load champion list")
        return {}
    data = res.json()["data"]
    return {int(info["key"]): name for name, info in data.items()}

def get_puuid(riot_id, tag_line):
    url = f"https://asia.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{riot_id}/{tag_line}"
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        return res.json()["puuid"]
    else:
        print("❌ Failed to get PUUID:", res.json())
        return None

def get_match_ids(puuid, total=300):
    all_ids = []
    for start in range(0, total, 100):
        url = f"https://sea.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start={start}&count=100"
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            all_ids.extend(res.json())
        else:
            print(f"❌ Failed to fetch matches from {start}")
            break
        time.sleep(1)
    return all_ids

# 非同步下載所有 match details
async def fetch_match(session, match_id, sem):
    url = f"https://sea.api.riotgames.com/lol/match/v5/matches/{match_id}"
    async with sem:
        async with session.get(url, headers=headers) as res:
            if res.status == 200:
                return await res.json()
            else:
                return None

async def fetch_all_matches(match_ids):
    sem = asyncio.Semaphore(10)  # 同時最多 10 個連線
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
                    "kills": p["kills"],
                    "deaths": p["deaths"],
                    "assists": p["assists"],
                    "win": p["win"]
                })
                break
    return pd.DataFrame(records)

def analyze_enemy_champions_on_losses(puuid, match_data):
    losses = []
    for match in match_data:
        if not match:
            continue
        info = match["info"]
        team_id = None
        player_lost = False
        for p in info["participants"]:
            if p["puuid"] == puuid:
                team_id = p["teamId"]
                player_lost = not p["win"]
                break
        if not player_lost:
            continue
        enemy_team_id = 100 if team_id == 200 else 200
        for p in info["participants"]:
            if p["teamId"] == enemy_team_id:
                losses.append(p["championName"])
    counter = Counter(losses)
    df = pd.DataFrame(counter.items(), columns=["champion", "loss_count"])
    return df.sort_values(by="loss_count", ascending=False).reset_index(drop=True)

def summarize(df):
    summary = df.groupby("champion").agg(
        games_played=("win", "count"),
        wins=("win", "sum"),
        avg_kills=("kills", "mean"),
        avg_deaths=("deaths", "mean"),
        avg_assists=("assists", "mean")
    ).reset_index()
    summary["win_rate"] = (summary["wins"] / summary["games_played"] * 100).round(2)
    return summary.sort_values(by="games_played", ascending=False)

# === 主執行區 ===
if __name__ == "__main__":
    champion_map = get_champion_id_name_map()
    puuid = get_puuid(riot_id, tag_line)

    if puuid:
        print("🔍 Getting match list...")
        match_ids = get_match_ids(puuid, total=300)

        print("⚡ Fetching match details asynchronously...")
        match_data = asyncio.run(fetch_all_matches(match_ids))

        print("📊 Analyzing player stats...")
        df = collect_player_stats(puuid, match_data)
        result = summarize(df)
        result.to_csv("hikaru_300games_summary.csv", index=False)

        print("🛡️ Analyzing most lost-to enemy champions...")
        loss_df = analyze_enemy_champions_on_losses(puuid, match_data)
        loss_df.to_csv("hikaru_most_lost_to.csv", index=False)

        print("✅ Done! Two files saved:")
        print(" - hikaru_300games_summary.csv")
        print(" - hikaru_most_lost_to.csv")
    else:
        print("❌ Failed to get puuid.")
