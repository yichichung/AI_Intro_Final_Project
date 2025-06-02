# Pull Request: Initial Commit from Local Backup with Git LFS

🔧 **Summary**  
This PR includes the full migration of the LoL_BP_Project from local development to GitHub with:

- ✅ Cleaned Git history (previous large-file commits removed)  
- ✅ Proper usage of Git LFS for large files:  
  - `models/pick_lgb_v3.txt`  
  - `data/stats1.csv`  
  - `data/stats2.csv`  
  - Other large `.csv`, `.parquet`, `.pkl` files  
- ✅ Organized directory structure: `data/`, `models/`, `scripts/`, `ui/`, `output/`, `reports/`  
- ✅ Git ignore rules for intermediate files, system cache, and local logs  
- ✅ Dev branch is ready to be merged into `main`  

---

## Overview

本專案設計一套完整的英雄推薦與勝率預測系統，應用於《英雄聯盟》的選角階段（Ban/Pick Phase）。我們使用歷史職業賽資料，訓練模型預測在當前選角狀態下：

- 哪些英雄最可能勝出（Top-k recommendation）
- 目前藍方相對紅方的預測勝率（Win Rate Curve）

此系統已具備完整介面（Streamlit），支援使用者逐步輸入選角順序，並即時回饋推薦與預測勝率。

---

## 🧪 Evaluation vs Spec

| 評分項目                 | 預估得分  | 說明                                                                 |
|------------------------|---------|--------------------------------------------------------------------|
| 🎯 Originality (15%)   | 10–12   | 應用於實際選角，模型融合 pick 組合與統計勝率；若能整合 WR 模型得更高             |
| ⚙️ Difficulty (10%)     | 7       | 中等難度；含模型訓練與推薦，非簡單分類，但可加強應用深度                              |
| 🗂️ Clarity (15%)        | 13–15   | 文件完整、架構清楚；影片說明補足後應達滿分                                      |
| ✅ Completeness (50%)   | 38–45   | 目前尚未整合 WR 預測曲線；若補齊分析與應用層面將更完整                           |

> **預估總分：** 78–89/100（若整合勝率曲線與分析，預估可達 90+）

---

## 🔜 改進方向

| 功能                | 狀態 | 說明                                         |
|-------------------|:----:|--------------------------------------------|
| 勝率預測曲線         | ⛔   | 模型已訓練，尚未整合到 `app.py` 曲線顯示             |
| 推薦邏輯整合 WR      | ⛔   | 可考慮根據 WR 模型挑選「最大化勝率提升」之英雄            |
| 模型多版本切換       | ⛔   | 比較 `pick_v2` / `pick_v3` 效果                  |
| 使用者回饋 / 團隊推薦   | ⛔   | UI 中可納入位置、陣營等更細的互動欄位                  |

---


## 🎯 目標

**直接部署** `app.py`（含勝率曲線）

---

## 📂 需要的專案結構

```bash
LoL_BP_Project/
│
├── models/
│   └── wr_lgb_v4.txt          ⬅️ 英雄勝率模型（由 `train_wr.py` 訓練）
│
├── output/
│   └── champ_wr.parquet       ⬅️ 每隻英雄藍/紅方勝率（由 `data_preprocess_v2.py` 計算）
│
├── data/
│   └── champs.csv             ⬅️ id → 英雄名稱 對應表
│
├── ui/
│   └── app.py                 ⬅️ Streamlit 主程式
```

---

## ✅ 安裝環境（只需一次）

```bash
conda activate lolbp    # 或你的 Python 環境
pip install -U streamlit lightgbm pandas scikit-learn
```

---

## ✅ 下載與準備檔案

| 檔案路徑                         | 來源 / 用途                                        |
|----------------------------------|---------------------------------------------------|
| `models/wr_lgb_v4.txt`           | `train_wr.py` 訓練的 LightGBM 勝率模型             |
| `output/champ_wr.parquet`        | 英雄藍方／紅方平均勝率，用於 pick 模型             |
| `data/champs.csv`                | 英雄 ID → 英文名稱 對應表                         |
| `ui/app.py`                      | Streamlit 前端程式，實作推薦與勝率曲線顯示         |

> ✅ 你可以把整理好的 `app.py` 直接放進 `ui/` 資料夾使用。

---

## ✅ 執行指令

```bash
cd ~/LoL_BP_Project/ui
streamlit run app.py
```

啟動後你會看到：

```
Local URL: http://localhost:8501
Network URL: http://<你的IP>:8501
```

---

## 📂 Git LFS Tracked Files

```bash
git lfs install
```

- `models/pick_lgb_v3.txt`
- `models/pick_encoder_v3.pkl`
- `data/stats1.csv`
- `data/stats2.csv`
- `output/pick_dataset.parquet`
- `output/match_dataset.parquet`
- `output/champ_wr.parquet`

> ⚠️ Contributors should install Git LFS before cloning.

---

## 🗂️ 專案目錄總覽與用途

### 🔹 `data/`  
存放原始 CSV 資料：  
- `matches.csv`, `participants.csv`, `teamstats.csv`, `teambans.csv`, `stats1.csv`, `stats2.csv`  
  OraclesElixir 歷史對戰資料  
- `champs.csv`  
  英雄 ID ↔ 英雄名稱 對應表  

### 🔹 `output/`  
存放前處理後 Parquet 檔案：  
- `match_dataset.parquet`  
- `pick_dataset.parquet`  
- `champ_wr.parquet`  

### 🔹 `models/`  
存放訓練後模型與編碼器：  
- `pick_lgb_v3.txt`  
- `pick_encoder_v3.pkl`  
- `wr_lgb_v4.txt`  

### 🔹 `reports/`  
存放訓練紀錄與評估：  
- `train_pick_v3.log`, `train_pick_v3_continue.log`  
- `pick_cv_v3.txt`  
- `wr_cv_v4.txt`  

### 🔹 `scripts/`  
主要處理腳本：  
- `data_preprocess_v2.py`  
- `train_pick_v3_tqdm.py`  
- `train_wr.py`  
- `search.py`  
- `app.py` (可移至 `ui/`)  

### 🔹 `ui/`  
Streamlit 前端 App：  
- `app.py`  
- `ngrok/` (外網分享工具)  

---

## 🧠 系統功能與已完成目標

1. **機器學習模型**  
   - LightGBM + Optuna 自動調參  
   - Top-5 推薦 accuracy ≈ 67.3%

2. **Ban/Pick Draft 支援**   
   - 自動過濾已選／已禁英雄  

3. **互動式介面 (Streamlit)**  
   - 雙方已選／已禁英雄  
   - 設定步驟、陣營、角色  
   - 顯示 Top-5 推薦與勝率  
   - Ngrok 外網訪問  

---

## 🚀 專案執行說明

1. **資料預處理**  
   ```bash
   cd scripts/
   python data_preprocess_v2.py
   ```
2. **訓練勝率模型 (WR)**  
   ```bash
   python train_wr.py
   ```
3. **訓練推薦模型 (Pick)**  
   ```bash
   python train_pick_v3_tqdm.py --n_trials 10 --num_boost 400
   ```
4. **CLI 測試**  
   ```bash
   python search.py --team blue --ban 1,2,3 --topk 5
   ```
5. **啟動 Web UI**  
   ```bash
   cd ui/
   streamlit run app.py
   ```
6. **(選用) Ngrok 外網分享**  
   ```bash
   ngrok config add-authtoken 2xcbYiu9hjsQMdKfWcKWEIyixtw_235mjXu9LBF1M8dvjn1tF
   ngrok http 8501
   ```

---

## 🎯 逐步勝率預測曲線

**示例效果：**

| Step | Blue Picks           | Red Picks            | Predicted Blue WR |
|-----:|----------------------|----------------------|------------------:|
|    1 | [1]                  | []                   |            52.1%  |
|    2 | [1]                  | [11]                 |            50.3%  |
|    3 | [1]                  | [11, 64]             |            48.7%  |
|  ... | ...                  | ...                  |             ...   |
|   10 | [1,105,55,238,61]    | [11,64,103,157,121]  |            59.8%  |

---

## 🔍 如何實作

1. **組合中間步驟**  
   - 以完整 `picks_b`、`picks_r` 列表為基礎  
2. **計算每步勝率 (Python)**  
   ```python
   from copy import deepcopy
   import pandas as pd

   def compute_progression_wr(picks_b, picks_r, model):
       progression = []
       for step in range(1, 11):
           temp_b = deepcopy(picks_b[:min(step,5)])
           temp_r = deepcopy(picks_r[:max(0,step-5)])
           while len(temp_b) < 5:
               temp_b.append(0)
           while len(temp_r) < 5:
               temp_r.append(0)
           X_step = pd.DataFrame([{
               **{f"blue{i+1}": temp_b[i] for i in range(5)},
               **{f"red{i+1}": temp_r[i] for i in range(5)}
           }])
           win_prob = model.predict(X_step)[0]
           progression.append(dict(step=step, blue=temp_b, red=temp_r, wr=win_prob))
       return progression
   ```
3. **在 `app.py` 顯示折線圖**  
   ```python
   wr_prog = compute_progression_wr(picks_b, picks_r, WR_MODEL)
   df_prog = pd.DataFrame(wr_prog).set_index("step")
   st.line_chart(df_prog["wr"])
   ```

---

## 💡 改進方向與未完成項目

| 功能                        | 狀態  | 說明                                          |
|----------------------------|:-----:|-----------------------------------------------|
| Ban/Pick 階段支援          | ✅    | 已整合 DRAFT_PHASES                           |
| 勝率預測整合                | ⛔     | `wr_lgb_v4.txt` 尚未使用                      |
| 多模型比較                  | ⛔     | 尚未支援 v2/v3 輸出比較                       |

---
_  


## UI 程式碼設計說明（`ui/app.py`）

目前版本支援功能如下：
- 使用者透過側欄輸入已選 / 已禁英雄（藍方與紅方）
- 設定目前選角順序、角色代碼、隊伍陣營
- 推薦 Top-5 可選英雄，附帶預測機率與名稱（由 `champs.csv` 對應）
- 英雄選項為動態過濾（不重複於 Ban 與 Pick 名單中）
- 目前使用的模型檔案：
  - `models/pick_lgb_v3.txt`
  - `models/pick_encoder_v3.pkl`
- 特徵欄位來源：`output/pick_dataset.parquet`

---
