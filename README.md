Pull Request: Initial Commit from Local Backup with Git LFS
🔧 Summary

直接執行: 39行:　目標：直接部署 app.py（含勝率曲線）
專案目錄總覽與用途： 100行
專案執行說明整體步驟總覽: 223行
UI 程式碼設計說明與期望改動（ui/app.py）:328行

This PR includes the full migration of the LoL_BP_Project from local development to GitHub with:
✅ Cleaned Git history (previous large-file commits removed)


✅ Proper usage of Git LFS for large files:


models/pick_lgb_v3.txt


data/stats1.csv


data/stats2.csv


Other large .csv, .parquet, .pkl files


✅ Organized directory structure:


data/, models/, scripts/, ui/, output/, reports/


✅ Git ignore rules for intermediate files, system cache, and local logs


✅ Dev branch is ready to be merged into main

目標：直接部署 app.py（含勝率曲線）
📂 需要的專案結構（只需這些）
bash
Copy code
LoL_BP_Project/
│
├── models/
│   └── wr_lgb_v4.txt          ⬅️ 英雄勝率模型（由 train_wr.py 訓練）
│
├── output/
│   └── champ_wr.parquet       ⬅️ 每隻英雄藍/紅方勝率（由 data_preprocess_v2.py 計算）
│
├── data/
│   └── champs.csv             ⬅️ id 對應英雄名稱的表
│
├── ui/
│   └── app.py                 ⬅️ 主程式，從這邊執行
✅ 安裝環境（只需一次）
bash
Copy code
conda activate lolbp  # 或你的 Python 環境
pip install -U streamlit lightgbm pandas scikit-learn
✅ 下載與準備檔案
檔案路徑	來源/用途
models/wr_lgb_v4.txt	train_wr.py 訓練的 LightGBM 勝率模型
output/champ_wr.parquet	每隻英雄的藍方與紅方平均勝率，用於 pick 模型
data/champs.csv	英雄 id 對應名稱，建議為英文名稱欄位
ui/app.py	Streamlit 前端程式，實作推薦與曲線顯示

✅ 你可以把我剛剛整理好的 app.py 丟進 ui/ 資料夾直接使用。

✅ 執行指令
bash
Copy code
cd ~/LoL_BP_Project/ui
streamlit run app.py
啟動後會顯示：

nginx
Copy code
Local URL: http://localhost:8501
Network URL: http://你的IP:8501


📂 Git LFS Tracked Files
bash
CopyEdit
- models/pick_lgb_v3.txt
- models/pick_encoder_v3.pkl
- data/stats1.csv
- data/stats2.csv
- output/pick_dataset.parquet
- output/match_dataset.parquet
- output/champ_wr.parquet

Note: Contributors should install Git LFS before cloning:
nginx
CopyEdit
git lfs install


專案目錄總覽與用途

🔹 data/
儲存原始資料（CSV 格式）：
檔案名稱
功能
matches.csv, participants.csv, teamstats.csv, teambans.csv, stats1.csv, stats2.csv
從 OraclesElixir 下載的歷史對戰資料（比賽結果、玩家資訊、ban/pick 順序）
champs.csv
每個英雄的英文名稱與對應 ID，用於介面顯示與轉換


🔹 output/
儲存經前處理後的中繼檔案（parquet 格式）：
檔案名稱
功能
match_dataset.parquet
每場比賽的 ban、pick、勝負、時間資訊彙整表
pick_dataset.parquet
轉成單筆 pick 訓練資料
champ_wr.parquet
每位英雄在藍方/紅方的平均勝率，用於作為模型特徵


🔹 models/
儲存訓練好的 LightGBM 模型與編碼器：
檔案
用途
pick_lgb_v3.txt
英雄推薦模型（top-5 預測）
pick_encoder_v3.pkl
LabelEncoder，用來轉換英雄 ID 為模型 label
wr_lgb_v4.txt
勝率模型（你目前沒有直接用這個做預測）


🔹 reports/
儲存模型訓練過程紀錄：
檔案
說明
train_pick_v3.log、train_pick_v3_continue.log
訓練 log（Optuna 10 次 trial，每次 top-5 accuracy）
pick_cv_v3.txt
最佳 trial 的參數與結果（top-5 accuracy ≈ 0.6732）
wr_cv_v4.txt
英雄勝率模型的 AUC 訓練結果


🔹 scripts/
主要程式碼：
檔案
說明
data_preprocess_v2.py
將原始 CSV 轉換為模型訓練格式（pick + match 資料集 + win-rate）
train_pick_v3_tqdm.py
使用 Optuna 訓練推薦模型並儲存
search.py
終端機測試推薦模型用 CLI（支援 --team --ban --topk）
app.py
Streamlit 網頁版互動系統，支援完整 ban/pick 順序與推薦邏輯（含 phase 判斷、勝率排序等）


🔹 ui/
專門用來放前端執行的 App
檔案
說明
app.py
同上，整合版 Streamlit App
ngrok / ngrok.zip
用來啟用外部瀏覽器連線用的 Ngrok 工具（連結 http://xxx.ngrok.io）


🧠 系統功能與已完成目標
✅ 機器學習模型
使用 LightGBM 訓練英雄推薦模型（pick order、角色、ban/pick 英雄、勝率等作為特徵）


使用 Optuna 自動調參（10 trial）


預測 top-5 機率最高的英雄


模型 accuracy：約 Top-5 ≈ 67.3%



✅ Ban/Pick Draft 支援（根據順序）
使用 DRAFT_PHASES 指定 20 步驟（6 ban + 14 pick）


使用者只需選擇「目前是第幾步驟」→ 自動顯示哪一方該 ban/pick


介面自動刪除已選與已禁英雄，避免推薦重複



✅ 互動式介面（Streamlit）
支援藍/紅雙方：


已選英雄


已禁英雄


可設定：


第幾 pick 順序


陣營、角色


即時推薦對應的英雄與勝率


支援外網訪問（透過 Ngrok）


專案執行說明
整體步驟總覽
1.預處理資料 → scripts/data_preprocess_v2.py


2.訓練英雄勝率模型（WR 模型） → scripts/train_wr.py


3.訓練英雄選角模型（Pick 模型） → scripts/train_pick_v3_tqdm.py


4.啟動前測試 CLI 預測效果 → scripts/search.py


5.啟動網頁推薦系統 → ui/app.py（用 streamlit）


6.（選用）用 ngrok 建立公開網址



🔹 Step 1：資料預處理
進入 scripts/ 目錄，執行：
bash
Copy code
python data_preprocess_v2.py

這會產生 3 個主要的預處理資料集：
檔案路徑
說明
output/match_dataset.parquet
整理過的賽事資訊（含藍紅方英雄、隊伍勝負等）
output/pick_dataset.parquet
Ban/Pick 模型訓練資料（以單一英雄為預測目標）
output/champ_wr.parquet
每位英雄在藍方 / 紅方的平均勝率


🔹 Step 2：訓練模型
✅ 英雄勝率模型
bash
Copy code
python train_wr.py

輸出：
models/wr_lgb_v4.txt：使用 champ_wr 特徵建構的 LightGBM 勝率預測模型


reports/wr_cv_v4.txt：包含 AUC 評估數值


🔎 此模型未直接應用於推薦，但可作為未來擴充用來 根據目前隊伍組合預測勝率 的基礎。
✅ 英雄推薦模型（Pick 預測）
bash
Copy code
python train_pick_v3_tqdm.py --n_trials 10 --num_boost 400

可背景執行並紀錄 log：
bash
Copy code
nohup python train_pick_v3_tqdm.py --n_trials 10 --num_boost 400 > reports/train_pick_v3.log 2>&1 &
tail -f reports/train_pick_v3.log

輸出：
models/pick_lgb_v3.txt：推薦模型


models/pick_encoder_v3.pkl：LabelEncoder


reports/pick_cv_v3.txt：Top-5 accuracy 與最佳參數



🔹 Step 3：啟動 Web UI（Streamlit）
bash
Copy code
cd ui/
streamlit run app.py

若需對外連線，可使用 ngrok ：
bash
Copy code
ngrok config add-authtoken <your_token>
ngrok http 8501

將顯示網址分享給使用者。

🗂️ 專案資料夾說明
資料夾
用途
data/
原始資料（CSV 格式）
output/
Parquet 格式的預處理資料，供模型與 UI 使用
models/
LightGBM 訓練後的模型檔案與 encoder
reports/
訓練日誌與模型評估報告
scripts/
所有 Python 處理腳本，包括訓練、查詢、資料轉換
ui/
Streamlit 前端應用程式，主程式為 app.py


📁 UI 程式碼設計說明（ui/app.py）
目前版本支援功能如下：
使用者透過側欄輸入已選 / 已禁英雄（藍方與紅方）


設定目前選角順序、角色代碼、隊伍陣營


推薦 Top-5 可選英雄，附帶預測機率與名稱（champs.csv 對應）


英雄選項為動態過濾（不重複於 Ban 與 Pick 名單中）


目前使用的模型檔案：
models/pick_lgb_v3.txt


models/pick_encoder_v3.pkl


特徵欄位來自 output/pick_dataset.parquet



💡 改進方向與未完成項目
項目
狀態
說明
支援 Ban/Pick 階段步驟（DRAFT_PHASES）
✅ 已整合
使用 step = st.number_input(...) 對應至動作類型
每步驟過濾推薦名單
✅ 已實作
根據 pick/bans，自動排除重複英雄
胜率預測整合
⛔ 尚未整合
wr_lgb_v4.txt 可用來計算目前藍/紅陣營勝率（未使用）
英雄名稱顯示
✅ 已完成
由 champs.csv 對應 ID to name
支援最多 10 選角記錄
⛔ 僅支援前 5 名
特徵欄位為固定長度（5 pick + 5 ban）
多模型比較
⛔ 尚未支援
可比較 v2 / v3 預測模型輸出效果


目標：逐步勝率預測曲線
🎯 效果示意：
在 UI 或報表中呈現出這樣的效果：
Pick Step
Blue Picks
Red Picks
Predicted Blue Win Rate
1
1


52.1%
2
1
11
50.3%
3
1
11, 64
48.7%
4
1, 105
11, 64
51.4%
5
1, 105
11, 64, 103
52.7%
...
...
...
...
10
1,105,55,238,61
11,64,103,157,121
59.8%


🔍 如何實作（逐步構建勝率曲線）
🔢 1. 組合所有可能的中間步驟
你只要有一組完整的 ban pick list，例如：
python
Copy code
picks_b = [1, 105, 55, 238, 61]
picks_r = [11, 64, 103, 157, 121]

就可以逐步填入，預測每一步的勝率變化。
🧪 2. 實際 Python 程式碼
python
Copy code
from copy import deepcopy
import pandas as pd

def compute_progression_wr(picks_b, picks_r, model):
    progression = []
    for step in range(1, 11):
        temp_b = deepcopy(picks_b[:min(step, 5)])
        temp_r = deepcopy(picks_r[:max(0, step - 5)])
        # Pad to 5
        while len(temp_b) < 5:
            temp_b.append(0)
        while len(temp_r) < 5:
            temp_r.append(0)
        X_step = pd.DataFrame([{
            f"blue{i+1}": temp_b[i] for i in range(5)
        } | {
            f"red{i+1}": temp_r[i] for i in range(5)
        }])
        win_prob = model.predict(X_step)[0]
        progression.append(dict(step=step, blue=temp_b[:step], red=temp_r[:step-5], wr=win_prob))
    return progression

📊 3. 加在 app.py 顯示
你可以在 UI 裡加入一張 line_chart 或 table 呈現：
python
Copy code
wr_prog = compute_progression_wr(picks_b, picks_r, WR_MODEL)
df_prog = pd.DataFrame(wr_prog)
st.line_chart(df_prog.set_index("step")["wr"])


🚀 好處與應用場景
應用
說明
教學與分析
可以看到哪一步選角最影響勝率（拐點）
AI 模擬推薦
讓模型選出能 最大化勝率上升幅度的 pick
視覺化 UI 強化
使用者對當下選角的勝率變化有感知
