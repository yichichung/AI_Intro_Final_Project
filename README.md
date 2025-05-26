Pull Request: Initial Commit from Local Backup with Git LFS
🔧 Summary
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


當初為什麼訓練 wr_lgb_v4.txt
這個模型是使用 match_dataset.parquet 中的欄位訓練來預測某一方是否會獲勝（win=1 or 0），使用的特徵有：
藍方/紅方的 picks（藍紅各 5 名英雄）


每位英雄的角色編碼


ban 英雄


duration（對局時間）


baronKills, dragonKills, towerKills（teamstats.csv 來的）


patch（版本號碼）


透過這些特徵，訓練出一個能夠根據雙方組合，預測哪一隊會獲勝的模型，AUC 通常約 0.79 左右。

✅ 它的用途
雖然你目前最主要使用的是 pick_lgb_v3.txt（針對單一 pick 預測 top-5），但 wr_lgb_v4.txt 可以用來做更高層次的事情：
🌟 延伸應用一：模擬整場比賽勝率
根據使用者輸入藍紅雙方全部 pick 完後的組合，由 wr 模型輸出勝率


這可以變成 團隊組合強度預測 工具


甚至可在 ban 階段顯示「目前藍方預測勝率為 48.3%」



🌟 延伸應用二：推薦 Ban（或破壞對方組合）
模擬「如果紅方 ban 掉某一隻英雄，藍方勝率會下降/上升多少」


這可用來推薦 破壞力最高的 ban



🌟 延伸應用三：納入 pick 模型的特徵
你目前的 pick_lgb_v3.txt 模型，是根據單一英雄預測其會被選中的機率


若你納入 wr_lgb 的預測值作為 pick 特徵，可能能提升 Top-5 accuracy


例如：
python
CopyEdit
# 建立 pick 候選者，分別加入該角色後整體 wr 模型預測勝率
# 以此作為額外特徵，加入 pick 模型中


⚠️ 為何目前沒使用它？
因為你目前的 train_pick_v3.py 專注於單一 pick 預測，而非整隊勝率，且預測過程速度也會慢一點。當時先完成 baseline 成果比較實際。

🛠️ 現在如何使用 wr 模型？
若你想立即使用：
load_model("models/wr_lgb_v4.txt")


建立完整的 team 組合（藍紅各 5 名英雄）


預測 model.predict(X) → 回傳 win 機率



📈 對 Final Project 的加分貢獻
✅ 若你加入 wr 預測（可在畫面上展示每一步對雙方勝率的影響），將：
提升 系統的策略深度


展示 你對機器學習模型整合與應用的能力


在報告中作為 模型整合、策略分析亮點
