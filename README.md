LoL_BP_Project/
├── data/               # 9 份原始 csv
├── output/             # ⇨ parquet 產出
│   ├── pick_dataset.parquet
│   └── match_dataset.parquet
├── scripts/
│   ├── data_preprocess_v3.py
│   ├── train_pick_v2.py      # Pick Rank‑LGBM
│   ├── train_wr.py           # 勝率 LightGBM
│   └── search.py             # Beam + WR re‑rank
├── models/             # 訓練後由腳本自動存放
└── reports/            # Top‑5 / AUC / log
