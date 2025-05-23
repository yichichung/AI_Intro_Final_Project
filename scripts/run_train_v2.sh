#!/usr/bin/env bash
# run_train_v2.sh —— 安全、自動背景執行訓練用腳本
set -euo pipefail

ROOT="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"  # ~/LoL_BP_Project
cd "$ROOT"

# 🔪 殺掉舊的 train_pick_v2.py
OLD_PIDS=$(pgrep -f "python scripts/train_pick_v2.py" || true)
if [[ -n "$OLD_PIDS" ]]; then
  echo "[run_train_v2] Found running: $OLD_PIDS → killing…"
  kill $OLD_PIDS || true
  sleep 2
fi

# 📁 準備 log 檔
LOG_DIR="$ROOT/reports"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_${TS}.log"

# 🚀 背景訓練
echo "[run_train_v2] Starting training → $LOG_FILE"
nohup python scripts/train_pick_v2.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "$PID" > "$LOG_DIR/last_train_pid.txt"

cat <<EOF
✅ Training started
PID  : $PID
LOG  : $LOG_FILE

📡 Monitor:  tail -f $LOG_FILE
❌ Stop   :  kill \$(cat $LOG_DIR/last_train_pid.txt)
EOF
