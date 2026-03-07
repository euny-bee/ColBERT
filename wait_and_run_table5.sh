#!/bin/bash
# Wait for run_remaining.py to finish, then start run_table5.py
cd /mnt/c/Users/dmsdu/ColBERT
source ~/colbert-env/bin/activate

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for run_remaining.py (PID 2998) to finish..."

while kill -0 2998 2>/dev/null; do
    sleep 30
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_remaining.py finished! Waiting 10s for GPU cleanup..."
sleep 10

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting run_table5.py..."
python run_table5.py > table5.log 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_table5.py finished!"
