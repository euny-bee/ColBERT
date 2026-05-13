#!/bin/bash
# run_all.sh
# reindex_2bit.py → run_and_export.py 순서로 실행
# 로그: ~/ColBERT/run_all.log

COLBERT_DIR="$HOME/ColBERT"
LOG="$COLBERT_DIR/run_all.log"

echo_ts() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"
}

cd "$COLBERT_DIR" || exit 1
echo "" >> "$LOG"
echo_ts "=========================================="
echo_ts "START: $(date)"
echo_ts "=========================================="

# Step 1: reindex
echo_ts "[1/2] reindex_2bit.py 시작..."
python reindex_2bit.py >> "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
    echo_ts "ERROR: reindex_2bit.py 실패 (exit $RC). 중단."
    exit $RC
fi
echo_ts "[1/2] reindex_2bit.py 완료."

# Step 2: export
echo_ts "[2/2] run_and_export.py 시작..."
python run_and_export.py >> "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
    echo_ts "ERROR: run_and_export.py 실패 (exit $RC)."
    exit $RC
fi
echo_ts "[2/2] run_and_export.py 완료."

echo_ts "=========================================="
echo_ts "ALL DONE: $(date)"
echo_ts "=========================================="
