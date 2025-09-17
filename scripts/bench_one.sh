#!/usr/bin/env bash
# 用法：MODEL=TranAD|iT_BASE|iTAC_AD DATASET=SMD:machine-1-1 SEED=0 ./scripts/bench_one.sh
set -e
MODEL="${MODEL:-iTAC_AD}"
DATASET="${DATASET:-synthetic}"
SEED="${SEED:-0}"

export ITAC_SEED="$SEED"
export ITAC_NO_TB=1 ITAC_NO_PLOTS=1 ITAC_SAVE=1 ITAC_LOG_EVERY=50
# 下面这几项可由上层设置覆盖
export AAC_ALPHA="${AAC_ALPHA:-1.0}" AAC_BETA="${AAC_BETA:-0.5}" AAC_TAU="${AAC_TAU:-0.9}"
export AAC_W_MIN="${AAC_W_MIN:-0.0}" AAC_W_MAX="${AAC_W_MAX:-1.0}"

python main.py --model "$MODEL" --dataset "$DATASET" --retrain

# 评测（POT/F1）
RUN_DIR=$(ls -1dt vendor/tranad/outputs/* | head -n1)
RUN_DIR="$RUN_DIR" POT_Q="${POT_Q:-0.98}" POT_LEVEL="${POT_LEVEL:-0.99}" EVENT_IOU="${EVENT_IOU:-0.1}" \
bash scripts/eval_event.sh "$RUN_DIR"

echo "[bench_one] done -> $RUN_DIR/metrics.json"
