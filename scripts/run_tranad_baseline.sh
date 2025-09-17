#!/usr/bin/env bash
# scripts/run_tranad_baseline.sh
# TranAD 基线对比脚本

set -e

# 设置环境变量
export ITAC_FORCE_CPU=1
export ITAC_NO_TB=1
export ITAC_NO_PLOTS=1
export ITAC_SAVE=1
export ITAC_LOG_EVERY=10

echo "=== TranAD 基线对比测试 ==="

# 训练TranAD模型
echo "开始训练 TranAD 基线..."
cd vendor/tranad
python main.py \
    --model TranAD \
    --dataset synthetic \
    --epochs 5 \
    --batch_size 8 \
    --retrain

# 获取最新的checkpoint
LATEST_CKPT=$(ls -t outputs/*/tranad_ckpt.pt | head -1)
echo "使用checkpoint: $LATEST_CKPT"

# 评测TranAD模型
echo "开始评测 TranAD..."
cd ../..
python eval.py \
    --checkpoint "vendor/tranad/$LATEST_CKPT" \
    --output_dir "eval_results_tranad_baseline" \
    --pot_q 0.98 \
    --pot_level 0.99 \
    --event_iou 0.1 \
    --device cpu

echo "✅ TranAD 基线测试完成！"
echo "结果保存在: eval_results_tranad_baseline/"