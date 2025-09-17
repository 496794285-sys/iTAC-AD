#!/usr/bin/env bash
# scripts/run_synthetic.sh
# iTAC-AD 在 synthetic 数据集上的快速测试脚本

set -e

# 设置环境变量
export ITAC_FORCE_CPU=1
export ITAC_NO_TB=1
export ITAC_NO_PLOTS=1
export ITAC_SAVE=1
export ITAC_LOG_EVERY=10

# AAC 配置
export AAC_TAU=0.9
export AAC_ALPHA=1.0
export AAC_BETA=0.5
export AAC_W_MIN=0.0
export AAC_W_MAX=1.0

# 模型配置
export ITAC_ENCODER=itr
export ITAC_DECODER=tranad
export ITAC_D_MODEL=128
export ITAC_N_HEADS=8
export ITAC_E_LAYERS=2

echo "=== iTAC-AD Synthetic 数据集测试 ==="
echo "编码器: $ITAC_ENCODER"
echo "解码器: $ITAC_DECODER"
echo "模型维度: $ITAC_D_MODEL"

# 训练模型
echo "开始训练..."
python vendor/tranad/main.py \
    --model iTAC_AD \
    --dataset synthetic \
    --retrain \
    --epochs 5 \
    --batch_size 8

# 获取最新的checkpoint
LATEST_CKPT=$(ls -t vendor/tranad/outputs/*/itac_ad_ckpt.pt | head -1)
echo "使用checkpoint: $LATEST_CKPT"

# 评测模型
echo "开始评测..."
python eval.py \
    --checkpoint "$LATEST_CKPT" \
    --output_dir "eval_results_synthetic" \
    --pot_q 0.98 \
    --pot_level 0.99 \
    --event_iou 0.1 \
    --device cpu

echo "✅ Synthetic 数据集测试完成！"
echo "结果保存在: eval_results_synthetic/"