#!/usr/bin/env bash
# scripts/test_simple.sh
# 简单的iTAC-AD测试脚本

set -e

echo "=== 简单iTAC-AD测试 ==="

# 设置环境变量
export ITAC_FORCE_CPU=1
export ITAC_LOG_EVERY=5

# 训练模型
echo "开始训练..."
python vendor/tranad/main.py \
    --model iTAC_AD \
    --dataset synthetic \
    --retrain

echo "✅ 简单测试完成！"
