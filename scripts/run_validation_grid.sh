#!/bin/bash

# iTAC-AD 验证网格脚本
# 测试 GRL λ 和 AAC 上限的不同组合

echo "=== iTAC-AD 验证网格测试 ==="
echo "测试 GRL λ: 0.3 / 0.5"
echo "测试 AAC 上限: 0.25 / 0.35"
echo ""

# 设置基础环境变量
export AAC_WIN=512 AAC_QP=0.90 AAC_Q0=0.015
export AAC_A=0.6 AAC_B=4.0 AAC_C=0.25 AAC_D=3.5 AAC_D0=0.03
export AAC_WMIN=0.0
export ITAC_USE_GRL=1 ITAC_GRL_WARMUP=300

# 创建结果目录
RESULTS_DIR="outputs/validation_grid_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "结果将保存到: $RESULTS_DIR"
echo ""

# 网格测试
for L in 0.3 0.5; do
  for W in 0.25 0.35; do
    echo "--- 测试组合: GRL_λ=$L, AAC_WMAX=$W ---"
    
    # 设置当前测试的参数
    export ITAC_GRL_LAMBDA=$L
    export AAC_WMAX=$W
    
    # 运行测试
    echo "开始训练..."
    LOG_FILE="$RESULTS_DIR/grid_L${L}_W${W}.log"
    ./scripts/run_synthetic.sh > "$LOG_FILE" 2>&1
    status=$?
    
    # 检查是否成功完成
    if [ $status -eq 0 ]; then
      echo "✓ 测试完成: GRL_λ=$L, AAC_WMAX=$W"
      
      # 解析日志中记录的输出目录并复制结果
      OUTPUT_DIR=$(grep -Eo 'outputs/[0-9]{8}-[0-9]{6}-itac' "$LOG_FILE" | tail -1)
      if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
        cp -r "$OUTPUT_DIR" "$RESULTS_DIR/L${L}_W${W}_output"
        echo "  输出已复制到: $RESULTS_DIR/L${L}_W${W}_output"
      else
        echo "  ⚠ 未在日志中找到有效的输出目录"
      fi
    else
      echo "✗ 测试失败: GRL_λ=$L, AAC_WMAX=$W"
    fi
    
    echo ""
  done
done

echo "=== 验证网格测试完成 ==="
echo "所有结果保存在: $RESULTS_DIR"
echo ""
echo "查看结果:"
echo "  ls -la $RESULTS_DIR"
echo "  查看日志: cat $RESULTS_DIR/grid_L*.log"
echo "  查看指标: cat $RESULTS_DIR/L*_W*_output/metrics.json"
