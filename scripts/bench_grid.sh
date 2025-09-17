#!/usr/bin/env bash
set -e
# 只支持iTAC_AD模型的数据集列表
DATASETS=(
  "synthetic"
)
SEEDS=(0 1 2)

# 组合：iTAC_AD + 两个消融（无AAC/无GRL）
run_combo() {
  local tag="$1"
  local aac_alpha="$2"
  local aac_beta="$3"
  local grl_lambda="$4"
  
  for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "==> $tag | $ds | seed=$seed"
      MODEL="iTAC_AD" AAC_ALPHA="$aac_alpha" AAC_BETA="$aac_beta" ITAC_GRL_LAMBDA="$grl_lambda" SEED="$seed" DATASET="$ds" ./scripts/bench_one.sh
      RUN_DIR=$(ls -1dt vendor/tranad/outputs/* | head -n1)
      python tools/aggregate_results.py --run "$RUN_DIR" --tag "$tag" --dataset "$ds" --seed "$seed" \
        --out results/all.csv
    done
  done
}

mkdir -p results

# 运行不同的组合
run_combo "iTAC_AD.full" "1.0" "0.5" "1.0"
run_combo "iTAC_AD.no_aac" "0" "0" "1.0"
run_combo "iTAC_AD.no_grl" "1.0" "0.5" "0"

echo "[bench_grid] done. aggregate -> results/all.csv"
