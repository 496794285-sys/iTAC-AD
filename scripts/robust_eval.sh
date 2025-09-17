#!/usr/bin/env bash
# 用法：
# CSV=data/PSM/test.csv CKPT=outputs/your_ckpt_dir WINDOW=100 LABEL=label ./scripts/robust_eval.sh
set -e
CSV="${CSV:?need CSV path}"
CKPT="${CKPT:?need ckpt dir}"
WINDOW="${WINDOW:-100}"
LABEL="${LABEL:-}"
OUTDIR="results/robust"
mkdir -p "$OUTDIR"

run_pred () { # csv outtag window
  local csv="$1"; local tag="$2"; local win="${3:-$WINDOW}"
  itacad predict --csv "$csv" --ckpt "$CKPT" --window "$win" ${LABEL:+--label-col "$LABEL"} \
    --out "$OUTDIR/$tag" >/dev/null
  jq -c --arg tag "$tag" --arg win "$win" '. + {tag:$tag, window:($win|tonumber)}' "$OUTDIR/$tag/metrics.json" \
    >> "$OUTDIR/robust.jsonl"
}

# 清空旧结果
: > "$OUTDIR/robust.jsonl"

# 原始
run_pred "$CSV" "orig" "$WINDOW"

# 缺失：1%, 5%, 10%
for r in 0.01 0.05 0.10; do
  pert="$OUTDIR/miss_${r}.csv"
  python tools/perturb_csv.py --in "$CSV" --out "$pert" ${LABEL:+--label-col "$LABEL"} --missing "$r"
  run_pred "$pert" "miss_${r}" "$WINDOW"
done

# 尖峰污染（测试鲁棒性，不改训练）：1%, 5%
for r in 0.01 0.05; do
  pert="$OUTDIR/spike_${r}.csv"
  python tools/perturb_csv.py --in "$CSV" --out "$pert" ${LABEL:+--label-col "$LABEL"} --spike "$r" --spike-scale 4.0
  run_pred "$pert" "spike_${r}" "$WINDOW"
done

# 窗口错配：训练假设为 WINDOW，测试改 120/150
for w in 120 150; do
  run_pred "$CSV" "win_${w}" "$w"
done

# 汇总到 CSV
python - <<'PY'
import json, pandas as pd, os
p = "results/robust/robust.jsonl"
rows = [json.loads(l) for l in open(p)]
df = pd.DataFrame(rows)
df.to_csv("results/robustness.csv", index=False)
print(df)
PY

echo "[robust_eval] done -> results/robustness.csv"