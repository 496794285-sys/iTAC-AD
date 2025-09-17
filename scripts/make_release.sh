#!/usr/bin/env bash
# 用法：TAG=v0.1.0 CKPT=outputs/your_ckpt_dir ./scripts/make_release.sh
set -e
TAG="${TAG:?set TAG like v0.1.0}"
CKPT="${CKPT:?set CKPT dir}"

# 导出模型
mkdir -p exports
itacad export --ckpt "$CKPT" --format onnx --L 100 --D 25 --out exports/itacad_L100_D25.onnx
itacad export --ckpt "$CKPT" --format ts   --L 100 --D 25 --out exports/itacad_L100_D25.ts

# 固化环境
bash scripts/freeze_env.sh

# 结果表/图（若已存在 results/all.csv，将附加聚合表）
python tools/make_tables.py --csv results/all.csv || true
python tools/make_figs.py || true

# 收集文件
OUTDIR="release_$TAG"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"
cp -r "$CKPT" "$OUTDIR/ckpt"
cp -r exports "$OUTDIR/exports"
cp -r results "$OUTDIR/results" || true
cp environment-lock.yml "$OUTDIR/" || true
cp -r itacad itac_eval tools scripts pyproject.toml README.md "$OUTDIR/"

# 打包
tar czf "$OUTDIR.tar.gz" "$OUTDIR"
echo "[release] packed -> $OUTDIR.tar.gz"

# 可选：打 git tag
git tag -f "$TAG" && git push -f origin "$TAG" || true