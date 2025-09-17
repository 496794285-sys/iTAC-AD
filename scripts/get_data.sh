#!/usr/bin/env bash
# 统一下载/整理常见数据集到 data/ 目录
set -e
mkdir -p data
echo ">>> 提示：请根据许可自行下载以下数据集并解压到 data/ 下："
echo "SMD: data/SMD/{train,test}/*.csv"
echo "MSL/SMAP: data/MSL/*.csv  data/SMAP/*.csv"
echo "PSM: data/PSM/train.csv data/PSM/test.csv"
echo "SWaT/WADI: 体量较大，建议单独准备"
echo ">>> 若你已有下载脚本，可在此处补充 wget/curl + md5sum 校验"