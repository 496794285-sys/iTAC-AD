#!/usr/bin/env bash
# 依据 TranAD 官方复现所用公开源获取 SMD/SMAP/MSL，
# 并保持与社区常用目录结构一致。
# 参考来源：
# - SMD: OmniAnomaly/Dropbox 原始包
# - SMAP/MSL: NASA telemanom 数据包 + labeled_anomalies.csv
set -euo pipefail

# 工程根目录（支持空格路径）
PROJ_DIR="${PROJ_DIR:-$(pwd)}"
DATA_DIR="$PROJ_DIR/data"
RAW_DIR="$DATA_DIR/_raw"
mkdir -p "$RAW_DIR"

echo "==> 目标目录: $DATA_DIR"

# -------- SMD（Server Machine Dataset）---------
# 官方常用直链（Dropbox，Merlion 文档中引用）
SMD_URL="https://www.dropbox.com/s/x53ph5cru62kv0f/ServerMachineDataset.tar.gz?dl=1"
SMD_TAR="$RAW_DIR/ServerMachineDataset.tar.gz"
if [ ! -d "$DATA_DIR/SMD" ]; then
  echo "==> 下载 SMD ..."
  curl -L "$SMD_URL" -o "$SMD_TAR"
  tar -xzf "$SMD_TAR" -C "$DATA_DIR"
  # 解压得到 ServerMachineDataset/{train,test,test_label}
  mv "$DATA_DIR/ServerMachineDataset" "$DATA_DIR/SMD"
else
  echo "==> 跳过 SMD（已存在）"
fi

# -------- SMAP / MSL（NASA, telemanom）---------
# TranAD 与大量论文复现都从 telemanom 包拿原始数据与标签
# 使用GitHub上的telemanom数据源
if [ ! -d "$RAW_DIR/telemanom" ]; then
  echo "==> 从GitHub下载telemanom数据..."
  mkdir -p "$RAW_DIR/telemanom"
  cd "$RAW_DIR/telemanom"
  
  # 下载SMAP数据
  curl -L "https://raw.githubusercontent.com/khundman/telemanom/master/data/train/SMAP" -o "SMAP_train"
  curl -L "https://raw.githubusercontent.com/khundman/telemanom/master/data/test/SMAP" -o "SMAP_test"
  
  # 下载MSL数据  
  curl -L "https://raw.githubusercontent.com/khundman/telemanom/master/data/train/MSL" -o "MSL_train"
  curl -L "https://raw.githubusercontent.com/khundman/telemanom/master/data/test/MSL" -o "MSL_test"
  
  # 下载标签文件
  curl -L "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv" -o "labeled_anomalies.csv"
  
  cd "$PROJ_DIR"
else
  echo "==> 跳过 telemanom 下载（已存在）"
fi

# 规范化到 data/SMAP 与 data/MSL（TranAD 的 preprocess.py 会读取）
mkdir -p "$DATA_DIR/SMAP" "$DATA_DIR/MSL"
# 创建适当的目录结构
for ds in SMAP MSL; do
  mkdir -p "$DATA_DIR/$ds/train" "$DATA_DIR/$ds/test"
  # 复制数据文件
  cp "$RAW_DIR/telemanom/${ds}_train" "$DATA_DIR/$ds/train/"
  cp "$RAW_DIR/telemanom/${ds}_test" "$DATA_DIR/$ds/test/"
  # 复制标签文件
  cp "$RAW_DIR/telemanom/labeled_anomalies.csv" "$DATA_DIR/$ds/"
done

echo "==> 数据拉取完毕。接下来调用 TranAD 的预处理脚本统一格式..."
# 运行 TranAD 的 preprocess.py（你改过的 TRANAD-0 里也有同名脚本）
# 仅处理我们已准备的三个数据集
python3 preprocess.py SMD SMAP MSL

echo "==> OK：SMD/SMAP/MSL 预处理完成。目录位于 $DATA_DIR 下。"

cat <<'NOTE'

【可选说明】
- SWaT/WADI：需向 iTrust 申请下载，然后放到 data/SWaT 与 data/WADI，再执行
  python3 preprocess.py SWaT WADI
- PSM：TranAD 论文默认不含 PSM；若你要额外比较，可用 Kaggle CLI 下载：
    pip install kaggle
    # 把 kaggle.json 放到 ~/.kaggle 并 chmod 600 ~/.kaggle/kaggle.json
    kaggle datasets download -d ljolm08/pooled-server-metrics-psm -p data/PSM
    (cd data/PSM && unzip -q *.zip)
  之后依据你当前仓库的 PSM loader 要求组织 train/test 文件。

NOTE
