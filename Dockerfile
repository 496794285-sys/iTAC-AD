FROM python:3.10-slim

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
# 复制项目（如需提速，可先只拷依赖文件）
COPY pyproject.toml README.md /app/
COPY itac_ad /app/itac_ad
COPY itac_eval /app/itac_eval
COPY itacad /app/itacad
COPY scripts /app/scripts
COPY tools /app/tools

# 安装依赖（torch CPU 版）
RUN pip install -U pip && \
    pip install --no-cache-dir numpy pandas scipy matplotlib pyyaml onnx && \
    pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install -e .

ENTRYPOINT ["itacad"]