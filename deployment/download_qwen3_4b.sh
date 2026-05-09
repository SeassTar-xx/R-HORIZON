#!/usr/bin/env bash
# 将 Qwen3-4B 下载到数据盘 models_store（需可访问 Hugging Face 或配置镜像）。
set -euo pipefail

export HF_REPO_ID="${HF_REPO_ID:-Qwen/Qwen3-4B}"
export MODEL_DEST="${MODEL_DEST:-/root/autodl-tmp/models_store/Qwen3-4B}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

mkdir -p "$MODEL_DEST" "$HF_HUB_CACHE"

PY="${PYTHON_BIN:-/root/miniconda3/envs/r-horizon/bin/python}"
exec "$PY" -c '
import os
from huggingface_hub import snapshot_download
repo = os.environ["HF_REPO_ID"]
dest = os.environ["MODEL_DEST"]
print(f"repo={repo} -> {dest}")
snapshot_download(repo_id=repo, local_dir=dest)
print("ok")
'

STORE="$(dirname "$MODEL_DEST")"
ln -sfn "$MODEL_DEST" "$STORE/qwen3-4b"
echo "Symlink: $STORE/qwen3-4b -> $MODEL_DEST"
