#!/usr/bin/env bash
# 启动 OpenAI 兼容的本地 Chat API（evaluation/qwen3_local_server.py）。
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/models/qwen3-4b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "错误: 未找到模型权重 $MODEL_PATH/config.json"
  echo "请先在有网络环境下执行: bash deployment/download_qwen3_4b.sh"
  exit 1
fi

PY="${PYTHON_BIN:-/root/miniconda3/envs/r-horizon/bin/python}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$PY" "$ROOT/evaluation/qwen3_local_server.py" \
  --model_path "$MODEL_PATH" --host "$HOST" --port "$PORT"
