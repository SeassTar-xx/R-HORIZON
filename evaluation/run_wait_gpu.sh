#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage:"
  echo "  Single file: bash evaluation/run_wait_gpu.sh <input_jsonl> <output_dir> [min_free_mib]"
  echo "  All datasets: bash evaluation/run_wait_gpu.sh --all <output_dir> [min_free_mib]"
  exit 1
fi

MODEL_KEY="qwen2.5-3b-local"
VLLM_MODEL="Qwen/Qwen2.5-3B-Instruct"
SERVE_NAME="qwen2.5-3b-local"
PORT=8000
MODE="single"
INPUT_FILE=""
OUTPUT_DIR=""
MIN_FREE_MIB=""

if [ "${1}" = "--all" ]; then
  MODE="all"
  if [ "$#" -lt 2 ]; then
    echo "Usage: bash evaluation/run_wait_gpu.sh --all <output_dir> [min_free_mib]"
    exit 1
  fi
  OUTPUT_DIR="$2"
  MIN_FREE_MIB="${3:-11000}"
else
  if [ "$#" -lt 2 ]; then
    echo "Usage: bash evaluation/run_wait_gpu.sh <input_jsonl> <output_dir> [min_free_mib]"
    exit 1
  fi
  INPUT_FILE="$1"
  OUTPUT_DIR="$2"
  MIN_FREE_MIB="${3:-11000}"
fi

source /home/data/anaconda3/etc/profile.d/conda.sh
conda activate r-horizon
export PYTHONPATH="/home/data/XuXin/R-HORIZON/training"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, -v min_free="$MIN_FREE_MIB" '
      {
        gsub(/ /, "", $1); gsub(/ /, "", $2);
        if ($2 >= min_free) {
          if ($2 > best_free) {
            best_free = $2; best_gpu = $1;
          }
        }
      }
      END { if (best_gpu != "") print best_gpu; }'
}

echo "[INFO] Waiting for a GPU with free memory >= ${MIN_FREE_MIB} MiB ..."
while true; do
  GPU_ID="$(pick_gpu || true)"
  if [ -n "${GPU_ID}" ]; then
    echo "[INFO] Selected GPU ${GPU_ID}"
    break
  fi
  sleep 30
done

cleanup() {
  if tmux has-session -t rh_eval_vllm 2>/dev/null; then
    tmux kill-session -t rh_eval_vllm
  fi
}
trap cleanup EXIT

tmux new-session -d -s rh_eval_vllm \
  "source /home/data/anaconda3/etc/profile.d/conda.sh && conda activate r-horizon && CUDA_VISIBLE_DEVICES=${GPU_ID} vllm serve ${VLLM_MODEL} --host 127.0.0.1 --port ${PORT} --served-model-name ${SERVE_NAME} --dtype auto --tensor-parallel-size 1 --pipeline-parallel-size 1 --gpu-memory-utilization 0.35 --max-model-len 2048 --max-num-seqs 4 --trust-remote-code"

echo "[INFO] Waiting for vLLM API readiness ..."
python - <<'PY'
import requests
import time
url = "http://127.0.0.1:8000/v1/chat/completions"
payload = {
    "model": "qwen2.5-3b-local",
    "messages": [{"role": "user", "content": "Reply with OK only."}],
    "temperature": 0,
    "max_tokens": 8,
}
for _ in range(120):
    try:
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code == 200:
            print("[INFO] vLLM is ready")
            raise SystemExit(0)
    except Exception:
        pass
    time.sleep(5)
print("[ERROR] vLLM was not ready in time")
raise SystemExit(1)
PY

mkdir -p "${OUTPUT_DIR}"

if [ "${MODE}" = "single" ]; then
  echo "[INFO] Starting evaluation pipeline for single file: ${INPUT_FILE}"
  bash evaluation/run.sh "${INPUT_FILE}" "${OUTPUT_DIR}" "${MODEL_KEY}"
  echo "[INFO] Evaluation completed for single file."
else
  echo "[INFO] Starting evaluation pipeline for all benchmark datasets..."
  DATASETS=(
    "evaluation/data/R-HORIZON-Math500/Math500-combined-n2.jsonl"
    "evaluation/data/R-HORIZON-AIME24/AIME24-combined-n2.jsonl"
    "evaluation/data/R-HORIZON-AIME25/AIME25-combined-n2.jsonl"
    "evaluation/data/R-HORIZON-AMC23/AMC23-combined-n2.jsonl"
    "evaluation/data/R-HORIZON-Websearch/Websearch-combined-n2.jsonl"
  )

  for DATA_FILE in "${DATASETS[@]}"; do
    if [ ! -f "${DATA_FILE}" ]; then
      echo "[WARN] Skip missing dataset file: ${DATA_FILE}"
      continue
    fi
    echo "[INFO] Running dataset: ${DATA_FILE}"
    bash evaluation/run.sh "${DATA_FILE}" "${OUTPUT_DIR}" "${MODEL_KEY}"
    echo "[INFO] Finished dataset: ${DATA_FILE}"
  done
  echo "[INFO] Evaluation completed for all datasets."
fi
