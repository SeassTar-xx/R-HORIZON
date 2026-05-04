#!/usr/bin/env bash
set -euo pipefail

# Pipeline-style evaluator:
# - Keep Qwen inference on one dataset at a time (single local model endpoint).
# - As soon as one dataset inference finishes, start extract+judge in background.
# - Immediately move Qwen inference to next dataset to overlap compute and API post-process.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
if [ ! -f "${CONDA_SH}" ]; then
  CONDA_SH="/home/data/anaconda3/etc/profile.d/conda.sh"
fi
source "${CONDA_SH}"
conda activate r-horizon
cd "${REPO_ROOT}"

MODEL_KEY="${MODEL_KEY:-qwen3-4b-local}"
OUTPUT_ROOT="${1:-/root/autodl-tmp/evaluation-results/pipelined_$(date +%Y%m%d_%H%M%S)}"
MAX_BG_POSTPROC="${MAX_BG_POSTPROC:-1}"  # keep conservative for DeepSeek API stability

mkdir -p "${OUTPUT_ROOT}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] MODEL_KEY=${MODEL_KEY}"
echo "[INFO] MAX_BG_POSTPROC=${MAX_BG_POSTPROC}"

DATASETS=(
  "evaluation/data/R-HORIZON-Math500/Math500-combined-n2.jsonl"
  "evaluation/data/R-HORIZON-AIME24/AIME24-combined-n2.jsonl"
  "evaluation/data/R-HORIZON-AIME25/AIME25-combined-n2.jsonl"
  "evaluation/data/R-HORIZON-AMC23/AMC23-combined-n2.jsonl"
)

declare -a BG_PIDS=()
declare -a BG_NAMES=()
FAIL=0

cleanup_finished_jobs() {
  local new_pids=()
  local new_names=()
  local i
  for i in "${!BG_PIDS[@]}"; do
    local pid="${BG_PIDS[$i]}"
    local name="${BG_NAMES[$i]}"
    if kill -0 "${pid}" 2>/dev/null; then
      new_pids+=("${pid}")
      new_names+=("${name}")
    else
      if wait "${pid}"; then
        echo "[POST OK] ${name}"
      else
        echo "[POST FAIL] ${name}"
        FAIL=1
      fi
    fi
  done
  BG_PIDS=("${new_pids[@]}")
  BG_NAMES=("${new_names[@]}")
}

wait_for_slot() {
  while true; do
    cleanup_finished_jobs
    if [ "${#BG_PIDS[@]}" -lt "${MAX_BG_POSTPROC}" ]; then
      break
    fi
    sleep 5
  done
}

run_postprocess_bg() {
  local input_file="$1"
  local outdir="$2"
  local base
  base="$(basename "${input_file}" .jsonl)"

  (
    set -euo pipefail
    python evaluation/extract.py \
      --input "${outdir}/${base}_result.json" \
      --output "${outdir}/${base}_result_judged.json" \
      --model_name deepseek-reasoner

    python evaluation/judge.py \
      --raw_input "${input_file}" \
      --prediction "${outdir}/${base}_result_judged.json" \
      --output "${outdir}/${base}_result_judged_stat.txt"
  ) > "${outdir}/${base}_postprocess.log" 2>&1 &

  local pid=$!
  BG_PIDS+=("${pid}")
  BG_NAMES+=("${base}")
  echo "[POST START] ${base}, pid=${pid}"
}

for input_file in "${DATASETS[@]}"; do
  if [ ! -f "${input_file}" ]; then
    echo "[WARN] Missing dataset, skip: ${input_file}"
    continue
  fi

  ds="$(basename "$(dirname "${input_file}")" | sed 's/^R-HORIZON-//')"
  base="$(basename "${input_file}" .jsonl)"
  tag="$(echo "${base}" | sed "s/^${ds}-//")"
  outdir="${OUTPUT_ROOT}/${ds}/${tag}"
  mkdir -p "${outdir}"

  echo "[INF START] ${input_file} -> ${outdir}"
  python evaluation/inference.py \
    --input "${input_file}" \
    --output "${outdir}/${base}_result.json" \
    --model_name "${MODEL_KEY}"
  echo "[INF DONE] ${base}"

  wait_for_slot
  run_postprocess_bg "${input_file}" "${outdir}"
done

while [ "${#BG_PIDS[@]}" -gt 0 ]; do
  cleanup_finished_jobs
  [ "${#BG_PIDS[@]}" -gt 0 ] && sleep 5
done

if [ "${FAIL}" -ne 0 ]; then
  echo "[DONE WITH WARNINGS] Some post-process jobs failed."
  exit 1
fi

echo "[DONE] All datasets finished with pipelined post-process."
