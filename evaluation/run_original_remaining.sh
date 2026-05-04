#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
if [ ! -f "${CONDA_SH}" ]; then
  CONDA_SH="/home/data/anaconda3/etc/profile.d/conda.sh"
fi
source "${CONDA_SH}"
conda activate r-horizon
cd "${REPO_ROOT}"

MODEL_KEY="${MODEL_KEY:-qwen3-4b-local}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/evaluation-results/result_original}"
INFER_MAX_WORKERS="${INFER_MAX_WORKERS:-2}"
MAX_BG_POSTPROC="${MAX_BG_POSTPROC:-1}"
mkdir -p "${OUT_ROOT}"

echo "[INFO] MODEL_KEY=${MODEL_KEY}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] INFER_MAX_WORKERS=${INFER_MAX_WORKERS}"
echo "[INFO] MAX_BG_POSTPROC=${MAX_BG_POSTPROC}"

# Ordered schedule:
# 1) Finish Math500 remaining tags
# 2) Then AIME24 remaining tags
# 3) Then AIME25 remaining tags
# 4) Then AMC23 remaining tags
# NOTE: n2 is intentionally excluded because it is already completed.
DATASETS=(
  "Math500|n4|evaluation/data/R-HORIZON-Math500/Math500-combined-n4.jsonl"
  "Math500|n8|evaluation/data/R-HORIZON-Math500/Math500-combined-n8.jsonl"
  "Math500|n16|evaluation/data/R-HORIZON-Math500/Math500-combined-n16.jsonl"
  "Math500|origin|evaluation/data/R-HORIZON-Math500/Math500-origin.jsonl"
  "AIME24|n3|evaluation/data/R-HORIZON-AIME24/AIME24-combined-n3.jsonl"
  "AIME24|n4|evaluation/data/R-HORIZON-AIME24/AIME24-combined-n4.jsonl"
  "AIME24|n5|evaluation/data/R-HORIZON-AIME24/AIME24-combined-n5.jsonl"
  "AIME24|origin|evaluation/data/R-HORIZON-AIME24/AIME24-origin.jsonl"
  "AIME25|n3|evaluation/data/R-HORIZON-AIME25/AIME25-combined-n3.jsonl"
  "AIME25|n4|evaluation/data/R-HORIZON-AIME25/AIME25-combined-n4.jsonl"
  "AIME25|n5|evaluation/data/R-HORIZON-AIME25/AIME25-combined-n5.jsonl"
  "AIME25|origin|evaluation/data/R-HORIZON-AIME25/AIME25-origin.jsonl"
  "AMC23|n4|evaluation/data/R-HORIZON-AMC23/AMC23-combined-n4.jsonl"
  "AMC23|n6|evaluation/data/R-HORIZON-AMC23/AMC23-combined-n6.jsonl"
  "AMC23|n8|evaluation/data/R-HORIZON-AMC23/AMC23-combined-n8.jsonl"
  "AMC23|origin|evaluation/data/R-HORIZON-AMC23/AMC23-origin.jsonl"
)

target_lines_for_tag() {
  local dataset="$1"
  local tag="$2"
  if [ "${dataset}" = "Math500" ] && { [ "${tag}" = "n8" ] || [ "${tag}" = "n16" ] || [ "${tag}" = "origin" ]; }; then
    echo 100
  else
    echo 0
  fi
}

prepare_input_for_task() {
  local dataset="$1"
  local tag="$2"
  local input_file="$3"
  local out_dir="$4"
  local limit
  limit="$(target_lines_for_tag "${dataset}" "${tag}")"
  if [ "${limit}" -le 0 ]; then
    echo "${input_file}"
    return 0
  fi
  local subset_file="${out_dir}/__input_first_${limit}.jsonl"
  python - "${input_file}" "${subset_file}" "${limit}" <<'PY'
import sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
    for i, line in enumerate(f):
        if i >= n:
            break
        g.write(line)
PY
  echo "${subset_file}"
}

is_done() {
  local input_file="$1"
  local result_json="$2"
  local judged_json="$3"
  local stat_txt="$4"
  if [ ! -f "${result_json}" ] || [ ! -f "${judged_json}" ] || [ ! -f "${stat_txt}" ]; then
    return 1
  fi
  local target c1 c2 c3
  target="$(wc -l < "${input_file}")"
  c1="$(wc -l < "${result_json}")"
  c2="$(wc -l < "${judged_json}")"
  c3="$(wc -l < "${stat_txt}")"
  if [ "${c1}" -eq "${target}" ] && [ "${c2}" -eq "${target}" ] && [ "${c3}" -eq "${target}" ]; then
    return 0
  fi
  return 1
}

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

wait_for_post_slot() {
  while true; do
    cleanup_finished_jobs
    if [ "${#BG_PIDS[@]}" -lt "${MAX_BG_POSTPROC}" ]; then
      break
    fi
    sleep 5
  done
}

start_postprocess_bg() {
  local dataset="$1"
  local tag="$2"
  local eval_input="$3"
  local result_json="$4"
  local judged_json="$5"
  local stat_txt="$6"
  (
    set -euo pipefail
    python evaluation/extract.py \
      --input "${result_json}" \
      --output "${judged_json}" \
      --model_name deepseek-reasoner
    python evaluation/judge.py \
      --raw_input "${eval_input}" \
      --prediction "${judged_json}" \
      --output "${stat_txt}"
  ) > "${stat_txt%.txt}_post.log" 2>&1 &
  local pid="$!"
  BG_PIDS+=("${pid}")
  BG_NAMES+=("${dataset}/${tag}")
  echo "[POST START] ${dataset}/${tag} pid=${pid}"
}

for item in "${DATASETS[@]}"; do
  IFS="|" read -r dataset tag input_file <<< "${item}"
  if [ ! -f "${input_file}" ]; then
    echo "[WARN] Missing dataset file, skip: ${input_file}"
    continue
  fi

  out_dir="${OUT_ROOT}/${dataset}/${tag}"
  mkdir -p "${out_dir}"
  base="$(basename "${input_file}" .jsonl)"
  eval_input="$(prepare_input_for_task "${dataset}" "${tag}" "${input_file}" "${out_dir}")"
  result_json="${out_dir}/${base}_result.json"
  judged_json="${out_dir}/${base}_result_judged.json"
  stat_txt="${out_dir}/${base}_result_judged_stat.txt"

  if is_done "${eval_input}" "${result_json}" "${judged_json}" "${stat_txt}"; then
    echo "[SKIP] ${dataset}/${tag} already complete."
    continue
  fi

  echo "[INF START] ${dataset}/${tag}"
  python evaluation/inference.py \
    --input "${eval_input}" \
    --output "${result_json}" \
    --model_name "${MODEL_KEY}" \
    --max_workers "${INFER_MAX_WORKERS}"
  echo "[INF DONE] ${dataset}/${tag}"

  wait_for_post_slot
  start_postprocess_bg "${dataset}" "${tag}" "${eval_input}" "${result_json}" "${judged_json}" "${stat_txt}"
done

while [ "${#BG_PIDS[@]}" -gt 0 ]; do
  cleanup_finished_jobs
  [ "${#BG_PIDS[@]}" -gt 0 ] && sleep 5
done

if [ "${FAIL}" -ne 0 ]; then
  echo "[ALL DONE WITH WARNINGS] Some post-process tasks failed."
  exit 1
fi
echo "[ALL DONE] Remaining original benchmarks finished."
