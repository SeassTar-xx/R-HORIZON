#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
source "${CONDA_SH}"
conda activate r-horizon
cd "${REPO_ROOT}"

OUT="${1:-/root/autodl-tmp/evaluation-results/result_auto_20260429_004737}"
MODEL_NAME="qwen3-4b-local"

mkdir -p "${OUT}"
echo "[RESUME] output=${OUT}"

# Continue Math500 post-process in background (supports resume by key).
(
  python evaluation/extract.py \
    --input "${OUT}/Math500-combined-n2_result.json" \
    --output "${OUT}/Math500-combined-n2_result_judged.json" \
    --model_name deepseek-reasoner
  python evaluation/judge.py \
    --raw_input evaluation/data/R-HORIZON-Math500/Math500-combined-n2.jsonl \
    --prediction "${OUT}/Math500-combined-n2_result_judged.json" \
    --output "${OUT}/Math500-combined-n2_result_judged_stat.txt"
) > "${OUT}/Math500-combined-n2_post.log" 2>&1 &

DATASETS=(
  "evaluation/data/R-HORIZON-AIME24/AIME24-combined-n2.jsonl"
  "evaluation/data/R-HORIZON-AIME25/AIME25-combined-n2.jsonl"
  "evaluation/data/R-HORIZON-AMC23/AMC23-combined-n2.jsonl"
)

for f in "${DATASETS[@]}"; do
  b="$(basename "${f}" .jsonl)"
  echo "[INF START] ${f}"
  python evaluation/inference.py \
    --input "${f}" \
    --output "${OUT}/${b}_result.json" \
    --model_name "${MODEL_NAME}"
  echo "[INF DONE] ${b}"

  (
    python evaluation/extract.py \
      --input "${OUT}/${b}_result.json" \
      --output "${OUT}/${b}_result_judged.json" \
      --model_name deepseek-reasoner
    python evaluation/judge.py \
      --raw_input "${f}" \
      --prediction "${OUT}/${b}_result_judged.json" \
      --output "${OUT}/${b}_result_judged_stat.txt"
  ) > "${OUT}/${b}_post.log" 2>&1 &
done

wait
echo "[DONE] resumed remaining datasets."
