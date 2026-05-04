#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
source "${CONDA_SH}"
conda activate r-horizon
cd "${REPO_ROOT}"

MODEL_KEY="qwen3-4b-local"
RESULT_ROOT="evaluation/result_qwenn3_eval/Math500"

start_api() {
  tmux kill-session -t qwen3_api 2>/dev/null || true
  tmux new-session -d -s qwen3_api \
    "source ${CONDA_SH} && conda activate r-horizon && CUDA_VISIBLE_DEVICES=0 python evaluation/qwen3_local_server.py --model_path /root/models/Qwen3-4B --host 127.0.0.1 --port 8000"
  python - <<'PY'
import time
import requests

url = "http://127.0.0.1:8000/v1/chat/completions"
payload = {
    "model": "qwen3-4b-local",
    "messages": [{"role": "user", "content": "Reply OK only."}],
    "temperature": 0.0,
    "max_tokens": 8,
}
for _ in range(90):
    try:
        r = requests.post(url, json=payload, timeout=8)
        if r.status_code == 200:
            print("[API] ready")
            raise SystemExit(0)
    except Exception:
        pass
    time.sleep(2)
print("[API] not ready")
raise SystemExit(1)
PY
}

start_api

DATASET_ORDER=(
  "evaluation/data/R-HORIZON-Math500/Math500-combined-n2.jsonl"
  "evaluation/data/R-HORIZON-Math500/Math500-combined-n4.jsonl"
  "evaluation/data/R-HORIZON-Math500/Math500-combined-n8.jsonl"
  "evaluation/data/R-HORIZON-Math500/Math500-combined-n16.jsonl"
  "evaluation/data/R-HORIZON-Math500/Math500-origin.jsonl"
)

for input_file in "${DATASET_ORDER[@]}"; do
  [ -f "${input_file}" ] || continue
  base="$(basename "${input_file}" .jsonl)"
  tag="${base#Math500-}"
  outdir="${RESULT_ROOT}/${tag}"
  mkdir -p "${outdir}"
  echo "[DATASET] ${input_file} -> ${outdir}"

  while true; do
    target="$(wc -l < "${input_file}")"
    result_json="${outdir}/${base}_result.json"
    c1="$( [ -f "${result_json}" ] && wc -l < "${result_json}" || echo 0 )"

    if [[ "${c1}" -ge "${target}" ]]; then
      echo "[OK] ${base} result ready ${c1}/${target}"
      break
    fi

    pkill -f "python evaluation/inference.py" 2>/dev/null || true

    if ! bash evaluation/run.sh "${input_file}" "${outdir}" "${MODEL_KEY}"; then
      echo "[WARN] run failed, restarting API..."
      start_api
      continue
    fi
  done
done

echo "[DONE] Math500-only evaluation finished."
