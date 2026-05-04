#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
source "${CONDA_SH}"
conda activate r-horizon
cd "${REPO_ROOT}"

RESULT_ROOT="evaluation/result_qwenn3_eval"
MODEL_KEY="qwen3-4b-local"
MAX_ROUNDS_PER_DATASET=8
WATCH_INTERVAL_SEC=30
STALL_LIMIT_SEC=900

cleanup_eval_processes() {
  # Ensure there is only one active eval pipeline at any time.
  pkill -f "bash evaluation/run.sh" 2>/dev/null || true
  pkill -f "python evaluation/inference.py" 2>/dev/null || true
  pkill -f "python evaluation/extract.py" 2>/dev/null || true
  pkill -f "python evaluation/judge.py" 2>/dev/null || true
}

start_api() {
  tmux kill-session -t qwen3_api 2>/dev/null || true
  tmux new-session -d -s qwen3_api \
    "source ${CONDA_SH} && conda activate r-horizon && CUDA_VISIBLE_DEVICES=0 python evaluation/qwen3_local_server.py --model_path /root/models/Qwen3-4B --host 127.0.0.1 --port 8000"
  python - <<'PY'
import time, requests, sys
url = "http://127.0.0.1:8000/v1/chat/completions"
payload = {
    "model": "qwen3-4b-local",
    "messages": [{"role": "user", "content": "Reply OK only."}],
    "temperature": 0.0,
    "max_tokens": 8,
}
for _ in range(60):
    try:
        r = requests.post(url, json=payload, timeout=8)
        if r.status_code == 200:
            print("[API] ready")
            raise SystemExit(0)
    except Exception:
        pass
    time.sleep(2)
print("[API] not ready in time")
raise SystemExit(1)
PY
}

repair_bad_rows() {
  local result_json="$1"
  local judged_json="$2"
  local stat_txt="$3"
  [ -f "${result_json}" ] || return 0
  local before
  before="$(wc -l < "${result_json}" || echo 0)"
  python - "${result_json}" <<'PY'
import json, sys
path = sys.argv[1]
rows = []
seen = set()
with open(path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        k = obj.get("key")
        if k in seen:
            continue
        seen.add(k)
        resp = obj.get("response", "")
        # Remove obviously broken generations, let run.sh refill them.
        if isinstance(resp, str) and (len(resp) < 300 or resp.count("!") > 40):
            continue
        rows.append(obj)
with open(path, "w", encoding="utf-8") as f:
    for obj in rows:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
print(f"repaired_rows={len(rows)}")
PY
  local after
  after="$(wc -l < "${result_json}" || echo 0)"
  if [[ "${after}" -lt "${before}" ]]; then
    rm -f "${judged_json}" "${stat_txt}"
    echo "[REPAIR] trimmed ${before}->${after}, removed stale judged/stat"
  fi
}

dataset_done() {
  local input_file="$1"
  local outdir="$2"
  local base
  base="$(basename "${input_file}" .jsonl)"
  local rj="${outdir}/${base}_result.json"
  local ej="${outdir}/${base}_result_judged.json"
  local sj="${outdir}/${base}_result_judged_stat.txt"
  local target
  target="$(wc -l < "${input_file}")"
  local c1 c2 c3
  c1="$( [ -f "${rj}" ] && wc -l < "${rj}" || echo 0 )"
  c2="$( [ -f "${ej}" ] && wc -l < "${ej}" || echo 0 )"
  c3="$( [ -f "${sj}" ] && wc -l < "${sj}" || echo 0 )"
  echo "[CHECK] ${base} input=${target} result=${c1} judged=${c2} stat=${c3}"
  [[ "${c1}" -eq "${target}" && "${c2}" -eq "${target}" && "${c3}" -eq "${target}" ]]
}

run_with_watchdog() {
  local input_file="$1"
  local outdir="$2"
  local base="$3"
  local result_json="${outdir}/${base}_result.json"

  bash evaluation/run.sh "${input_file}" "${outdir}" "${MODEL_KEY}" &
  local job_pid=$!

  local last_count=-1
  local stalled_for=0
  while kill -0 "${job_pid}" 2>/dev/null; do
    local cur_count=0
    if [[ -f "${result_json}" ]]; then
      cur_count="$(wc -l < "${result_json}" || echo 0)"
    fi

    if [[ "${cur_count}" -gt "${last_count}" ]]; then
      last_count="${cur_count}"
      stalled_for=0
    else
      stalled_for=$((stalled_for + WATCH_INTERVAL_SEC))
    fi

    if [[ "${stalled_for}" -ge "${STALL_LIMIT_SEC}" ]]; then
      echo "[WATCHDOG] No progress for ${STALL_LIMIT_SEC}s, terminating current run.sh (pid=${job_pid})"
      kill "${job_pid}" 2>/dev/null || true
      wait "${job_pid}" 2>/dev/null || true
      cleanup_eval_processes
      return 124
    fi

    sleep "${WATCH_INTERVAL_SEC}"
  done

  wait "${job_pid}"
  return $?
}

cleanup_eval_processes
start_api

for input_file in evaluation/data/R-HORIZON-*/*.jsonl; do
  ds="$(basename "$(dirname "${input_file}")" | sed 's/^R-HORIZON-//')"
  base="$(basename "${input_file}" .jsonl)"
  tag="$(echo "${base}" | sed "s/^${ds}-//")"
  outdir="${RESULT_ROOT}/${ds}/${tag}"
  mkdir -p "${outdir}"

  echo "[DATASET] ${input_file} -> ${outdir}"
  ok=0
  round=0
  while true; do
    round=$((round + 1))
    echo "[ROUND ${round}] running..."
    cleanup_eval_processes
    repair_bad_rows "${outdir}/${base}_result.json" "${outdir}/${base}_result_judged.json" "${outdir}/${base}_result_judged_stat.txt"
    if ! run_with_watchdog "${input_file}" "${outdir}" "${base}"; then
      echo "[WARN] run.sh failed, restarting API..."
      start_api
      continue
    fi
    if dataset_done "${input_file}" "${outdir}"; then
      echo "[OK] ${base} completed."
      ok=1
      break
    fi
    echo "[WARN] incomplete dataset, restarting API and retrying..."
    start_api
  done

  if [[ "${ok}" -ne 1 ]]; then
    echo "[FAIL] ${base} incomplete after retries."
  fi
done

echo "[DONE] Supervised full evaluation finished."
