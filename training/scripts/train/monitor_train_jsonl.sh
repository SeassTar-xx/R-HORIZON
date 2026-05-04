#!/usr/bin/env bash
# 监控 train_metrics_steps_all.jsonl：累计新增 LINE_DELTA 行则成功退出；
# 若连续 STALL_MINUTES 分钟行数未变，认为卡死：可选杀训练进程并退出 3；
# 总等待超过 MAX_WAIT_MINUTES 分钟退出 4（避免无限轮询）。
set -euo pipefail

OUT="${OUTPUT_DIR:?请设置 OUTPUT_DIR}"
J="${OUT}/stats/train_metrics_steps_all.jsonl"
LINE_DELTA="${LINE_DELTA:-50}"
POLL_SEC="${POLL_SEC:-120}"
STALL_MINUTES="${STALL_MINUTES:-50}"
MAX_WAIT_MINUTES="${MAX_WAIT_MINUTES:-720}"
KILL_ON_STALL="${KILL_ON_STALL:-1}"

mkdir -p "$(dirname "$J")"
touch "$J" 2>/dev/null || true

start_lines=$(wc -l <"$J" | tr -d ' ')
target=$((start_lines + LINE_DELTA))
start_ts=$(date +%s)
last_lines="$start_lines"
last_prog_ts="$start_ts"

echo "[watchdog] OUT=$OUT start_lines=$start_lines target_lines=$target stall_after=${STALL_MINUTES}m max_wait=${MAX_WAIT_MINUTES}m poll=${POLL_SEC}s"

while true; do
  sleep "$POLL_SEC"
  now=$(date +%s)
  lines=$(wc -l <"$J" | tr -d ' ')
  if [[ "$lines" -gt "$last_lines" ]]; then
    last_prog_ts="$now"
    last_lines="$lines"
    tail -1 "$J" | python3 -c "import sys,json; d=json.load(sys.stdin); print('[watchdog]', 'lines=', '$lines', 'step=', d.get('step'), 'dense_llm_calls=', d.get('dense_llm_calls'))" 2>/dev/null || echo "[watchdog] lines=$lines"
  fi

  if [[ "$lines" -ge "$target" ]]; then
    echo "[watchdog] OK: reached $lines lines (>= $target)"
    exit 0
  fi

  stall_secs=$((now - last_prog_ts))
  if [[ "$stall_secs" -ge $((STALL_MINUTES * 60)) ]]; then
    echo "[watchdog] STALL: no new metrics for ${STALL_MINUTES}m (lines=$lines)" >&2
    if [[ "$KILL_ON_STALL" == "1" ]]; then
      pkill -f "verl.trainer.main_ppo" 2>/dev/null || true
    fi
    exit 3
  fi

  waited_min=$(( (now - start_ts) / 60 ))
  if [[ "$waited_min" -ge "$MAX_WAIT_MINUTES" ]]; then
    echo "[watchdog] TIMEOUT after ${MAX_WAIT_MINUTES}m (lines=$lines, target=$target)" >&2
    exit 4
  fi
done
