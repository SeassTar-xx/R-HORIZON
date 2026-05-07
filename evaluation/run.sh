#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE=$1
OUTPUT_DIR=$2
MODEL_NAME=$3
MAX_WORKERS="${4:-1}"
EXTRACT_WORKERS="${5:-4}"
INPUT_FILENAME=$(basename "$INPUT_FILE" .jsonl)

# 仅允许数字，避免异常 $4/$5 破坏整行解析（曾见 exit 127: _workers: command not found）
case "${MAX_WORKERS}" in
  ''|*[!0-9]*) MAX_WORKERS=1 ;;
esac
case "${EXTRACT_WORKERS}" in
  ''|*[!0-9]*) EXTRACT_WORKERS=4 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[ERROR] python/python3 not in PATH" >&2
    exit 127
  fi
fi

mkdir -p "$OUTPUT_DIR"

# 抽取阶段使用的 config.json 中 extract.* 的键名（默认与 config.example.json 一致）
EXTRACT_MODEL_NAME="${EXTRACT_MODEL_NAME:-extract-llm}"

# Step1 Inference（SKIP_INFERENCE=1 时跳过，仅跑 extract+judge，用于补跑抽取）
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"
if [[ "${SKIP_INFERENCE}" != "1" ]]; then
  "${PYTHON_BIN}" evaluation/inference.py --input "$INPUT_FILE" --output "$OUTPUT_DIR/${INPUT_FILENAME}_result.json" --model_name "$MODEL_NAME" --max_workers="${MAX_WORKERS}"
else
  echo "[INFO] SKIP_INFERENCE=1: skip inference, require existing ${OUTPUT_DIR}/${INPUT_FILENAME}_result.json"
  [[ -f "${OUTPUT_DIR}/${INPUT_FILENAME}_result.json" ]] || { echo "[ERROR] missing inference result" >&2; exit 1; }
fi
# # Step2 use llm to extract nested answer from inference result
"${PYTHON_BIN}" evaluation/extract.py --input "$OUTPUT_DIR/${INPUT_FILENAME}_result.json" --output "$OUTPUT_DIR/${INPUT_FILENAME}_result_judged.json" --model_name "${EXTRACT_MODEL_NAME}" --max_workers="${EXTRACT_WORKERS}"
# # Step3 use verify script to judge
"${PYTHON_BIN}" evaluation/judge.py --raw_input "$INPUT_FILE" --prediction "$OUTPUT_DIR/${INPUT_FILENAME}_result_judged.json" --output "$OUTPUT_DIR/${INPUT_FILENAME}_result_judged_stat.txt"
#python evaluation/judge.py --raw_input test1.jsonl --prediction evaluation/result/test1_result_judged.json --output evaluation/result/test1_result_judged_stat.txt