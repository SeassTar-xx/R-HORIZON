#!/usr/bin/env bash
# GRPO 训练入口（在 tmux 中启动）。密钥从数据盘 env 文件读取，勿提交仓库。
set -euo pipefail
ROOT="/root/R-HORIZON"
ENVF="${ENVF:-/root/autodl-tmp/.env.r_horizon}"
if [[ -f "$ENVF" ]]; then
  # shellcheck disable=SC1090
  source "$ENVF"
fi
if [[ -z "${LLM_JUDGE_API_KEY:-}" ]]; then
  echo "请设置 LLM_JUDGE_API_KEY（或创建 $ENVF 并 export LLM_JUDGE_API_KEY=...）"
  exit 1
fi

export PYTHONPATH="${ROOT}/training"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RH_CKPT_DIR="${RH_CKPT_DIR:-/root/autodl-tmp/r_horizon_ckpts}"
export RH_STATS_DIR="${RH_STATS_DIR:-/root/autodl-tmp/r_horizon_grpo_logs}"
mkdir -p "$RH_CKPT_DIR" "$RH_STATS_DIR/grpo_qwen3_4b_n234"

cd "${ROOT}/training"
LOG="${RH_STATS_DIR}/grpo_qwen3_4b_n234/train_console.log"
echo "[$(date -Is)] starting GRPO -> logging to $LOG"
exec /root/autodl-tmp/miniconda_envs/r-horizon-grpo/bin/python -u -m verl.trainer.main_ppo 2>&1 | tee -a "$LOG"
