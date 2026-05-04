#!/usr/bin/env bash
set -euo pipefail

# Single-GPU tiny GRPO run for constrained servers.
# Designed for "can run + quick smoke fine-tune" verification.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
source "${CONDA_SH}"
conda activate r-horizon-grpo

export PYTHONPATH="${REPO_ROOT}/training:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1

MODEL_PATH="${MODEL_PATH:-/root/models/Qwen3-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/training/checkpoints/grpo-qwen3-4b-tiny}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/training/data}"
MASTER_PORT="${MASTER_PORT:-29531}"

mkdir -p "${OUTPUT_DIR}" "${DATA_DIR}"

TRAIN_FILE="${DATA_DIR}/grpo_tiny_train.parquet"
VAL_FILE="${DATA_DIR}/grpo_tiny_val.parquet"
export TRAIN_FILE VAL_FILE

# Build a tiny local RLHF-style dataset if missing.
python - <<'PY'
import os
import pandas as pd

train_file = os.environ["TRAIN_FILE"]
val_file = os.environ["VAL_FILE"]

def build_rows():
    qs = [
        ("计算 7 + 5 的结果，只输出最终数字。", "12"),
        ("计算 9 * 6 的结果，只输出最终数字。", "54"),
        ("求 100 - 37 的结果，只输出最终数字。", "63"),
        ("计算 81 / 9 的结果，只输出最终数字。", "9"),
        ("若 x=8，求 3x+2 的值，只输出最终数字。", "26"),
        ("计算 14 + 19 的结果，只输出最终数字。", "33"),
        ("计算 11 * 11 的结果，只输出最终数字。", "121"),
        ("求 2 的 5 次方，只输出最终数字。", "32"),
    ]
    rows = []
    for i, (q, ans) in enumerate(qs):
        rows.append({
            "prompt": [{"role": "user", "content": q}],
            "data_source": "math",
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ans},
            "extra_info": {"index": i},
        })
    return rows

rows = build_rows()
if not os.path.exists(train_file):
    pd.DataFrame(rows[:6]).to_parquet(train_file, index=False)
if not os.path.exists(val_file):
    pd.DataFrame(rows[6:]).to_parquet(val_file, index=False)
print("train_file=", train_file)
print("val_file=", val_file)
PY

RAY_ADDRESS="127.0.0.1:${MASTER_PORT}"
ray start --head --port="${MASTER_PORT}" --num-gpus=1 --include-dashboard=false >/dev/null 2>&1 || true

EXP_NAME="grpo-qwen3-4b-tiny-$(date +%Y%m%d-%H%M%S)"
STATS_DIR="${OUTPUT_DIR}/stats"
mkdir -p "${STATS_DIR}"

ray job submit \
  --address "${RAY_ADDRESS}" \
  --runtime-env="${REPO_ROOT}/training/verl/trainer/runtime_env.yaml" \
  --working-dir="${REPO_ROOT}" \
  -- python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_reward_clip=True \
  data.train_files="[\"${TRAIN_FILE}\"]" \
  data.val_files="[\"${VAL_FILE}\"]" \
  data.train_batch_size=4 \
  data.val_batch_size=2 \
  data.max_prompt_length=256 \
  data.max_response_length=128 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
  actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.n_val=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  critic.model.trust_remote_code=True \
  critic.model.use_remove_padding=False \
  critic.ppo_micro_batch_size_per_gpu=1 \
  trainer.project_name=R-HORIZON-RL-Training \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=5 \
  trainer.test_freq=1 \
  trainer.save_freq=1 \
  trainer.stats_path="${STATS_DIR}" \
  trainer.default_local_dir="${OUTPUT_DIR}"

ray stop >/dev/null 2>&1 || true
echo "[DONE] Tiny GRPO run finished."
