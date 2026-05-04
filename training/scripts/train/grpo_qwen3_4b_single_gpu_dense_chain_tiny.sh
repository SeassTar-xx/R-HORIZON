#!/usr/bin/env bash
set -euo pipefail

# Single-GPU tiny GRPO run with dense chain reward.
# This script is intended for pipeline smoke-start and reward integration checks.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
source "${CONDA_SH}"
conda activate /root/autodl-tmp/miniconda_envs/r-horizon-grpo

export PYTHONPATH="${REPO_ROOT}/training:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1

MODEL_PATH="${MODEL_PATH:-/root/models/Qwen3-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/training/checkpoints/grpo-qwen3-4b-dense-tiny}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/training/data}"

mkdir -p "${OUTPUT_DIR}" "${DATA_DIR}"

TRAIN_FILE="${DATA_DIR}/grpo_dense_tiny_train.parquet"
VAL_FILE="${DATA_DIR}/grpo_dense_tiny_val.parquet"
export TRAIN_FILE VAL_FILE

python - <<'PY'
import os
import pandas as pd

train_file = os.environ["TRAIN_FILE"]
val_file = os.environ["VAL_FILE"]

rows = [
    {
        "prompt": [{"role": "user", "content": "已知a=3。问题1：求a+2。问题2：用问题1结果乘4。按 Problem 1/2 分步作答，并给出最终答案。"}],
        "data_source": "train-math-multi-query",
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ["5", "20"]},
        "extra_info": {
            "index": 0,
            "composed_query_num": 2,
            "dependencies": [["v2", "a1*4"]],
            "difficulty": "easy",
        },
    },
    {
        "prompt": [{"role": "user", "content": "设x=7。问题1：求x-1。问题2：将问题1结果平方。请按 Problem 1/2 依次推理并作答。"}],
        "data_source": "train-math-multi-query",
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ["6", "36"]},
        "extra_info": {
            "index": 1,
            "composed_query_num": 2,
            "dependencies": [["v2", "a1^2"]],
            "difficulty": "easy",
        },
    },
    {
        "prompt": [{"role": "user", "content": "已知b=10。问题1：求b/2。问题2：把问题1结果加上9。按 Problem 1/2 输出。"}],
        "data_source": "train-math-multi-query",
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ["5", "14"]},
        "extra_info": {
            "index": 2,
            "composed_query_num": 2,
            "dependencies": [["v2", "a1+9"]],
            "difficulty": "easy",
        },
    },
]

# Replicate samples to provide enough throughput for GPU and stable multi-step PPO.
expanded_train = []
expanded_val = []
for i in range(24):
    base = dict(rows[i % 3])
    base["extra_info"] = dict(base["extra_info"])
    base["extra_info"]["index"] = i
    if i < 18:
        expanded_train.append(base)
    else:
        expanded_val.append(base)

pd.DataFrame(expanded_train).to_parquet(train_file, index=False)
pd.DataFrame(expanded_val).to_parquet(val_file, index=False)
print("train_file=", train_file)
print("val_file=", val_file)
PY

EXP_NAME="grpo-qwen3-4b-dense-tiny-$(date +%Y%m%d-%H%M%S)"
STATS_DIR="${OUTPUT_DIR}/stats"
mkdir -p "${STATS_DIR}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_reward_clip=True \
  data.train_files="[\"${TRAIN_FILE}\"]" \
  data.val_files="[\"${VAL_FILE}\"]" \
  data.min_composed_query_num=2 \
  data.train_batch_size=2 \
  data.val_batch_size=2 \
  data.max_prompt_length=512 \
  data.max_response_length=192 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.trust_remote_code=True \
  +actor_rollout_ref.model.lora_rank=16 \
  +actor_rollout_ref.model.lora_alpha=32 \
  +actor_rollout_ref.model.target_modules="[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\"gate_proj\",\"up_proj\",\"down_proj\"]" \
  +actor_rollout_ref.model.keep_wrap_policy_for_hf=True \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.use_liger=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  +actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_k=0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
  actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.n_val=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  +critic.model.trust_remote_code=True \
  +critic.model.fsdp_config.use_orig_params=True \
  critic.model.use_remove_padding=False \
  critic.ppo_micro_batch_size_per_gpu=1 \
  reward_model.reward_manager=dense_chain \
  reward_model.dense_reward.max_workers=2 \
  reward_model.dense_reward.enable_llm_judge=False \
  reward_model.dense_reward.api_key="${DEEPSEEK_API_KEY:-}" \
  trainer.project_name=R-HORIZON-RL-Training \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=6 \
  +trainer.skip_validation=True \
  trainer.test_freq=1000 \
  trainer.save_freq=1000 \
  trainer.stats_path="${STATS_DIR}" \
  trainer.default_local_dir="${OUTPUT_DIR}"

echo "[DONE] Dense tiny GRPO run finished."

