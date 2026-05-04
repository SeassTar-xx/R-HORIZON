#!/usr/bin/env bash
set -euo pipefail

# Formal GRPO with dense chain reward, combined pkl (n>=2 only), LoRA checkpoints, AIME every 20 steps.
# 默认：total_training_steps=500，save_freq=100（可用环境变量 TOTAL_TRAINING_STEPS / SAVE_FREQ 覆盖）。

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  conda activate /root/autodl-tmp/miniconda_envs/r-horizon-grpo 2>/dev/null || true
fi

export PYTHONPATH="${REPO_ROOT}/training:${PYTHONPATH:-}"
# 单卡默认可见 GPU 0（避免容器内未设置时 Ray/worker 错位）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
# 不显式设置 PYTORCH_CUDA_ALLOC_CONF 可避免部分环境下 expandable_segments 相关断言崩溃。
# 若需限制碎片，可在启动前手动: export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then export PYTORCH_CUDA_ALLOC_CONF; fi
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1

# 默认断开误连的远程 Ray（否则常见：GPU 空闲、metrics 永不增加）。设为 0 可保留 RAY_ADDRESS。
CLEAR_RAY_ADDRESS="${CLEAR_RAY_ADDRESS:-1}"
if [[ "${CLEAR_RAY_ADDRESS}" == "1" ]]; then unset RAY_ADDRESS; fi

# 当前环境 vLLM 未注册 Qwen3ForCausalLM，Qwen3 训练请用 HF；其余已支持模型可 ROLLOUT_BACKEND=vllm
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-hf}"
# vLLM 要求 top_k=-1；HF 常用 top_k=0
if [[ "${ROLLOUT_BACKEND}" == "vllm" ]]; then
  ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
else
  ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-0}"
fi

# 大文件默认写到数据盘（AutoDL 等）：checkpoint / 缓存 / Ray&Python 临时文件，避免撑满系统盘。
DATA_DISK_ROOT="${DATA_DISK_ROOT:-/root/autodl-tmp}"
if [[ -d "${DATA_DISK_ROOT}" ]]; then
  export TMPDIR="${TMPDIR:-${DATA_DISK_ROOT}/tmp}"
  mkdir -p "${TMPDIR}"
  export RAY_TMPDIR="${RAY_TMPDIR:-${DATA_DISK_ROOT}/ray_tmp}"
  mkdir -p "${RAY_TMPDIR}"
  export HF_HOME="${HF_HOME:-${DATA_DISK_ROOT}/huggingface}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
  export TORCH_HOME="${TORCH_HOME:-${DATA_DISK_ROOT}/torch}"
  export WANDB_DIR="${WANDB_DIR:-${DATA_DISK_ROOT}/wandb}"
  mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}" "${WANDB_DIR}"
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${DATA_DISK_ROOT}/training/checkpoints}"
else
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/training/checkpoints}"
fi
mkdir -p "${CHECKPOINT_ROOT}"

# 单机独占 GPU 时可设 RAY_STOP_BEFORE_TRAIN=1；默认 0（勿误杀共享/远程 Ray 集群）。
if [[ "${RAY_STOP_BEFORE_TRAIN:-0}" == "1" ]]; then
  ray stop --force 2>/dev/null || true
fi

MODEL_PATH="${MODEL_PATH:-/root/models/Qwen3-4B}"
EXP_TAG="${EXP_TAG:-formal500-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/grpo-qwen3-4b-dense-${EXP_TAG}}"
STATS_DIR="${OUTPUT_DIR}/stats"
TRAIN_PKL="${TRAIN_PKL:-${REPO_ROOT}/training/data/combined_mixed_k1234_key_var_sd43_passrate0.25_skywork_or1_train_7b_math_key_variables_filtered.pkl}"
VAL_FILE="${VAL_FILE:-${REPO_ROOT}/training/data/grpo_dense_tiny_val.parquet}"

mkdir -p "${OUTPUT_DIR}" "${STATS_DIR}"

if [[ -f "${OUTPUT_DIR}/latest_checkpointed_iteration.txt" ]]; then
  echo "[INFO] 在同一 OUTPUT_DIR 检测到 checkpoint（latest=$(tr -d '\n' <"${OUTPUT_DIR}/latest_checkpointed_iteration.txt")）；trainer.resume_mode=auto 将从该步继续。"
else
  echo "[INFO] 未检测到 latest_checkpointed_iteration.txt，将从头训练（新目录或尚未保存过 ckpt）。"
fi

# 若存在 global_step_* 目录但与 tracker 不一致，可用下面命令同步后再训（将 tracker 指到最大 global_step_N）：
#   python3 - <<'PY'
#   import glob, os
#   root = os.environ["OUTPUT_DIR"]
#   dirs = sorted(glob.glob(os.path.join(root, "global_step_*")), key=lambda p: int(p.rsplit("_",1)[-1]))
#   assert dirs
#   last = int(dirs[-1].rsplit("_",1)[-1])
#   open(os.path.join(root, "latest_checkpointed_iteration.txt"), "w").write(str(last))
#   print("tracker ->", last)
#   PY

# AIME 评测间隔：默认拉大以降低阻塞（加速）；需要密集监控时再改小。
AIME_EVAL_FREQ="${AIME_EVAL_FREQ:-250}"
ENABLE_LLM_JUDGE="${ENABLE_LLM_JUDGE:-true}"

# LLM-judge：优先环境变量，否则从仓库 evaluation/config.json 的 extract.deepseek-reasoner 读取
if [[ -z "${DEEPSEEK_API_KEY:-}" && -f "${REPO_ROOT}/evaluation/config.json" ]]; then
  DEEPSEEK_API_KEY="$(python3 -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); print(d['extract']['deepseek-reasoner']['api_key'])" "${REPO_ROOT}/evaluation/config.json")"
fi
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "[WARN] DEEPSEEK_API_KEY 仍为空：LLM judge 无法调用 DeepSeek（dense_llm_calls 将为 0）。请设置环境变量或在 evaluation/config.json 中配置 extract.deepseek-reasoner.api_key。" >&2
else
  echo "[INFO] 已加载 DeepSeek API key（长度 ${#DEEPSEEK_API_KEY}），LLM judge 将发起远程调用。"
fi

if [[ ! -f "${TRAIN_PKL}" ]]; then
  echo "ERROR: train pkl not found: ${TRAIN_PKL}" >&2
  exit 1
fi
# Val file: reuse tiny val if present (only for dataloader init; main val is AIME bench + skip default val)
if [[ ! -f "${VAL_FILE}" ]]; then
  VAL_FILE="${REPO_ROOT}/training/data/grpo_dense_tiny_train.parquet"
fi

python3 -m verl.trainer.main_ppo \
  hydra.run.dir="${OUTPUT_DIR}/hydra_run" \
  hydra.job.chdir=false \
  algorithm.adv_estimator=grpo \
  algorithm.use_reward_clip=True \
  data.train_files="[\"${TRAIN_PKL}\"]" \
  data.val_files="[\"${VAL_FILE}\"]" \
  data.min_composed_query_num=2 \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-1}" \
  data.val_batch_size="${VAL_BATCH_SIZE:-4}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-1536}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-512}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.trust_remote_code=True \
  +actor_rollout_ref.model.lora_rank=16 \
  +actor_rollout_ref.model.lora_alpha=32 \
  +actor_rollout_ref.model.target_modules="[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\"gate_proj\",\"up_proj\",\"down_proj\"]" \
  +actor_rollout_ref.model.keep_wrap_policy_for_hf=True \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.use_liger=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${LR:-3e-6}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  +actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
  +actor_rollout_ref.actor.entropy_logits_chunk_size="${ENTROPY_CHUNK:-24}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH:-2}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH:-1}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOK:-4096}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.rollout.name="${ROLLOUT_BACKEND}" \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GEN_GPU_MEM:-0.58}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_BATCHED_TOK:-6144}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS:-32}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-2}" \
  actor_rollout_ref.rollout.n_val=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  +critic.model.trust_remote_code=True \
  +critic.model.fsdp_config.use_orig_params=True \
  critic.model.use_remove_padding=False \
  critic.ppo_micro_batch_size_per_gpu=1 \
  reward_model.reward_manager=dense_chain \
  reward_model.dense_reward.max_workers="${DENSE_WORKERS:-20}" \
  reward_model.dense_reward.timeout="${DENSE_LLM_TIMEOUT:-40}" \
  reward_model.dense_reward.enable_llm_judge="${ENABLE_LLM_JUDGE}" \
  reward_model.dense_reward.api_key="${DEEPSEEK_API_KEY:-}" \
  trainer.project_name=R-HORIZON-RL-Training \
  trainer.experiment_name="grpo-qwen3-4b-dense-${EXP_TAG}" \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-500}" \
  trainer.resume_mode=auto \
  +trainer.skip_validation=True \
  trainer.test_freq=100000 \
  trainer.save_freq="${SAVE_FREQ:-100}" \
  +trainer.save_resolved_config_path="${OUTPUT_DIR}/resolved_config.yaml" \
  trainer.stats_path="${STATS_DIR}" \
  trainer.stats_save_freq=1 \
  +trainer.aime_eval_freq="${AIME_EVAL_FREQ}" \
  +trainer.aime_eval_batch_size="${AIME_EVAL_BATCH:-4}" \
  trainer.default_local_dir="${OUTPUT_DIR}"

echo "[DONE] Formal run finished. Output: ${OUTPUT_DIR}"
