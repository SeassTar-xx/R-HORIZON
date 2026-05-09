#!/usr/bin/env bash
# 在 shell 中 source 此文件以统一数据盘路径（推理 / 训练共用 MODEL_PATH）。
export AUTODL_DATA="${AUTODL_DATA:-/root/autodl-tmp}"
export HF_HOME="${HF_HOME:-$AUTODL_DATA/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

# 推理与评测（较轻依赖）：系统 conda
export CONDA_ENV_INFERENCE="${CONDA_ENV_INFERENCE:-/root/miniconda3/envs/r-horizon}"

# GRPO / verl 训练（requirements.txt 全量）：数据盘 conda
export CONDA_ENV_GRPO="${CONDA_ENV_GRPO:-/root/autodl-tmp/miniconda_envs/r-horizon-grpo}"

# 本地权重目录（/root/models -> autodl-tmp/models_store）
export MODEL_PATH="${MODEL_PATH:-/root/models/qwen3-4b}"
