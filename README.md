# LongHorizon-RL

LongHorizon-RL studies how to train small and mid-sized language models for
dependency-based long-horizon mathematical reasoning. The project focuses on
process-aware reward shaping: instead of supervising only the final answer, it
turns a long reasoning trajectory into several interpretable reward dimensions
that can be used by reinforcement learning.

The current implementation is built around composed mathematical reasoning
samples. Each sample contains multiple sub-problems. Later sub-problems depend on
answers produced by earlier ones, so the model must maintain intermediate state,
use cross-step dependencies correctly, and still produce complete parseable
answers.

## Motivation

Long-Horizon Reasoning requires a model to maintain context, intermediate
variables, and cross-step dependencies over an extended reasoning chain. This is
harder than solving isolated single-step problems. It also creates a supervision
problem: a reward based only on the final answer is too sparse, while requiring
every sub-problem to be correct before giving any signal can make reinforcement
learning unstable.

This repository explores a denser alternative for mathematical reasoning tasks.
The reward manager decomposes the quality of a generated trajectory into five
process-level signals:

- **Progress reward**: whether each sub-problem is solved correctly and
  efficiently.
- **Complementarity reward**: whether later sub-problems correctly use previous
  answers or explicit dependency variables.
- **Negativity reward**: whether the trajectory avoids verifiable mistakes and
  handles self-correction reasonably.
- **Consistency reward**: whether stated answers are supported by their
  reasoning process.
- **Format reward**: whether the output is complete, stable, and easy to parse.

These signals are combined into a single reward for GRPO/PPO-style RL training.

## Repository Structure

```text
data_contruction/          Scripts for filtering, key-variable selection, and composed sample construction
evaluation/                Inference, extraction, and judging utilities
training/                  verl-based RL training code and reward managers
requirements.txt           Python dependencies used by the project scripts
```

## Installation

```bash
git clone <your-repository-url>
cd <your-repository-directory>

conda create -n longhorizon-rl python=3.10 -y
conda activate longhorizon-rl

pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip install -r requirements.txt
```

Adjust the PyTorch and CUDA versions according to your hardware.

## Data Construction

The data construction pipeline converts independent mathematical problems into
composed long-horizon samples with explicit answer dependencies.

### 1. Filter Integer-Valued Samples

```bash
python data_contruction/step1_filt_integer_samples.py
```

This step keeps samples whose inputs contain valid integer variables and whose
targets are pure integers. Ambiguous numeric expressions such as fractions,
floats, or unresolved LaTeX commands are filtered out.

### 2. Select Key Variables

```bash
python data_contruction/step2_select_key_varible.py
```

This step identifies key variables: integers in the problem statement that
strongly affect the final answer. Configure the API endpoint and key in the
script or through your local environment before running it.

### 3. Compose Dependent Problems

```bash
python data_contruction/step3_combine_problems.py
```

This step links multiple sub-problems into one long-horizon sample. A later
problem can use the previous answer as an intermediate variable, forming a
verifiable cross-problem dependency chain.

## Training

The training code is based on `verl` and supports GRPO-style optimization with
the process-aware dense reward manager.

Prepare your training and validation data under `training/data/`, then launch a
Ray job from the `training/` directory:

```bash
export MODEL_PATH="/path/to/your/base/model"
export TRAIN_DATA_DIR="./data"
export OUTPUT_DIR="./checkpoints/process-aware-run"
mkdir -p "$OUTPUT_DIR"

ray start --head --port=29500 --num-gpus=4 --include-dashboard=false

ray job submit \
  --address "127.0.0.1:29500" \
  --runtime-env="./verl/trainer/runtime_env.yaml" \
  --working-dir="." \
  -- python3 -m verl.trainer.main_ppo \
    reward_model.reward_manager=dense_chain \
    algorithm.adv_estimator=grpo \
    data.train_files='["'"${TRAIN_DATA_DIR}/your_train.parquet"'"]' \
    data.val_files='["'"${TRAIN_DATA_DIR}/your_val.parquet"'"]' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1

ray stop
```

### Training Details

All methods use the same data construction and inference settings. For
reinforcement learning baselines, GRPO is used as the optimization framework,
and these baselines differ only in reward design. The default training
configuration follows the TRACE-RL experiment setting:

- total training steps: `800`
- actor learning rate: `5e-7`
- rollout temperature: `0.8`
- training batch size: `4` prompts per step
- group size: `16` candidate responses per prompt

Here, batch size denotes the number of prompts used in each training step, while
group size denotes the number of candidate responses sampled for each prompt.
Experiments are conducted on NVIDIA RTX 4090 GPUs with 24GB memory, using
gradient accumulation and group-wise rollout sampling to support GRPO training.

The process-aware reward weights are configured in
`training/verl/trainer/config/ppo_trainer.yaml`:

```yaml
data:
  train_batch_size: 4

actor_rollout_ref:
  actor:
    optim:
      lr: 5e-7
  rollout:
    temperature: 0.8
    n: 16

reward_model:
  reward_manager: dense_chain
  dense_reward:
    reward_weights:
      prog: 0.4
      comp: 0.2
      neg: 0.15
      cons: 0.15
      form: 0.1

trainer:
  total_training_steps: 800
  n_gpus_per_node: 4
```

For the process-aware reward in TRACE-RL, the weights are set to `0.4`, `0.2`,
`0.15`, `0.15`, and `0.1` for progression, complementarity, negation,
consistency, and format rewards, respectively. For reward terms requiring
semantic judgment, including complementarity, negation, consistency, and format
clarity, DeepSeek is used as an OpenAI-compatible LLM judge with numeric-output
prompts.

## Evaluation

Create a local evaluation config from the template:

```bash
cp evaluation/config.example.json evaluation/config.json
```

Edit `evaluation/config.json` with your inference endpoint, model name, and API
keys. Then run:

```bash
sh evaluation/run.sh evaluation/data/<dataset>/<file>.jsonl evaluation/result my-vllm-model
```

For training-time AIME-style checks, the default lookup paths are:

```text
evaluation/data/AIME24/AIME24-origin.jsonl
evaluation/data/AIME24/AIME24-combined-n2.jsonl
evaluation/data/AIME25/AIME25-origin.jsonl
evaluation/data/AIME25/AIME25-combined-n2.jsonl
```

You can modify these paths in
`training/verl/trainer/aime_eval_metrics.py` if your local evaluation files use a
different layout.

## Data Format

Each composed sample is expected to contain a chat prompt plus metadata used by
the reward manager:

```json
{
  "prompt": [
    {
      "role": "user",
      "content": "Problem 1: ...\n\nProblem 2: ..."
    }
  ],
  "reward_model": {
    "style": "rule",
    "ground_truth": ["answer1,answer2"]
  },
  "extra_info": {
    "composed_query_num": 2,
    "dependencies": [["variable2", "answer1 + c"]]
  }
}
```

The reward manager also accepts several equivalent target layouts, including
`target`, `group_targets`, comma-separated targets, and nested boxed LaTeX
answers.

## Current Scope

This repository currently focuses on mathematical long-horizon reasoning with
Qwen-style causal language models and GRPO training. The implementation is still
research-oriented: paths, data names, and launch parameters should be adjusted to
your local cluster and dataset layout.

## Citation

The paper draft is still in progress. Add the final citation here once the title,
author list, and venue information are fixed.
