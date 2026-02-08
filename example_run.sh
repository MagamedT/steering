#!/bin/bash
#SBATCH -J steeringJob
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH -p h100
#SBATCH --gres=gpu:8
#SBATCH --tmp=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=example@gmail.com

set -euo pipefail

CODE_ROOT="steering_run"
EXP_ROOT="steering_run/data"  # root for all outputs
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

SEEDS=(1 4 3)
MODELS=(
  openai-community/gpt2-medium
  openai-community/gpt2
  google/gemma-3-1b-it
  Qwen/Qwen3-8B
)
CONCEPTS=(joy evil)
MODEL_GENERATING_CONCEPT="google/gemma-3-12b-it"
CONTEXTS_FILE="${CODE_ROOT}/contexts_modified.jsonl"         
EVAL_PARQUET="/workspace/fineweb_eval_parquet/sample/10BT/000_00000.parquet" 

mkdir -p "${EXP_ROOT}"

for SEED in "${SEEDS[@]}"; do
  RUN_DIR="${EXP_ROOT}/run_seed${SEED}"
  PROMPTS_DIR="${RUN_DIR}/prompts"
  STEERING_DIR="${RUN_DIR}/steering_vectors"
  PLOT_DIR="${RUN_DIR}/plot_data"
  LOG_ODDS_DIR="${RUN_DIR}/log_odds"
  XENT_DIR="${RUN_DIR}/cross_entropy"
  MMLU_DIR="${RUN_DIR}/mmlu"
  BEHAVIOR_DIR="${RUN_DIR}/behavior_data"

  mkdir -p "${PROMPTS_DIR}" "${STEERING_DIR}" "${PLOT_DIR}" "${LOG_ODDS_DIR}" "${XENT_DIR}" "${MMLU_DIR}"

  python3 "${CODE_ROOT}/generate_prompts.py" \
    --model_generating_concept "${MODEL_GENERATING_CONCEPT}" \
    --models "${MODELS[@]}" \
    --concepts "${CONCEPTS[@]}" \
    --out_dir "${PROMPTS_DIR}" \
    --seed "${SEED}"

  python3 "${CODE_ROOT}/generate_steering_vectors.py" \
    --models "${MODELS[@]}" \
    --in_dir "${PROMPTS_DIR}" \
    --save_dir "${STEERING_DIR}" \
    --seed "${SEED}"

  python3 "${CODE_ROOT}/generate_plot_data.py" \
    --models "${MODELS[@]}" \
    --steer_dir "${STEERING_DIR}" \
    --contexts_file "${CONTEXTS_FILE}" \
    --out_dir "${PLOT_DIR}" \
    --seed "${SEED}"

  python3 "${CODE_ROOT}/generate_behavior.py" \
  --models "${MODELS[@]}" \
  --judge_model "${MODEL_GENERATING_CONCEPT}" \
  --steer_dir "${STEERING_DIR}" \
  --contexts_file "${CONTEXTS_FILE}" \
  --out_dir "${BEHAVIOR_DIR}" \
  --layers 6

  python3 "${CODE_ROOT}/generate_log_odds.py" \
    --models "${MODELS[@]}" \
    --prompts_dir "${PROMPTS_DIR}" \
    --out_dir "${LOG_ODDS_DIR}"

  python3 "${CODE_ROOT}/generate_cross_entropy.py" \
    --models "${MODELS[@]}" \
    --steer_dir "${STEERING_DIR}" \
    --eval_parquet "${EVAL_PARQUET}" \
    --out_dir "${XENT_DIR}" \
    --seed "${SEED}" \
    --layers 3

  python3 "${CODE_ROOT}/generate_mmlu.py" \
    --models "${MODELS[@]}" \
    --tasks HIGH_SCHOOL_COMPUTER_SCIENCE \
    --steer_dir "${STEERING_DIR}" \
    --out_dir "${MMLU_DIR}" \
    --seed "${SEED}"

done
