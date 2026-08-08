#!/usr/bin/env bash
# Run the full-resolution Gemma 1B experiment with one-GPU model copies.

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-.}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

RUN_ROOT="${RUN_ROOT:-data/gemma1b-real-run-agent-middle-layer-13}"
CONTEXTS_FILE="${CONTEXTS_FILE:-$CODE_ROOT/data/contexts.jsonl}"
EVAL_PARQUET="${EVAL_PARQUET:-}"
MODEL="google/gemma-3-1b-it"
BINARY_JUDGE="google/gemma-3-12b-it"
CONTINUOUS_JUDGE="AtlaAI/Selene-1-Mini-Llama-3.1-8B"
SEED=1234
LAYER=13
COMMON=(--model_parallel_size 1 --local_files_only)

if [[ ! -f "$CONTEXTS_FILE" ]]; then
  echo "Missing contexts file: $CONTEXTS_FILE" >&2
  exit 1
fi
if [[ -z "$EVAL_PARQUET" || ! -f "$EVAL_PARQUET" ]]; then
  echo "Set EVAL_PARQUET to a local evaluation parquet." >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"/{prompts,steering,next_token_probs,concept_probs_binary,concept_probs_continuous,concept_probs_continuous_rescored,cross_entropy,mmlu,log_odds,plots,logs}

run_stage() {
  local name="$1"
  shift
  echo "[$(date -u +%FT%TZ)] START $name" | tee -a "$RUN_ROOT/run.log"
  "$@" 2>&1 | tee "$RUN_ROOT/logs/$name.log"
  echo "[$(date -u +%FT%TZ)] DONE  $name" | tee -a "$RUN_ROOT/run.log"
}

if [[ "${SKIP_PROMPTS:-0}" != "1" ]]; then
  run_stage prompts python experiments/generate_prompts.py \
    --model_generating_concept "$BINARY_JUDGE" --models "$MODEL" --concepts joy \
    --out_dir "$RUN_ROOT/prompts" --seed "$SEED" --contrastive "${COMMON[@]}"
fi

# Log odds uses the steered model name in the negative-prompt filename.
cp "$RUN_ROOT/prompts/joy_negative.jsonl" \
  "$RUN_ROOT/prompts/joy_google-gemma-3-1b-it_negative.jsonl"

run_stage steering python experiments/generate_steering_vectors.py \
  --models "$MODEL" --in_dir "$RUN_ROOT/prompts" --save_dir "$RUN_ROOT/steering" \
  --layers "$LAYER" --seed "$SEED" --contrastive "${COMMON[@]}"

run_stage next_token_probs python experiments/generate_next_token_probs.py \
  --models "$MODEL" --steer_dir "$RUN_ROOT/steering" \
  --contexts_file "$CONTEXTS_FILE" --out_dir "$RUN_ROOT/next_token_probs" \
  --layers "$LAYER" --seed "$SEED" "${COMMON[@]}"

run_stage concept_probs_binary python experiments/generate_concept_probs.py \
  --models "$MODEL" --judge_model "$BINARY_JUDGE" --steer_dir "$RUN_ROOT/steering" \
  --contexts_file "$CONTEXTS_FILE" --out_dir "$RUN_ROOT/concept_probs_binary" \
  --layer "$LAYER" --seed "$SEED" "${COMMON[@]}"

run_stage concept_probs_continuous python experiments/generate_concept_probs_continuous.py \
  --models "$MODEL" --judge_model "$CONTINUOUS_JUDGE" --steer_dir "$RUN_ROOT/steering" \
  --contexts_file "$CONTEXTS_FILE" --out_dir "$RUN_ROOT/concept_probs_continuous" \
  --layer "$LAYER" --seed "$SEED" "${COMMON[@]}"

run_stage continuous_rescore python experiments/rescore_concept_probs_continuous.py \
  --input_dir "$RUN_ROOT/concept_probs_continuous" \
  --output_dir "$RUN_ROOT/concept_probs_continuous_rescored" \
  --judge_model "$CONTINUOUS_JUDGE" --judge_batch_size 16 --judge_max_new_tokens 512 \
  "${COMMON[@]}"

run_stage cross_entropy python experiments/generate_cross_entropy.py \
  --models "$MODEL" --steer_dir "$RUN_ROOT/steering" \
  --eval_parquet "$EVAL_PARQUET" --out_dir "$RUN_ROOT/cross_entropy" \
  --layer "$LAYER" --seed "$SEED" "${COMMON[@]}"

run_stage mmlu python experiments/generate_mmlu.py \
  --models "$MODEL" --tasks HIGH_SCHOOL_COMPUTER_SCIENCE \
  --max_problems_per_task 8 --steer_dir "$RUN_ROOT/steering" \
  --out_dir "$RUN_ROOT/mmlu" --layer "$LAYER" --seed "$SEED" "${COMMON[@]}"

run_stage log_odds python experiments/generate_log_odds.py \
  --models "$MODEL" --prompts_dir "$RUN_ROOT/prompts" --out_dir "$RUN_ROOT/log_odds" \
  --concepts joy "${COMMON[@]}"

echo "[$(date -u +%FT%TZ)] EXPERIMENTS COMPLETE" | tee -a "$RUN_ROOT/run.log"
