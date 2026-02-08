# Towards Understanding Steering Strength codebase

Codebase for steering-vector experiments used for [Towards Understanding Steering Strength](https://arxiv.org/abs/2602.02712).  

## What This Repository Does

This repository implements activation steering for causal LMs:

1. Generate concept-positive and concept-negative prompt datasets.
2. Compute per-layer steering vectors from activation mean differences.
3. Sweep steering strength `alpha` and evaluate effects with:
   - next token probability curves (`generate_plot_data.py`),
   - concept presence probabilities judge scores (`generate_behavior.py`),
   - cross-entropy curves (`generate_cross_entropy.py`),
   - MMLU accuracy (`generate_mmlu.py`),
   - compute token log-odds (`generate_log_odds.py`).

## Project Structure

- `generate_prompts.py`: builds concept prompt datasets (`*_positive.jsonl`, `*_negative.jsonl`).
- `generate_steering_vectors.py`: computes steering vectors and saves `layer_<i>.pt`.
- `generate_plot_data.py`: runs alpha sweeps and saves token probability curves (`.npz`).
- `generate_behavior.py`: concept presence probabilities sweeps with a judge model (`.npz`).
- `generate_cross_entropy.py`: cross-entropy vs alpha (`.npz`).
- `dataset_eval_processing.py`: downloads a filtered eval parquet shard (for cross-entropy runs).
- `generate_mmlu.py`: MMLU vs alpha (`.json`).
- `generate_log_odds.py`: non-steered token log-odds baseline (`.npz`).
- `plot_probs.py`: plotting/analysis utilities.
- `actors/`: GPU actor implementations used by torchmonarch.
- `example_run.sh`: example Slurm batch script showing how to run all experiments.

## Requirements

- Python 3.10+ (3.12 recommended)
- CUDA GPU(s)
- `torchmonarch` runtime (see `build_monarch.sh` for a setup path)

Install Python deps:

```bash
pip install -r requirements.txt
```

Some models require Hugging Face authentication (for gated access). Log in before running scripts that load those models:

```bash
hf auth login
```

## Data Formats

### Prompt files (`prompts/`)

Each line is JSON. Typical schema:

```json
{"concept": "joy", "kind": "positive", "text": "..."}
{"concept": "joy", "kind": "negative", "text": "..."}
```

### Context file (`contexts.jsonl`)

Expected JSONL shape:

```json
{"negative": ["neg prompt 1", "neg prompt 2"]}
{"joy": ["positive prompt 1", "positive prompt 2"]}
{"evil": ["positive prompt 1", "positive prompt 2"]}
```

### Steering vectors (`steering_vectors/.../layer_<i>.pt`)

Saved object includes:

- `model`
- `concept`
- `concept_slug`
- `layer_idx`
- `hidden_size`
- `steering_vector`

## Typical Workflow

For a single scheduled run on Slurm, use `example_run.sh` as a template:

```bash
sbatch example_run.sh
```

### 1) Generate prompts

```bash
python generate_prompts.py \
  --model_generating_concept google/gemma-3-12b-it \
  --models openai-community/gpt2 google/gemma-3-1b-it \
  --concepts joy evil \
  --out_dir prompts
```

### 2) Compute steering vectors

```bash
python generate_steering_vectors.py \
  --models openai-community/gpt2 google/gemma-3-1b-it \
  --in_dir prompts \
  --save_dir steering_vectors
```

### 3) Probability curves vs alpha

```bash
python generate_plot_data.py \
  --models openai-community/gpt2 google/gemma-3-1b-it \
  --steer_dir steering_vectors \
  --contexts_file contexts_modified.jsonl \
  --out_dir plot_data
```

## Optional Evaluations

### Behavior score (judge model)

```bash
python generate_behavior.py \
  --models openai-community/gpt2 \
  --judge_model google/gemma-3-12b-it \
  --steer_dir steering_vectors \
  --contexts_file contexts_modified.jsonl \
  --out_dir behavior_data
```

### Cross-entropy

Prepare an evaluation parquet first (example: one FineWeb shard):

```bash
python dataset_eval_processing.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --remote_name sample-10BT \
  --split train \
  --file_idx 0 \
  --out_dir fineweb_eval_parquet
```

```bash
python generate_cross_entropy.py \
  --models openai-community/gpt2 \
  --steer_dir steering_vectors \
  --eval_parquet fineweb_eval_parquet/sample/10BT/000_00000.parquet \
  --out_dir cross_entropy
```

### MMLU

```bash
python generate_mmlu.py \
  --models openai-community/gpt2 \
  --tasks HIGH_SCHOOL_COMPUTER_SCIENCE \
  --steer_dir steering_vectors \
  --out_dir mmlu
```

### Log-Odds

```bash
python generate_log_odds.py \
  --models openai-community/gpt2 \
  --prompts_dir prompts \
  --out_dir log_odds
```

## Output Layout

Most outputs are grouped by model slug and concept slug:

```text
<out_dir>/
  <model_slug>/
    <concept_slug>/
      layer_*.pt
      layer_*_ctx_*.npz
      layer_*_behavior.npz
      layer_*_cross_entropy.npz
      layer_*_mmlu.json
      log_odds_topk.npz
```

## Notes

- To change experiment settings (for example batch size, steering-vector normalization, or contrastive prompts), edit the `@dataclass` config blocks in the corresponding files under `actors/` (for example `actors/steering_vector_actor.py`, `actors/steering_plot_actor.py`, `actors/behavior_score_actor.py`, `actors/cross_entropy_actor.py`, `actors/log_odds_actor.py`).


## Citation

If you use this code, please use the following to cite our work:

```bibtex
@article{taimeskhanov2026towards,
  title={Towards Understanding Steering Strength},
  author={Taimeskhanov, Magamed and Vaiter, Samuel and Garreau, Damien},
  journal={arXiv preprint arXiv:2602.02712},
  year={2026}
}
```
