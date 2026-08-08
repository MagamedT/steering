# Towards Understanding Steering Strength codebase

Codebase for steering-vector experiments used in [Towards Understanding Steering Strength](https://arxiv.org/abs/2602.02712).

## What This Repository Does

This repository implements activation steering for causal LMs:

1. Generate concept-positive and concept-negative prompt datasets (`experiments/generate_prompts.py`).
2. Compute per-layer steering vectors from activation mean differences (`experiments/generate_steering_vectors.py`).
3. Sweep steering strength `alpha` and evaluate effects with:
   - next token probability curves (`experiments/generate_next_token_probs.py`),
   - concept presence probabilities judge scores (`experiments/generate_concept_probs.py`),
   - cross-entropy curves (`experiments/generate_cross_entropy.py`),
   - MMLU accuracy (`experiments/generate_mmlu.py`),
   - compute token log-odds (`experiments/generate_log_odds.py`).

## Project Structure

- `experiments/generate_*.py`: stable CLI entry points; argument parsing only.
- `experiments/runners.py`: experiment orchestration and job construction.
- `actors/tasks/`: endpoint implementations grouped by evaluation task.
- `actors/model_actors.py`: binds task endpoints to distributed model lifecycles.
- `steering/data.py`: prompt/context discovery and parquet evaluation streams.
- `steering/modeling.py`: transformer-block, steering-vector, and tokenizer helpers.
- `steering/batching.py`, `steering/naming.py`, `steering/mmlu.py`: focused shared helpers.
- `steering/runtime/`: topology, model placement, actor pools, and scheduling.
- `tests/`: CPU end-to-end coverage and topology/planner unit tests.
- `docs/`: architecture and operational constraints.

The public experiment commands remain at their original paths; reusable code is
kept out of the CLI modules and generic catch-all utility files are avoided.

## Requirements

- Python 3.10+ (3.12 recommended)
- CUDA GPU(s)
- `torchmonarch` runtime (see `build_monarch.sh` for a setup path)

Install Python deps:

```bash
pip install -r requirements.txt
```

Run experiment commands from the repository root. Because the entry points live
in `experiments/` and import the sibling `actors` package, add the repository
root to Python's module search path first:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
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

This command builds concept-positive and concept-negative prompt JSONL files used to build the steering vectors.

```bash
python experiments/generate_prompts.py \
  --model_generating_concept google/gemma-3-12b-it \
  --models openai-community/gpt2 google/gemma-3-1b-it \
  --concepts joy evil \
  --out_dir prompts
```

Every GPU experiment uses one logical-actor runtime. The default
`--model_parallel_size auto` chooses the smallest compatible group, including
a one-rank actor when the model fits on one GPU. Use `1` only to force
one-rank placement:

```bash
python experiments/generate_prompts.py \
  --model_generating_concept Qwen/Qwen2.5-72B \
  --concepts joy \
  --out_dir prompts \
  --contrastive --model_parallel_size auto --dtype bfloat16
```

Contrastive positive and negative generation are independent scheduler jobs.
Plot contexts and explicitly requested layers are split the same way. If the
planner selects two GPUs per Qwen2.5-72B replica, four GPUs can therefore run
two model replicas concurrently whenever at least two such jobs are available.

### 2) Compute steering vectors

This command computes per-layer steering vectors from prompt activations.

```bash
python experiments/generate_steering_vectors.py \
  --models openai-community/gpt2 google/gemma-3-1b-it \
  --in_dir prompts \
  --save_dir steering_vectors
```

The same command handles one- and multi-GPU replicas. Automatic placement also
packs multiple replicas when enough GPUs and independent jobs are available:

```bash
python experiments/generate_steering_vectors.py \
  --models google/gemma-3-27b-it \
  --in_dir prompts \
  --save_dir steering_vectors \
  --model_parallel_size auto
```

See [unified model parallelism](docs/model_parallelism.md) for supported
experiments, planning controls, topology, and current constraints.

### 3) Probability curves vs alpha

This command sweeps steering strength and saves next-token probability curves.

```bash
python experiments/generate_next_token_probs.py \
  --models openai-community/gpt2 google/gemma-3-1b-it \
  --steer_dir steering_vectors \
  --contexts_file data/contexts.jsonl \
  --out_dir plot_data
```

The plot launcher supports the same automatic model placement, plus CLI
controls for bounded alpha sweeps:

```bash
python experiments/generate_next_token_probs.py \
  --models Qwen/Qwen2.5-72B \
  --steer_dir steering_vectors \
  --contexts_file data/contexts.jsonl \
  --out_dir plot_data \
  --model_parallel_size auto --dtype bfloat16 \
  --alpha_steps 41 --batch_size 8 --layers 39
```

## Optional Evaluations

### Behavior score (judge model)

This command measures concept presence in a steered model with a judge model across alpha values.

```bash
python experiments/generate_concept_probs.py \
  --models openai-community/gpt2 \
  --judge_model google/gemma-3-12b-it \
  --steer_dir steering_vectors \
  --contexts_file data/contexts.jsonl \
  --out_dir behavior_data
```

With `--model_parallel_size auto`, the generator and judge are planned
together and sharded across the same logical-actor GPU group. The continuous
concept-probability launcher uses the same runtime.

### Cross-entropy

Prepare an evaluation parquet first (example: one FineWeb shard):

This command downloads and writes an evaluation parquet shard for scoring.

```bash
python experiments/generate_eval_dataset.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --remote_name sample-10BT \
  --split train \
  --file_idx 0 \
  --out_dir fineweb_eval_parquet
```

This command computes cross-entropy curves over steering strengths.

```bash
python experiments/generate_cross_entropy.py \
  --models openai-community/gpt2 \
  --steer_dir steering_vectors \
  --eval_parquet fineweb_eval_parquet/sample/10BT/000_00000.parquet \
  --out_dir cross_entropy
```

### MMLU

This command evaluates MMLU accuracy as steering strength changes.

```bash
python experiments/generate_mmlu.py \
  --models openai-community/gpt2 \
  --tasks HIGH_SCHOOL_COMPUTER_SCIENCE \
  --steer_dir steering_vectors \
  --out_dir mmlu
```

### Log-Odds

This command computes token log-odds as in the paper from prompts in the sets $P$ and $N$.

```bash
python experiments/generate_log_odds.py \
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

- All GPU launchers accept `--model_parallel_size {1,auto}`, `--dtype`,
  `--max_gpus`, `--gpu_utilization`, `--inference_headroom`, and
  `--plan_only`.
- Experiment defaults live in the corresponding config dataclasses under
  `actors/tasks/`.


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
