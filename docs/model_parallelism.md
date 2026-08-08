# Running models on one or more GPUs

A **model worker** is one loaded copy of a model. A worker uses one GPU when the model fits, or several GPUs when the model must be split. The code calls this a `logical_actor`, but the user-facing idea is simply one running model copy.

- `--model_parallel_size auto` is the default. It estimates memory and chooses how many GPUs each model worker needs.
- `--model_parallel_size 1` forces every model worker onto one GPU.

The same experiment code handles both cases.

## Where the work happens

- `utils/runtime/actor.py` loads a model and sets up its GPUs.
- `utils/runtime/placement.py` estimates memory and chooses a GPU count.
- `utils/runtime/pool.py` starts and stops model workers.
- `utils/runtime/scheduler.py` assigns jobs to free model workers.
- `actors/model_actors.py` loads the models used by each experiment task.

Prompt generation, steering-vector extraction, probability curves, log-odds, cross-entropy, MMLU, concept scoring, and rescoring all use this shared code.

Concept scoring loads both a generator and a judge in each model worker, so the memory estimate includes both models.

## How automatic GPU selection works

With `--model_parallel_size auto`, the program:

1. estimates the model-weight memory without loading the weights;
2. reserves some free GPU memory for activations and generation;
3. chooses the smallest supported GPU count that can hold the model;
4. uses any remaining GPUs for more independent model workers.

For example, if one model copy needs two GPUs and four GPUs are available, two independent jobs can run at the same time.

Every GPU holding part of the same model receives the same input. All of those GPUs run the model, while the first GPU writes the output file and returns the result.

A model is loaded once for a set of jobs. When several model names are requested, the program finishes one model's jobs before loading the next model.

## Check a layout without loading the model

Use `--plan_only` to print the GPU layout:

```bash
python experiments/generate_cross_entropy.py \
  --models Qwen/Qwen2.5-72B \
  --steer_dir steering_vectors \
  --eval_parquet data/eval.parquet \
  --model_parallel_size auto \
  --dtype bfloat16 \
  --plan_only
```

The default settings keep 10% of currently free GPU memory unused and add 20% above the estimated model weights:

```text
--gpu_utilization 0.90 --inference_headroom 1.20
```

Use `--max_gpus N` to limit the number of GPUs. Use `--local_files_only` to prevent Hugging Face downloads.

## Current limits

- Multi-GPU loading only works when the model's Transformers configuration describes how to split the model.
- A generator and judge loaded together must support the same GPU count.
- All GPUs must be on one host.
- The memory estimate covers model weights plus the chosen extra space. Actual memory also depends on batch size, prompt length, and generated length.
- Splitting a model across GPUs works best with a fast connection between those GPUs.
