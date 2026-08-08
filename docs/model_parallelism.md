# Unified model parallelism

Every GPU experiment uses the same logical-actor runtime. A logical actor owns
one model replica and spans one or more contiguous Monarch worker processes,
with one process per GPU.

- `--model_parallel_size auto` (the default) estimates model memory, chooses
  the smallest compatible group, and packs as many replicas as possible on the
  visible GPUs. It selects one rank when the model fits on one GPU.
- `--model_parallel_size 1` explicitly forces one-rank logical actors.

There is no separate single-GPU implementation. A one-GPU run is the `k=1`
case of the same actor, scheduler, model loader, and endpoint code.

The unified runtime is used by prompt generation, steering-vector extraction,
next-token probability plots, log-odds, cross-entropy, MMLU, binary concept
probabilities, continuous concept probabilities, and continuous-score
rescoring. Concept-probability actors plan the generator and judge together
because both models coexist on each logical actor.

## Placement and scheduling

For `auto`, the controller:

1. builds meta-device model skeletons and estimates model-weight memory;
2. applies the configured inference headroom and current free-VRAM budget;
3. chooses the smallest tensor-parallel size compatible with every resident
   model;
4. packs up to `floor(available_gpus / GPUs_per_actor)` replicas, bounded by
   the number of independent jobs.

For example, when Qwen2.5-72B needs two GPUs, four visible GPUs can host two
concurrent logical actors. If there are at least two concept/context jobs, the
scheduler keeps both two-GPU replicas active.

Every endpoint call is broadcast to every rank in its logical actor so model
collectives remain aligned. All ranks execute forward and generation calls;
only tensor-parallel rank zero writes output files and returns the scheduler
result. Log-odds, cross-entropy, and token-probability actors collectively
materialize full-vocabulary logits when the language head is sharded.

Models are grouped by model id so one actor pool never reloads weights during a
run. Prompt generation creates one pool per generation phase. Experiments with
several requested models create one pool per model topology in sequence.

## Planning without loading weights

All GPU launchers accept `--plan_only`. For example:

```bash
python experiments/generate_cross_entropy.py \
  --models Qwen/Qwen2.5-72B \
  --steer_dir steering_vectors \
  --eval_parquet data/eval.parquet \
  --model_parallel_size auto \
  --dtype bfloat16 \
  --plan_only
```

The default placement reserve keeps 10% of currently free VRAM unused and
adds 20% headroom above estimated model-weight memory. Both controls are
explicit:

```text
--gpu_utilization 0.90 --inference_headroom 1.20
```

Use `--max_gpus N` to cap the allocation and `--local_files_only` to prevent
Hub access.

## Constraints

- Multi-GPU loading requires every sharded model to provide a Transformers
  `base_model_tp_plan` and a compatible tensor-parallel divisor.
- Generator and judge models in concept-probability experiments must share a
  compatible group size because they live on the same GPU group.
- Placement is single-host; the rank layout assumes all workers belong to one
  host.
- The estimate covers weights plus configurable headroom. Activation and KV
  cache usage still depends on batch size, sequence length, and generation
  length.
- Tensor parallelism communicates at every layer and benefits from a fast
  intra-node interconnect.
