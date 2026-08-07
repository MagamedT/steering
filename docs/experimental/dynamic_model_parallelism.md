# Experimental dynamic model parallelism

Dynamic model parallelism is integrated into the normal steering-vector
launcher but is disabled by default. Without an explicit opt-in, the launcher
keeps the existing behavior of one model replica per GPU actor.

When enabled, one **logical actor** is a contiguous slice of `k` Monarch
actors/processes, with one process per GPU. PyTorch tensor parallelism makes the
slice hold one sharded model replica.

The controller:

1. creates a meta-device model skeleton and estimates weight memory;
2. applies configurable inference headroom and the GPUs' current free-memory
   budget;
3. chooses the smallest compatible tensor-parallel size `k`;
4. packs `floor(available_gpus / k)` logical actors onto the Monarch process
   mesh and schedules concepts across those groups.

Every endpoint call is broadcast to the full logical-actor slice so tensor
parallel collectives cannot deadlock. Only tensor-parallel rank zero writes
steering-vector files.

## Plan without loading weights

From the repository root:

```bash
python experiments/generate_steering_vectors.py \
  --models google/gemma-3-27b-it \
  --in_dir data/run_seed1/prompts \
  --model_parallel_size auto \
  --plan_only
```

The default sizing reserves 10% of currently free VRAM and adds 20% above model
weight memory for inference. Both knobs are explicit:

```bash
--gpu_utilization 0.90 --inference_headroom 1.20
```

## Run the experimental path

```bash
python experiments/generate_steering_vectors.py \
  --models google/gemma-3-27b-it \
  --in_dir data/run_seed1/prompts \
  --save_dir steering_vectors \
  --model_parallel_size auto \
  --dtype bfloat16 \
  --batch_size 8 \
  --layers 13
```

Use `--local_files_only` to guarantee the run does not contact the Hub. Use
`--max_gpus N` to cap the allocation.

## Current constraints

- The experimental path accepts one model per invocation so every logical
  actor has the same topology.
- It currently supports `--pairing product` only.
- Multi-GPU loading requires a Transformers model whose config defines
  `base_model_tp_plan`. A clear planning error is raised otherwise.
- It is single-host for now. The rank layout assumes all worker GPUs belong to
  one host.
- The estimate covers model weights plus a configurable multiplier; real
  activation use still depends on batch size and sequence length.
- Tensor parallelism communicates at every layer and benefits strongly from a
  fast intra-node interconnect.
