"""Assign experiment jobs to the available model workers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from actors import (
    DistributedConceptProbsActor,
    DistributedContinuousConceptProbsActor,
    DistributedCrossEntropyActor,
    DistributedLogOddsActor,
    DistributedMMLUActor,
    DistributedPromptActor,
    DistributedRescoreActor,
    DistributedSteeringActor,
    DistributedTokenActor,
    SteeringConfig,
)
from utils.data import discover_concepts, load_contexts_for_concept
from utils.runtime.pool import (
    ModelSpec,
    actor_pool,
    plan_models,
    print_plan,
)
from utils.runtime.scheduler import run_ranked_jobs


def prompt_phases(args) -> list[tuple[str, str, str | None]]:
    """Choose the model and mode for each prompt-generation phase."""
    if args.contrastive:
        return [(args.model_generating_concept, "both", None)]
    if not args.models:
        raise ValueError("--models must be provided when --contrastive is not set")
    return [
        (args.model_generating_concept, "related", None),
        *((model, "unrelated", model) for model in args.models),
    ]


def prompt_phase_jobs(concepts: list[str], mode: str) -> list[tuple[str, str]]:
    """Build concept jobs for one prompt-generation phase."""
    if mode == "both":
        return [(concept, "both") for concept in concepts]
    return [(concept, mode) for concept in concepts]


def plot_work_items(model_jobs, contexts_file, layers):
    """Split token plots into one job per context and layer."""
    requested_layers = [None] if layers == [None] else [int(index) for index in layers]
    work_items = []
    for model_name, concept_slug, concept_label in model_jobs:
        contexts, _ = load_contexts_for_concept(
            str(contexts_file),
            concept_slug=concept_slug,
            concept_label=concept_label,
        )
        work_items.extend(
            (model_name, concept_slug, concept_label, context_index, layer)
            for context_index in range(len(contexts))
            for layer in requested_layers
        )
    return work_items


def _single_model_actor_args(args, plan, model_name: str):
    """Build common constructor arguments for one-model actors."""
    return (
        model_name,
        args.dtype,
        plan.logical_actors,
        plan.gpus_per_actor,
        getattr(args, "local_files_only", False),
        getattr(args, "trust_remote_code", False),
    )


async def run_prompts(args, cfg) -> None:
    """Plan and run prompt-generation jobs."""
    concepts = list(args.concepts)
    if not concepts:
        raise ValueError("At least one concept is required")
    if args.contrastive and args.models:
        print("NOTE: --models is ignored in contrastive mode.", flush=True)

    for phase_index, (model_name, mode, negative_tag) in enumerate(prompt_phases(args)):
        jobs = prompt_phase_jobs(concepts, mode)
        specs = [ModelSpec(model_name, args.dtype)]
        plan = plan_models(args, specs, desired_actors=len(jobs))
        print_plan(plan, stage="prompts", models=specs, mode=mode, jobs=len(jobs))
        if getattr(args, "plan_only", False):
            continue

        async with actor_pool(
            args,
            plan,
            f"prompts_{phase_index}",
            DistributedPromptActor,
            *_single_model_actor_args(args, plan, model_name),
        ) as pool:
            async def run_one(rank: int, concept: str, job_mode: str):
                result = await pool.call(
                    rank,
                    "generate_for_concept",
                    concept,
                    asdict(cfg),
                    args.out_dir,
                    rank,
                    job_mode,
                    negative_tag,
                )
                result["job_mode"] = job_mode
                return result

            async for rank, result in run_ranked_jobs(
                jobs, plan.logical_actors, run_one
            ):
                if isinstance(result, Exception):
                    raise result
                print(
                    f"[actor {rank}; {plan.gpus_per_actor} GPU(s)] "
                    f"phase={mode} concept={result['concept']!r} "
                    f"related={result['related']} unrelated={result['unrelated']} "
                    f"-> {result['files']}",
                    flush=True,
                )


async def run_steering(args) -> None:
    """Plan and run steering-vector jobs."""
    prompt_root = Path(args.in_dir)
    if not prompt_root.exists():
        raise RuntimeError(f"--in_dir {str(prompt_root)!r} does not exist")
    concepts = discover_concepts(prompt_root)
    if not concepts:
        raise RuntimeError(f"No positive/negative concept pairs found under {prompt_root}")

    cfg = SteeringConfig(
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
        seed=args.seed,
        n_positive=args.n_positive,
        n_negative=args.n_negative,
        contrastive=args.contrastive,
        block_per_pass=args.block_per_pass,
        progress_every=args.progress_every,
    )
    requested = [None] if args.layers == [None] else [int(i) for i in args.layers]
    all_jobs = [(model, slug, label) for model in args.models for slug, label in concepts]

    if args.pairing != "product":
        from experiments.generate_steering_vectors import pair_jobs

        all_jobs = pair_jobs(list(args.models), concepts, mode=args.pairing)

    for model_index, model_name in enumerate(dict.fromkeys(args.models)):
        jobs = [(slug, label) for model, slug, label in all_jobs if model == model_name]
        if not jobs:
            continue
        specs = [ModelSpec(model_name, args.dtype)]
        plan = plan_models(args, specs, desired_actors=len(jobs))
        print_plan(plan, stage="steering", models=specs, jobs=len(jobs))
        if getattr(args, "plan_only", False):
            continue

        async with actor_pool(
            args,
            plan,
            f"steering_{model_index}",
            DistributedSteeringActor,
            *_single_model_actor_args(args, plan, model_name),
        ) as pool:
            async def run_one(rank: int, concept_slug: str, concept_label: str):
                return await pool.call(
                    rank,
                    "compute_for",
                    concept_slug,
                    concept_label,
                    requested,
                    asdict(cfg),
                    str(prompt_root),
                    args.save_dir,
                    args.layer_path,
                    rank,
                )

            async for rank, result in run_ranked_jobs(
                jobs, plan.logical_actors, run_one
            ):
                if isinstance(result, Exception):
                    raise result
                if "error" in result:
                    raise RuntimeError(result["error"])
                print(
                    f"[actor {rank}; {plan.gpus_per_actor} GPU(s)] "
                    f"model={model_name!r} concept={result['concept']!r} "
                    f"saved={len(result['saved'])}",
                    flush=True,
                )


async def run_next_token_probs(args, cfg, jobs, steer_dir, out_dir, contexts_file):
    """Plan and run next-token probability jobs."""
    for model_index, model_name in enumerate(dict.fromkeys(args.models)):
        model_jobs = [job for job in jobs if job[0] == model_name]
        work_items = plot_work_items(model_jobs, contexts_file, args.layers)
        if not work_items:
            continue
        specs = [ModelSpec(model_name, args.dtype)]
        plan = plan_models(args, specs, desired_actors=len(work_items))
        print_plan(
            plan,
            stage="next_token_probs",
            models=specs,
            concepts=len(model_jobs),
            jobs=len(work_items),
        )
        if getattr(args, "plan_only", False):
            continue

        async with actor_pool(
            args,
            plan,
            f"next_token_probs_{model_index}",
            DistributedTokenActor,
            *_single_model_actor_args(args, plan, model_name),
        ) as pool:
            async def run_one(rank, job_model, slug, label, context_index, layer):
                return await pool.call(
                    rank,
                    "compute_plot_curves",
                    model_name=job_model,
                    concept_slug=slug,
                    concept_label=label,
                    block_idx_to_steer=[None] if layer is None else [layer],
                    contexts_file=str(contexts_file),
                    steer_dir=str(steer_dir),
                    save_dir=str(out_dir),
                    layer_path=args.layer_path,
                    cfg_dict=asdict(cfg),
                    rank_hint=rank,
                    context_indices=[context_index],
                )

            async for rank, result in run_ranked_jobs(
                work_items, plan.logical_actors, run_one
            ):
                if isinstance(result, Exception):
                    raise result
                if "error" in result:
                    raise RuntimeError(result["error"])
                print(
                    f"[actor {rank}; {plan.gpus_per_actor} GPU(s)] "
                    f"model={model_name!r} contexts={result['context_indices']} "
                    f"layers={result['layers']} finished",
                    flush=True,
                )


async def _run_single_model_jobs(
    args,
    *,
    stage,
    actor_cls,
    jobs,
    endpoint,
    invoke,
    continue_on_error=False,
):
    """Run independent jobs that share one loaded model."""
    for model_index, model_name in enumerate(dict.fromkeys(args.models)):
        model_jobs = [job[1:] for job in jobs if job[0] == model_name]
        if not model_jobs:
            continue
        specs = [ModelSpec(model_name, args.dtype)]
        plan = plan_models(args, specs, desired_actors=len(model_jobs))
        print_plan(plan, stage=stage, models=specs, jobs=len(model_jobs))
        if getattr(args, "plan_only", False):
            continue
        async with actor_pool(
            args,
            plan,
            f"{stage}_{model_index}",
            actor_cls,
            *_single_model_actor_args(args, plan, model_name),
        ) as pool:
            async def run_one(rank, concept_slug, concept_label):
                return await pool.call(
                    rank,
                    endpoint,
                    **invoke(rank, model_name, concept_slug, concept_label),
                )

            async for rank, result in run_ranked_jobs(
                model_jobs, plan.logical_actors, run_one
            ):
                if isinstance(result, Exception):
                    if continue_on_error:
                        print(f"[actor {rank}] FAILED: {result}", flush=True)
                        continue
                    raise result
                if isinstance(result, dict) and "error" in result:
                    raise RuntimeError(result["error"])
                print(
                    f"[actor {rank}; {plan.gpus_per_actor} GPU(s)] finished: {result}",
                    flush=True,
                )


async def run_cross_entropy(args, cfg, jobs, steer_dir, out_dir, eval_parquet):
    """Plan and run cross-entropy jobs."""
    await _run_single_model_jobs(
        args,
        stage="cross_entropy",
        actor_cls=DistributedCrossEntropyActor,
        jobs=jobs,
        endpoint="compute_cross_entropy_curves",
        invoke=lambda rank, model, slug, label: dict(
            model_name=model,
            concept_slug=slug,
            concept_label=label,
            block_idx_to_steer=None if args.layers == 0 else int(args.layers),
            exact_layer_idx=getattr(args, "layer", None),
            eval_parquet=str(eval_parquet),
            steer_dir=str(steer_dir),
            save_dir=str(out_dir),
            layer_path=args.layer_path,
            cfg_dict=asdict(cfg),
            rank_hint=rank,
        ),
    )


async def run_log_odds(args, cfg, jobs, prompts_path, out_dir):
    """Plan and run token log-odds jobs."""
    await _run_single_model_jobs(
        args,
        stage="log_odds",
        actor_cls=DistributedLogOddsActor,
        jobs=jobs,
        endpoint="compute_log_odds",
        invoke=lambda rank, model, slug, label: dict(
            model_name=model,
            concept_slug=slug,
            concept_label=label,
            prompts_dir=str(prompts_path),
            save_dir=str(out_dir),
            cfg_dict=asdict(cfg),
            rank_hint=rank,
        ),
    )


async def run_mmlu(args, cfg, jobs, steer_dir, out_dir):
    """Plan and run MMLU jobs."""
    await _run_single_model_jobs(
        args,
        stage="mmlu",
        actor_cls=DistributedMMLUActor,
        jobs=jobs,
        endpoint="compute_mmlu",
        continue_on_error=True,
        invoke=lambda rank, model, slug, label: dict(
            model_name=model,
            concept_slug=slug,
            concept_label=label,
            steer_dir=str(steer_dir),
            save_dir=str(out_dir),
            block_idx_to_steer=None if args.layers == 0 else int(args.layers),
            exact_layer_idx=getattr(args, "layer", None),
            layer_path=args.layer_path,
            cfg_dict=asdict(cfg),
            rank_hint=rank,
        ),
    )


async def run_concept_probs(
    args,
    cfg,
    jobs,
    steer_dir,
    out_dir,
    contexts_file,
    *,
    continuous: bool,
):
    """Plan and run concept-probability jobs."""
    actor_cls = (
        DistributedContinuousConceptProbsActor
        if continuous
        else DistributedConceptProbsActor
    )
    judge_dtype = getattr(args, "judge_dtype", args.dtype)
    for model_index, model_name in enumerate(dict.fromkeys(args.models)):
        model_jobs = [job[1:] for job in jobs if job[0] == model_name]
        if not model_jobs:
            continue
        specs = [ModelSpec(model_name, args.dtype), ModelSpec(args.judge_model, judge_dtype)]
        plan = plan_models(args, specs, desired_actors=len(model_jobs))
        print_plan(
            plan,
            stage="concept_probs_continuous" if continuous else "concept_probs",
            models=specs,
            jobs=len(model_jobs),
        )
        if getattr(args, "plan_only", False):
            continue

        actor_args = (
            model_name,
            args.dtype,
            args.judge_model,
            judge_dtype,
            plan.logical_actors,
            plan.gpus_per_actor,
            getattr(args, "local_files_only", False),
            getattr(args, "trust_remote_code", False),
        )
        async with actor_pool(
            args,
            plan,
            f"concept_probs_{model_index}",
            actor_cls,
            *actor_args,
        ) as pool:
            async def run_one(rank, concept_slug, concept_label):
                kwargs = dict(
                    model_name=model_name,
                    concept_slug=concept_slug,
                    concept_label=concept_label,
                    block_idx_to_steer=None if args.layers == 0 else int(args.layers),
                    contexts_file=str(contexts_file),
                    steer_dir=str(steer_dir),
                    save_dir=str(out_dir),
                    layer_path=args.layer_path,
                    cfg_dict=asdict(cfg),


                    rank_hint=rank,
                )
                if getattr(args, "layer", None) is not None:
                    kwargs["exact_layer_idx"] = int(args.layer)
                return await pool.call(rank, "compute_concept_probs_curves", **kwargs)

            async for rank, result in run_ranked_jobs(
                model_jobs, plan.logical_actors, run_one
            ):
                if isinstance(result, Exception):
                    raise result
                if "error" in result:
                    raise RuntimeError(result["error"])
                files = [
                    item.get("file")
                    for item in result.get("results", [])
                    if item.get("file")
                ]
                print(
                    f"[actor {rank}; {plan.gpus_per_actor} GPU(s)] finished -> "
                    f"{files[0] if files else '(no files)'}",
                    flush=True,
                )


async def run_rescore(args, cfg, files, input_root: Path, output_root: Path):
    """Plan and run rescoring jobs."""
    specs = [ModelSpec(args.judge_model, args.dtype)]
    plan = plan_models(args, specs, desired_actors=len(files))
    print_plan(
        plan,
        stage="rescore_concept_probs_continuous",
        models=specs,
        jobs=len(files),
    )
    if getattr(args, "plan_only", False):
        return

    async with actor_pool(
        args,
        plan,
        "rescore_concept_probs_continuous",
        DistributedRescoreActor,
        *_single_model_actor_args(args, plan, args.judge_model),
    ) as pool:
        jobs = [(str(path),) for path in files]

        async def run_one(rank: int, path: str):
            return await pool.call(
                rank,
                "rescore_file",
                path,
                str(input_root),
                str(output_root),
                asdict(cfg),
            )

        async for rank, result in run_ranked_jobs(
            jobs, plan.logical_actors, run_one
        ):
            if isinstance(result, Exception):
                raise result
            if not result.get("ok"):
                raise RuntimeError(f"Unexpected rescoring result: {result}")
            print(
                f"[actor {rank}; {plan.gpus_per_actor} GPU(s)] "
                f"rescored -> {result['file']}",
                flush=True,
            )
