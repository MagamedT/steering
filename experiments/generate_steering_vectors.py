import argparse
import asyncio

from monarch.actor import shutdown_context

from actors.tasks.steering import SteeringConfig
from steering.runtime.pool import add_distributed_args
from experiments.runners import run_steering


def pair_jobs(models, concepts, mode="product"):
    if mode == "product":
        return [
            (model, slug, label)
            for model in models
            for slug, label in concepts
        ]
    if mode == "zip":
        return [
            (model, *concepts[index])
            for index, model in enumerate(models[: len(concepts)])
        ]
    length = max(len(models), len(concepts))
    return [
        (models[index % len(models)], *concepts[index % len(concepts)])
        for index in range(length)
    ]


async def main_async(args):
    await run_steering(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--in_dir", default="prompts")
    parser.add_argument("--save_dir", default="steering_vectors")
    parser.add_argument("--layers", nargs="+", default=[None])
    parser.add_argument("--layer_path", default=None)
    parser.add_argument(
        "--pairing", choices=("product", "zip", "zip_cycle"), default="product"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=SteeringConfig.batch_size)
    parser.add_argument("--max_length", type=int, default=SteeringConfig.max_length)
    parser.add_argument("--block_per_pass", type=int, default=SteeringConfig.block_per_pass)
    parser.add_argument("--progress_every", type=int, default=SteeringConfig.progress_every)
    parser.add_argument("--n_positive", type=int, default=None)
    parser.add_argument("--n_negative", type=int, default=None)
    parser.add_argument("--contrastive", action="store_true")
    add_distributed_args(parser, default_dtype=SteeringConfig.dtype)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main_async(parse_args()))
    finally:
        shutdown_context().get()
