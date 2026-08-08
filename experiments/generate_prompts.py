import argparse
import asyncio

from monarch.actor import shutdown_context

from actors.tasks.prompts import GenConfig
from steering.runtime.pool import add_distributed_args
from experiments.runners import run_prompts


async def main_async(args):
    values = {
        "seed": args.seed,
        "name_of_model_instruct": args.model_generating_concept,
        "contrastive": args.contrastive,
        "n_related": args.n_related,
        "n_unrelated": args.n_unrelated,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }
    await run_prompts(args, GenConfig(**values))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_generating_concept", required=True)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--concepts", nargs="+", required=True)
    parser.add_argument("--out_dir", default="prompts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contrastive", action="store_true")
    parser.add_argument("--n_related", type=int, default=GenConfig.n_related)
    parser.add_argument("--n_unrelated", type=int, default=GenConfig.n_unrelated)
    parser.add_argument("--batch_size", type=int, default=GenConfig.batch_size)
    parser.add_argument("--max_new_tokens", type=int, default=GenConfig.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=GenConfig.temperature)
    parser.add_argument("--top_k", type=int, default=GenConfig.top_k)
    parser.add_argument("--top_p", type=float, default=GenConfig.top_p)
    add_distributed_args(parser, default_dtype="bfloat16")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main_async(parse_args()))
    finally:
        shutdown_context().get()
