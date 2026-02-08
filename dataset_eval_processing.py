#!/usr/bin/env python3
"""
download_one_fineweb_parquet.py

Downloads a single parquet shard from a FineWeb dataset repo on Hugging Face.

Example:
  python dataset_eval_processing.py \
    --dataset HuggingFaceFW/fineweb \
    --remote_name sample-10BT \
    --split train \
    --file_idx 0 \
    --out_dir ./fineweb_eval_parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, hf_hub_download


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu",
                   help="HF dataset repo id, e.g. HuggingFaceFW/fineweb or HuggingFaceFW/fineweb-edu")
    p.add_argument("--remote_name", type=str, default="sample-10BT",
                   help="Optional substring to filter parquet filenames by config/subset name (e.g. sample-10BT).")
    p.add_argument("--split", type=str, default="train",
                   help="Optional substring filter for split (train/validation/test).")
    p.add_argument("--contains", type=str, default="",
                   help="Additional substring filter applied to parquet filenames.")
    p.add_argument("--file_idx", type=int, default=0,
                   help="Index in the filtered, sorted parquet list to download.")
    p.add_argument("--out_dir", type=str, default="fineweb_eval_parquet",
                   help="Where to place the downloaded parquet (preserving repo subdirs).")
    return p.parse_args()


def _filter_parquets(files: List[str], remote_name: str, split: str, contains: str) -> List[str]:
    parq = [f for f in files if f.endswith(".parquet")]
    if remote_name:
        parq = [f for f in parq if remote_name in f]
    if split:
        s = split.lower()
        # Match either in basename or as a path component.
        parq_split = [f for f in parq if s in Path(f).name.lower() or f"/{s}/" in f.lower()]
        if parq_split:
            parq = parq_split
    if contains:
        parq = [f for f in parq if contains in f]
    return sorted(parq)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(repo_id=args.dataset, repo_type="dataset")

    parquets = _filter_parquets(files, args.remote_name, args.split, args.contains)
    if not parquets:
        raise RuntimeError(
            "No parquet files matched your filters.\n"
            f"dataset={args.dataset}\nremote_name={args.remote_name}\nsplit={args.split}\ncontains={args.contains}\n"
        )

    if not (0 <= args.file_idx < len(parquets)):
        raise RuntimeError(f"--file_idx {args.file_idx} out of range (0..{len(parquets)-1})")

    filename = parquets[args.file_idx]
    local_path = hf_hub_download(
        repo_id=args.dataset,
        repo_type="dataset",
        filename=filename,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )

    print(f"Found {len(parquets)} matching parquets.")
    print(f"Downloaded: {filename}")
    print(f"Local path: {local_path}")


if __name__ == "__main__":
    main()
