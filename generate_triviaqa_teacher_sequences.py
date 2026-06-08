#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate cached TriviaQA teacher continuations for joint KD.

This script only writes JSONL generated-sequence caches. It does not capture
layer activations or train replacement layers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset

from gptoss_kd_capture import get_dtype
from kd_teacher import load_teacher
from train_triviaqa_all_layers_kd import generate_or_load_teacher_sequences


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--teacher_adapter_dir", type=str, default="")
    parser.add_argument("--teacher_dtype", type=str, default="bfloat16")
    parser.add_argument("--teacher_device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher_device_map", type=str, default="cuda:0")
    parser.add_argument("--triviaqa_config", type=str, default="rc.nocontext")
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--generation_batch_size", type=int, default=128)
    parser.add_argument("--generation_cache_dir", type=str, required=True)
    parser.add_argument("--force_regenerate", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cache_dir = Path(args.generation_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Loading teacher on {args.teacher_device} ===")
    teacher, tokenizer = load_teacher(
        args.teacher_model,
        args.teacher_adapter_dir,
        get_dtype(args.teacher_dtype),
        args.teacher_device,
        args.teacher_device_map,
        args.trust_remote_code,
    )

    print("=== Loading TriviaQA ===")
    train = load_dataset("trivia_qa", args.triviaqa_config, split=f"train[:{args.train_size}]")
    validation = load_dataset("trivia_qa", args.triviaqa_config, split=f"validation[:{args.test_size}]")

    print("=== Generating train cache ===")
    generate_or_load_teacher_sequences(tokenizer, teacher, train, "train", args)
    print("=== Generating test cache ===")
    generate_or_load_teacher_sequences(tokenizer, teacher, validation, "test", args)

    print("[done] generated caches in", cache_dir)


if __name__ == "__main__":
    main()
