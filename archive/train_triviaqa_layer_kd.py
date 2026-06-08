#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer-local TriviaQA KD for CustomMoLELayer replacement.

Shared pieces live in:
  - kd_data.py: dataset/prompt loading
  - kd_teacher.py: teacher loading and layer-pair capture
  - kd_mole.py: CustomMoLELayer construction and GPT-OSS replacement wrapper
  - kd_layer_train.py: layer-local KD train/eval
  - kd_full_eval.py: full-model replacement logits eval

Future experiments:
  - Layer-wise KD: loop this script over --layer values or add a small multi-layer
    driver that reuses the same modules.
  - Full-replacement-after-KD: reuse kd_mole.GPTOSSLayerReplacement and add a
    separate full-model train loop against teacher logits/hidden states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gptoss_kd_capture import get_dtype, unwrap_for_introspection
from kd_data import load_triviaqa_prompts
from kd_full_eval import evaluate_full_replacement_logits
from kd_layer_train import evaluate_layer_local, train_layer_kd
from kd_losses import free_cuda
from kd_mole import build_custom_layer
from kd_teacher import capture_layer_pairs, load_teacher


def run_layer_kd(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading TriviaQA ===")
    train_prompts, test_prompts = load_triviaqa_prompts(args.triviaqa_config, args.train_size, args.test_size)
    (out_dir / "train_prompts_preview.txt").write_text("\n".join(train_prompts[:20]), encoding="utf-8")
    (out_dir / "test_prompts_preview.txt").write_text("\n".join(test_prompts[:20]), encoding="utf-8")

    print("=== Loading teacher for layer-pair capture ===")
    teacher, tokenizer = load_teacher(
        args.teacher_model,
        args.teacher_adapter_dir,
        get_dtype(args.teacher_dtype),
        args.device,
        args.teacher_device_map,
        args.trust_remote_code,
    )
    teacher_config = unwrap_for_introspection(teacher).config

    print("=== Capturing train layer pairs ===")
    train_pairs = capture_layer_pairs(
        teacher=teacher,
        tokenizer=tokenizer,
        prompts=train_prompts,
        layer_idx=args.layer,
        max_length=args.max_length,
        batch_size=args.capture_batch_size,
        cpu_dtype=get_dtype(args.cache_dtype),
    )
    print("=== Capturing test layer pairs ===")
    test_pairs = capture_layer_pairs(
        teacher=teacher,
        tokenizer=tokenizer,
        prompts=test_prompts,
        layer_idx=args.layer,
        max_length=args.max_length,
        batch_size=args.capture_batch_size,
        cpu_dtype=get_dtype(args.cache_dtype),
    )

    del teacher
    free_cuda()

    print("=== Building CustomMoLELayer ===")
    custom_layer, qwen_config = build_custom_layer(
        teacher_config=teacher_config,
        qwen_model_name_or_path=args.qwen_model,
        layer_idx=args.qwen_layer if args.qwen_layer >= 0 else args.layer,
        rank=args.rank,
        mole_alpha=args.mole_alpha,
        train_dtype=get_dtype(args.train_dtype),
        device=args.device,
        init_from_qwen=not args.no_init_from_qwen,
    )

    print("=== Layer-local KD training ===")
    train_summary = train_layer_kd(custom_layer, qwen_config, train_pairs, args)
    layer_local_eval = evaluate_layer_local(custom_layer, qwen_config, test_pairs, args)
    print("[layer-local eval]", layer_local_eval)

    full_eval = None
    if not args.skip_full_replacement_eval:
        full_eval = evaluate_full_replacement_logits(
            model_name_or_path=args.teacher_model,
            adapter_dir=args.teacher_adapter_dir,
            custom_layer=custom_layer,
            qwen_config=qwen_config,
            test_pairs=test_pairs,
            args=args,
        )
        print("[full replacement eval]", full_eval)

    summary = {
        "experiment": "layer_local_kd_then_full_replacement_eval",
        "args": vars(args),
        "train_pairs_shape": {
            "input_ids": list(train_pairs.input_ids.shape),
            "layer_input": list(train_pairs.layer_input.shape),
            "layer_output": list(train_pairs.layer_output.shape),
        },
        "test_pairs_shape": {
            "input_ids": list(test_pairs.input_ids.shape),
            "layer_input": list(test_pairs.layer_input.shape),
            "layer_output": list(test_pairs.layer_output.shape),
        },
        "train": train_summary,
        "layer_local_eval": layer_local_eval,
        "full_replacement_eval": full_eval,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_state:
        torch.save({k: v.detach().cpu() for k, v in custom_layer.state_dict().items()}, out_dir / "custom_mole_layer.pt")
    print(f"Saved summary to {out_dir / 'summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--teacher_adapter_dir", type=str, default="")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--triviaqa_config", type=str, default="rc.nocontext")
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--qwen_layer", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--mole_alpha", type=float, default=-1.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--capture_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--full_eval_batch_size", type=int, default=4)
    parser.add_argument("--teacher_dtype", type=str, default="bfloat16")
    parser.add_argument("--cache_dtype", type=str, default="bfloat16")
    parser.add_argument("--train_dtype", type=str, default="float32")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher_device_map", type=str, default="auto_if_cuda")
    parser.add_argument("--train_mode", choices=["adapter", "all"], default="adapter")
    parser.add_argument("--no_init_from_qwen", action="store_true")
    parser.add_argument("--skip_full_replacement_eval", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="triviaqa_layer_kd")
    parser.add_argument("--save_state", action="store_true")
    return parser


def main() -> None:
    run_layer_kd(build_parser().parse_args())


if __name__ == "__main__":
    main()
