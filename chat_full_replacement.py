#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive/chat generation with GPT-OSS replacement layers."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any

import torch

from full_replacement_utils import load_student_with_replacements


def apply_chat_template(tokenizer: Any, prompt: str, reasoning_effort: str | None) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    try:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("reasoning_effort", None)
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    if isinstance(ids, list):
        ids = torch.tensor([ids], dtype=torch.long)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--layer_checkpoint_dir", type=str, required=True)
    parser.add_argument("--student_device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_dtype", type=str, default="bfloat16")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--qwen_layer", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--mole_alpha", type=float, default=-1.0)
    parser.add_argument("--trainable", choices=["mole_router", "mole_only", "adapter", "projection_only", "all_replacement"], default="adapter")
    parser.add_argument("--reasoning_effort", type=str, default="medium")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    return parser


def generate_once(model: torch.nn.Module, tokenizer: Any, args: argparse.Namespace, prompt: str) -> str:
    input_ids = apply_chat_template(tokenizer, prompt, args.reasoning_effort).to(args.student_device)
    attention_mask = torch.ones_like(input_ids, device=args.student_device)
    do_sample = args.temperature > 0
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def main() -> None:
    args = build_parser().parse_args()
    load_args = SimpleNamespace(
        student_model=args.student_model,
        layer_checkpoint_dir=args.layer_checkpoint_dir,
        student_device=args.student_device,
        model_dtype=args.model_dtype,
        qwen_model=args.qwen_model,
        layers=args.layers,
        qwen_layer=args.qwen_layer,
        rank=args.rank,
        mole_alpha=args.mole_alpha,
        trainable=args.trainable,
        trust_remote_code=args.trust_remote_code,
        gradient_checkpointing=False,
    )
    model, _, _ = load_student_with_replacements(load_args)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.config.use_cache = False

    if args.prompt:
        print(generate_once(model, tokenizer, args, args.prompt))
        return

    print("Interactive mode. Empty line exits.")
    while True:
        try:
            prompt = input("user> ").strip()
        except EOFError:
            break
        if not prompt:
            break
        print("assistant>", generate_once(model, tokenizer, args, prompt))


if __name__ == "__main__":
    main()
