#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive/chat generation with GPT-OSS plus an optional PEFT LoRA adapter."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from kd_teacher import load_teacher
from gptoss_kd_capture import get_dtype


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


def first_device(model: torch.nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--adapter_dir", type=str, default="")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device_map", type=str, default="cuda:0")
    parser.add_argument("--reasoning_effort", type=str, default="medium")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--show_special_tokens", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    return parser


def generate_once(model: torch.nn.Module, tokenizer: Any, args: argparse.Namespace, prompt: str) -> str:
    device = first_device(model)
    input_ids = apply_chat_template(tokenizer, prompt, args.reasoning_effort).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
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
            use_cache=True,
        )
    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=not args.show_special_tokens)


def main() -> None:
    args = build_parser().parse_args()
    model, tokenizer = load_teacher(
        args.model,
        args.adapter_dir,
        get_dtype(args.dtype),
        args.device,
        args.device_map,
        args.trust_remote_code,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.prompt:
        print(generate_once(model, tokenizer, args, args.prompt))
        return

    print("Interactive GPT-OSS LoRA mode. Empty line exits.")
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
