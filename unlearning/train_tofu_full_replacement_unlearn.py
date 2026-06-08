#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint TOFU unlearning for a full GPT-OSS replacement student.

This script keeps the layerwise KD checkpoints immutable: it loads replacement
layers from --layer_checkpoint_dir, inserts them into a GPT-OSS backbone, trains
only selected replacement parameters, and writes tuned replacement states to a
new --output_dir.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import cycle
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None

from full_replacement_utils import (
    QABatch,
    SupervisedCollator,
    TofuQADataset,
    answer_token_count,
    kl_teacher_student_loss,
    load_student_with_replacements,
    model_loss,
    save_replacements,
)
from gptoss_kd_capture import get_dtype
from kd_losses import free_cuda


def make_optimizer(params: list[torch.nn.Parameter], args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.optim == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=args.weight_decay)
        except ImportError:
            print("[optim] bitsandbytes not available; falling back to torch AdamW")
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def load_teacher_for_kl(args: argparse.Namespace) -> torch.nn.Module | None:
    if args.normal_kl_weight <= 0 and args.retain_kl_weight <= 0:
        return None
    dtype = get_dtype(args.teacher_dtype)
    print(f"=== Loading KL teacher on {args.teacher_device} ===")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype,
        device_map={"": args.teacher_device},
        trust_remote_code=args.trust_remote_code,
    )
    if args.teacher_adapter_dir:
        if PeftModel is None:
            raise ImportError("peft is required when --teacher_adapter_dir is provided.")
        teacher = PeftModel.from_pretrained(teacher, args.teacher_adapter_dir)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--teacher_model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--teacher_adapter_dir", type=str, default="")
    parser.add_argument("--teacher_device", type=str, default="cuda:0")
    parser.add_argument("--teacher_dtype", type=str, default="bfloat16")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--layer_checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--qwen_layer", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--mole_alpha", type=float, default=-1.0)
    parser.add_argument("--student_device", type=str, default="cuda:1")
    parser.add_argument("--model_dtype", type=str, default="bfloat16")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--forget_split", type=str, default="forget10")
    parser.add_argument("--retain_split", type=str, default="retain90")
    parser.add_argument("--normal_split", type=str, default="holdout10")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--reasoning_effort", type=str, default="low")
    parser.add_argument("--forget_limit", type=int, default=-1)
    parser.add_argument("--retain_limit", type=int, default=-1)
    parser.add_argument("--normal_limit", type=int, default=-1)

    parser.add_argument("--trainable", choices=["mole_router", "mole_only", "adapter", "all_replacement"], default="mole_router")
    parser.add_argument("--optim", choices=["adamw_8bit", "adamw"], default="adamw_8bit")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--gamma_forget", type=float, default=1.0)
    parser.add_argument("--alpha_retain", type=float, default=1.0)
    parser.add_argument("--retain_ce_weight", type=float, default=None)
    parser.add_argument("--retain_kl_weight", type=float, default=0.0)
    parser.add_argument("--normal_kl_weight", type=float, default=0.0)
    parser.add_argument("--kl_temperature", type=float, default=1.0)
    parser.add_argument("--kl_token_chunk_size", type=int, default=64)
    parser.add_argument("--normal_kl_answer_only", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.retain_ce_weight is None:
        args.retain_ce_weight = args.alpha_retain
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    student, _, replacements = load_student_with_replacements(args)
    student.train()

    forget_data = TofuQADataset(
        tokenizer=tokenizer,
        dataset_name=args.forget_split,
        split=args.dataset_split,
        max_length=args.max_length,
        reasoning_effort=args.reasoning_effort if args.reasoning_effort != "none" else None,
        limit=args.forget_limit,
    )
    retain_data = TofuQADataset(
        tokenizer=tokenizer,
        dataset_name=args.retain_split,
        split=args.dataset_split,
        max_length=args.max_length,
        reasoning_effort=args.reasoning_effort if args.reasoning_effort != "none" else None,
        limit=args.retain_limit,
    )
    normal_data = TofuQADataset(
        tokenizer=tokenizer,
        dataset_name=args.normal_split,
        split=args.dataset_split,
        max_length=args.max_length,
        reasoning_effort=args.reasoning_effort if args.reasoning_effort != "none" else None,
        limit=args.normal_limit,
    )
    collator = SupervisedCollator(tokenizer)
    forget_loader = DataLoader(forget_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    retain_loader = DataLoader(retain_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    normal_loader = DataLoader(normal_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    eval_forget_loader = DataLoader(forget_data, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    eval_retain_loader = DataLoader(retain_data, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    forget_iter: Iterable[QABatch] = cycle(forget_loader)
    retain_iter: Iterable[QABatch] = cycle(retain_loader)
    normal_iter: Iterable[QABatch] = cycle(normal_loader)
    teacher = load_teacher_for_kl(args)

    params = [p for p in student.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters selected.")
    optimizer = make_optimizer(params, args)

    history: list[dict[str, float]] = []
    initial_eval = evaluate_losses(student, eval_forget_loader, eval_retain_loader, args, args.eval_batches)
    print("[eval step 0000]", initial_eval)
    history.append({"step": 0.0, **initial_eval})

    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        accum_forget = 0.0
        accum_retain = 0.0
        accum_retain_kl = 0.0
        accum_normal_kl = 0.0
        accum_total = 0.0
        for _ in range(args.grad_accum_steps):
            forget_batch = next(forget_iter)
            retain_batch = next(retain_iter)
            forget_loss = model_loss(student, forget_batch, args.student_device)
            retain_loss = model_loss(student, retain_batch, args.student_device)
            retain_kl = torch.zeros((), device=args.student_device)
            normal_kl = torch.zeros((), device=args.student_device)
            if teacher is not None and args.retain_kl_weight > 0:
                retain_kl = kl_teacher_student_loss(teacher, student, retain_batch, args, answer_only=True)
            if teacher is not None and args.normal_kl_weight > 0:
                normal_batch = next(normal_iter)
                normal_kl = kl_teacher_student_loss(
                    teacher, student, normal_batch, args, answer_only=args.normal_kl_answer_only
                )
            loss = (
                args.retain_ce_weight * retain_loss
                + args.retain_kl_weight * retain_kl
                + args.normal_kl_weight * normal_kl
                - args.gamma_forget * forget_loss
            ) / args.grad_accum_steps
            loss.backward()
            accum_forget += float(forget_loss.detach().item())
            accum_retain += float(retain_loss.detach().item())
            accum_retain_kl += float(retain_kl.detach().item())
            accum_normal_kl += float(normal_kl.detach().item())
            accum_total += float(loss.detach().item())

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            row = {
                "step": float(step),
                "forget_ce": accum_forget / args.grad_accum_steps,
                "retain_ce": accum_retain / args.grad_accum_steps,
                "retain_kl": accum_retain_kl / args.grad_accum_steps,
                "normal_kl": accum_normal_kl / args.grad_accum_steps,
                "objective": accum_total,
            }
            print(
                f"[train step {step:04d}] "
                f"forget_ce={row['forget_ce']:.6f} retain_ce={row['retain_ce']:.6f} "
                f"retain_kl={row['retain_kl']:.6f} normal_kl={row['normal_kl']:.6f} "
                f"objective={row['objective']:.6f}"
            )
            history.append(row)

        if args.eval_every > 0 and step % args.eval_every == 0:
            eval_row = evaluate_losses(student, eval_forget_loader, eval_retain_loader, args, args.eval_batches)
            print(f"[eval step {step:04d}]", eval_row)
            history.append({"step": float(step), **eval_row})

        if args.save_every > 0 and step % args.save_every == 0:
            save_replacements(replacements, output_dir / "checkpoints", step, final=False)
            print(f"[save] checkpoint step {step}")

    save_replacements(replacements, output_dir, args.steps, final=True)
    final_eval = evaluate_losses(student, eval_forget_loader, eval_retain_loader, args, args.eval_batches)
    summary = {
        "args": vars(args),
        "trainable_params": int(sum(p.numel() for p in params)),
        "replacement_layers": sorted(replacements),
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "history": history,
    }
    (output_dir / "unlearn_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[save] final replacement layers and summary written to {output_dir}")
    if teacher is not None:
        del teacher
    del student
    free_cuda()


if __name__ == "__main__":
    main()
