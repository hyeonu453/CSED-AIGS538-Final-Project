#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOFU unlearning for GPT-OSS full replacement layers.

This mirrors train_full_replacement_joint_kd.py's replacement loading,
trainable-parameter selection, optimizer loop, best_current saving, and final
checkpoint layout. The objective is adapted from open-unlearning-lora's
unlearn trainers:

  GradAscent: minimize -forget_nll
  GradDiff:   minimize gamma * (-forget_nll) + alpha * retain_nll
  SimNPO:     minimize gamma * simnpo_forget + alpha * retain_nll
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import cycle
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from full_replacement_utils import (
    IGNORE_INDEX,
    QABatch,
    SupervisedCollator,
    TofuQADataset,
    load_student_with_replacements,
    save_replacements,
)
from kd_losses import free_cuda


def make_optimizer(params: list[torch.nn.Parameter], args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.optim == "adamw_8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=args.weight_decay)
        except ImportError:
            print("[optim] bitsandbytes not available; falling back to torch AdamW")
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def compute_lr(step: int, args: argparse.Namespace) -> float:
    scheduler_name = getattr(args, "lr_scheduler", "constant")
    min_lr = float(getattr(args, "min_lr", 0.0))
    warmup_steps = int(getattr(args, "warmup_steps", 0))
    if warmup_steps > 0 and step <= warmup_steps:
        return float(args.lr) * max(1e-8, float(step) / float(warmup_steps))
    decay_steps = max(1, int(args.steps) - max(0, warmup_steps))
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    if scheduler_name == "constant":
        return float(args.lr)
    if scheduler_name == "cosine":
        return min_lr + (float(args.lr) - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if scheduler_name == "linear":
        return min_lr + (float(args.lr) - min_lr) * (1.0 - progress)
    raise ValueError(f"Unknown lr scheduler: {scheduler_name}")


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def get_model_input_device(model: torch.nn.Module, args: argparse.Namespace) -> torch.device:
    if getattr(args, "student_device_map", "single") == "single":
        return torch.device(args.student_device)
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        for param in model.parameters():
            return param.device
    return torch.device(args.student_device)


def batch_to_model_inputs(batch: QABatch, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    inputs = {
        "input_ids": batch.input_ids.to(device),
        "attention_mask": batch.attention_mask.to(device),
    }
    return inputs, batch.labels


def batch_nll_sum(model: torch.nn.Module, batch: QABatch, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    input_device = get_model_input_device(model, args)
    inputs, labels = batch_to_model_inputs(batch, input_device)
    outputs = model(**inputs, use_cache=False)
    logits = outputs.logits[..., :-1, :].contiguous()
    labels = labels.to(logits.device)[..., 1:].contiguous()
    token_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")(
        logits.transpose(-1, -2),
        labels,
    )
    seq_loss = token_loss.sum(dim=-1)
    token_count = labels.ne(IGNORE_INDEX).sum(dim=-1).clamp_min(1)
    return seq_loss, token_count


def model_ce_loss(model: torch.nn.Module, batch: QABatch, args: argparse.Namespace) -> torch.Tensor:
    seq_loss, token_count = batch_nll_sum(model, batch, args)
    return seq_loss.sum() / token_count.sum().clamp_min(1)


def simnpo_forget_loss(model: torch.nn.Module, batch: QABatch, args: argparse.Namespace) -> torch.Tensor:
    seq_loss, token_count = batch_nll_sum(model, batch, args)
    forget_score = seq_loss / token_count - float(args.simnpo_delta)
    return -F.logsigmoid(float(args.simnpo_beta) * forget_score).mean() * 2.0 / float(args.simnpo_beta)


def unlearn_losses(
    model: torch.nn.Module,
    forget_batch: QABatch,
    retain_batch: QABatch | None,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    forget_ce = model_ce_loss(model, forget_batch, args)
    retain_ce = forget_ce.new_zeros(())
    if retain_batch is not None:
        retain_ce = model_ce_loss(model, retain_batch, args)

    if args.method == "grad_ascent":
        forget_objective = -forget_ce
    elif args.method == "grad_diff":
        forget_objective = -forget_ce
    elif args.method == "simnpo":
        forget_objective = simnpo_forget_loss(model, forget_batch, args)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    retain_weight = 0.0 if args.method == "grad_ascent" else float(args.alpha_retain)
    loss = float(args.gamma_forget) * forget_objective + retain_weight * retain_ce
    return {
        "loss": loss,
        "forget_objective": forget_objective.detach(),
        "forget_ce": forget_ce.detach(),
        "retain_ce": retain_ce.detach(),
    }


@torch.no_grad()
def evaluate_unlearn(
    model: torch.nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader | None,
    args: argparse.Namespace,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    retain_iter = cycle(retain_loader) if retain_loader is not None else None
    totals = {"objective": 0.0, "forget_objective": 0.0, "forget_ce": 0.0, "retain_ce": 0.0}
    total_batches = 0
    for i, forget_batch in enumerate(forget_loader):
        if i >= max_batches:
            break
        retain_batch = next(retain_iter) if retain_iter is not None else None
        losses = unlearn_losses(model, forget_batch, retain_batch, args)
        totals["objective"] += float(losses["loss"].item())
        totals["forget_objective"] += float(losses["forget_objective"].item())
        totals["forget_ce"] += float(losses["forget_ce"].item())
        totals["retain_ce"] += float(losses["retain_ce"].item())
        total_batches += 1
    model.train()
    denom = max(1, total_batches)
    return {
        "objective": totals["objective"] / denom,
        "forget_objective": totals["forget_objective"] / denom,
        "forget_ce": totals["forget_ce"] / denom,
        "retain_ce": totals["retain_ce"] / denom,
        "batches": float(total_batches),
    }


def build_tofu_dataset(tokenizer, dataset_name: str, args: argparse.Namespace, limit: int) -> TofuQADataset:
    return TofuQADataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split=args.dataset_split,
        max_length=args.max_length,
        question_key=args.question_key,
        answer_key=args.answer_key,
        reasoning_effort=args.reasoning_effort if args.reasoning_effort != "none" else None,
        limit=limit,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--student_device", type=str, default="cuda:1")
    parser.add_argument("--student_device_map", choices=["single", "auto", "balanced", "balanced_low_0", "sequential"], default="single")
    parser.add_argument("--model_dtype", type=str, default="bfloat16")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--layer_checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--qwen_layer", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--mole_alpha", type=float, default=-1.0)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--forget_split", type=str, default="forget10")
    parser.add_argument("--retain_split", type=str, default="retain90")
    parser.add_argument("--eval_forget_split", type=str, default="")
    parser.add_argument("--eval_retain_split", type=str, default="")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--question_key", type=str, default="question")
    parser.add_argument("--answer_key", type=str, default="answer")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--reasoning_effort", type=str, default="low")
    parser.add_argument("--forget_limit", type=int, default=-1)
    parser.add_argument("--retain_limit", type=int, default=-1)
    parser.add_argument("--eval_forget_limit", type=int, default=-1)
    parser.add_argument("--eval_retain_limit", type=int, default=-1)

    parser.add_argument("--method", choices=["grad_ascent", "grad_diff", "simnpo"], default="grad_diff")
    parser.add_argument("--gamma_forget", type=float, default=1.0)
    parser.add_argument("--alpha_retain", type=float, default=1.0)
    parser.add_argument("--simnpo_beta", type=float, default=4.5)
    parser.add_argument("--simnpo_delta", type=float, default=0.0)

    parser.add_argument("--trainable", choices=["mole_router", "mole_only", "adapter", "all_replacement"], default="adapter")
    parser.add_argument("--optim", choices=["adamw_8bit", "adamw"], default="adamw_8bit")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", choices=["constant", "cosine", "linear"], default="constant")
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    student, _, replacements = load_student_with_replacements(args)
    student.train()

    eval_forget_split = args.eval_forget_split or args.forget_split
    eval_retain_split = args.eval_retain_split or args.retain_split
    collator = SupervisedCollator(tokenizer)

    forget_data = build_tofu_dataset(tokenizer, args.forget_split, args, args.forget_limit)
    retain_data = None
    if args.method != "grad_ascent" or args.alpha_retain > 0:
        retain_data = build_tofu_dataset(tokenizer, args.retain_split, args, args.retain_limit)
    eval_forget_data = build_tofu_dataset(tokenizer, eval_forget_split, args, args.eval_forget_limit)
    eval_retain_data = None
    if retain_data is not None:
        eval_retain_data = build_tofu_dataset(tokenizer, eval_retain_split, args, args.eval_retain_limit)

    forget_loader = DataLoader(forget_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    retain_loader = DataLoader(retain_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator) if retain_data is not None else None
    eval_forget_loader = DataLoader(eval_forget_data, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    eval_retain_loader = DataLoader(eval_retain_data, batch_size=args.batch_size, shuffle=False, collate_fn=collator) if eval_retain_data is not None else None
    forget_iter: Iterable[QABatch] = cycle(forget_loader)
    retain_iter: Iterable[QABatch] | None = cycle(retain_loader) if retain_loader is not None else None

    params = [p for p in student.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters selected.")
    optimizer = make_optimizer(params, args)

    history: list[dict[str, float]] = []
    initial_eval = evaluate_unlearn(student, eval_forget_loader, eval_retain_loader, args, args.eval_batches)
    best_eval_objective = float(initial_eval["objective"])
    best_eval_step = 0
    print("[eval step 0000]", initial_eval)
    history.append({"step": 0.0, **initial_eval, "best": 1.0})
    save_replacements(replacements, output_dir / "best_current", 0, final=True)
    print(f"[save] best_current checkpoint step 0 objective={best_eval_objective:.6f}")

    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        current_lr = compute_lr(step, args)
        set_optimizer_lr(optimizer, current_lr)
        accum = {"objective": 0.0, "forget_objective": 0.0, "forget_ce": 0.0, "retain_ce": 0.0}
        for _ in range(args.grad_accum_steps):
            forget_batch = next(forget_iter)
            retain_batch = next(retain_iter) if retain_iter is not None else None
            losses = unlearn_losses(student, forget_batch, retain_batch, args)
            loss = losses["loss"] / args.grad_accum_steps
            loss.backward()
            accum["objective"] += float(losses["loss"].detach().item())
            accum["forget_objective"] += float(losses["forget_objective"].item())
            accum["forget_ce"] += float(losses["forget_ce"].item())
            accum["retain_ce"] += float(losses["retain_ce"].item())

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            row = {
                "step": float(step),
                "train_objective": accum["objective"] / args.grad_accum_steps,
                "train_forget_objective": accum["forget_objective"] / args.grad_accum_steps,
                "train_forget_ce": accum["forget_ce"] / args.grad_accum_steps,
                "train_retain_ce": accum["retain_ce"] / args.grad_accum_steps,
                "lr": float(current_lr),
            }
            print(
                f"[train step {step:04d}] objective={row['train_objective']:.6f} "
                f"forget_obj={row['train_forget_objective']:.6f} "
                f"forget_ce={row['train_forget_ce']:.6f} retain_ce={row['train_retain_ce']:.6f} "
                f"lr={row['lr']:.6e}"
            )
            history.append(row)

        if args.eval_every > 0 and step % args.eval_every == 0:
            eval_row = evaluate_unlearn(student, eval_forget_loader, eval_retain_loader, args, args.eval_batches)
            print(f"[eval step {step:04d}]", eval_row)
            is_best = float(eval_row["objective"]) < best_eval_objective
            history.append({"step": float(step), **eval_row, "best": float(is_best)})
            if is_best:
                best_eval_objective = float(eval_row["objective"])
                best_eval_step = int(step)
                save_replacements(replacements, output_dir / "best_current", step, final=True)
                print(f"[save] best_current checkpoint step {step} objective={best_eval_objective:.6f}")

        if args.save_every > 0 and step % args.save_every == 0:
            save_replacements(replacements, output_dir / "checkpoints", step, final=False)
            print(f"[save] checkpoint step {step}")

    save_replacements(replacements, output_dir, args.steps, final=True)
    final_eval = evaluate_unlearn(student, eval_forget_loader, eval_retain_loader, args, args.eval_batches)
    summary = {
        "args": vars(args),
        "trainable_params": int(sum(p.numel() for p in params)),
        "replacement_layers": sorted(replacements),
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "best_eval": {"step": float(best_eval_step), "objective": float(best_eval_objective)},
        "history": history,
    }
    (output_dir / "unlearn_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[save] final replacement layers and summary written to {output_dir}")
    del student
    free_cuda()


if __name__ == "__main__":
    main()
