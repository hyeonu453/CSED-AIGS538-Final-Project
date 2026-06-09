#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint full-replacement KL distillation for GPT-OSS replacement layers.

Loads previously layerwise-trained replacement layers, inserts them into a
GPT-OSS student backbone, freezes the backbone, trains selected replacement
parameters against an online teacher KL objective, and saves the tuned
replacement layer states to a new output directory.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None

from gptoss_kd_capture import get_dtype
from kd_losses import free_cuda
from full_replacement_utils import (
    IGNORE_INDEX,
    QABatch,
    SupervisedCollator,
    TofuQADataset,
    load_student_with_replacements,
    save_replacements,
)


class TriviaQAChatDataset(Dataset):
    def __init__(
        self,
        tokenizer: Any,
        split_expr: str,
        max_length: int,
        config_name: str = "rc.nocontext",
        limit: int = -1,
        reasoning_effort: str | None = "low",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reasoning_effort = reasoning_effort
        self.data = load_dataset("trivia_qa", config_name, split=split_expr)
        if limit > 0:
            self.data = self.data.select(range(min(limit, len(self.data))))

    def __len__(self) -> int:
        return len(self.data)

    def _apply_chat_template(self, question: str) -> list[int]:
        messages = [{"role": "user", "content": question}]
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        try:
            ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("reasoning_effort", None)
            ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        if hasattr(ids, "keys") and "input_ids" in ids:
            ids = ids["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)[: self.max_length]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data[idx]
        ids = self._apply_chat_template(row["question"])
        if not ids:
            eos = self.tokenizer.eos_token_id
            ids = [int(eos)] if eos is not None else [0]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "labels": torch.tensor(ids, dtype=torch.long),
        }


class JsonlGeneratedTextDataset(Dataset):
    def __init__(self, tokenizer: Any, jsonl_path: str, max_length: int, limit: int = -1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Generated JSONL not found: {path}")
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        if limit > 0:
            rows = rows[:limit]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        text = row["text"]
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        ids = encoded["input_ids"]
        if not ids:
            eos = self.tokenizer.eos_token_id
            ids = [int(eos)] if eos is not None else [0]
        prompt_tokens = int(row.get("prompt_tokens", 0))
        prompt_tokens = min(max(0, prompt_tokens), len(ids))
        labels = [IGNORE_INDEX] * prompt_tokens + ids[prompt_tokens:]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class PromptOnlyCollator:
    def __init__(self, tokenizer: Any, padding_side: str = "right"):
        self.tokenizer = tokenizer
        self.padding_side = padding_side

    def _pad(self, rows: list[torch.Tensor], value: int) -> torch.Tensor:
        if self.padding_side == "left":
            rows = [torch.flip(row, dims=[0]) for row in rows]
            padded = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=value)
            return torch.flip(padded, dims=[1])
        return torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=value)

    def __call__(self, instances: list[dict[str, torch.Tensor]]) -> QABatch:
        input_ids = self._pad([x["input_ids"] for x in instances], int(self.tokenizer.pad_token_id))
        attention_mask = input_ids.ne(int(self.tokenizer.pad_token_id)).long()
        labels = self._pad([x["labels"] for x in instances], IGNORE_INDEX)
        return QABatch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def load_teacher(args: argparse.Namespace) -> torch.nn.Module:
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


def build_dataset_and_collator(tokenizer: Any, args: argparse.Namespace, eval_mode: bool = False):
    if args.dataset == "tofu_retain":
        split = args.eval_tofu_split if eval_mode else args.tofu_split
        limit = args.eval_limit if eval_mode else args.train_limit
        return (
            TofuQADataset(
                tokenizer=tokenizer,
                dataset_name=split,
                split=args.dataset_split,
                max_length=args.max_length,
                reasoning_effort=args.reasoning_effort if args.reasoning_effort != "none" else None,
                limit=limit,
            ),
            SupervisedCollator(tokenizer),
            True,
        )
    if args.dataset == "triviaqa_prompt":
        split_expr = args.eval_triviaqa_split if eval_mode else args.triviaqa_split
        limit = args.eval_limit if eval_mode else args.train_limit
        return (
            TriviaQAChatDataset(
                tokenizer=tokenizer,
                split_expr=split_expr,
                max_length=args.max_length,
                config_name=args.triviaqa_config,
                limit=limit,
                reasoning_effort=args.reasoning_effort if args.reasoning_effort != "none" else None,
            ),
            PromptOnlyCollator(tokenizer),
            False,
        )
    if args.dataset == "triviaqa_generated":
        jsonl_path = args.eval_generated_jsonl if eval_mode else args.train_generated_jsonl
        if not jsonl_path:
            raise ValueError("--train_generated_jsonl/--eval_generated_jsonl are required for triviaqa_generated")
        limit = args.eval_limit if eval_mode else args.train_limit
        return (
            JsonlGeneratedTextDataset(
                tokenizer=tokenizer,
                jsonl_path=jsonl_path,
                max_length=args.max_length,
                limit=limit,
            ),
            PromptOnlyCollator(tokenizer),
            True,
        )
    raise ValueError(f"Unknown dataset: {args.dataset}")


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


def make_loss_weights(
    batch: QABatch,
    args: argparse.Namespace,
    answer_only: bool,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch.labels[:, 1:].to(device)
    shifted_input_ids = batch.input_ids[:, 1:].to(device)
    attention_mask = batch.attention_mask[:, 1:].to(device).bool()
    answer_mask = labels.ne(IGNORE_INDEX) & attention_mask
    prompt_mask = attention_mask & ~answer_mask
    if answer_only:
        prompt_mask = torch.zeros_like(prompt_mask)
    weights = (
        answer_mask.float() * float(args.answer_loss_weight)
        + prompt_mask.float() * float(args.prompt_loss_weight)
    )
    targets = torch.where(answer_mask, labels, shifted_input_ids)
    return targets, weights


def weighted_ce_loss(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if not bool(weights.gt(0).any().item()):
        return student_logits.sum() * 0.0
    token_loss = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(weights)
    return (token_loss * weights).sum() / weights.sum().clamp_min(1e-8)


def weighted_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    weights: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not bool(weights.gt(0).any().item()):
        zero = student_logits.sum() * 0.0
        return zero, zero, zero

    temperature = float(args.kl_temperature)
    token_chunk_size = max(1, int(args.kl_token_chunk_size))
    denom = weights.sum().clamp_min(1e-8)
    forward_sum = student_logits.new_zeros((), dtype=torch.float32)
    reverse_sum = student_logits.new_zeros((), dtype=torch.float32)

    for start in range(0, student_logits.shape[1], token_chunk_size):
        end = min(start + token_chunk_size, student_logits.shape[1])
        chunk_weights = weights[:, start:end]
        if not bool(chunk_weights.gt(0).any().item()):
            continue
        teacher_chunk = teacher_logits[:, start:end, :].to(student_logits.device, dtype=torch.float32)
        student_chunk = student_logits[:, start:end, :].float()
        teacher_log_probs = F.log_softmax(teacher_chunk / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_chunk / temperature, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_probs = student_log_probs.exp()
        forward_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
        reverse_token = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        forward_sum = forward_sum + (forward_token * chunk_weights).sum()
        reverse_sum = reverse_sum + (reverse_token * chunk_weights).sum()
        del teacher_chunk, student_chunk, teacher_log_probs, student_log_probs, teacher_probs, student_probs

    scale = temperature * temperature
    forward_kl = forward_sum / denom * scale
    reverse_kl = reverse_sum / denom * scale
    if args.kl_direction == "forward":
        kl = forward_kl
    elif args.kl_direction == "reverse":
        kl = reverse_kl
    elif args.kl_direction == "both":
        kl = 0.5 * (forward_kl + reverse_kl)
    else:
        raise ValueError(f"Unknown KL direction: {args.kl_direction}")
    return kl, forward_kl, reverse_kl


def joint_kd_loss(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    batch: QABatch,
    args: argparse.Namespace,
    answer_only: bool,
) -> dict[str, torch.Tensor]:
    student_inputs = {
        "input_ids": batch.input_ids.to(args.student_device),
        "attention_mask": batch.attention_mask.to(args.student_device),
    }
    teacher_inputs = {
        "input_ids": batch.input_ids.to(args.teacher_device),
        "attention_mask": batch.attention_mask.to(args.teacher_device),
    }
    with torch.no_grad():
        teacher_logits = teacher(**teacher_inputs, use_cache=False).logits[:, :-1, :].detach()
    student_logits = student(**student_inputs, use_cache=False).logits[:, :-1, :]
    targets, weights = make_loss_weights(batch, args, answer_only, args.student_device)
    ce = weighted_ce_loss(student_logits, targets, weights)
    kl, forward_kl, reverse_kl = weighted_kl_loss(teacher_logits, student_logits, weights, args)
    loss = float(args.ce_weight) * ce + float(args.kl_weight) * kl
    return {
        "loss": loss,
        "ce": ce.detach(),
        "kl": kl.detach(),
        "forward_kl": forward_kl.detach(),
        "reverse_kl": reverse_kl.detach(),
        "tokens": weights.sum().detach(),
    }


@torch.no_grad()
def evaluate_joint_kd(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    loader: DataLoader,
    answer_only: bool,
    args: argparse.Namespace,
    max_batches: int,
) -> dict[str, float]:
    student.eval()
    totals = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "forward_kl": 0.0, "reverse_kl": 0.0, "tokens": 0.0}
    total_batches = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        losses = joint_kd_loss(teacher, student, batch, args, answer_only=answer_only)
        for key in totals:
            totals[key] += float(losses[key].item())
        total_batches += 1
    student.train()
    denom = max(1, total_batches)
    return {
        "loss": totals["loss"] / denom,
        "ce": totals["ce"] / denom,
        "kl": totals["kl"] / denom,
        "forward_kl": totals["forward_kl"] / denom,
        "reverse_kl": totals["reverse_kl"] / denom,
        "tokens": totals["tokens"] / denom,
        "batches": float(total_batches),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--teacher_model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--teacher_adapter_dir", type=str, default="")
    parser.add_argument("--teacher_device", type=str, default="cuda:0")
    parser.add_argument("--student_device", type=str, default="cuda:1")
    parser.add_argument("--teacher_dtype", type=str, default="bfloat16")
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

    parser.add_argument("--dataset", choices=["tofu_retain", "triviaqa_prompt", "triviaqa_generated"], default="tofu_retain")
    parser.add_argument("--tofu_split", type=str, default="retain90")
    parser.add_argument("--eval_tofu_split", type=str, default="holdout10")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--triviaqa_config", type=str, default="rc.nocontext")
    parser.add_argument("--triviaqa_split", type=str, default="train[:1000]")
    parser.add_argument("--eval_triviaqa_split", type=str, default="validation[:100]")
    parser.add_argument("--train_generated_jsonl", type=str, default="")
    parser.add_argument("--eval_generated_jsonl", type=str, default="")
    parser.add_argument("--train_limit", type=int, default=-1)
    parser.add_argument("--eval_limit", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--reasoning_effort", type=str, default="low")
    parser.add_argument("--kl_answer_only", action="store_true")
    parser.add_argument("--ce_weight", type=float, default=0.7)
    parser.add_argument("--kl_weight", type=float, default=0.3)
    parser.add_argument("--kl_direction", choices=["forward", "reverse", "both"], default="forward")
    parser.add_argument("--kl_temperature", type=float, default=2.0)
    parser.add_argument("--kl_token_chunk_size", type=int, default=64)
    parser.add_argument("--prompt_loss_weight", type=float, default=0.0)
    parser.add_argument("--answer_loss_weight", type=float, default=1.0)

    parser.add_argument("--trainable", choices=["mole_router", "mole_only", "adapter", "projection_only", "all_replacement"], default="mole_router")
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
    teacher = load_teacher(args)
    student.train()

    train_data, train_collator, dataset_answer_only = build_dataset_and_collator(tokenizer, args, eval_mode=False)
    eval_data, eval_collator, eval_answer_only = build_dataset_and_collator(tokenizer, args, eval_mode=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=train_collator)
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, collate_fn=eval_collator)
    train_iter: Iterable[QABatch] = cycle(train_loader)
    answer_only = args.kl_answer_only or dataset_answer_only
    eval_answer_only = args.kl_answer_only or eval_answer_only

    params = [p for p in student.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters selected.")
    optimizer = make_optimizer(params, args)

    history: list[dict[str, float]] = []
    initial_eval = evaluate_joint_kd(teacher, student, eval_loader, eval_answer_only, args, args.eval_batches)
    best_eval_loss = float(initial_eval["loss"])
    best_eval_step = 0
    print("[eval step 0000]", initial_eval)
    history.append({"step": 0.0, **initial_eval, "best": 1.0})
    save_replacements(replacements, output_dir / "best_current", 0, final=True)
    print(f"[save] best_current checkpoint step 0 loss={best_eval_loss:.6f}")

    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        current_lr = compute_lr(step, args)
        set_optimizer_lr(optimizer, current_lr)
        accum_loss = 0.0
        accum_ce = 0.0
        accum_kl = 0.0
        accum_forward_kl = 0.0
        accum_reverse_kl = 0.0
        accum_tokens = 0.0
        for _ in range(args.grad_accum_steps):
            batch = next(train_iter)
            losses = joint_kd_loss(teacher, student, batch, args, answer_only=answer_only)
            loss = losses["loss"] / args.grad_accum_steps
            loss.backward()
            accum_loss += float(losses["loss"].detach().item())
            accum_ce += float(losses["ce"].item())
            accum_kl += float(losses["kl"].item())
            accum_forward_kl += float(losses["forward_kl"].item())
            accum_reverse_kl += float(losses["reverse_kl"].item())
            accum_tokens += float(losses["tokens"].item())

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            row = {
                "step": float(step),
                "train_loss": accum_loss / args.grad_accum_steps,
                "train_ce": accum_ce / args.grad_accum_steps,
                "train_kl": accum_kl / args.grad_accum_steps,
                "train_forward_kl": accum_forward_kl / args.grad_accum_steps,
                "train_reverse_kl": accum_reverse_kl / args.grad_accum_steps,
                "train_tokens": accum_tokens / args.grad_accum_steps,
                "lr": float(current_lr),
            }
            print(
                f"[train step {step:04d}] loss={row['train_loss']:.6f} ce={row['train_ce']:.6f} "
                f"kl={row['train_kl']:.6f} fkl={row['train_forward_kl']:.6f} "
                f"rkl={row['train_reverse_kl']:.6f} tokens={row['train_tokens']:.1f} lr={row['lr']:.6e}"
            )
            history.append(row)

        if args.eval_every > 0 and step % args.eval_every == 0:
            eval_row = evaluate_joint_kd(teacher, student, eval_loader, eval_answer_only, args, args.eval_batches)
            print(f"[eval step {step:04d}]", eval_row)
            is_best = float(eval_row["loss"]) < best_eval_loss
            history.append({"step": float(step), **eval_row, "best": float(is_best)})
            if is_best:
                best_eval_loss = float(eval_row["loss"])
                best_eval_step = int(step)
                save_replacements(replacements, output_dir / "best_current", step, final=True)
                print(f"[save] best_current checkpoint step {step} loss={best_eval_loss:.6f}")

        if args.save_every > 0 and step % args.save_every == 0:
            save_replacements(replacements, output_dir / "checkpoints", step, final=False)
            print(f"[save] checkpoint step {step}")

    save_replacements(replacements, output_dir, args.steps, final=True)
    final_eval = evaluate_joint_kd(teacher, student, eval_loader, eval_answer_only, args, args.eval_batches)
    summary = {
        "args": vars(args),
        "trainable_params": int(sum(p.numel() for p in params)),
        "replacement_layers": sorted(replacements),
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "best_eval": {"step": float(best_eval_step), "loss": float(best_eval_loss)},
        "history": history,
    }
    (output_dir / "joint_kd_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[save] final replacement layers and summary written to {output_dir}")
    del teacher
    del student
    free_cuda()


if __name__ == "__main__":
    main()
