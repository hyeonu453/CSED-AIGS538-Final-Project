#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOFU unlearning for the GPT-OSS full replacement student.

This script keeps the layerwise KD checkpoints immutable: it loads replacement
layers from --init_replacement_dir, inserts them into a GPT-OSS backbone, trains
only the selected replacement parameters, and writes newly tuned layers to
--output_dir.

Default objective follows open-unlearning-lora GradDiff:
  loss = alpha * CE(retain answers) - gamma * CE(forget answers)
where labels are masked so loss is computed only on assistant answer tokens.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from gptoss_kd_capture import get_decoder_layers, get_dtype, unwrap_for_introspection
from kd_losses import free_cuda
from kd_mole import GPTOSSLayerReplacement, build_custom_layer

IGNORE_INDEX = -100


def normalize_input_ids(tokenized_output: Any) -> list[int]:
    if hasattr(tokenized_output, "keys") and "input_ids" in tokenized_output:
        ids = tokenized_output["input_ids"]
    else:
        ids = tokenized_output
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return list(ids)


def apply_prompt_template(tokenizer: Any, question: str, reasoning_effort: str | None) -> str:
    messages = [{"role": "user", "content": question}]
    kwargs: dict[str, Any] = {}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def tokenize_qa(tokenizer: Any, question: str, answer: str, max_length: int, reasoning_effort: str | None) -> dict[str, torch.Tensor]:
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    kwargs: dict[str, Any] = {}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    try:
        full_ids = normalize_input_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, **kwargs))
        prompt_ids = normalize_input_ids(tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, **kwargs))
    except TypeError:
        full_ids = normalize_input_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
        prompt_ids = normalize_input_ids(tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True))

    eos = tokenizer.eos_token_id
    if eos is not None and (not full_ids or full_ids[-1] != eos):
        full_ids = full_ids + [eos]

    full_ids = full_ids[:max_length]
    label_prefix = min(len(prompt_ids), len(full_ids))
    labels = [IGNORE_INDEX] * label_prefix + full_ids[label_prefix:]
    if len(labels) < len(full_ids):
        labels.extend([IGNORE_INDEX] * (len(full_ids) - len(labels)))
    if not any(label != IGNORE_INDEX for label in labels):
        labels = [IGNORE_INDEX] * len(full_ids)

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
    }


class TofuQADataset(Dataset):
    def __init__(self, split_name: str, max_samples: int, tokenizer: Any, max_length: int, reasoning_effort: str | None):
        split_expr = "train" if max_samples <= 0 else f"train[:{max_samples}]"
        self.rows = load_dataset("locuslab/TOFU", name=split_name, split=split_expr)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reasoning_effort = reasoning_effort

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        item = tokenize_qa(
            self.tokenizer,
            row["question"],
            row["answer"],
            self.max_length,
            self.reasoning_effort,
        )
        item["index"] = torch.tensor(idx, dtype=torch.long)
        return item


class ForgetRetainTorchDataset(Dataset):
    def __init__(self, forget: Dataset, retain: Dataset, anchor: str = "forget", seed: int = 42):
        if anchor not in {"forget", "retain"}:
            raise ValueError("anchor must be 'forget' or 'retain'")
        self.forget = forget
        self.retain = retain
        self.anchor = anchor
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.forget) if self.anchor == "forget" else len(self.retain)

    def __getitem__(self, idx: int) -> dict[str, dict[str, torch.Tensor]]:
        if self.anchor == "forget":
            retain_idx = int(torch.randint(0, len(self.retain), (1,), generator=self.generator).item())
            return {"forget": self.forget[idx], "retain": self.retain[retain_idx]}
        forget_idx = int(torch.randint(0, len(self.forget), (1,), generator=self.generator).item())
        return {"forget": self.forget[forget_idx], "retain": self.retain[idx]}


class SupervisedCollator:
    def __init__(self, tokenizer: Any, padding_side: str = "right"):
        self.tokenizer = tokenizer
        self.padding_side = padding_side

    def pad(self, tensors: list[torch.Tensor], value: int) -> torch.Tensor:
        if self.padding_side == "right":
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=value)
        flipped = [torch.flip(t, dims=[0]) for t in tensors]
        return torch.flip(torch.nn.utils.rnn.pad_sequence(flipped, batch_first=True, padding_value=value), dims=[1])

    def collate_leaf(self, instances: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = self.pad([x["input_ids"] for x in instances], self.tokenizer.pad_token_id)
        labels = self.pad([x["labels"] for x in instances], IGNORE_INDEX)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, Any]:
        if "input_ids" in instances[0]:
            return self.collate_leaf(instances)
        return {key: self([instance[key] for instance in instances]) for key in instances[0].keys()}


def ce_loss_from_model(model: torch.nn.Module, batch: dict[str, torch.Tensor], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    model_batch = {key: value.to(device) for key, value in batch.items()}
    outputs = model(**model_batch, use_cache=False)
    return outputs.loss, outputs.logits


def set_joint_trainable(model: torch.nn.Module, mode: str) -> tuple[int, int]:
    for param in model.parameters():
        param.requires_grad_(False)
    if mode == "all_replacement":
        prefixes = None
    elif mode == "mlp_mole_router":
        prefixes = ("custom_layer.mlp.router",)
    elif mode == "mlp_mole_only":
        prefixes = tuple()
    elif mode == "adapter":
        prefixes = ("custom_layer.proj_down", "custom_layer.proj_up", "custom_layer.mlp.router")
    else:
        raise ValueError(f"Unknown trainable mode: {mode}")

    for module in model.modules():
        if not isinstance(module, GPTOSSLayerReplacement):
            continue
        for name, param in module.named_parameters():
            train = False
            if mode == "all_replacement":
                train = True
            elif name.endswith(("_mole_A", "_mole_B")):
                train = True
            elif prefixes and name.startswith(prefixes):
                train = True
            param.requires_grad_(train)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def make_optimizer(params: list[torch.nn.Parameter], args: argparse.Namespace):
    if args.optim == "adamw_8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("bitsandbytes is required for --optim adamw_8bit") from exc
        return bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optim == "sgd":
        return torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def load_replacement_student(args: argparse.Namespace) -> tuple[torch.nn.Module, Any, list[int]]:
    dtype = get_dtype(args.model_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(args.student_device)
    inspected = unwrap_for_introspection(model)
    teacher_config = inspected.config
    layers = get_decoder_layers(inspected)
    layer_indices = parse_layers(args.layers, len(layers))

    for layer in layer_indices:
        state_path = Path(args.init_replacement_dir) / f"layer_{layer:02d}" / "custom_mole_layer.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing initial replacement checkpoint: {state_path}")
        custom_layer, qwen_config = build_custom_layer(
            teacher_config=teacher_config,
            qwen_model_name_or_path=args.qwen_model,
            layer_idx=args.qwen_layer if args.qwen_layer >= 0 else layer,
            rank=args.rank,
            mole_alpha=args.mole_alpha,
            train_dtype=dtype,
            device=args.student_device,
            init_from_qwen=False,
        )
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        custom_layer.load_state_dict(state, strict=True)
        replacement = GPTOSSLayerReplacement(custom_layer, qwen_config).to(device=args.student_device, dtype=dtype)
        layers[layer] = replacement
        print(f"[student] replaced layer {layer} from {state_path}")

    model.config.use_cache = False
    model.train()
    return model, teacher_config, layer_indices


def parse_layers(spec: str, num_layers: int) -> list[int]:
    if spec == "all":
        return list(range(num_layers))
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    result = sorted(set(layers))
    for layer in result:
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"Invalid layer {layer}; model has {num_layers} layers")
    return result


def save_replacement_layers(model: torch.nn.Module, layer_indices: list[int], output_dir: Path) -> None:
    layers = get_decoder_layers(unwrap_for_introspection(model))
    for layer in layer_indices:
        module = layers[layer]
        if not isinstance(module, GPTOSSLayerReplacement):
            raise TypeError(f"Layer {layer} is not a GPTOSSLayerReplacement")
        layer_dir = output_dir / f"layer_{layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        state = {key: value.detach().cpu() for key, value in module.custom_layer.state_dict().items()}
        torch.save(state, layer_dir / "custom_mole_layer.pt")
    print(f"[save] wrote tuned replacement layers to {output_dir}")


def evaluate_ce(model: torch.nn.Module, loader: DataLoader, key: str, device: str, max_batches: int) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            leaf = batch[key]
            labels = leaf["labels"].to(device)
            valid = labels.ne(IGNORE_INDEX)
            loss, _ = ce_loss_from_model(model, leaf, device)
            tokens = int(valid.sum().item())
            total_loss += float(loss.item()) * max(1, tokens)
            total_tokens += tokens
            batches += 1
            if max_batches > 0 and batches >= max_batches:
                break
    model.train()
    mean = total_loss / max(1, total_tokens)
    return {"loss": mean, "ppl": math.exp(min(mean, 50.0)), "tokens": float(total_tokens), "batches": float(batches)}


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    forget = TofuQADataset(args.forget_split, args.forget_size, tokenizer, args.max_length, args.reasoning_effort)
    retain = TofuQADataset(args.retain_split, args.retain_size, tokenizer, args.max_length, args.reasoning_effort)
    train_dataset = ForgetRetainTorchDataset(forget, retain, anchor=args.anchor, seed=args.seed)
    collator = SupervisedCollator(tokenizer, padding_side=args.padding_side)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, drop_last=False)

    model, teacher_config, layer_indices = load_replacement_student(args)
    trainable, total = set_joint_trainable(model, args.trainable_mode)
    print(f"[student] trainable={trainable:,} total={total:,} mode={args.trainable_mode}")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(params, args)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.steps,
    ) if args.steps > 0 else None

    summary: dict[str, Any] = {
        "args": vars(args),
        "layers": layer_indices,
        "trainable_params": int(trainable),
        "total_params": int(total),
        "losses": [],
    }
    (output_dir / "run_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    model.train()
    step = 0
    optimizer.zero_grad(set_to_none=True)
    while step < args.steps:
        for batch in loader:
            forget_loss, _ = ce_loss_from_model(model, batch["forget"], args.student_device)
            retain_loss, _ = ce_loss_from_model(model, batch["retain"], args.student_device)
            loss = args.alpha * retain_loss - args.gamma * forget_loss
            scaled = loss / args.grad_accum_steps
            scaled.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step == 0 or (step + 1) % args.log_every == 0:
                row = {
                    "step": int(step + 1),
                    "loss": float(loss.item()),
                    "forget_ce": float(forget_loss.item()),
                    "retain_ce": float(retain_loss.item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                summary["losses"].append(row)
                print(
                    f"[step {step + 1:05d}] loss={row['loss']:.6e} "
                    f"forget_ce={row['forget_ce']:.6e} retain_ce={row['retain_ce']:.6e} lr={row['lr']:.3e}"
                )
                (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

            step += 1
            if step >= args.steps:
                break

    if args.eval_batches != 0:
        eval_loader = DataLoader(train_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator)
        summary["eval_forget"] = evaluate_ce(model, eval_loader, "forget", args.student_device, args.eval_batches)
        summary["eval_retain"] = evaluate_ce(model, eval_loader, "retain", args.student_device, args.eval_batches)
        print("[eval forget]", summary["eval_forget"])
        print("[eval retain]", summary["eval_retain"])

    save_replacement_layers(model, layer_indices, output_dir)
    summary["saved_layers_dir"] = str(output_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del model
    free_cuda()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--init_replacement_dir", type=str, default="output/triviaqa_teacher_generated_len1024_new768")
    parser.add_argument("--output_dir", type=str, default="output/tofu_replacement_unlearn_graddiff")
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--qwen_layer", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--mole_alpha", type=float, default=-1.0)
    parser.add_argument("--forget_split", type=str, default="forget10")
    parser.add_argument("--retain_split", type=str, default="retain90")
    parser.add_argument("--forget_size", type=int, default=0)
    parser.add_argument("--retain_size", type=int, default=0)
    parser.add_argument("--anchor", choices=["forget", "retain"], default="forget")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--reasoning_effort", type=str, default="low")
    parser.add_argument("--padding_side", choices=["left", "right"], default="right")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_batches", type=int, default=20, help="0 disables eval; positive value caps eval batches")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0, help="forget CE ascent weight")
    parser.add_argument("--alpha", type=float, default=1.0, help="retain CE descent weight")
    parser.add_argument("--optim", choices=["adamw_torch", "adamw_8bit", "sgd"], default="adamw_8bit")
    parser.add_argument("--trainable_mode", choices=["mlp_mole_router", "mlp_mole_only", "adapter", "all_replacement"], default="mlp_mole_router")
    parser.add_argument("--student_device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_dtype", type=str, default="bfloat16")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
