#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke train loop for replacing one GPT-OSS 20B decoder layer with CustomMoLELayer.

The test builds a tiny teacher dataset from one or more prompts:

    GPT-OSS layer input hidden state -> GPT-OSS layer output hidden state

Then it trains modeling.CustomMoLELayer, which projects GPT-OSS hidden states down
to a Qwen2.5-3B hidden size, applies Qwen-style attention plus the custom MoLE MLP,
and projects back to the GPT-OSS hidden size.

This is intentionally a small approximation/replacement smoke test, not a full
distillation run.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from gptoss_kd_capture import get_decoder_layers, get_dtype, get_first_device, unwrap_for_introspection
from modeling import CustomMoLELayer

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - optional dependency
    PeftModel = None


@dataclass
class LayerPairBatch:
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    layer_input: torch.Tensor
    layer_output: torch.Tensor


def load_prompt_file(path: str | Path, max_prompts: int) -> list[str]:
    prompts = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return prompts[:max_prompts] if max_prompts > 0 else prompts


def load_teacher(
    model_name_or_path: str,
    adapter_dir: str,
    dtype: torch.dtype,
    device: str,
    device_map: str | None,
    trust_remote_code: bool,
) -> tuple[nn.Module, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_device_map: str | None = device_map
    if device_map == "auto_if_cuda":
        resolved_device_map = "auto" if device.startswith("cuda") else None

    teacher = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=resolved_device_map,
        trust_remote_code=trust_remote_code,
    )
    if adapter_dir:
        if PeftModel is None:
            raise ImportError("peft is required when --teacher_adapter_dir is provided.")
        teacher = PeftModel.from_pretrained(teacher, adapter_dir)
    if resolved_device_map is None:
        teacher.to(device)
    teacher.eval()
    return teacher, tokenizer


def capture_teacher_layer_pairs(
    teacher: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layer_idx: int,
    max_length: int,
) -> LayerPairBatch:
    input_device = get_first_device(teacher)
    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    model_batch = {key: value.to(input_device) for key, value in batch.items()}

    inspected = unwrap_for_introspection(teacher)
    layers = get_decoder_layers(inspected)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"Invalid layer {layer_idx}; teacher has {len(layers)} layers.")

    saved: dict[str, torch.Tensor] = {}

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError(f"Could not capture input hidden states for layer {layer_idx}.")
        layer_output = output[0] if isinstance(output, tuple) else output
        if not isinstance(layer_output, torch.Tensor):
            raise RuntimeError(f"Unexpected layer output type: {type(layer_output)}")
        saved["x"] = inputs[0].detach().cpu().contiguous()
        saved["y"] = layer_output.detach().cpu().contiguous()

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = teacher(**model_batch, use_cache=False)
    finally:
        handle.remove()

    if "x" not in saved or "y" not in saved:
        raise RuntimeError("Teacher layer hook did not capture input/output hidden states.")

    return LayerPairBatch(
        input_ids=model_batch["input_ids"].detach().cpu(),
        attention_mask=model_batch.get("attention_mask").detach().cpu()
        if model_batch.get("attention_mask") is not None
        else None,
        layer_input=saved["x"],
        layer_output=saved["y"],
    )


def make_mole_config(teacher_config: Any, rank: int, mole_alpha: float) -> Any:
    config = copy.deepcopy(teacher_config)
    config.rank = int(rank)
    config.mole_alpha = float(mole_alpha if mole_alpha > 0 else rank)
    if not hasattr(config, "num_experts"):
        config.num_experts = 32
    if not hasattr(config, "num_experts_per_tok"):
        config.num_experts_per_tok = 4
    if not hasattr(config, "initializer_range"):
        config.initializer_range = 0.02
    return config


def copy_qwen_layer_weights(custom_layer: CustomMoLELayer, qwen_layer: nn.Module) -> None:
    custom_layer.self_attn.load_state_dict(qwen_layer.self_attn.state_dict(), strict=True)
    custom_layer.input_layernorm.load_state_dict(qwen_layer.input_layernorm.state_dict(), strict=True)
    custom_layer.post_attention_layernorm.load_state_dict(qwen_layer.post_attention_layernorm.state_dict(), strict=True)
    custom_layer.mlp.experts.gate_proj.load_state_dict(qwen_layer.mlp.gate_proj.state_dict(), strict=True)
    custom_layer.mlp.experts.up_proj.load_state_dict(qwen_layer.mlp.up_proj.state_dict(), strict=True)
    custom_layer.mlp.experts.down_proj.load_state_dict(qwen_layer.mlp.down_proj.state_dict(), strict=True)


def build_custom_layer(
    teacher_config: Any,
    qwen_model_name_or_path: str,
    layer_idx: int,
    rank: int,
    mole_alpha: float,
    init_from_qwen: bool,
    train_dtype: torch.dtype,
    device: str,
) -> tuple[CustomMoLELayer, Qwen2RotaryEmbedding]:
    qwen_config = AutoConfig.from_pretrained(qwen_model_name_or_path, trust_remote_code=True)
    mole_config = make_mole_config(teacher_config, rank=rank, mole_alpha=mole_alpha)
    custom_layer = CustomMoLELayer(mole_config, qwen_config, layer_idx=layer_idx)

    if init_from_qwen:
        qwen = AutoModelForCausalLM.from_pretrained(
            qwen_model_name_or_path,
            torch_dtype=train_dtype,
            device_map=None,
            trust_remote_code=True,
        )
        qwen_layers = get_decoder_layers(qwen)
        qwen_layer_idx = min(layer_idx, len(qwen_layers) - 1)
        copy_qwen_layer_weights(custom_layer, qwen_layers[qwen_layer_idx])
        del qwen
        gc.collect()

    custom_layer.to(device=device, dtype=train_dtype)
    rotary = Qwen2RotaryEmbedding(qwen_config, device=device)
    rotary.to(device=device)
    return custom_layer, rotary


def set_trainable_parameters(model: CustomMoLELayer, mode: str) -> None:
    for param in model.parameters():
        param.requires_grad_(False)

    if mode == "all":
        for param in model.parameters():
            param.requires_grad_(True)
        return

    if mode != "adapter":
        raise ValueError(f"Unknown train mode: {mode}")

    trainable_prefixes = (
        "proj_down",
        "proj_up",
        "mlp.router",
    )
    trainable_suffixes = ("_mole_A", "_mole_B")
    for name, param in model.named_parameters():
        if name.startswith(trainable_prefixes) or name.endswith(trainable_suffixes):
            param.requires_grad_(True)


def masked_token_view(x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return x.reshape(-1, x.shape[-1])
    mask = attention_mask.to(x.device).bool()
    return x[mask]


def normalized_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    return (pred - target).pow(2).sum() / (target.pow(2).sum() + eps)


def train_replacement(args: argparse.Namespace) -> dict[str, Any]:
    teacher_dtype = get_dtype(args.teacher_dtype)
    train_dtype = get_dtype(args.train_dtype)

    prompts = load_prompt_file(args.prompts_file, args.max_prompts)
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompts_file}")

    print("=== Loading teacher ===")
    teacher, tokenizer = load_teacher(
        args.teacher_model,
        args.teacher_adapter_dir,
        teacher_dtype,
        args.device,
        args.teacher_device_map,
        args.trust_remote_code,
    )
    teacher_config = unwrap_for_introspection(teacher).config

    print("=== Capturing teacher layer pair ===")
    pair = capture_teacher_layer_pairs(
        teacher=teacher,
        tokenizer=tokenizer,
        prompts=prompts,
        layer_idx=args.layer,
        max_length=args.max_length,
    )
    print(f"input_ids={tuple(pair.input_ids.shape)}")
    print(f"teacher layer input={tuple(pair.layer_input.shape)} output={tuple(pair.layer_output.shape)}")

    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=== Building CustomMoLELayer ===")
    model, rotary = build_custom_layer(
        teacher_config=teacher_config,
        qwen_model_name_or_path=args.qwen_model,
        layer_idx=args.qwen_layer if args.qwen_layer >= 0 else args.layer,
        rank=args.rank,
        mole_alpha=args.mole_alpha,
        init_from_qwen=not args.no_init_from_qwen,
        train_dtype=train_dtype,
        device=args.device,
    )
    set_trainable_parameters(model, args.train_mode)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"CustomMoLELayer params: trainable={trainable:,} total={total:,} mode={args.train_mode}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    x = pair.layer_input.to(device=args.device, dtype=train_dtype)
    y = pair.layer_output.to(device=args.device, dtype=train_dtype)
    attention_mask = pair.attention_mask.to(args.device) if pair.attention_mask is not None else None
    position_ids = torch.arange(x.shape[1], device=args.device).unsqueeze(0).expand(x.shape[0], -1)

    losses: list[dict[str, float]] = []
    model.train()
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            rotary_input = model.proj_down(x)
            position_embeddings = rotary(rotary_input, position_ids)

        pred = model(
            x,
            attention_mask=None,
            position_ids=None,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        pred_tokens = masked_token_view(pred, attention_mask)
        target_tokens = masked_token_view(y, attention_mask)
        loss = F.mse_loss(pred_tokens.float(), target_tokens.float())
        nmse = normalized_mse(pred_tokens, target_tokens)

        if step == 0 or step % args.log_every == 0 or step == args.steps:
            row = {"step": float(step), "mse": float(loss.item()), "nmse": float(nmse.item())}
            losses.append(row)
            print(f"[step {step:04d}] mse={row['mse']:.6e} nmse={row['nmse']:.6e}")

        if step == args.steps:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()

    summary = {
        "teacher_model": args.teacher_model,
        "teacher_adapter_dir": args.teacher_adapter_dir or None,
        "qwen_model": args.qwen_model,
        "teacher_layer": args.layer,
        "qwen_layer": args.qwen_layer if args.qwen_layer >= 0 else args.layer,
        "prompt_count": len(prompts),
        "input_shape": list(pair.layer_input.shape),
        "output_shape": list(pair.layer_output.shape),
        "trainable_params": int(trainable),
        "total_params": int(total),
        "losses": losses,
    }

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if args.save_state:
            torch.save(
                {key: value.detach().cpu() for key, value in model.state_dict().items()},
                out_dir / "custom_mole_layer_state.pt",
            )
        print(f"Saved summary to {out_dir / 'summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--teacher_adapter_dir", type=str, default="")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--prompts_file", type=str, default="smoke_prompts_layer_replacement.txt")
    parser.add_argument("--max_prompts", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--qwen_layer", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--mole_alpha", type=float, default=-1.0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--teacher_dtype", type=str, default="bfloat16")
    parser.add_argument("--train_dtype", type=str, default="float32")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher_device_map", type=str, default="auto_if_cuda")
    parser.add_argument("--train_mode", choices=["adapter", "all"], default="adapter")
    parser.add_argument("--no_init_from_qwen", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    parser.add_argument("--out_dir", type=str, default="mole_layer_replacement_smoke")
    parser.add_argument("--save_state", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_replacement(args)


if __name__ == "__main__":
    main()
