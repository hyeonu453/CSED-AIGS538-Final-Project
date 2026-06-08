#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TriviaQA KD test for replacing one GPT-OSS 20B layer with CustomMoLELayer.

Default flow:
1. Load ~1000 TriviaQA train prompts and ~100 validation/test prompts.
2. Capture GPT-OSS teacher layer input/output hidden states.
3. Train CustomMoLELayer to approximate one GPT-OSS decoder layer.
4. Evaluate layer-local approximation on held-out TriviaQA prompts.
5. Replace the real GPT-OSS decoder layer with the trained CustomMoLELayer and
   compare full-model logits against the original teacher.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from gptoss_kd_capture import get_decoder_layers, get_dtype, get_first_device, unwrap_for_introspection
from modeling import CustomMoLELayer

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None


@dataclass
class LayerPairs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    layer_input: torch.Tensor
    layer_output: torch.Tensor


class GPTOSSLayerReplacement(nn.Module):
    def __init__(self, custom_layer: CustomMoLELayer, qwen_config: Any):
        super().__init__()
        self.custom_layer = custom_layer
        self.rotary = Qwen2RotaryEmbedding(qwen_config, device=next(custom_layer.parameters()).device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del position_embeddings
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
            position_ids = position_ids.expand(hidden_states.shape[0], -1)
        rotary_input = self.custom_layer.proj_down(hidden_states)
        qwen_position_embeddings = self.rotary(rotary_input, position_ids)
        return self.custom_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=qwen_position_embeddings,
            **kwargs,
        )


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def normalized_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    return (pred - target).pow(2).sum() / (target.pow(2).sum() + eps)


def mean_cosine(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    pred = pred / (pred.norm(dim=-1, keepdim=True) + eps)
    target = target / (target.norm(dim=-1, keepdim=True) + eps)
    return (pred * target).sum(dim=-1).mean()


def masked_tokens(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return x[attention_mask.to(x.device).bool()]


def load_triviaqa_prompts(config_name: str, train_n: int, test_n: int) -> tuple[list[str], list[str]]:
    train = load_dataset("trivia_qa", config_name, split=f"train[:{train_n}]")
    validation = load_dataset("trivia_qa", config_name, split=f"validation[:{test_n}]")

    def format_prompt(row: dict[str, Any]) -> str:
        return f"Question: {row['question']}\nAnswer:"

    return [format_prompt(row) for row in train], [format_prompt(row) for row in validation]


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
    resolved_device_map = device_map
    if device_map == "auto_if_cuda":
        resolved_device_map = "auto" if device.startswith("cuda") else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=resolved_device_map,
        trust_remote_code=trust_remote_code,
    )
    if adapter_dir:
        if PeftModel is None:
            raise ImportError("peft is required when --teacher_adapter_dir is provided.")
        model = PeftModel.from_pretrained(model, adapter_dir)
    if resolved_device_map is None:
        model.to(device)
    model.eval()
    return model, tokenizer


def capture_layer_pairs(
    teacher: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layer_idx: int,
    max_length: int,
    batch_size: int,
    cpu_dtype: torch.dtype,
) -> LayerPairs:
    input_device = get_first_device(teacher)
    inspected = unwrap_for_introspection(teacher)
    layers = get_decoder_layers(inspected)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"Invalid layer {layer_idx}; teacher has {len(layers)} layers.")

    all_input_ids: list[torch.Tensor] = []
    all_attention_masks: list[torch.Tensor] = []
    all_x: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        batch = tokenizer(
            chunk,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        model_batch = {key: value.to(input_device) for key, value in batch.items()}
        saved: dict[str, torch.Tensor] = {}

        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            y = output[0] if isinstance(output, tuple) else output
            if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(y, torch.Tensor):
                raise RuntimeError("Failed to capture teacher layer input/output.")
            saved["x"] = inputs[0].detach().to("cpu", dtype=cpu_dtype).contiguous()
            saved["y"] = y.detach().to("cpu", dtype=cpu_dtype).contiguous()

        handle = layers[layer_idx].register_forward_hook(hook)
        try:
            with torch.no_grad():
                _ = teacher(**model_batch, use_cache=False)
        finally:
            handle.remove()

        if "x" not in saved or "y" not in saved:
            raise RuntimeError(f"No layer pair captured for batch starting at {start}.")
        all_input_ids.append(model_batch["input_ids"].detach().cpu())
        all_attention_masks.append(model_batch["attention_mask"].detach().cpu())
        all_x.append(saved["x"])
        all_y.append(saved["y"])
        print(f"[capture] {min(start + batch_size, len(prompts))}/{len(prompts)}")

    return LayerPairs(
        input_ids=torch.cat(all_input_ids, dim=0),
        attention_mask=torch.cat(all_attention_masks, dim=0),
        layer_input=torch.cat(all_x, dim=0),
        layer_output=torch.cat(all_y, dim=0),
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
    train_dtype: torch.dtype,
    device: str,
    init_from_qwen: bool,
) -> tuple[CustomMoLELayer, Any]:
    qwen_config = AutoConfig.from_pretrained(qwen_model_name_or_path, trust_remote_code=True)
    custom_layer = CustomMoLELayer(make_mole_config(teacher_config, rank, mole_alpha), qwen_config, layer_idx)
    if init_from_qwen:
        qwen = AutoModelForCausalLM.from_pretrained(
            qwen_model_name_or_path,
            torch_dtype=train_dtype,
            device_map=None,
            trust_remote_code=True,
        )
        qwen_layers = get_decoder_layers(qwen)
        copy_qwen_layer_weights(custom_layer, qwen_layers[min(layer_idx, len(qwen_layers) - 1)])
        del qwen
        free_cuda()
    custom_layer.to(device=device, dtype=train_dtype)
    return custom_layer, qwen_config


def set_trainable_parameters(model: CustomMoLELayer, mode: str) -> None:
    for param in model.parameters():
        param.requires_grad_(mode == "all")
    if mode == "all":
        return
    if mode != "adapter":
        raise ValueError(f"Unknown train mode: {mode}")
    for name, param in model.named_parameters():
        if name.startswith(("proj_down", "proj_up", "mlp.router")) or name.endswith(("_mole_A", "_mole_B")):
            param.requires_grad_(True)


def forward_custom_layer(
    model: CustomMoLELayer,
    rotary: Qwen2RotaryEmbedding,
    x: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
    rotary_input = model.proj_down(x)
    position_embeddings = rotary(rotary_input, position_ids)
    return model(
        x,
        attention_mask=attention_mask,
        position_ids=None,
        position_embeddings=position_embeddings,
        use_cache=False,
    )


def train_custom_layer(
    model: CustomMoLELayer,
    qwen_config: Any,
    train_pairs: LayerPairs,
    test_pairs: LayerPairs,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, float]]:
    set_trainable_parameters(model, args.train_mode)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[student] trainable={trainable:,} total={total:,} mode={args.train_mode}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    rotary = Qwen2RotaryEmbedding(qwen_config, device=args.device).to(args.device)
    rng = torch.Generator().manual_seed(args.seed)
    losses: list[dict[str, float]] = []

    x_train = train_pairs.layer_input
    y_train = train_pairs.layer_output
    mask_train = train_pairs.attention_mask

    model.train()
    for step in range(args.steps + 1):
        indices = torch.randint(0, x_train.shape[0], (args.train_batch_size,), generator=rng)
        x = x_train[indices].to(args.device, dtype=get_dtype(args.train_dtype))
        y = y_train[indices].to(args.device, dtype=get_dtype(args.train_dtype))
        mask = mask_train[indices].to(args.device)

        optimizer.zero_grad(set_to_none=True)
        pred = forward_custom_layer(model, rotary, x, attention_mask=None)
        pred_tokens = masked_tokens(pred, mask)
        target_tokens = masked_tokens(y, mask)
        loss = F.mse_loss(pred_tokens.float(), target_tokens.float())
        nmse = normalized_mse(pred_tokens, target_tokens)

        if step == 0 or step % args.log_every == 0 or step == args.steps:
            row = {"step": float(step), "mse": float(loss.item()), "nmse": float(nmse.item())}
            losses.append(row)
            print(f"[train step {step:04d}] mse={row['mse']:.6e} nmse={row['nmse']:.6e}")
        if step == args.steps:
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()

    local_eval = evaluate_layer_local(model, qwen_config, test_pairs, args)
    train_summary = {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "losses": losses,
    }
    return train_summary, local_eval


def evaluate_layer_local(
    model: CustomMoLELayer,
    qwen_config: Any,
    pairs: LayerPairs,
    args: argparse.Namespace,
) -> dict[str, float]:
    rotary = Qwen2RotaryEmbedding(qwen_config, device=args.device).to(args.device)
    model.eval()
    mse_sum = 0.0
    norm_sum = 0.0
    cos_sum = 0.0
    n_tokens = 0
    with torch.no_grad():
        for start in range(0, pairs.layer_input.shape[0], args.eval_batch_size):
            idx = slice(start, start + args.eval_batch_size)
            x = pairs.layer_input[idx].to(args.device, dtype=get_dtype(args.train_dtype))
            y = pairs.layer_output[idx].to(args.device, dtype=get_dtype(args.train_dtype))
            mask = pairs.attention_mask[idx].to(args.device)
            pred = forward_custom_layer(model, rotary, x, attention_mask=None)
            pred_tokens = masked_tokens(pred, mask)
            target_tokens = masked_tokens(y, mask)
            diff = (pred_tokens.float() - target_tokens.float()).pow(2)
            mse_sum += float(diff.sum().item())
            norm_sum += float(target_tokens.float().pow(2).sum().item())
            cos_sum += float(mean_cosine(pred_tokens, target_tokens).item()) * int(pred_tokens.shape[0])
            n_tokens += int(pred_tokens.shape[0])
    return {
        "tokens": float(n_tokens),
        "mse_mean": mse_sum / max(1, n_tokens * pairs.layer_output.shape[-1]),
        "nmse": mse_sum / max(1e-12, norm_sum),
        "mean_cos": cos_sum / max(1, n_tokens),
    }


def evaluate_full_replacement(
    model_name_or_path: str,
    adapter_dir: str,
    tokenizer: Any,
    custom_layer: CustomMoLELayer,
    qwen_config: Any,
    test_pairs: LayerPairs,
    args: argparse.Namespace,
) -> dict[str, float]:
    print("=== Loading teacher for full replacement eval ===")
    teacher, _ = load_teacher(
        model_name_or_path,
        adapter_dir,
        get_dtype(args.teacher_dtype),
        args.device,
        args.teacher_device_map,
        args.trust_remote_code,
    )
    input_device = get_first_device(teacher)
    inspected = unwrap_for_introspection(teacher)
    layers = get_decoder_layers(inspected)

    logits_mse_sum = 0.0
    logits_norm_sum = 0.0
    kl_sum = 0.0
    top1_match_sum = 0
    n_tokens = 0

    with torch.no_grad():
        original_logits: list[torch.Tensor] = []
        for start in range(0, test_pairs.input_ids.shape[0], args.full_eval_batch_size):
            idx = slice(start, start + args.full_eval_batch_size)
            batch = {
                "input_ids": test_pairs.input_ids[idx].to(input_device),
                "attention_mask": test_pairs.attention_mask[idx].to(input_device),
            }
            out = teacher(**batch, use_cache=False)
            original_logits.append(out.logits.detach().cpu())
        original_logits_cpu = torch.cat(original_logits, dim=0)

    original_layer = layers[args.layer]
    custom_layer = custom_layer.to(device=input_device, dtype=get_dtype(args.teacher_dtype))
    replacement = GPTOSSLayerReplacement(custom_layer, qwen_config).to(device=input_device, dtype=get_dtype(args.teacher_dtype))
    replacement.eval()
    layers[args.layer] = replacement

    with torch.no_grad():
        offset = 0
        for start in range(0, test_pairs.input_ids.shape[0], args.full_eval_batch_size):
            idx = slice(start, start + args.full_eval_batch_size)
            batch = {
                "input_ids": test_pairs.input_ids[idx].to(input_device),
                "attention_mask": test_pairs.attention_mask[idx].to(input_device),
            }
            replaced_logits = teacher(**batch, use_cache=False).logits.detach().cpu()
            base_logits = original_logits_cpu[offset : offset + replaced_logits.shape[0]]
            offset += replaced_logits.shape[0]
            mask = test_pairs.attention_mask[idx].bool()
            base_tokens = base_logits[mask]
            replaced_tokens = replaced_logits[mask]

            diff = (replaced_tokens.float() - base_tokens.float()).pow(2)
            logits_mse_sum += float(diff.sum().item())
            logits_norm_sum += float(base_tokens.float().pow(2).sum().item())
            base_log_probs = F.log_softmax(base_tokens.float(), dim=-1)
            replaced_log_probs = F.log_softmax(replaced_tokens.float(), dim=-1)
            kl = F.kl_div(replaced_log_probs, base_log_probs.exp(), reduction="sum")
            kl_sum += float(kl.item())
            top1_match_sum += int((base_tokens.argmax(dim=-1) == replaced_tokens.argmax(dim=-1)).sum().item())
            n_tokens += int(base_tokens.shape[0])

    layers[args.layer] = original_layer
    del teacher
    free_cuda()

    vocab = int(original_logits_cpu.shape[-1])
    return {
        "tokens": float(n_tokens),
        "logits_mse_mean": logits_mse_sum / max(1, n_tokens * vocab),
        "logits_nmse": logits_mse_sum / max(1e-12, logits_norm_sum),
        "kl_base_to_replaced_mean": kl_sum / max(1, n_tokens),
        "top1_match": top1_match_sum / max(1, n_tokens),
    }


def main() -> None:
    args = build_parser().parse_args()
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
        teacher, tokenizer, train_prompts, args.layer, args.max_length, args.capture_batch_size, get_dtype(args.cache_dtype)
    )
    print("=== Capturing test layer pairs ===")
    test_pairs = capture_layer_pairs(
        teacher, tokenizer, test_prompts, args.layer, args.max_length, args.capture_batch_size, get_dtype(args.cache_dtype)
    )

    del teacher
    free_cuda()

    print("=== Building CustomMoLELayer ===")
    custom_layer, qwen_config = build_custom_layer(
        teacher_config,
        args.qwen_model,
        args.qwen_layer if args.qwen_layer >= 0 else args.layer,
        args.rank,
        args.mole_alpha,
        get_dtype(args.train_dtype),
        args.device,
        not args.no_init_from_qwen,
    )

    print("=== KD training ===")
    train_summary, layer_local_eval = train_custom_layer(custom_layer, qwen_config, train_pairs, test_pairs, args)
    print("[layer-local eval]", layer_local_eval)

    full_eval: dict[str, float] | None = None
    if not args.skip_full_replacement_eval:
        full_eval = evaluate_full_replacement(
            args.teacher_model,
            args.teacher_adapter_dir,
            tokenizer,
            custom_layer,
            qwen_config,
            test_pairs,
            args,
        )
        print("[full replacement eval]", full_eval)

    summary = {
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
    parser.add_argument("--out_dir", type=str, default="triviaqa_mole_layer_replacement")
    parser.add_argument("--save_state", action="store_true")
    return parser


if __name__ == "__main__":
    main()
