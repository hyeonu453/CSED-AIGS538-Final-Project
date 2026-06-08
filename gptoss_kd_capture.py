#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference utilities for GPT-OSS 20B attention-only LoRA knowledge distillation.

This module intentionally has no CLI entrypoint. Import it from training or
cache-building code to load a GPT-OSS teacher with an optional PEFT adapter and
capture, per decoder layer:

  - decoder layer output hidden states
  - MoE router logits/top-k probabilities/top-k expert ids

The implementation borrows the PEFT-introspection idea from
gptoss_raw_expert_lora_bank_peft.py: forward through the PEFT-wrapped teacher so
LoRA is active, but unwrap only for finding GPT-OSS internals and registering
hooks. Router logits are obtained through router.forward(...), not by directly
reading router.weight, so LoRA-wrapped router modules also work.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - optional dependency
    PeftModel = None


@dataclass(frozen=True)
class GPTOSSTeacherBundle:
    model: nn.Module
    tokenizer: Any
    introspection_model: nn.Module


@dataclass(frozen=True)
class LayerRouterCapture:
    logits: torch.Tensor
    topk_weights: torch.Tensor
    topk_indices: torch.Tensor
    num_experts: int
    top_k: int
    moe_name: str


@dataclass(frozen=True)
class LayerKDCapture:
    hidden_states: torch.Tensor
    router: Optional[LayerRouterCapture]


@dataclass(frozen=True)
class GPTOSSKDCapture:
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    layers: Dict[int, LayerKDCapture]


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def get_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def unwrap_for_introspection(model: nn.Module) -> nn.Module:
    current = model
    seen: set[int] = set()
    for _ in range(12):
        if id(current) in seen:
            break
        seen.add(id(current))

        if hasattr(current, "model"):
            current = current.model
            continue
        if hasattr(current, "base_model"):
            current = current.base_model
            continue
        break
    return current


def get_first_device(model: nn.Module) -> torch.device:
    for param in model.parameters():
        return param.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def first_tensor_device(module: nn.Module) -> torch.device:
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cpu")


def get_decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    candidates = (
        "model.layers",
        "model.decoder.layers",
        "transformer.h",
        "gpt_neox.layers",
        "backbone.layers",
    )
    for path in candidates:
        obj: Any = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and isinstance(obj, (nn.ModuleList, list, tuple)):
            return obj

    best: Optional[Sequence[nn.Module]] = None
    best_len = 0
    for _, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > best_len:
            best = module
            best_len = len(module)
    if best is None:
        raise RuntimeError("Could not find decoder layers automatically.")
    return best


def find_moe_modules_in_layer(layer: nn.Module) -> list[tuple[str, nn.Module]]:
    found: list[tuple[str, nn.Module]] = []
    for name, module in layer.named_modules():
        has_router = any(hasattr(module, attr) for attr in ("router", "gate", "router_gate"))
        if has_router and hasattr(module, "experts"):
            found.append((name, module))
    return found


def get_router_module(moe: nn.Module) -> nn.Module:
    for attr in ("router", "gate", "router_gate"):
        if hasattr(moe, attr):
            return getattr(moe, attr)
    raise RuntimeError(f"Could not find router/gate in {moe.__class__.__name__}")


def get_experts_module(moe: nn.Module) -> Any:
    if hasattr(moe, "experts"):
        return getattr(moe, "experts")
    raise RuntimeError(f"Could not find experts in {moe.__class__.__name__}")


def get_num_experts(experts: Any, router_logits: Optional[torch.Tensor] = None) -> int:
    for attr in ("num_experts", "num_local_experts"):
        if hasattr(experts, attr):
            return int(getattr(experts, attr))
    for attr in ("gate_up_proj", "down_proj"):
        if hasattr(experts, attr):
            return int(getattr(experts, attr).shape[0])
    if router_logits is not None:
        return int(router_logits.shape[-1])
    raise RuntimeError(f"Cannot infer number of experts from {experts.__class__.__name__}")


def get_top_k(moe: nn.Module, default: int = 4) -> int:
    for attr in ("top_k", "num_experts_per_tok", "topk", "num_selected_experts"):
        if hasattr(moe, attr):
            try:
                return int(getattr(moe, attr))
            except Exception:
                pass
    router = getattr(moe, "router", None)
    if router is not None and hasattr(router, "top_k"):
        return int(router.top_k)
    return default


def router_forward_logits(router: nn.Module, hidden_flat: torch.Tensor) -> torch.Tensor:
    output = router(hidden_flat)
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise RuntimeError(f"Unexpected router output type: {type(output)}")
    return output


def compute_router_topk(router_logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(router_logits.float(), dim=-1)
    k = min(int(top_k), int(probs.shape[-1]))
    topk_weights, topk_indices = torch.topk(probs, k=k, dim=-1)
    return topk_weights, topk_indices


def _as_cpu_tensor(
    tensor: torch.Tensor,
    *,
    dtype: Optional[torch.dtype],
    keep_batch_shape: bool,
    valid_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if valid_mask is not None and not keep_batch_shape:
        tensor = tensor[valid_mask.to(tensor.device).bool()]
    tensor = tensor.detach()
    if dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype=dtype)
    return tensor.cpu().contiguous()


def load_gptoss_attn_lora_teacher(
    model_name_or_path: str = "openai/gpt-oss-20b",
    adapter_dir: str | Path | None = None,
    *,
    dtype: str | torch.dtype = torch.bfloat16,
    device: str = "cuda",
    device_map: str | Mapping[str, Any] | None = "auto_if_cuda",
    trust_remote_code: bool = True,
    attn_implementation: Optional[str] = None,
) -> GPTOSSTeacherBundle:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_device_map: str | Mapping[str, Any] | None = device_map
    if device_map == "auto_if_cuda":
        resolved_device_map = "auto" if str(device).startswith("cuda") else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=get_dtype(dtype),
        device_map=resolved_device_map,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )

    if adapter_dir is not None:
        if PeftModel is None:
            raise ImportError("peft is required when adapter_dir is provided.")
        adapter_path = resolve_path(adapter_dir)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))

    if resolved_device_map is None:
        model.to(device)
    model.eval()
    return GPTOSSTeacherBundle(
        model=model,
        tokenizer=tokenizer,
        introspection_model=unwrap_for_introspection(model),
    )


@contextmanager
def capture_gptoss_kd_hooks(
    teacher: nn.Module,
    *,
    layers: Optional[Iterable[int]] = None,
    cpu_dtype: Optional[torch.dtype] = torch.bfloat16,
    keep_batch_shape: bool = True,
    capture_router_logits: bool = True,
    attention_mask: Optional[torch.Tensor] = None,
) -> Iterator[Dict[int, LayerKDCapture]]:
    introspection_model = unwrap_for_introspection(teacher)
    decoder_layers = get_decoder_layers(introspection_model)
    layer_indices = tuple(range(len(decoder_layers))) if layers is None else tuple(layers)
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= len(decoder_layers):
            raise ValueError(f"Invalid layer {layer_idx}; teacher has {len(decoder_layers)} layers.")

    valid_mask = None
    if attention_mask is not None and not keep_batch_shape:
        valid_mask = attention_mask.reshape(-1).bool()

    captured_hidden: Dict[int, torch.Tensor] = {}
    captured_router: Dict[int, LayerRouterCapture] = {}
    captured_layers: Dict[int, LayerKDCapture] = {}
    handles: list[Any] = []

    def make_layer_hook(layer_idx: int):
        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                raise RuntimeError(
                    f"Unexpected decoder layer output for layer {layer_idx}: {type(hidden)}"
                )
            captured_hidden[layer_idx] = _as_cpu_tensor(
                hidden,
                dtype=cpu_dtype,
                keep_batch_shape=keep_batch_shape,
                valid_mask=valid_mask,
            )
            captured_layers[layer_idx] = LayerKDCapture(
                hidden_states=captured_hidden[layer_idx],
                router=captured_router.get(layer_idx),
            )

        return hook

    def make_moe_hook(layer_idx: int, moe_name: str, moe: nn.Module):
        router = get_router_module(moe)
        experts = get_experts_module(moe)
        top_k = get_top_k(moe, default=4)

        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError(f"Could not read MoE input for layer {layer_idx}.")
            hidden = inputs[0]
            if hidden.dim() != 3:
                raise RuntimeError(
                    f"Expected MoE hidden shape [B, T, D], got {tuple(hidden.shape)}"
                )
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            router_device = first_tensor_device(router)
            with torch.no_grad():
                logits = router_forward_logits(router, hidden_flat.to(router_device))
                topk_weights, topk_indices = compute_router_topk(logits, top_k)
                num_experts = get_num_experts(experts, logits)

            if keep_batch_shape:
                out_shape = hidden.shape[:2] + (logits.shape[-1],)
                logits_for_store = logits.reshape(out_shape)
                topk_shape = hidden.shape[:2] + (topk_indices.shape[-1],)
                weights_for_store = topk_weights.reshape(topk_shape)
                indices_for_store = topk_indices.reshape(topk_shape)
                mask_for_store = None
            else:
                logits_for_store = logits
                weights_for_store = topk_weights
                indices_for_store = topk_indices
                mask_for_store = valid_mask

            captured_router[layer_idx] = LayerRouterCapture(
                logits=_as_cpu_tensor(
                    logits_for_store,
                    dtype=torch.float32 if capture_router_logits else None,
                    keep_batch_shape=keep_batch_shape,
                    valid_mask=mask_for_store,
                )
                if capture_router_logits
                else torch.empty(0),
                topk_weights=_as_cpu_tensor(
                    weights_for_store,
                    dtype=torch.float32,
                    keep_batch_shape=keep_batch_shape,
                    valid_mask=mask_for_store,
                ),
                topk_indices=_as_cpu_tensor(
                    indices_for_store,
                    dtype=None,
                    keep_batch_shape=keep_batch_shape,
                    valid_mask=mask_for_store,
                ).long(),
                num_experts=num_experts,
                top_k=int(topk_indices.shape[-1]),
                moe_name=moe_name,
            )
            if layer_idx in captured_hidden:
                captured_layers[layer_idx] = LayerKDCapture(
                    hidden_states=captured_hidden[layer_idx],
                    router=captured_router[layer_idx],
                )

        return hook

    for layer_idx in layer_indices:
        layer = decoder_layers[layer_idx]
        handles.append(layer.register_forward_hook(make_layer_hook(layer_idx)))

        moe_candidates = find_moe_modules_in_layer(layer)
        if moe_candidates:
            moe_name, moe = moe_candidates[0]
            handles.append(moe.register_forward_hook(make_moe_hook(layer_idx, moe_name, moe)))

    try:
        yield captured_layers
    finally:
        for handle in handles:
            handle.remove()


def capture_gptoss_kd_batch(
    teacher: nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    layers: Optional[Iterable[int]] = None,
    cpu_dtype: Optional[torch.dtype] = torch.bfloat16,
    keep_batch_shape: bool = True,
    capture_router_logits: bool = True,
) -> GPTOSSKDCapture:
    input_device = get_first_device(teacher)
    model_batch = {key: value.to(input_device) for key, value in batch.items()}
    attention_mask = model_batch.get("attention_mask")

    with torch.no_grad():
        with capture_gptoss_kd_hooks(
            teacher,
            layers=layers,
            cpu_dtype=cpu_dtype,
            keep_batch_shape=keep_batch_shape,
            capture_router_logits=capture_router_logits,
            attention_mask=attention_mask,
        ) as captured:
            _ = teacher(**model_batch, use_cache=False)
            layers_out = dict(captured)

    if not layers_out:
        raise RuntimeError("No GPT-OSS layer outputs were captured.")

    return GPTOSSKDCapture(
        input_ids=model_batch["input_ids"].detach().cpu(),
        attention_mask=attention_mask.detach().cpu() if attention_mask is not None else None,
        layers=layers_out,
    )


def tokenize_prompts(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_length: int = 1024,
    padding: bool | str = True,
) -> Dict[str, torch.Tensor]:
    return tokenizer(
        list(prompts),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=padding,
    )


def capture_gptoss_kd_prompts(
    teacher: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    layers: Optional[Iterable[int]] = None,
    max_length: int = 1024,
    padding: bool | str = True,
    cpu_dtype: Optional[torch.dtype] = torch.bfloat16,
    keep_batch_shape: bool = True,
    capture_router_logits: bool = True,
) -> GPTOSSKDCapture:
    batch = tokenize_prompts(
        tokenizer,
        prompts,
        max_length=max_length,
        padding=padding,
    )
    return capture_gptoss_kd_batch(
        teacher,
        batch,
        layers=layers,
        cpu_dtype=cpu_dtype,
        keep_batch_shape=keep_batch_shape,
        capture_router_logits=capture_router_logits,
    )
