from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptoss_kd_capture import get_decoder_layers, get_first_device, unwrap_for_introspection

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

    resolved_device_map: Any = device_map
    if device_map in (None, "", "none", "None"):
        resolved_device_map = None
    elif device_map == "auto_if_cuda":
        resolved_device_map = "auto" if device.startswith("cuda") else None
    elif isinstance(device_map, str) and (device_map.startswith("cuda") or device_map == "cpu"):
        resolved_device_map = {"": device_map}

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=resolved_device_map,
        trust_remote_code=trust_remote_code,
    )
    if adapter_dir:
        if PeftModel is None:
            raise ImportError("peft is required when adapter_dir is provided.")
        model = PeftModel.from_pretrained(model, adapter_dir)
    if resolved_device_map is None:
        model.to(device)
    model.eval()
    return model, tokenizer


def capture_layer_pairs_group(
    teacher: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layer_indices: list[int],
    max_length: int,
    batch_size: int,
    cpu_dtype: torch.dtype,
) -> dict[int, LayerPairs]:
    if not layer_indices:
        return {}

    input_device = get_first_device(teacher)
    inspected = unwrap_for_introspection(teacher)
    layers = get_decoder_layers(inspected)
    unique_layers = sorted(set(int(layer_idx) for layer_idx in layer_indices))
    for layer_idx in unique_layers:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Invalid layer {layer_idx}; teacher has {len(layers)} layers.")

    forward_target = inspected
    if hasattr(inspected, "lm_head") and hasattr(inspected, "model"):
        forward_target = inspected.model
    if not hasattr(forward_target, "forward"):
        forward_target = teacher

    all_input_ids: list[torch.Tensor] = []
    all_attention_masks: list[torch.Tensor] = []
    all_x: dict[int, list[torch.Tensor]] = {layer_idx: [] for layer_idx in unique_layers}
    all_y: dict[int, list[torch.Tensor]] = {layer_idx: [] for layer_idx in unique_layers}

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        batch = tokenizer(
            chunk,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        model_batch = {key: value.to(input_device) for key, value in batch.items()}
        saved: dict[int, dict[str, torch.Tensor]] = {layer_idx: {} for layer_idx in unique_layers}
        handles: list[Any] = []

        def make_hook(layer_idx: int):
            def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                layer_output = output[0] if isinstance(output, tuple) else output
                if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(layer_output, torch.Tensor):
                    raise RuntimeError(f"Failed to capture teacher layer {layer_idx} input/output.")
                saved[layer_idx]["x"] = inputs[0].detach().to("cpu", dtype=cpu_dtype).contiguous()
                saved[layer_idx]["y"] = layer_output.detach().to("cpu", dtype=cpu_dtype).contiguous()
            return hook

        for layer_idx in unique_layers:
            handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
        try:
            with torch.no_grad():
                _ = forward_target(**model_batch, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        for layer_idx in unique_layers:
            if "x" not in saved[layer_idx] or "y" not in saved[layer_idx]:
                raise RuntimeError(f"No layer pair captured for layer {layer_idx}, batch starting at {start}.")
            all_x[layer_idx].append(saved[layer_idx]["x"])
            all_y[layer_idx].append(saved[layer_idx]["y"])
        all_input_ids.append(model_batch["input_ids"].detach().cpu())
        all_attention_masks.append(model_batch["attention_mask"].detach().cpu())
        print(f"[capture group {unique_layers}] {min(start + batch_size, len(prompts))}/{len(prompts)}")

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_masks, dim=0)
    return {
        layer_idx: LayerPairs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_input=torch.cat(all_x[layer_idx], dim=0),
            layer_output=torch.cat(all_y[layer_idx], dim=0),
        )
        for layer_idx in unique_layers
    }


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

    forward_target = inspected
    if hasattr(inspected, "lm_head") and hasattr(inspected, "model"):
        forward_target = inspected.model
    if not hasattr(forward_target, "forward"):
        forward_target = teacher

    all_input_ids: list[torch.Tensor] = []
    all_attention_masks: list[torch.Tensor] = []
    all_x: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
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
            layer_output = output[0] if isinstance(output, tuple) else output
            if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(layer_output, torch.Tensor):
                raise RuntimeError("Failed to capture teacher layer input/output.")
            saved["x"] = inputs[0].detach().to("cpu", dtype=cpu_dtype).contiguous()
            saved["y"] = layer_output.detach().to("cpu", dtype=cpu_dtype).contiguous()

        handle = layers[layer_idx].register_forward_hook(hook)
        try:
            with torch.no_grad():
                _ = forward_target(**model_batch, use_cache=False)
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
