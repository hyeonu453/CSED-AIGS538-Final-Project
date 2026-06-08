from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from gptoss_kd_capture import get_decoder_layers, get_dtype, get_first_device, unwrap_for_introspection
from kd_losses import free_cuda
from kd_mole import GPTOSSLayerReplacement
from kd_teacher import LayerPairs, load_teacher
from modeling import CustomMoLELayer


def evaluate_full_replacement_logits(
    model_name_or_path: str,
    adapter_dir: str,
    custom_layer: CustomMoLELayer,
    qwen_config: Any,
    test_pairs: LayerPairs,
    args: Any,
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
            original_logits.append(teacher(**batch, use_cache=False).logits.detach().cpu())
        original_logits_cpu = torch.cat(original_logits, dim=0)

    original_layer = layers[args.layer]
    custom_layer = custom_layer.to(device=input_device, dtype=get_dtype(args.teacher_dtype))
    replacement = GPTOSSLayerReplacement(custom_layer, qwen_config).to(
        device=input_device,
        dtype=get_dtype(args.teacher_dtype),
    )
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
            base_logits = original_logits_cpu[offset:offset + replaced_logits.shape[0]]
            offset += replaced_logits.shape[0]
            mask = test_pairs.attention_mask[idx].bool()
            base_tokens = base_logits[mask]
            replaced_tokens = replaced_logits[mask]

            diff = (replaced_tokens.float() - base_tokens.float()).pow(2)
            logits_mse_sum += float(diff.sum().item())
            logits_norm_sum += float(base_tokens.float().pow(2).sum().item())
            base_log_probs = F.log_softmax(base_tokens.float(), dim=-1)
            replaced_log_probs = F.log_softmax(replaced_tokens.float(), dim=-1)
            kl_sum += float(F.kl_div(replaced_log_probs, base_log_probs.exp(), reduction="sum").item())
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
