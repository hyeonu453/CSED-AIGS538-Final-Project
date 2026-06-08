from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from gptoss_kd_capture import get_dtype
from kd_losses import masked_tokens, mean_cosine, normalized_mse
from kd_mole import forward_custom_layer, set_trainable_parameters
from kd_teacher import LayerPairs
from modeling import CustomMoLELayer


def train_layer_kd(
    model: CustomMoLELayer,
    qwen_config: Any,
    train_pairs: LayerPairs,
    args: Any,
) -> dict[str, Any]:
    set_trainable_parameters(model, args.train_mode)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[student] trainable={trainable:,} total={total:,} mode={args.train_mode}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler_name = getattr(args, "lr_scheduler", "constant")
    min_lr = float(getattr(args, "min_lr", 0.0))
    warmup_steps = int(getattr(args, "warmup_steps", 0))

    def lr_scale(step_idx: int) -> float:
        if warmup_steps > 0 and step_idx < warmup_steps:
            return max(1e-8, float(step_idx + 1) / float(warmup_steps))
        decay_steps = max(1, int(args.steps) - max(0, warmup_steps))
        progress = min(1.0, max(0.0, float(step_idx - warmup_steps) / float(decay_steps)))
        min_scale = min_lr / float(args.lr) if args.lr > 0 else 0.0
        if scheduler_name == "constant":
            return 1.0
        if scheduler_name == "cosine":
            return min_scale + (1.0 - min_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))
        if scheduler_name == "linear":
            return min_scale + (1.0 - min_scale) * (1.0 - progress)
        raise ValueError(f"Unknown lr scheduler: {scheduler_name}")

    def set_lr(step_idx: int) -> float:
        lr = float(args.lr) * lr_scale(step_idx)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr
    rotary = Qwen2RotaryEmbedding(qwen_config, device=args.device).to(args.device)
    rng = torch.Generator().manual_seed(args.seed)
    losses: list[dict[str, float]] = []
    best_nmse = float("inf")
    best_step = 0
    best_state_dict: dict[str, torch.Tensor] | None = None

    x_train = train_pairs.layer_input
    y_train = train_pairs.layer_output
    mask_train = train_pairs.attention_mask
    train_dtype = get_dtype(args.train_dtype)

    model.train()
    for step in range(args.steps + 1):
        indices = torch.randint(0, x_train.shape[0], (args.train_batch_size,), generator=rng)
        x = x_train[indices].to(args.device, dtype=train_dtype)
        y = y_train[indices].to(args.device, dtype=train_dtype)
        mask = mask_train[indices].to(args.device)

        current_lr = set_lr(step)
        optimizer.zero_grad(set_to_none=True)
        pred = forward_custom_layer(model, rotary, x, attention_mask=None)
        pred_tokens = masked_tokens(pred, mask)
        target_tokens = masked_tokens(y, mask)
        loss = F.mse_loss(pred_tokens.float(), target_tokens.float())
        nmse = normalized_mse(pred_tokens, target_tokens)

        nmse_value = float(nmse.item())
        if nmse_value < best_nmse:
            best_nmse = nmse_value
            best_step = int(step)
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if step == 0 or step % args.log_every == 0 or step == args.steps:
            row = {"step": float(step), "mse": float(loss.item()), "nmse": nmse_value, "lr": float(current_lr)}
            losses.append(row)
            print(f"[train step {step:04d}] mse={row['mse']:.6e} nmse={row['nmse']:.6e} lr={row['lr']:.6e}")
        if step == args.steps:
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "losses": losses,
        "best_step": int(best_step),
        "best_train_nmse": float(best_nmse),
        "best_state_dict": best_state_dict,
    }


def evaluate_layer_local(
    model: CustomMoLELayer,
    qwen_config: Any,
    pairs: LayerPairs,
    args: Any,
) -> dict[str, float]:
    rotary = Qwen2RotaryEmbedding(qwen_config, device=args.device).to(args.device)
    train_dtype = get_dtype(args.train_dtype)
    model.eval()
    mse_sum = 0.0
    norm_sum = 0.0
    cos_sum = 0.0
    n_tokens = 0
    with torch.no_grad():
        for start in range(0, pairs.layer_input.shape[0], args.eval_batch_size):
            idx = slice(start, start + args.eval_batch_size)
            x = pairs.layer_input[idx].to(args.device, dtype=train_dtype)
            y = pairs.layer_output[idx].to(args.device, dtype=train_dtype)
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
