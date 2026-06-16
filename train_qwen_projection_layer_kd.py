#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from gptoss_kd_capture import get_dtype
from kd_losses import masked_tokens, mean_cosine, normalized_mse
from kd_qwen_projection import ProjectionOnlyQwenLayer, set_projection_trainable
from kd_teacher import LayerPairs


def load_pairs(path: str) -> LayerPairs:
    print(f"[cache] loading {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def move_batch(pairs: LayerPairs, indices: Any, device: str, dtype: torch.dtype):
    x = pairs.layer_input[indices].to(device, dtype=dtype)
    y = pairs.layer_output[indices].to(device, dtype=dtype)
    mask = pairs.attention_mask[indices].to(device)
    return x, y, mask


def evaluate(model: ProjectionOnlyQwenLayer, pairs: LayerPairs, args: argparse.Namespace) -> dict[str, float]:
    dtype = get_dtype(args.train_dtype)
    model.eval()
    mse_sum = 0.0
    norm_sum = 0.0
    cos_sum = 0.0
    n_tokens = 0
    with torch.no_grad():
        for start in range(0, pairs.layer_input.shape[0], args.eval_batch_size):
            idx = slice(start, start + args.eval_batch_size)
            x, y, mask = move_batch(pairs, idx, args.device, dtype)
            pred = model(x, attention_mask=None)
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


def train(model: ProjectionOnlyQwenLayer, train_pairs: LayerPairs, test_pairs: LayerPairs, args: argparse.Namespace):
    set_projection_trainable(model, train_qwen_layer=args.train_qwen_layer)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[student] trainable={trainable:,} total={total:,} train_qwen_layer={args.train_qwen_layer}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    rng = torch.Generator().manual_seed(args.seed)
    dtype = get_dtype(args.train_dtype)
    history: list[dict[str, float]] = []
    best_nmse = float("inf")
    best_state = None
    best_step = 0

    model.train()
    for step in range(args.steps + 1):
        indices = torch.randint(0, train_pairs.layer_input.shape[0], (args.train_batch_size,), generator=rng)
        x, y, mask = move_batch(train_pairs, indices, args.device, dtype)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, attention_mask=None)
        pred_tokens = masked_tokens(pred, mask)
        target_tokens = masked_tokens(y, mask)
        loss = F.mse_loss(pred_tokens.float(), target_tokens.float())
        nmse = normalized_mse(pred_tokens, target_tokens)
        nmse_value = float(nmse.item())

        if nmse_value < best_nmse:
            best_nmse = nmse_value
            best_step = int(step)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if step == 0 or step % args.log_every == 0 or step == args.steps:
            eval_row = evaluate(model, test_pairs, args) if args.eval_during_train else {}
            row = {
                "step": float(step),
                "train_mse": float(loss.item()),
                "train_nmse": nmse_value,
                **{f"eval_{key}": float(value) for key, value in eval_row.items()},
            }
            print(json.dumps(row, sort_keys=True))
            history.append(row)

        if step == args.steps:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    final_eval = evaluate(model, test_pairs, args)
    return {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "best_step": int(best_step),
        "best_train_nmse": float(best_nmse),
        "final_eval": final_eval,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-hidden-size", type=int, default=2880)
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--qwen-layer", type=int, default=0)
    parser.add_argument("--train-dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--train-qwen-layer", action="store_true")
    parser.add_argument("--eval-during-train", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_pairs = load_pairs(args.train_cache)
    test_pairs = load_pairs(args.test_cache)
    teacher_config = SimpleNamespace(hidden_size=args.teacher_hidden_size)
    qwen_config = AutoConfig.from_pretrained(args.qwen_model, trust_remote_code=args.trust_remote_code)
    print(
        f"[qwen] model={args.qwen_model} hidden={qwen_config.hidden_size} "
        f"layers={getattr(qwen_config, 'num_hidden_layers', '?')}"
    )

    model = ProjectionOnlyQwenLayer(
        teacher_config=teacher_config,
        qwen_model_name_or_path=args.qwen_model,
        qwen_layer_idx=args.qwen_layer,
        train_dtype=get_dtype(args.train_dtype),
        trust_remote_code=args.trust_remote_code,
        init_from_qwen=True,
    ).to(args.device, dtype=get_dtype(args.train_dtype))

    summary = train(model, train_pairs, test_pairs, args)
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, output_dir / "projection_qwen_layer.pt")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[save] wrote {output_dir}")


if __name__ == "__main__":
    main()

