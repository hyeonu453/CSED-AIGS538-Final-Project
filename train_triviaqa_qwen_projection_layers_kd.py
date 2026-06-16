#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from gptoss_kd_capture import get_decoder_layers, get_dtype, unwrap_for_introspection
from kd_losses import free_cuda, masked_tokens, mean_cosine, normalized_mse
from kd_qwen_projection import ProjectionOnlyQwenLayer, set_projection_trainable
from kd_teacher import LayerPairs, capture_layer_pairs_group, load_teacher
from train_triviaqa_all_layers_kd import (
    delete_layer_cache_files,
    ensure_layer_group_cached,
    generate_or_load_teacher_sequences,
    load_triviaqa_chat_prompts,
    parse_layers,
)


def layer_cache_path(cache_dir: Path, split: str, layer: int) -> Path:
    return cache_dir / f"layer_{layer:02d}_{split}.pt"


def load_pairs(path: Path) -> LayerPairs:
    print(f"[cache] loading {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def move_batch(pairs: LayerPairs, indices: Any, device: str, dtype: torch.dtype):
    x = pairs.layer_input[indices].to(device, dtype=dtype)
    y = pairs.layer_output[indices].to(device, dtype=dtype)
    mask = pairs.attention_mask[indices].to(device)
    return x, y, mask


def evaluate_projection_layer(
    model: ProjectionOnlyQwenLayer,
    pairs: LayerPairs,
    args: argparse.Namespace,
) -> dict[str, float]:
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


def train_projection_layer(
    model: ProjectionOnlyQwenLayer,
    train_pairs: LayerPairs,
    test_pairs: LayerPairs,
    args: argparse.Namespace,
) -> dict[str, Any]:
    set_projection_trainable(model, train_qwen_layer=args.train_qwen_layer)
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    total = sum(p.numel() for p in model.parameters())
    print(f"[student] trainable={trainable:,} total={total:,} train_qwen_layer={args.train_qwen_layer}")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    rng = torch.Generator().manual_seed(args.seed)
    dtype = get_dtype(args.train_dtype)
    losses: list[dict[str, float]] = []
    best_nmse = float("inf")
    best_step = 0
    best_state_dict: dict[str, torch.Tensor] | None = None

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
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if step == 0 or step % args.log_every == 0 or step == args.steps:
            row = {"step": float(step), "mse": float(loss.item()), "nmse": nmse_value}
            losses.append(row)
            print(f"[train step {step:04d}] mse={row['mse']:.6e} nmse={row['nmse']:.6e}")

        if step == args.steps:
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    layer_eval = evaluate_projection_layer(model, test_pairs, args)
    return {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "losses": losses,
        "best_step": int(best_step),
        "best_train_nmse": float(best_nmse),
        "layer_local_eval": layer_eval,
        "best_state_dict": best_state_dict,
    }


def save_layer_summary(
    output_dir: Path,
    layer: int,
    train_summary: dict[str, Any],
    layer_eval: dict[str, float],
    train_pairs: LayerPairs,
    test_pairs: LayerPairs,
) -> None:
    layer_dir = output_dir / f"layer_{layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "layer": layer,
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
        "layer_local_eval": layer_eval,
    }
    (layer_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def train_one_layer(
    layer: int,
    teacher_config: Any,
    train_pairs: LayerPairs,
    test_pairs: LayerPairs,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(f"\n=== Training Qwen projection-only replacement layer {layer} ===")
    qwen_layer = args.qwen_layer if args.qwen_layer >= 0 else layer
    model = ProjectionOnlyQwenLayer(
        teacher_config=teacher_config,
        qwen_model_name_or_path=args.qwen_model,
        qwen_layer_idx=qwen_layer,
        train_dtype=get_dtype(args.train_dtype),
        trust_remote_code=args.trust_remote_code,
        init_from_qwen=True,
    ).to(device=args.device, dtype=get_dtype(args.train_dtype))

    train_summary = train_projection_layer(model, train_pairs, test_pairs, args)
    best_state_dict = train_summary.pop("best_state_dict", None)
    layer_eval = train_summary["layer_local_eval"]
    print(f"[layer {layer} local eval]", layer_eval)

    layer_dir = Path(args.output_dir) / f"layer_{layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    state_path = layer_dir / "projection_qwen_layer.pt"
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, state_path)
    if best_state_dict is not None:
        torch.save(best_state_dict, layer_dir / "projection_qwen_layer_best.pt")
    save_layer_summary(Path(args.output_dir), layer, train_summary, layer_eval, train_pairs, test_pairs)
    del model
    free_cuda()
    return {
        "layer": layer,
        "train": train_summary,
        "layer_local_eval": layer_eval,
        "state_path": str(state_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--teacher_adapter_dir", type=str, default="")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--triviaqa_config", type=str, default="rc.nocontext")
    parser.add_argument("--sequence_source", choices=["chat_prompt", "teacher_generated"], default="chat_prompt")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--generation_cache_dir", type=str, default="cache/generated_sequences")
    parser.add_argument("--force_regenerate", action="store_true")
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--qwen_layer", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=32, help="Accepted for CLI compatibility; unused.")
    parser.add_argument("--mole_alpha", type=float, default=-1.0, help="Accepted for CLI compatibility; unused.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--capture_batch_size", type=int, default=8)
    parser.add_argument("--capture_layer_group_size", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--full_eval_batch_size", type=int, default=2)
    parser.add_argument("--teacher_dtype", type=str, default="bfloat16")
    parser.add_argument("--cache_dtype", type=str, default="bfloat16")
    parser.add_argument("--train_dtype", type=str, default="float32")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher_device_map", type=str, default="auto_if_cuda")
    parser.add_argument("--train_mode", choices=["projection_only"], default="projection_only")
    parser.add_argument("--train_qwen_layer", action="store_true")
    parser.add_argument("--skip_full_replacement_eval", action="store_true", default=True)
    parser.add_argument("--skip_trained_layers", action="store_true")
    parser.add_argument("--delete_layer_cache_after_train", action="store_true")
    parser.add_argument("--force_recache", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default="cache/triviaqa_qwen_projection_layers")
    parser.add_argument("--output_dir", type=str, default="output/triviaqa_qwen_projection_layers")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print("=== Loading teacher for cache capture ===")
    teacher, tokenizer = load_teacher(
        args.teacher_model,
        args.teacher_adapter_dir,
        get_dtype(args.teacher_dtype),
        args.device,
        args.teacher_device_map,
        args.trust_remote_code,
    )

    print("=== Loading TriviaQA chat prompts ===")
    train_prompts, test_prompts = load_triviaqa_chat_prompts(
        tokenizer,
        teacher,
        args.triviaqa_config,
        args.train_size,
        args.test_size,
        args,
    )
    (output_dir / "train_prompts_preview.txt").write_text("\n".join(train_prompts[:20]), encoding="utf-8")
    (output_dir / "test_prompts_preview.txt").write_text("\n".join(test_prompts[:20]), encoding="utf-8")

    inspected = unwrap_for_introspection(teacher)
    teacher_config = SimpleNamespace(hidden_size=int(inspected.config.hidden_size))
    layers_to_train = parse_layers(args.layers, len(get_decoder_layers(inspected)))
    print("layers:", layers_to_train)

    layer_results: list[dict[str, Any]] = []
    group_size = max(1, int(args.capture_layer_group_size))
    for group_start in range(0, len(layers_to_train), group_size):
        group = layers_to_train[group_start:group_start + group_size]
        active_group: list[int] = []
        for layer in group:
            state_path = output_dir / f"layer_{layer:02d}" / "projection_qwen_layer.pt"
            if args.skip_trained_layers and state_path.exists():
                print(f"[train] skipping layer {layer}; found {state_path}")
                layer_results.append({"layer": layer, "skipped": True, "state_path": str(state_path)})
            else:
                active_group.append(layer)
        if not active_group:
            continue

        ensure_layer_group_cached(cache_dir, "train", active_group, teacher, tokenizer, train_prompts, args)
        ensure_layer_group_cached(cache_dir, "test", active_group, teacher, tokenizer, test_prompts, args)

        for layer in active_group:
            train_pairs = load_pairs(layer_cache_path(cache_dir, "train", layer))
            test_pairs = load_pairs(layer_cache_path(cache_dir, "test", layer))
            result = train_one_layer(layer, teacher_config, train_pairs, test_pairs, args)
            layer_results.append(result)
            del train_pairs, test_pairs
            if args.delete_layer_cache_after_train:
                delete_layer_cache_files(cache_dir, layer)

    del teacher
    free_cuda()
    summary = {
        "experiment": "triviaqa_qwen_projection_only_layer_kd",
        "args": vars(args),
        "layers": layers_to_train,
        "layer_results": layer_results,
        "full_replacement_eval": None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

