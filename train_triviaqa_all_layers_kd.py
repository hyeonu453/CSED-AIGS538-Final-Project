#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layerwise KD over many/all GPT-OSS decoder layers, then teacher-forced full
replacement evaluation.

Input sequences can be either raw TriviaQA questions wrapped with the teacher
tokenizer chat template and an empty assistant generation prompt, or cached
teacher-generated continuations from those same prompts.

Artifacts are separated into:
  cache/: layer input/output caches
  output/: trained per-layer replacement states and summaries
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset

from gptoss_kd_capture import get_decoder_layers, get_dtype, get_first_device, unwrap_for_introspection
from kd_layer_train import evaluate_layer_local, train_layer_kd
from kd_losses import free_cuda
from kd_mole import GPTOSSLayerReplacement, build_custom_layer
from kd_teacher import LayerPairs, capture_layer_pairs, capture_layer_pairs_group, load_teacher


def apply_question_chat_template(tokenizer: Any, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return question


def load_jsonl_texts(path: Path, expected_n: int) -> list[str] | None:
    if not path.exists():
        return None
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[generation cache] invalid json in {path} at line {line_no}: {exc}")
                return None
    if len(rows) < expected_n:
        print(f"[generation cache] incomplete {path}: {len(rows)}/{expected_n} rows")
        return None
    return [row["text"] for row in rows[:expected_n]]


def generate_or_load_teacher_sequences(
    tokenizer: Any,
    teacher: torch.nn.Module,
    rows: Any,
    split: str,
    args: argparse.Namespace,
) -> list[str]:
    cache_dir = Path(args.generation_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_n{len(rows)}_maxlen{args.max_length}_new{args.max_new_tokens}_genv2.jsonl"
    if not args.force_regenerate:
        cached = load_jsonl_texts(cache_path, len(rows))
        if cached is not None:
            print(f"[generation cache] loading {cache_path}")
            return cached

    print(f"[generation cache] generating {split} sequences -> {cache_path}")
    input_device = get_first_device(teacher)
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    records: list[dict[str, Any]] = []
    pad_token_id = tokenizer.pad_token_id
    try:
        for start_idx in range(0, len(rows), args.generation_batch_size):
            end_idx = min(start_idx + args.generation_batch_size, len(rows))
            chunk = [rows[i] for i in range(start_idx, end_idx)]
            prompts = [apply_question_chat_template(tokenizer, row["question"]) for row in chunk]
            prompt_max_length = max(1, args.max_length - args.max_new_tokens)
            batch = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=prompt_max_length,
            )
            batch = {key: value.to(input_device) for key, value in batch.items()}
            prompt_width = int(batch["input_ids"].shape[1])
            with torch.no_grad():
                output_ids = teacher.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            output_ids_cpu = output_ids.detach().cpu()
            input_ids_cpu = batch["input_ids"].detach().cpu()
            attention_mask_cpu = batch["attention_mask"].detach().cpu().bool()
            for offset, (row, prompt) in enumerate(zip(chunk, prompts)):
                prompt_ids = input_ids_cpu[offset][attention_mask_cpu[offset]]
                generated_ids = output_ids_cpu[offset, prompt_width:]
                if pad_token_id is not None:
                    end = int(generated_ids.numel())
                    while end > 0 and int(generated_ids[end - 1].item()) == int(pad_token_id):
                        end -= 1
                    generated_ids = generated_ids[:end]
                clean_ids = torch.cat([prompt_ids, generated_ids], dim=0)
                text = tokenizer.decode(clean_ids.tolist(), skip_special_tokens=False)
                records.append({
                    "idx": start_idx + offset,
                    "question_id": row.get("question_id", ""),
                    "question": row["question"],
                    "prompt": prompt,
                    "prompt_tokens": int(prompt_ids.numel()),
                    "generated_tokens": int(generated_ids.numel()),
                    "text": text,
                })
            print(f"[generation] {min(start_idx + args.generation_batch_size, len(rows))}/{len(rows)}")
    finally:
        tokenizer.padding_side = old_padding_side

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(cache_path)
    return [record["text"] for record in records]


def load_triviaqa_chat_prompts(
    tokenizer: Any,
    teacher: torch.nn.Module,
    config_name: str,
    train_n: int,
    test_n: int,
    args: argparse.Namespace,
) -> tuple[list[str], list[str]]:
    train = load_dataset("trivia_qa", config_name, split=f"train[:{train_n}]")
    validation = load_dataset("trivia_qa", config_name, split=f"validation[:{test_n}]")

    if args.sequence_source == "chat_prompt":
        return (
            [apply_question_chat_template(tokenizer, row["question"]) for row in train],
            [apply_question_chat_template(tokenizer, row["question"]) for row in validation],
        )
    if args.sequence_source == "teacher_generated":
        return (
            generate_or_load_teacher_sequences(tokenizer, teacher, train, "train", args),
            generate_or_load_teacher_sequences(tokenizer, teacher, validation, "test", args),
        )
    raise ValueError(f"Unknown sequence source: {args.sequence_source}")

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
    deduped = sorted(set(layers))
    for layer in deduped:
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"Invalid layer {layer}; teacher has {num_layers} layers.")
    return deduped


def layer_cache_path(cache_dir: Path, split: str, layer: int) -> Path:
    return cache_dir / f"layer_{layer:02d}_{split}.pt"


def delete_layer_cache_files(cache_dir: Path, layer: int) -> None:
    for split in ("train", "test"):
        path = layer_cache_path(cache_dir, split, layer)
        if path.exists():
            path.unlink()
            print(f"[cache] deleted {path}")


def load_or_capture_pairs(
    cache_dir: Path,
    split: str,
    layer: int,
    teacher: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    args: argparse.Namespace,
) -> LayerPairs:
    path = layer_cache_path(cache_dir, split, layer)
    if path.exists() and not args.force_recache:
        print(f"[cache] loading {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    print(f"[cache] capturing {split} layer {layer}")
    pairs = capture_layer_pairs(
        teacher=teacher,
        tokenizer=tokenizer,
        prompts=prompts,
        layer_idx=layer,
        max_length=args.max_length,
        batch_size=args.capture_batch_size,
        cpu_dtype=get_dtype(args.cache_dtype),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pairs, path)
    print(f"[cache] saved {path}")
    return pairs


def ensure_layer_group_cached(
    cache_dir: Path,
    split: str,
    layers: list[int],
    teacher: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    args: argparse.Namespace,
) -> None:
    missing = [
        layer
        for layer in layers
        if args.force_recache or not layer_cache_path(cache_dir, split, layer).exists()
    ]
    if not missing:
        return

    if len(missing) == 1 and args.capture_layer_group_size <= 1:
        load_or_capture_pairs(cache_dir, split, missing[0], teacher, tokenizer, prompts, args)
        return

    print(f"[cache] capturing {split} layer group {missing}")
    pairs_by_layer = capture_layer_pairs_group(
        teacher=teacher,
        tokenizer=tokenizer,
        prompts=prompts,
        layer_indices=missing,
        max_length=args.max_length,
        batch_size=args.capture_batch_size,
        cpu_dtype=get_dtype(args.cache_dtype),
    )
    for layer, pairs in pairs_by_layer.items():
        path = layer_cache_path(cache_dir, split, layer)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pairs, path)
        print(f"[cache] saved {path}")
    del pairs_by_layer


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
    print(f"\n=== Training replacement layer {layer} ===")
    custom_layer, qwen_config = build_custom_layer(
        teacher_config=teacher_config,
        qwen_model_name_or_path=args.qwen_model,
        layer_idx=args.qwen_layer if args.qwen_layer >= 0 else layer,
        rank=args.rank,
        mole_alpha=args.mole_alpha,
        train_dtype=get_dtype(args.train_dtype),
        device=args.device,
        init_from_qwen=not args.no_init_from_qwen,
    )
    train_summary = train_layer_kd(custom_layer, qwen_config, train_pairs, args)
    best_state_dict = train_summary.pop("best_state_dict", None)
    layer_eval = evaluate_layer_local(custom_layer, qwen_config, test_pairs, args)
    print(f"[layer {layer} local eval]", layer_eval)

    layer_dir = Path(args.output_dir) / f"layer_{layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    state_path = layer_dir / "custom_mole_layer.pt"
    torch.save({key: value.detach().cpu() for key, value in custom_layer.state_dict().items()}, state_path)
    if best_state_dict is not None:
        best_state_path = layer_dir / "custom_mole_layer_best.pt"
        torch.save(best_state_dict, best_state_path)
    save_layer_summary(Path(args.output_dir), layer, train_summary, layer_eval, train_pairs, test_pairs)

    del custom_layer
    free_cuda()
    return {
        "layer": layer,
        "train": train_summary,
        "layer_local_eval": layer_eval,
        "state_path": str(state_path),
    }


def evaluate_full_replacement_many(
    layers_to_replace: list[int],
    teacher_config: Any,
    test_pairs: LayerPairs,
    args: argparse.Namespace,
) -> dict[str, float]:
    print("\n=== Loading teacher for all-layer replacement eval ===")
    teacher, _ = load_teacher(
        args.teacher_model,
        args.teacher_adapter_dir,
        get_dtype(args.teacher_dtype),
        args.device,
        args.teacher_device_map,
        args.trust_remote_code,
    )
    input_device = get_first_device(teacher)
    inspected = unwrap_for_introspection(teacher)
    teacher_layers = get_decoder_layers(inspected)

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

    originals: dict[int, torch.nn.Module] = {}
    for layer in layers_to_replace:
        state_path = Path(args.output_dir) / f"layer_{layer:02d}" / "custom_mole_layer.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing trained state for layer {layer}: {state_path}")
        custom_layer, qwen_config = build_custom_layer(
            teacher_config=teacher_config,
            qwen_model_name_or_path=args.qwen_model,
            layer_idx=args.qwen_layer if args.qwen_layer >= 0 else layer,
            rank=args.rank,
            mole_alpha=args.mole_alpha,
            train_dtype=get_dtype(args.teacher_dtype),
            device=str(input_device),
            init_from_qwen=False,
        )
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        custom_layer.load_state_dict(state, strict=True)
        custom_layer.to(device=input_device, dtype=get_dtype(args.teacher_dtype))
        replacement = GPTOSSLayerReplacement(custom_layer, qwen_config).to(
            device=input_device,
            dtype=get_dtype(args.teacher_dtype),
        )
        replacement.eval()
        originals[layer] = teacher_layers[layer]
        teacher_layers[layer] = replacement

    logits_mse_sum = 0.0
    logits_norm_sum = 0.0
    kl_sum = 0.0
    top1_match_sum = 0
    n_tokens = 0
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

    for layer, original in originals.items():
        teacher_layers[layer] = original
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
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    teacher_config = inspected.config
    layers_to_train = parse_layers(args.layers, len(get_decoder_layers(inspected)))
    print("layers:", layers_to_train)

    layer_results: list[dict[str, Any]] = []
    last_test_pairs: LayerPairs | None = None
    group_size = max(1, int(args.capture_layer_group_size))
    for group_start in range(0, len(layers_to_train), group_size):
        group = layers_to_train[group_start:group_start + group_size]
        active_group: list[int] = []
        for layer in group:
            state_path = output_dir / f"layer_{layer:02d}" / "custom_mole_layer.pt"
            if args.skip_trained_layers and state_path.exists():
                print(f"[train] skipping layer {layer}; found {state_path}")
                layer_results.append({
                    "layer": layer,
                    "skipped": True,
                    "state_path": str(state_path),
                })
            else:
                active_group.append(layer)
        if not active_group:
            continue

        ensure_layer_group_cached(cache_dir, "train", active_group, teacher, tokenizer, train_prompts, args)
        ensure_layer_group_cached(cache_dir, "test", active_group, teacher, tokenizer, test_prompts, args)

        for layer in active_group:
            train_path = layer_cache_path(cache_dir, "train", layer)
            test_path = layer_cache_path(cache_dir, "test", layer)
            print(f"[cache] loading {train_path}")
            train_pairs = torch.load(train_path, map_location="cpu", weights_only=False)
            print(f"[cache] loading {test_path}")
            test_pairs = torch.load(test_path, map_location="cpu", weights_only=False)
            last_test_pairs = test_pairs
            result = train_one_layer(layer, teacher_config, train_pairs, test_pairs, args)
            layer_results.append(result)
            del train_pairs
            if args.delete_layer_cache_after_train:
                delete_layer_cache_files(cache_dir, layer)

    if last_test_pairs is None and not args.skip_full_replacement_eval:
        eval_cache_layer = layers_to_train[-1]
        print(f"[cache] preparing test cache for full replacement eval from layer {eval_cache_layer}")
        last_test_pairs = load_or_capture_pairs(cache_dir, "test", eval_cache_layer, teacher, tokenizer, test_prompts, args)

    del teacher
    free_cuda()

    full_eval = None
    if not args.skip_full_replacement_eval:
        if last_test_pairs is None:
            raise RuntimeError("No test cache was built.")
        full_eval = evaluate_full_replacement_many(layers_to_train, teacher_config, last_test_pairs, args)
        print("[full replacement eval]", full_eval)

    summary = {
        "experiment": "layerwise_kd_many_layers_then_full_replacement_eval",
        "args": vars(args),
        "layers": layers_to_train,
        "layer_results": layer_results,
        "full_replacement_eval": full_eval,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to {output_dir / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--teacher_adapter_dir", type=str, default="")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen2.5-3B")
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
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--mole_alpha", type=float, default=-1.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", choices=["constant", "cosine", "linear"], default="constant")
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
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
    parser.add_argument("--train_mode", choices=["adapter", "projection_only", "all"], default="adapter")
    parser.add_argument("--no_init_from_qwen", action="store_true")
    parser.add_argument("--skip_full_replacement_eval", action="store_true")
    parser.add_argument("--skip_trained_layers", action="store_true")
    parser.add_argument("--delete_layer_cache_after_train", action="store_true")
    parser.add_argument("--force_recache", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default="cache/triviaqa_all_layers")
    parser.add_argument("--output_dir", type=str, default="output/triviaqa_all_layers")
    return parser


if __name__ == "__main__":
    main()
