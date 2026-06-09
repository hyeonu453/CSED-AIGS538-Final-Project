#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from merge_replacement_layers import copy_replacement_layers, parse_layers


def sanitize_stage_name(stage: str) -> str:
    return stage.replace(",", "_").replace("-", "to").replace(" ", "")


def str_arg(value: object) -> str:
    return str(value)


def add_arg(cmd: list[str], name: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, str) and value == "":
        return
    cmd.extend([name, str_arg(value)])


def build_train_command(args: argparse.Namespace, stage: str, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(args.train_script),
    ]
    add_arg(cmd, "--teacher_model", args.teacher_model)
    add_arg(cmd, "--student_model", args.student_model)
    add_arg(cmd, "--teacher_adapter_dir", args.teacher_adapter_dir)
    add_arg(cmd, "--teacher_device", args.teacher_device)
    add_arg(cmd, "--student_device", args.student_device)
    add_arg(cmd, "--teacher_dtype", args.teacher_dtype)
    add_arg(cmd, "--model_dtype", args.model_dtype)
    add_arg(cmd, "--qwen_model", args.qwen_model)
    add_arg(cmd, "--layer_checkpoint_dir", args.current_set)
    add_arg(cmd, "--output_dir", output_dir)
    add_arg(cmd, "--layers", stage)
    add_arg(cmd, "--rank", args.rank)
    add_arg(cmd, "--dataset", args.dataset)
    add_arg(cmd, "--train_generated_jsonl", args.train_generated_jsonl)
    add_arg(cmd, "--eval_generated_jsonl", args.eval_generated_jsonl)
    add_arg(cmd, "--max_length", args.max_length)
    add_arg(cmd, "--trainable", args.trainable)
    add_arg(cmd, "--optim", args.optim)
    add_arg(cmd, "--lr", args.lr)
    add_arg(cmd, "--lr_scheduler", args.lr_scheduler)
    add_arg(cmd, "--min_lr", args.min_lr)
    add_arg(cmd, "--warmup_steps", args.warmup_steps)
    add_arg(cmd, "--weight_decay", args.weight_decay)
    add_arg(cmd, "--grad_clip", args.grad_clip)
    add_arg(cmd, "--save_every", args.save_every)
    add_arg(cmd, "--ce_weight", args.ce_weight)
    add_arg(cmd, "--kl_weight", args.kl_weight)
    add_arg(cmd, "--kl_direction", args.kl_direction)
    add_arg(cmd, "--kl_temperature", args.kl_temperature)
    add_arg(cmd, "--prompt_loss_weight", args.prompt_loss_weight)
    add_arg(cmd, "--answer_loss_weight", args.answer_loss_weight)
    add_arg(cmd, "--steps", args.steps)
    add_arg(cmd, "--batch_size", args.batch_size)
    add_arg(cmd, "--grad_accum_steps", args.grad_accum_steps)
    add_arg(cmd, "--eval_batches", args.eval_batches)
    add_arg(cmd, "--log_every", args.log_every)
    add_arg(cmd, "--eval_every", args.eval_every)
    add_arg(cmd, "--kl_token_chunk_size", args.kl_token_chunk_size)
    add_arg(cmd, "--seed", args.seed)
    if args.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if args.no_trust_remote_code:
        cmd.append("--no_trust_remote_code")
    if args.extra_train_args:
        cmd.extend(args.extra_train_args)
    return cmd


def run_and_log(cmd: list[str], log_path: Path, env: dict[str, str], dry_run: bool) -> int:
    print("[run]", " ".join(cmd))
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[command] " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def ensure_current_set(args: argparse.Namespace) -> None:
    current_set = Path(args.current_set)
    if current_set.exists():
        return
    if not args.initial_checkpoint:
        raise FileNotFoundError(f"Current set does not exist and --initial_checkpoint was not given: {current_set}")
    if args.dry_run:
        print(f"[dry-run] would initialize {current_set} from {args.initial_checkpoint}")
        return
    current_set.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.initial_checkpoint, current_set)
    print(f"[init] copied {args.initial_checkpoint} -> {current_set}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run staged joint KD and merge each stage's best_current into a current-set.")
    parser.add_argument("--stages", nargs="+", required=True, help="Layer specs to train and merge, e.g. 0-5 6-11 12-17.")
    parser.add_argument("--current_set", required=True, help="Mutable checkpoint root used as layer_checkpoint_dir and merge target.")
    parser.add_argument("--initial_checkpoint", default="", help="Optional checkpoint root copied to current_set if current_set is missing.")
    parser.add_argument("--output_root", default="./output/progressive-joint-kd")
    parser.add_argument("--train_script", default="train_full_replacement_joint_kd.py")
    parser.add_argument("--overwrite_stage", action="store_true", help="Delete an existing stage output before re-running it.")
    parser.add_argument("--no_merge", action="store_true", help="Run training stages without copying best_current into current_set.")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--student_model", default="openai/gpt-oss-20b")
    parser.add_argument("--teacher_model", default="openai/gpt-oss-20b")
    parser.add_argument("--teacher_adapter_dir", default="./lora")
    parser.add_argument("--teacher_device", default="cuda:0")
    parser.add_argument("--student_device", default="cuda:1")
    parser.add_argument("--teacher_dtype", default="bfloat16")
    parser.add_argument("--model_dtype", default="bfloat16")
    parser.add_argument("--qwen_model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--dataset", default="triviaqa_generated")
    parser.add_argument("--train_generated_jsonl", default="./cache/triviaqa_teacher_generated_len1024_new512_n10k/generated/train_n10000_maxlen1024_new512_genv2.jsonl")
    parser.add_argument("--eval_generated_jsonl", default="./cache/triviaqa_teacher_generated_len1024_new512_n10k/generated/test_n500_maxlen1024_new512_genv2.jsonl")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--trainable", default="adapter")
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", default="cosine")
    parser.add_argument("--min_lr", type=float, default=5e-7)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--kl_direction", default="forward")
    parser.add_argument("--kl_temperature", type=float, default=2.0)
    parser.add_argument("--prompt_loss_weight", type=float, default=0.0)
    parser.add_argument("--answer_loss_weight", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--eval_batches", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--kl_token_chunk_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_trust_remote_code", action="store_true")
    parser.add_argument("--extra_train_args", nargs=argparse.REMAINDER, default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    args.current_set = str(Path(args.current_set))
    args.initial_checkpoint = str(Path(args.initial_checkpoint)) if args.initial_checkpoint else ""
    args.train_script = Path(args.train_script)
    if not args.train_script.is_absolute():
        args.train_script = repo_root / args.train_script

    ensure_current_set(args)
    output_root = Path(args.output_root)
    run_manifest: list[dict[str, object]] = []
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for index, stage in enumerate(args.stages):
        parse_layers(stage)
        stage_dir = output_root / f"stage_{index:02d}_{sanitize_stage_name(stage)}"
        if stage_dir.exists() and args.overwrite_stage and not args.dry_run:
            shutil.rmtree(stage_dir)
        if stage_dir.exists() and not args.overwrite_stage and not args.dry_run:
            raise FileExistsError(f"Stage output already exists: {stage_dir}. Use --overwrite_stage to replace it.")

        cmd = build_train_command(args, stage, stage_dir)
        exit_code = run_and_log(cmd, stage_dir / "train.log", env, args.dry_run)
        if exit_code != 0:
            raise SystemExit(f"Stage {stage} failed with exit code {exit_code}")

        merge_manifest = None
        if not args.no_merge:
            source = stage_dir / "best_current"
            merge_manifest = copy_replacement_layers(
                source_root=source,
                target_root=Path(args.current_set),
                layers=parse_layers(stage),
                backup_root=Path(args.current_set) / "_backups",
                dry_run=args.dry_run,
            )
            print("[merge]", json.dumps(merge_manifest, indent=2, ensure_ascii=False))

        run_manifest.append(
            {
                "stage_index": index,
                "layers": stage,
                "stage_dir": str(stage_dir),
                "command": cmd,
                "merged": not args.no_merge,
                "merge_manifest": merge_manifest,
            }
        )

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "current_set": args.current_set,
            "stages": run_manifest,
        }
        (output_root / "progressive_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[done] wrote {output_root / 'progressive_manifest.json'}")


if __name__ == "__main__":
    main()
