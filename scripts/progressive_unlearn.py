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


def add_arg(cmd: list[str], name: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, str) and value == "":
        return
    cmd.extend([name, str(value)])


def build_train_command(args: argparse.Namespace, stage: str, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(args.train_script),
    ]
    add_arg(cmd, "--student_model", args.student_model)
    add_arg(cmd, "--student_device", args.student_device)
    add_arg(cmd, "--model_dtype", args.model_dtype)
    add_arg(cmd, "--qwen_model", args.qwen_model)
    add_arg(cmd, "--layer_checkpoint_dir", args.current_set)
    add_arg(cmd, "--output_dir", output_dir)
    add_arg(cmd, "--layers", stage)
    add_arg(cmd, "--qwen_layer", args.qwen_layer)
    add_arg(cmd, "--rank", args.rank)
    add_arg(cmd, "--mole_alpha", args.mole_alpha)

    add_arg(cmd, "--forget_split", args.forget_split)
    add_arg(cmd, "--retain_split", args.retain_split)
    add_arg(cmd, "--eval_forget_split", args.eval_forget_split)
    add_arg(cmd, "--eval_retain_split", args.eval_retain_split)
    add_arg(cmd, "--dataset_split", args.dataset_split)
    add_arg(cmd, "--question_key", args.question_key)
    add_arg(cmd, "--answer_key", args.answer_key)
    add_arg(cmd, "--max_length", args.max_length)
    add_arg(cmd, "--reasoning_effort", args.reasoning_effort)
    add_arg(cmd, "--forget_limit", args.forget_limit)
    add_arg(cmd, "--retain_limit", args.retain_limit)
    add_arg(cmd, "--eval_forget_limit", args.eval_forget_limit)
    add_arg(cmd, "--eval_retain_limit", args.eval_retain_limit)

    add_arg(cmd, "--method", args.method)
    add_arg(cmd, "--gamma_forget", args.gamma_forget)
    add_arg(cmd, "--alpha_retain", args.alpha_retain)
    add_arg(cmd, "--simnpo_beta", args.simnpo_beta)
    add_arg(cmd, "--simnpo_delta", args.simnpo_delta)

    add_arg(cmd, "--trainable", args.trainable)
    add_arg(cmd, "--optim", args.optim)
    add_arg(cmd, "--lr", args.lr)
    add_arg(cmd, "--lr_scheduler", args.lr_scheduler)
    add_arg(cmd, "--min_lr", args.min_lr)
    add_arg(cmd, "--warmup_steps", args.warmup_steps)
    add_arg(cmd, "--weight_decay", args.weight_decay)
    add_arg(cmd, "--grad_clip", args.grad_clip)
    add_arg(cmd, "--save_every", args.save_every)
    add_arg(cmd, "--steps", args.steps)
    add_arg(cmd, "--batch_size", args.batch_size)
    add_arg(cmd, "--grad_accum_steps", args.grad_accum_steps)
    add_arg(cmd, "--eval_batches", args.eval_batches)
    add_arg(cmd, "--log_every", args.log_every)
    add_arg(cmd, "--eval_every", args.eval_every)
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
    parser = argparse.ArgumentParser(description="Run staged TOFU unlearning and merge each stage's best_current into a current-set.")
    parser.add_argument("--stages", nargs="+", required=True, help="Layer specs to unlearn and merge, e.g. 0-5 6-11 12-17.")
    parser.add_argument("--current_set", required=True, help="Mutable checkpoint root used as layer_checkpoint_dir and merge target.")
    parser.add_argument("--initial_checkpoint", default="", help="Optional checkpoint root copied to current_set if current_set is missing.")
    parser.add_argument("--output_root", default="./output/progressive-unlearn")
    parser.add_argument("--train_script", default="unlearning/train_tofu_full_replacement_open_unlearn.py")
    parser.add_argument("--overwrite_stage", action="store_true")
    parser.add_argument("--no_merge", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--student_model", default="openai/gpt-oss-20b")
    parser.add_argument("--student_device", default="cuda:1")
    parser.add_argument("--model_dtype", default="bfloat16")
    parser.add_argument("--qwen_model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--qwen_layer", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--mole_alpha", type=float, default=-1.0)

    parser.add_argument("--forget_split", default="forget10")
    parser.add_argument("--retain_split", default="retain90")
    parser.add_argument("--eval_forget_split", default="")
    parser.add_argument("--eval_retain_split", default="")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--question_key", default="question")
    parser.add_argument("--answer_key", default="answer")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--reasoning_effort", default="low")
    parser.add_argument("--forget_limit", type=int, default=-1)
    parser.add_argument("--retain_limit", type=int, default=-1)
    parser.add_argument("--eval_forget_limit", type=int, default=-1)
    parser.add_argument("--eval_retain_limit", type=int, default=-1)

    parser.add_argument("--method", default="grad_diff", choices=["grad_ascent", "grad_diff", "simnpo"])
    parser.add_argument("--gamma_forget", type=float, default=1.0)
    parser.add_argument("--alpha_retain", type=float, default=1.0)
    parser.add_argument("--simnpo_beta", type=float, default=4.5)
    parser.add_argument("--simnpo_delta", type=float, default=0.0)

    parser.add_argument("--trainable", default="adapter")
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=50)
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
        (output_root / "progressive_unlearn_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[done] wrote {output_root / 'progressive_unlearn_manifest.json'}")


if __name__ == "__main__":
    main()
