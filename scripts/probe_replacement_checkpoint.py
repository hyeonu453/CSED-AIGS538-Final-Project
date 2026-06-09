#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PROMPTS = [
    "Which American-born Sinclair won the Nobel Prize for Literature in 1930?",
    "Who wrote the novel Babbitt?",
    "What is the capital of France?",
]


def add_arg(cmd: list[str], name: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, str) and value == "":
        return
    cmd.extend([name, str(value)])


def build_command(args: argparse.Namespace, prompt: str) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    chat_script = Path(args.chat_script)
    if not chat_script.is_absolute():
        chat_script = repo_root / chat_script
    cmd = [sys.executable, str(chat_script)]
    add_arg(cmd, "--student_model", args.student_model)
    add_arg(cmd, "--layer_checkpoint_dir", args.checkpoint)
    add_arg(cmd, "--student_device", args.student_device)
    add_arg(cmd, "--model_dtype", args.model_dtype)
    add_arg(cmd, "--qwen_model", args.qwen_model)
    add_arg(cmd, "--layers", args.layers)
    add_arg(cmd, "--rank", args.rank)
    add_arg(cmd, "--trainable", args.trainable)
    add_arg(cmd, "--reasoning_effort", args.reasoning_effort)
    add_arg(cmd, "--max_new_tokens", args.max_new_tokens)
    add_arg(cmd, "--temperature", args.temperature)
    add_arg(cmd, "--prompt", prompt)
    return cmd


def run_probe(args: argparse.Namespace, prompt: str) -> dict[str, object]:
    cmd = build_command(args, prompt)
    print("[probe]", " ".join(cmd))
    if args.dry_run:
        return {"prompt": prompt, "command": cmd, "returncode": 0, "output": ""}
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    print(proc.stdout)
    return {
        "prompt": prompt,
        "command": cmd,
        "returncode": proc.returncode,
        "output": proc.stdout,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run chat probes against one replacement checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Replacement checkpoint root, e.g. output/current-set/best_current.")
    parser.add_argument("--layers", default="all", help="Layers to replace while probing, e.g. all, 0-17, 0-23.")
    parser.add_argument("--prompt", action="append", default=[], help="Prompt to test. Can be repeated.")
    parser.add_argument("--output_jsonl", default="", help="Optional JSONL file for probe outputs.")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--chat_script", default="chat_full_replacement.py")
    parser.add_argument("--student_model", default="openai/gpt-oss-20b")
    parser.add_argument("--student_device", default="cuda:0")
    parser.add_argument("--model_dtype", default="bfloat16")
    parser.add_argument("--qwen_model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--trainable", default="adapter")
    parser.add_argument("--reasoning_effort", default="medium")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prompts = args.prompt or DEFAULT_PROMPTS
    results = [run_probe(args, prompt) for prompt in prompts]
    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as f:
            for result in results:
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                    "checkpoint": args.checkpoint,
                    "layers": args.layers,
                    **result,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
