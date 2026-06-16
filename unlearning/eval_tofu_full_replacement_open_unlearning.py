#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from transformers import AutoTokenizer


PROJECT_ROOT = "/workspace/CSED-AIGS538-Final-Project"
OPEN_UNLEARNING_ROOT = "/workspace/538/open-unlearning-lora"
OPEN_UNLEARNING_SRC = os.path.join(OPEN_UNLEARNING_ROOT, "src")

for path in (PROJECT_ROOT, OPEN_UNLEARNING_SRC):
    if path not in sys.path:
        sys.path.insert(0, path)

from evals.tofu import TOFUEvaluator  # noqa: E402
from full_replacement_utils import load_student_with_replacements  # noqa: E402
from trainer.utils import seed_everything  # noqa: E402
from utils.logging import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a GPT-OSS full layer-replacement student with "
            "open-unlearning-lora's TOFU evaluator."
        )
    )
    parser.add_argument(
        "--student-model",
        "--base-model",
        dest="student_model",
        default="openai/gpt-oss-20b",
        help="GPT-OSS backbone into which replacement layers are inserted.",
    )
    parser.add_argument(
        "--layer-checkpoint-dir",
        "--layer_checkpoint_dir",
        dest="layer_checkpoint_dir",
        required=True,
        help="Root containing layer_XX/custom_mole_layer.pt replacement checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/538/outputs/tofu_gptoss20b_full_replacement_eval",
    )
    parser.add_argument("--task-name", default="tofu_gptoss20b_full_replacement_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forget-split", default="forget10")
    parser.add_argument("--retain-split", default="retain90")
    parser.add_argument("--holdout-split", default="holdout10")
    parser.add_argument(
        "--retain-logs-path",
        default=None,
        help="Retain-model evaluation logs path for forget_quality. If omitted, forget_quality is disabled.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument(
        "--student-device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--model-dtype",
        "--torch-dtype",
        dest="model_dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--qwen-layer", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--mole-alpha", type=float, default=-1.0)
    parser.add_argument(
        "--trainable",
        choices=["mole_router", "mole_only", "adapter", "all_replacement"],
        default="adapter",
        help="Replacement trainability mode used only to configure requires_grad after loading.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="Passed to tokenizer.apply_chat_template via template_args.",
    )
    parser.add_argument(
        "--date-string",
        default=None,
        help="Optional gpt-oss chat-template date string.",
    )
    return parser.parse_args()


def build_eval_cfg(args: argparse.Namespace):
    with initialize_config_dir(
        version_base=None,
        config_dir=os.path.join(OPEN_UNLEARNING_ROOT, "configs"),
    ):
        cfg = compose(
            config_name="eval.yaml",
            overrides=[
                "experiment=eval/tofu/default",
                f"task_name={args.task_name}",
                f"seed={args.seed}",
                f"forget_split={args.forget_split}",
                f"holdout_split={args.holdout_split}",
                f"paths.root_dir={OPEN_UNLEARNING_ROOT}",
            ],
        )

    with open_dict(cfg):
        cfg.paths.output_dir = args.output_dir
        cfg.eval.tofu.output_dir = args.output_dir
        cfg.eval.tofu.batch_size = args.eval_batch_size
        cfg.eval.tofu.forget_split = args.forget_split
        cfg.eval.tofu.holdout_split = args.holdout_split
        cfg.eval.tofu.overwrite = True
        cfg.eval.tofu.retain_logs_path = args.retain_logs_path

        if args.retain_logs_path is None and "forget_quality" in cfg.eval.tofu.metrics:
            del cfg.eval.tofu.metrics["forget_quality"]

        privleak_cfg = cfg.eval.tofu.metrics.privleak
        retain_mia_cfg = OmegaConf.create(
            {
                "handler": "mia_min_k",
                "batch_size": args.eval_batch_size,
                "k": 0.4,
                "datasets": {
                    "TOFU_QA_retain": {
                        # The shared mia_auc implementation expects keys named
                        # forget/holdout; for this metric, "forget" means retain.
                        "access_key": "forget",
                        "handler": "QADataset",
                        "args": {
                            "hf_args": {
                                "path": "locuslab/TOFU",
                                "name": args.retain_split,
                                "split": "train",
                            },
                            "question_key": cfg.eval.tofu.question_key,
                            "answer_key": "answer",
                            "max_length": 512,
                        },
                    },
                    "TOFU_QA_holdout": {
                        "access_key": "holdout",
                        "handler": "QADataset",
                        "args": {
                            "hf_args": {
                                "path": "locuslab/TOFU",
                                "name": args.holdout_split,
                                "split": "train",
                            },
                            "question_key": cfg.eval.tofu.question_key,
                            "answer_key": "answer",
                            "max_length": 512,
                        },
                    },
                },
                "collators": {
                    "DataCollatorForSupervisedDataset": {
                        "handler": "DataCollatorForSupervisedDataset",
                        "args": {"padding_side": "right", "index": "index"},
                    }
                },
            }
        )
        cfg.eval.tofu.metrics = OmegaConf.create(
            {
                "privleak": privleak_cfg,
                "retain_mia_min_k": retain_mia_cfg,
            }
        )

    return cfg


def load_model_and_tokenizer(args: argparse.Namespace):
    load_args = SimpleNamespace(
        student_model=args.student_model,
        layer_checkpoint_dir=args.layer_checkpoint_dir,
        student_device=args.student_device,
        model_dtype=args.model_dtype,
        qwen_model=args.qwen_model,
        layers=args.layers,
        qwen_layer=args.qwen_layer,
        rank=args.rank,
        mole_alpha=args.mole_alpha,
        trainable=args.trainable,
        trust_remote_code=args.trust_remote_code,
        gradient_checkpointing=False,
    )
    model, _, _ = load_student_with_replacements(load_args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.config.use_cache = False
    return model, tokenizer


def build_template_args(args: argparse.Namespace) -> dict[str, object]:
    template_args: dict[str, object] = {
        "apply_chat_template": True,
        "system_prompt": args.system_prompt,
    }
    if args.date_string:
        template_args["date_string"] = args.date_string
    return template_args


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(log_file=os.path.join(args.output_dir, "full_replacement_eval.log"))
    seed_everything(args.seed)

    cfg = build_eval_cfg(args)
    model, tokenizer = load_model_and_tokenizer(args)
    template_args = build_template_args(args)

    evaluator = TOFUEvaluator(cfg.eval.tofu)
    summary = evaluator.evaluate(
        model=model,
        tokenizer=tokenizer,
        template_args=template_args,
        output_dir=args.output_dir,
        overwrite=True,
    )

    print("Evaluation summary:")

    def _to_builtin(value):
        if isinstance(value, dict):
            return {k: _to_builtin(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_to_builtin(v) for v in value]
        if hasattr(value, "item"):
            return value.item()
        return value

    print(OmegaConf.to_yaml(OmegaConf.create(_to_builtin(summary))))


if __name__ == "__main__":
    main()
