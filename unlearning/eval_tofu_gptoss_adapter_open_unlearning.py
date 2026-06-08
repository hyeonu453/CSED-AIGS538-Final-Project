#!/usr/bin/env python3
import argparse
import os
import sys

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel



REPO_ROOT = "/workspace/538/open-unlearning-lora"
REPO_SRC = os.path.join(REPO_ROOT, "src")
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from evals.tofu import TOFUEvaluator  # noqa: E402
from trainer.utils import seed_everything  # noqa: E402
from utils.logging import setup_logging  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a GPT-oss base model with an adapter-only checkpoint using open-unlearning-lora's TOFU evaluator."
    )
    parser.add_argument(
        "--base-model",
        default="openai/gpt-oss-20b",
        help="Base model to load with Transformers, then attach the adapter checkpoint to.",
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Optional directory containing the adapter-only checkpoint saved via model.save_pretrained(...). If omitted, evaluates the base model only.",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/538/outputs/tofu_gptoss20b_adapter_eval",
    )
    parser.add_argument("--task-name", default="tofu_gptoss20b_adapter_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forget-split", default="forget10")
    parser.add_argument("--holdout-split", default="holdout10")
    parser.add_argument(
        "--retain-logs-path",
        default=None,
        help="Retain-model evaluation logs path for forget_quality. If omitted, forget_quality is disabled.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--attn-implementation", default="eager")
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


def get_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def build_eval_cfg(args):
    with initialize_config_dir(version_base=None, config_dir=os.path.join(REPO_ROOT, "configs")):
        cfg = compose(
            config_name="eval.yaml",
            overrides=[
                "experiment=eval/tofu/default",
                f"task_name={args.task_name}",
                f"seed={args.seed}",
                f"forget_split={args.forget_split}",
                f"holdout_split={args.holdout_split}",
                f"paths.root_dir={REPO_ROOT}",
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

    return cfg


def load_model_and_tokenizer(args):
    torch_dtype = get_torch_dtype(args.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )

    if args.adapter_dir:
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    else:
        model = base_model
    model.eval()
    return model, tokenizer


def build_template_args(args):
    template_args = {
        "apply_chat_template": True,
        "system_prompt": args.system_prompt,
    }
    if args.date_string:
        template_args["date_string"] = args.date_string
    return template_args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(log_file=os.path.join(args.output_dir, "adapter_eval.log"))
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
