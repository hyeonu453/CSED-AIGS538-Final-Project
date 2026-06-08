#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer



@dataclass
class TrainRecipe:
    dataset_path: str = "locuslab/TOFU"
    dataset_name: str = "full"
    dataset_split: str = "train"
    question_key: str = "question"
    answer_key: str = "answer"
    max_seq_length: int = 512
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_train_epochs: float = 3.0
    warmup_epochs: float = 0.1
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 16
    logging_steps: int = 10
    seed: int = 42
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train gpt-oss-20b on TOFU with Unsloth attention-only LoRA."
    )
    parser.add_argument(
        "--model-name",
        default="unsloth/gpt-oss-20b-BF16",
        help="Unsloth model ID. Use unsloth/gpt-oss-20b for QLoRA or unsloth/gpt-oss-20b-BF16 for BF16 LoRA.",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/538/outputs/tofu_gptoss20b_unsloth_attn_lora_bf16",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Unsloth QLoRA path. Disable for BF16 LoRA.",
    )
    parser.add_argument(
        "--save-merged",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also export a merged checkpoint after adapter training.",
    )
    parser.add_argument(
        "--merged-save-method",
        default="merged_16bit",
        choices=["merged_16bit", "mxfp4"],
        help="Unsloth merge/export format.",
    )
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-epochs", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_recipe(args):
    recipe = TrainRecipe()
    recipe.max_seq_length = args.max_seq_length
    recipe.learning_rate = args.learning_rate
    recipe.weight_decay = args.weight_decay
    recipe.num_train_epochs = args.num_train_epochs
    recipe.warmup_epochs = args.warmup_epochs
    recipe.per_device_train_batch_size = args.per_device_train_batch_size
    recipe.gradient_accumulation_steps = args.gradient_accumulation_steps
    recipe.logging_steps = args.logging_steps
    recipe.seed = args.seed
    recipe.lora_r = args.lora_r
    recipe.lora_alpha = args.lora_alpha
    recipe.lora_dropout = args.lora_dropout
    return recipe


def format_dataset(dataset, tokenizer, recipe: TrainRecipe):
    def format_batch(batch):
        texts = []
        for question, answer in zip(batch[recipe.question_key], batch[recipe.answer_key]):
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    reasoning_effort="low",
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            texts.append(text)
        return {"text": texts}

    columns_to_remove = list(dataset.column_names)
    return dataset.map(format_batch, batched=True, remove_columns=columns_to_remove)


def compute_warmup_steps(dataset_len: int, recipe: TrainRecipe) -> int:
    world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
    denominator = (
        recipe.per_device_train_batch_size
        * recipe.gradient_accumulation_steps
        * world_size
    )
    if dataset_len == 0 or denominator == 0:
        return 0
    return int((recipe.warmup_epochs * dataset_len) // denominator)


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_run_recipe(output_dir: str, args, recipe: TrainRecipe):
    payload = {
        "cli_args": vars(args),
        "recipe": asdict(recipe),
        "notes": {
            "source_repo_alignment": {
                "dataset": "locuslab/TOFU full/train, question->answer supervised finetune",
                "hyperparameters": "aligned to open-unlearning-lora TOFU LoRA preset where practical",
                "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            }
        },
    }
    with open(os.path.join(output_dir, "run_recipe.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    recipe = build_recipe(args)
    ensure_output_dir(args.output_dir)
    save_run_recipe(args.output_dir, args, recipe)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=recipe.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=recipe.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=recipe.lora_alpha,
        lora_dropout=recipe.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=recipe.seed,
        use_rslora=False,
        loftq_config=None,
    )

    dataset = load_dataset(
        recipe.dataset_path,
        name=recipe.dataset_name,
        split=recipe.dataset_split,
    )
    dataset = format_dataset(dataset, tokenizer, recipe)
    warmup_steps = compute_warmup_steps(len(dataset), recipe)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            eos_token=None,
            pad_token=None,
            max_length=recipe.max_seq_length,
            per_device_train_batch_size=recipe.per_device_train_batch_size,
            gradient_accumulation_steps=recipe.gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=recipe.num_train_epochs,
            learning_rate=recipe.learning_rate,
            logging_steps=recipe.logging_steps,
            optim="adamw_8bit",
            weight_decay=recipe.weight_decay,
            lr_scheduler_type="linear",
            seed=recipe.seed,
            output_dir=args.output_dir,
            report_to="none",
            save_strategy="epoch",
            save_total_limit=2,
            packing=False,
        ),
    )

    trainer_stats = trainer.train()
    trainer.save_state()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics_path = os.path.join(args.output_dir, "train_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(trainer_stats.metrics, f, indent=2)

    if args.save_merged:
        merged_dir = os.path.join(args.output_dir, f"merged_{args.merged_save_method}")
        model.save_pretrained_merged(
            merged_dir,
            tokenizer,
            save_method=args.merged_save_method,
        )

    print(f"Training complete. Outputs saved to: {args.output_dir}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Train examples: {len(dataset)}")


if __name__ == "__main__":
    main()
