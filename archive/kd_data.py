from __future__ import annotations

from typing import Any

from datasets import load_dataset


def format_qa_prompt(row: dict[str, Any]) -> str:
    return f"Question: {row['question']}\nAnswer:"


def load_triviaqa_prompts(config_name: str, train_n: int, test_n: int) -> tuple[list[str], list[str]]:
    train = load_dataset("trivia_qa", config_name, split=f"train[:{train_n}]")
    validation = load_dataset("trivia_qa", config_name, split=f"validation[:{test_n}]")
    return [format_qa_prompt(row) for row in train], [format_qa_prompt(row) for row in validation]
