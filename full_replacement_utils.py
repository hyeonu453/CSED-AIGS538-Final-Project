from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM

from gptoss_kd_capture import get_decoder_layers, get_dtype, unwrap_for_introspection
from kd_mole import GPTOSSLayerReplacement, build_custom_layer

IGNORE_INDEX = -100


@dataclass
class QABatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class TofuQADataset(Dataset):
    def __init__(
        self,
        tokenizer: Any,
        dataset_name: str,
        split: str,
        max_length: int,
        question_key: str = "question",
        answer_key: str = "answer",
        reasoning_effort: str | None = "low",
        limit: int = -1,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.question_key = question_key
        self.answer_key = answer_key
        self.reasoning_effort = reasoning_effort
        self.data = load_dataset("locuslab/TOFU", name=dataset_name, split=split)
        if limit > 0:
            self.data = self.data.select(range(min(limit, len(self.data))))

    def __len__(self) -> int:
        return len(self.data)

    def _normalize_ids(self, value: Any) -> list[int]:
        if hasattr(value, "keys") and "input_ids" in value:
            value = value["input_ids"]
        if isinstance(value, torch.Tensor):
            value = value.tolist()
        if value and isinstance(value[0], list):
            value = value[0]
        return list(value)

    def _apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool) -> Any:
        kwargs: dict[str, Any] = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("reasoning_effort", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data[idx]
        question = row[self.question_key]
        answer = row[self.answer_key]
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        prompt_ids = self._normalize_ids(
            self._apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        )
        full_ids = self._normalize_ids(
            self._apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        )
        eos = self.tokenizer.eos_token_id
        if eos is not None and (not full_ids or full_ids[-1] != eos):
            full_ids.append(int(eos))

        full_ids = full_ids[: self.max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [IGNORE_INDEX] * prompt_len + full_ids[prompt_len:]
        if len(labels) < len(full_ids):
            labels.extend([IGNORE_INDEX] * (len(full_ids) - len(labels)))
        labels = labels[: len(full_ids)]
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class SupervisedCollator:
    def __init__(self, tokenizer: Any, padding_side: str = "right"):
        self.tokenizer = tokenizer
        self.padding_side = padding_side

    def _pad(self, rows: list[torch.Tensor], value: int) -> torch.Tensor:
        if self.padding_side == "left":
            rows = [torch.flip(row, dims=[0]) for row in rows]
            padded = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=value)
            return torch.flip(padded, dims=[1])
        return torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=value)

    def __call__(self, instances: list[dict[str, torch.Tensor]]) -> QABatch:
        input_ids = self._pad([x["input_ids"] for x in instances], int(self.tokenizer.pad_token_id))
        labels = self._pad([x["labels"] for x in instances], IGNORE_INDEX)
        attention_mask = input_ids.ne(int(self.tokenizer.pad_token_id)).long()
        return QABatch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


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
            raise ValueError(f"Invalid layer {layer}; model has {num_layers} layers.")
    return deduped


def batch_to_device(batch: QABatch, device: str) -> dict[str, torch.Tensor]:
    return {
        "input_ids": batch.input_ids.to(device),
        "attention_mask": batch.attention_mask.to(device),
        "labels": batch.labels.to(device),
    }


def answer_token_count(batch: QABatch) -> int:
    return int((batch.labels != IGNORE_INDEX).sum().item())


def set_replacement_trainable(custom_layer: torch.nn.Module, mode: str) -> None:
    for param in custom_layer.parameters():
        param.requires_grad_(False)
    if mode not in {"mole_router", "mole_only", "adapter", "projection_only", "all_replacement"}:
        raise ValueError(f"Unknown trainable mode: {mode}")

    for name, param in custom_layer.named_parameters():
        is_mole = name.endswith(("_mole_A", "_mole_B"))
        is_router = name.startswith("custom_layer.mlp.router")
        is_projection = name.startswith(("custom_layer.proj_down", "custom_layer.proj_up"))

        if mode == "mole_only":
            train = is_mole
        elif mode == "mole_router":
            train = is_mole or is_router
        elif mode == "adapter":
            train = is_projection or is_router or is_mole
        elif mode == "projection_only":
            train = is_projection
        else:  # all_replacement
            train = name.startswith("custom_layer")
        param.requires_grad_(train)


def load_student_with_replacements(args: Any) -> tuple[torch.nn.Module, Any, dict[int, GPTOSSLayerReplacement]]:
    dtype = get_dtype(args.model_dtype)
    print(f"=== Loading student backbone on {args.student_device} ===")
    model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=dtype,
        device_map={"": args.student_device},
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    inspected = unwrap_for_introspection(model)
    base_layers = get_decoder_layers(inspected)
    layers_to_replace = parse_layers(args.layers, len(base_layers))
    replacements: dict[int, GPTOSSLayerReplacement] = {}

    for layer in layers_to_replace:
        state_path = Path(args.layer_checkpoint_dir) / f"layer_{layer:02d}" / "custom_mole_layer.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing replacement checkpoint for layer {layer}: {state_path}")
        custom_layer, qwen_config = build_custom_layer(
            teacher_config=inspected.config,
            qwen_model_name_or_path=args.qwen_model,
            layer_idx=args.qwen_layer if args.qwen_layer >= 0 else layer,
            rank=args.rank,
            mole_alpha=args.mole_alpha,
            train_dtype=dtype,
            device=args.student_device,
            init_from_qwen=False,
        )
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        custom_layer.load_state_dict(state, strict=True)
        replacement = GPTOSSLayerReplacement(custom_layer, qwen_config).to(device=args.student_device, dtype=dtype)
        set_replacement_trainable(replacement, args.trainable)
        base_layers[layer] = replacement
        replacements[layer] = replacement
        print(f"[student] replaced layer {layer} from {state_path}")

    for _, param in model.named_parameters():
        if not any(param is p for repl in replacements.values() for p in repl.parameters()):
            param.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_repl = sum(p.numel() for repl in replacements.values() for p in repl.parameters())
    print(f"[student] trainable={trainable:,} replacement_total={total_repl:,} mode={args.trainable}")
    return model, inspected.config, replacements


def kl_teacher_student_loss(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    batch: QABatch,
    args: Any,
    answer_only: bool = True,
) -> torch.Tensor:
    student_inputs = {
        "input_ids": batch.input_ids.to(args.student_device),
        "attention_mask": batch.attention_mask.to(args.student_device),
    }
    teacher_inputs = {
        "input_ids": batch.input_ids.to(args.teacher_device),
        "attention_mask": batch.attention_mask.to(args.teacher_device),
    }
    with torch.no_grad():
        teacher_logits = teacher(**teacher_inputs, use_cache=False).logits[:, :-1, :].detach()
    student_logits = student(**student_inputs, use_cache=False).logits[:, :-1, :]

    if answer_only:
        mask = batch.labels[:, 1:].to(args.student_device).ne(IGNORE_INDEX)
    else:
        mask = batch.attention_mask[:, 1:].to(args.student_device).bool()

    temperature = float(args.kl_temperature)
    token_chunk_size = int(getattr(args, "kl_token_chunk_size", 64))
    token_chunk_size = max(1, token_chunk_size)
    kl_sum = student_logits.new_zeros((), dtype=torch.float32)
    token_count = mask.sum().clamp_min(1).to(dtype=torch.float32)

    for start in range(0, student_logits.shape[1], token_chunk_size):
        end = min(start + token_chunk_size, student_logits.shape[1])
        chunk_mask = mask[:, start:end]
        if not bool(chunk_mask.any().item()):
            continue
        teacher_chunk = teacher_logits[:, start:end, :].to(args.student_device, dtype=torch.float32)
        student_chunk = student_logits[:, start:end, :].float()
        teacher_log_probs = F.log_softmax(teacher_chunk / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_chunk / temperature, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
        kl_sum = kl_sum + (token_kl * chunk_mask).sum()
        del teacher_chunk, student_chunk, teacher_log_probs, student_log_probs, teacher_probs, token_kl

    return kl_sum / token_count * (temperature * temperature)


def model_loss(model: torch.nn.Module, batch: QABatch, device: str) -> torch.Tensor:
    inputs = batch_to_device(batch, device)
    outputs = model(**inputs, use_cache=False)
    return outputs.loss


def save_replacements(
    replacements: dict[int, GPTOSSLayerReplacement],
    output_dir: Path,
    step: int,
    final: bool = False,
) -> None:
    root = output_dir if final else output_dir / f"step_{step:06d}"
    for layer, replacement in replacements.items():
        layer_dir = root / f"layer_{layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        state = {
            key: value.detach().cpu()
            for key, value in replacement.custom_layer.state_dict().items()
        }
        torch.save(state, layer_dir / "custom_mole_layer.pt")
