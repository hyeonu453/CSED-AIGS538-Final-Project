from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from gptoss_kd_capture import get_decoder_layers
from kd_losses import free_cuda


class ProjectionOnlyQwenLayer(nn.Module):
    """GPT-OSS hidden -> Qwen decoder layer -> GPT-OSS hidden.

    This is a temporary probe module for testing Qwen-family decoder layers
    without the MoLE expert adapters.
    """

    def __init__(
        self,
        teacher_config: Any,
        qwen_model_name_or_path: str,
        qwen_layer_idx: int,
        train_dtype: torch.dtype,
        trust_remote_code: bool = True,
        init_from_qwen: bool = True,
    ):
        super().__init__()
        self.teacher_hidden_size = int(teacher_config.hidden_size)
        self.qwen_config = AutoConfig.from_pretrained(
            qwen_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.qwen_hidden_size = int(self.qwen_config.hidden_size)
        self.proj_down = nn.Linear(self.teacher_hidden_size, self.qwen_hidden_size, bias=True)
        self.proj_up = nn.Linear(self.qwen_hidden_size, self.teacher_hidden_size, bias=True)

        if not init_from_qwen:
            raise ValueError("ProjectionOnlyQwenLayer currently requires init_from_qwen=True.")

        qwen = AutoModelForCausalLM.from_pretrained(
            qwen_model_name_or_path,
            torch_dtype=train_dtype,
            device_map=None,
            trust_remote_code=trust_remote_code,
        )
        qwen_layers = get_decoder_layers(qwen)
        layer_idx = min(int(qwen_layer_idx), len(qwen_layers) - 1)
        self.qwen_layer = qwen_layers[layer_idx]
        self.rotary = getattr(getattr(qwen, "model", qwen), "rotary_emb", None)
        if self.rotary is None:
            raise RuntimeError(f"Could not find rotary_emb in {qwen_model_name_or_path}.")
        del qwen
        free_cuda()
        self.reset_projection_params()

    def reset_projection_params(self) -> None:
        nn.init.xavier_uniform_(self.proj_down.weight)
        nn.init.zeros_(self.proj_down.bias)
        nn.init.zeros_(self.proj_up.weight)
        nn.init.zeros_(self.proj_up.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        hidden_states = self.proj_down(hidden_states)
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
            position_ids = position_ids.expand(hidden_states.shape[0], -1)
        position_embeddings = self.rotary(hidden_states, position_ids)
        output = self.qwen_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        hidden_states = output[0] if isinstance(output, tuple) else output
        return self.proj_up(hidden_states)


def set_projection_trainable(model: ProjectionOnlyQwenLayer, train_qwen_layer: bool = False) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.proj_down.parameters():
        param.requires_grad_(True)
    for param in model.proj_up.parameters():
        param.requires_grad_(True)
    if train_qwen_layer:
        for param in model.qwen_layer.parameters():
            param.requires_grad_(True)

