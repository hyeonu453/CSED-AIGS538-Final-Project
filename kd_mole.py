from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from gptoss_kd_capture import get_decoder_layers
from kd_losses import free_cuda
from modeling import CustomMoLELayer


def make_qwen_replacement_config(qwen_config: Any, teacher_config: Any, teacher_layer_idx: int) -> Any:
    qwen_config = copy.deepcopy(qwen_config)
    teacher_layer_types = getattr(teacher_config, "layer_types", None)
    if teacher_layer_types is None:
        return qwen_config

    teacher_layer_type = teacher_layer_types[teacher_layer_idx]
    qwen_config.layer_types = ["full_attention"] * int(qwen_config.num_hidden_layers)
    qwen_layer_idx = teacher_layer_idx % int(qwen_config.num_hidden_layers)
    qwen_config.layer_types[qwen_layer_idx] = teacher_layer_type
    if teacher_layer_type == "sliding_attention":
        qwen_config.sliding_window = int(getattr(teacher_config, "sliding_window"))
        qwen_config.use_sliding_window = True
    else:
        qwen_config.sliding_window = None
        qwen_config.use_sliding_window = False
    return qwen_config


class GPTOSSLayerReplacement(nn.Module):
    def __init__(self, custom_layer: CustomMoLELayer, qwen_config: Any):
        super().__init__()
        self.custom_layer = custom_layer
        self.rotary = Qwen2RotaryEmbedding(qwen_config, device=next(custom_layer.parameters()).device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del position_embeddings
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
            position_ids = position_ids.expand(hidden_states.shape[0], -1)
        rotary_input = self.custom_layer.proj_down(hidden_states)
        qwen_position_embeddings = self.rotary(rotary_input, position_ids)
        return self.custom_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=qwen_position_embeddings,
            **kwargs,
        )


def make_mole_config(teacher_config: Any, rank: int, mole_alpha: float) -> Any:
    config = copy.deepcopy(teacher_config)
    config.rank = int(rank)
    config.mole_alpha = float(mole_alpha if mole_alpha > 0 else rank)
    if not hasattr(config, "num_experts"):
        config.num_experts = 32
    if not hasattr(config, "num_experts_per_tok"):
        config.num_experts_per_tok = 4
    if not hasattr(config, "initializer_range"):
        config.initializer_range = 0.02
    return config


def copy_qwen_layer_weights(custom_layer: CustomMoLELayer, qwen_layer: nn.Module) -> None:
    custom_layer.self_attn.load_state_dict(qwen_layer.self_attn.state_dict(), strict=True)
    custom_layer.input_layernorm.load_state_dict(qwen_layer.input_layernorm.state_dict(), strict=True)
    custom_layer.post_attention_layernorm.load_state_dict(qwen_layer.post_attention_layernorm.state_dict(), strict=True)
    custom_layer.mlp.experts.gate_proj.load_state_dict(qwen_layer.mlp.gate_proj.state_dict(), strict=True)
    custom_layer.mlp.experts.up_proj.load_state_dict(qwen_layer.mlp.up_proj.state_dict(), strict=True)
    custom_layer.mlp.experts.down_proj.load_state_dict(qwen_layer.mlp.down_proj.state_dict(), strict=True)


def build_custom_layer(
    teacher_config: Any,
    qwen_model_name_or_path: str,
    layer_idx: int,
    rank: int,
    mole_alpha: float,
    train_dtype: torch.dtype,
    device: str,
    init_from_qwen: bool,
) -> tuple[CustomMoLELayer, Any]:
    qwen_config = AutoConfig.from_pretrained(qwen_model_name_or_path, trust_remote_code=True)
    qwen_config = make_qwen_replacement_config(qwen_config, teacher_config, layer_idx)
    custom_layer = CustomMoLELayer(make_mole_config(teacher_config, rank, mole_alpha), qwen_config, layer_idx)
    if init_from_qwen:
        qwen = AutoModelForCausalLM.from_pretrained(
            qwen_model_name_or_path,
            torch_dtype=train_dtype,
            device_map=None,
            trust_remote_code=True,
        )
        qwen_layers = get_decoder_layers(qwen)
        copy_qwen_layer_weights(custom_layer, qwen_layers[min(layer_idx, len(qwen_layers) - 1)])
        del qwen
        free_cuda()
    custom_layer.to(device=device, dtype=train_dtype)
    return custom_layer, qwen_config


def set_trainable_parameters(model: CustomMoLELayer, mode: str) -> None:
    for param in model.parameters():
        param.requires_grad_(mode == "all")
    if mode == "all":
        return
    if mode != "adapter":
        raise ValueError(f"Unknown train mode: {mode}")
    for name, param in model.named_parameters():
        if name.startswith(("proj_down", "proj_up", "mlp.router")) or name.endswith(("_mole_A", "_mole_B")):
            param.requires_grad_(True)


def forward_custom_layer(
    model: CustomMoLELayer,
    rotary: Qwen2RotaryEmbedding,
    x: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
    rotary_input = model.proj_down(x)
    position_embeddings = rotary(rotary_input, position_ids)
    return model(
        x,
        attention_mask=attention_mask,
        position_ids=None,
        position_embeddings=position_embeddings,
        use_cache=False,
    )
