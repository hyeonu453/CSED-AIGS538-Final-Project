from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RMSNorm
import torch.nn as nn
import torch
import torch.nn.functional as F

class CustomTopKRouter(nn.Module):
    def __init__(self, config, qwen2_config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = qwen2_config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        self.bias = nn.Parameter(torch.zeros(self.num_experts))

    def forward(self, hidden_states):
        router_logits = F.linear(hidden_states, self.weight, self.bias)  
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  
        topk_scores = torch.nn.functional.softmax(router_top_value, dim=-1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits)
        router_scores.scatter_(dim=-1, index=router_indices, src=topk_scores)
        return router_logits, router_scores, router_indices
    
class CustomExperts(nn.Module):
    def __init__(self, config, qwen2_config):
        super().__init__()
        self.config = config
        self.qwen2_config = qwen2_config
        self.hidden_size = qwen2_config.hidden_size
        self.intermediate_size = qwen2_config.intermediate_size
        self.rank = config.rank
        self.num_experts = config.num_experts
        self.scaling = getattr(config, "mole_alpha", self.rank) / self.rank

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.gate_proj_mole_A = nn.Parameter(torch.zeros(self.num_experts, self.rank, self.hidden_size))
        self.gate_proj_mole_B = nn.Parameter(torch.zeros(self.num_experts, self.intermediate_size, self.rank))
        self.up_proj_mole_A = nn.Parameter(torch.zeros(self.num_experts, self.rank, self.hidden_size))
        self.up_proj_mole_B = nn.Parameter(torch.zeros(self.num_experts, self.intermediate_size, self.rank))
        self.down_proj_mole_A = nn.Parameter(torch.zeros(self.num_experts, self.rank, self.intermediate_size))
        self.down_proj_mole_B = nn.Parameter(torch.zeros(self.num_experts, self.hidden_size, self.rank))

        self.act_fn = nn.functional.silu

    def forward(self, hidden_states: torch.Tensor, routing_weights):
        def _calculate_mole(x, base_linear, A, B, alpha):
            N, r, in_dim = A.shape
            _, out_dim, _ = B.shape

            A_cat = A.reshape(N * r, in_dim)                          # [N*r, in_dim]
            B_cat = B.permute(1, 0, 2).contiguous().reshape(out_dim, N * r)  # [out_dim, N*r]

            z = x @ A_cat.T                                           # [T, N*r]
            z = z * alpha.repeat_interleave(r, dim=-1)                 # [T, N*r]
            delta = z @ B_cat.T                                       # [T, out_dim]

            base = base_linear(x)                                  # [T, out_dim]
            return base + self.scaling * delta
        
        gate_proj = self.act_fn(_calculate_mole(hidden_states, self.gate_proj, self.gate_proj_mole_A, self.gate_proj_mole_B, routing_weights))
        up_proj = _calculate_mole(hidden_states, self.up_proj, self.up_proj_mole_A, self.up_proj_mole_B, routing_weights)
        intermediate_states = gate_proj * up_proj
        down_proj = _calculate_mole(intermediate_states, self.down_proj, self.down_proj_mole_A, self.down_proj_mole_B, routing_weights)
        return down_proj
  

class CustomMLP(nn.Module):
    def __init__(self, config, qwen2_config):
        super().__init__()
        self.router = CustomTopKRouter(config, qwen2_config)
        self.experts = CustomExperts(config, qwen2_config)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        router_logits, router_scores, _ = self.router(hidden_states)
        hidden_states = self.experts(hidden_states, router_scores)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_scores
    
class CustomMoLELayer(nn.Module):
    def __init__(self, config, qwen2_config, layer_idx: int):
        super().__init__()
        self.config = config
        self.qwen2_hidden_size = qwen2_config.hidden_size
        self.proj_down = nn.Linear(config.hidden_size, qwen2_config.hidden_size, bias=True)
        self.self_attn = Qwen2Attention(config=qwen2_config, layer_idx=layer_idx)

        # self.mlp = Qwen2MLP(config)
        self.mlp = CustomMLP(config, qwen2_config)

        self.input_layernorm = Qwen2RMSNorm(qwen2_config.hidden_size, eps=qwen2_config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(qwen2_config.hidden_size, eps=qwen2_config.rms_norm_eps)

        self.proj_up = nn.Linear(qwen2_config.hidden_size, config.hidden_size, bias=True)
        self.reset_params()

    def reset_params(self):
        std = getattr(self.config, "initializer_range", 0.02)

        for name in [
            "gate_proj_mole_A",
            "up_proj_mole_A",
            "down_proj_mole_A",
        ]:
            nn.init.kaiming_uniform_(getattr(self.mlp.experts, name), a=5**0.5)

        for name in [
            "gate_proj_mole_B",
            "up_proj_mole_B",
            "down_proj_mole_B",
        ]:
            nn.init.zeros_(getattr(self.mlp.experts, name))

        nn.init.xavier_uniform_(self.proj_down.weight)
        nn.init.zeros_(self.proj_down.bias)

        nn.init.zeros_(self.proj_up.weight)
        nn.init.zeros_(self.proj_up.bias)

        nn.init.normal_(self.mlp.router.weight, mean=0.0, std=std)
        nn.init.zeros_(self.mlp.router.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.proj_down(hidden_states)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.proj_up(hidden_states)
        return hidden_states
