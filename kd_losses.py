from __future__ import annotations

import gc

import torch


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def masked_tokens(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return x[attention_mask.to(x.device).bool()]


def normalized_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    return (pred - target).pow(2).sum() / (target.pow(2).sum() + eps)


def mean_cosine(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    pred = pred / (pred.norm(dim=-1, keepdim=True) + eps)
    target = target / (target.norm(dim=-1, keepdim=True) + eps)
    return (pred * target).sum(dim=-1).mean()
