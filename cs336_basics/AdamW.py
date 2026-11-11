from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9,0.999),
        eps=1e-8,
    ):
        if lr < 0: raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0: raise ValueError(f"Invalid eps: {weight_decay}")
        if betas[0] < 0 or betas[0] >= 1 or betas[1] < 0 or betas[1] >= 1: raise ValueError(f"Invalid betas: {betas}")
        if eps < 0: raise ValueError(f"Invalid eps: {eps}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure = None):
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None: continue

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                
                t = state.get("t")
                m = state.get("m")
                v = state.get("v")
                grad = p.grad

                lr_t = lr * ( 1 - betas[1] ** t ) ** 0.5 / ( 1 - betas[0] ** t )
                p -= lr * weight_decay * p
                m.mul_(betas[0]).add_(grad, alpha = 1 - betas[0])
                v.mul_(betas[1]).add_(grad ** 2, alpha = 1 - betas[1])
                p -= lr_t * m / ( v ** 0.5 + eps )

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v