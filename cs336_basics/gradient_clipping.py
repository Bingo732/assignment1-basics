import torch
from collections.abc import Iterable

@torch.no_grad()
def gradientClipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float
) -> None:
    total = 0
    params_list = []
    eps = 1e-6

    for p in parameters:
        if p.grad is None: continue
        total += ( p.grad ** 2 ).sum()
        params_list.append(p.grad)
    
    l2 = total ** 0.5
    if l2 < max_l2_norm: return None

    for g in params_list:
        g *= ( max_l2_norm / ( l2 + eps ) )