import torch
from torch import nn
from einops import einsum

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype:  torch.dtype | None = None,
    ):
        super().__init__()
        sigma = (2.0 / (in_features + out_features)) ** 0.5
        w = torch.empty(out_features, in_features, device=device, dtype=dtype)
        self.weight = nn.Parameter(nn.init.trunc_normal_(w, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma))

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")