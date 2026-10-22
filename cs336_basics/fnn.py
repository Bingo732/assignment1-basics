import torch
from torch import nn, Tensor
from cs336_basics.linear import Linear
from cs336_basics.silu import SiLU
from jaxtyping import Float

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype:  torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        in_features: Float[Tensor, " ... d_model"],
    ) -> Float[Tensor, " ... d_model"]:
        return self.w2(SiLU(self.w1(in_features)) * self.w3(in_features))