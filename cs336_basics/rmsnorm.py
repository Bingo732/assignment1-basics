import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        w32 = self.weight.float()
        x32 = x.float()
        squre = x32 ** 2
        mean = squre.mean(dim=-1, keepdim=True)
        add = mean + self.eps
        rms = add.sqrt()
        return (x32 / rms * w32).to(x.dtype)