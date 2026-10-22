import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        freq = torch.arange(0, d_k, 2, device=device)
        angles = 1 / ( theta ** ( freq / d_k ) )
        pos = torch.arange(0, max_seq_len, 1, device=device)
        table = pos[:, None] * angles[None, :]
        table_cos = table.cos()
        table_sin = table.sin()
        self.register_buffer("cos", table_cos, persistent=False)
        self.register_buffer("sin", table_sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        cos_i  = self.cos[token_positions,:]
        sin_i  = self.sin[token_positions,:]
        out_even = x_even * cos_i - x_odd * sin_i
        out_odd  = x_even * sin_i + x_odd * cos_i
        stacked = torch.stack([out_even, out_odd], dim=-1)
        result = stacked.flatten(-2)
        return result
