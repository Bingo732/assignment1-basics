import torch
from torch import nn, Tensor
from jaxtyping import Float
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.mha import MultiHeadSelfAttention
from cs336_basics.ffn import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype:  torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, rope=rope, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        in_features: Float[Tensor, " batch sequence_length d_model"],
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        token_positions = torch.arange(0, in_features.shape[-2], 1, device=in_features.device)
        add1 = in_features + self.attn(in_features=self.ln1(in_features), token_positions=token_positions)
        return add1 + self.ffn(self.ln2(add1))
