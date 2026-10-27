import torch
from torch import Tensor, nn
from jaxtyping import Int, Float
from einops import einsum, rearrange
from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.sdpa import scaledDotProductAttention

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype:  torch.dtype | None = None,
        rope: RotaryPositionalEmbedding | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.rope = rope

    def forward(
        self,
        in_features: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        q = rearrange(self.q_proj(in_features), " ... sequence_length (heads d_head) -> ... heads sequence_length d_head", heads=self.num_heads)
        k = rearrange(self.k_proj(in_features), " ... sequence_length (heads d_head) -> ... heads sequence_length d_head", heads=self.num_heads)
        v = rearrange(self.v_proj(in_features), " ... sequence_length (heads d_head) -> ... heads sequence_length d_head", heads=self.num_heads)
        if self.rope is not None:
            assert token_positions is not None
            q = self.rope(q, token_positions.unsqueeze(-2))
            k = self.rope(k, token_positions.unsqueeze(-2))
        seq = torch.arange(in_features.shape[-2], device=in_features.device)
        mask = seq[:, None] >= seq[None, :]
        attn = scaledDotProductAttention(q, k, v, mask)
        attn = rearrange(attn, "... heads sequence_length d_head -> ... sequence_length (heads d_head)")
        return self.output_proj(attn)
