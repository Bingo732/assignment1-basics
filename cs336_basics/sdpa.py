from torch import Tensor
from jaxtyping import Bool, Float
from einops import einsum
from cs336_basics.softmax import softmax

def scaledDoProductAttention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scores = scores / (Q.shape[-1] ** 0.5)
    if mask is not None: scores = scores.masked_fill(~mask, float("-inf"))
    scores = softmax(scores, dim=-1)
    return einsum(scores, V, "... queries keys, ... keys d_v -> ... queries d_v")