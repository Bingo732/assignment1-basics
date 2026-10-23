import torch
from torch import Tensor
from jaxtyping import Float

def softmax(
    in_features: Float[Tensor, " ..."],
    dim: int
) -> Float[Tensor, " ..."]:
    x_max = torch.amax(in_features, dim=dim, keepdim=True)
    subtract = in_features - x_max
    x_exp = subtract.exp()
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum
