import torch
from torch import Tensor
from jaxtyping import Float
def SiLU(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return x * torch.sigmoid(x)