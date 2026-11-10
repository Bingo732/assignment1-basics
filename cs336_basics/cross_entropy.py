from jaxtyping import Float, Int
import torch
from torch import Tensor

def crossEntropy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
) -> Float[Tensor, ""]:
    m = inputs.amax(dim=-1)
    row_idx = torch.arange(0, inputs.shape[0], 1, device=inputs.device)
    z_y = inputs[row_idx, targets]
    y = inputs - m.unsqueeze(-1)
    result = - z_y + m + torch.log(torch.exp(y).sum(dim=-1))
    return result.mean()