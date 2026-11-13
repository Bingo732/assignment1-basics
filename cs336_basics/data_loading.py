import numpy as np
import numpy.typing as npt
import torch

def dataLoading(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    steps = np.arange(context_length+1)
    points = starts[:, None] + steps[None, :]
    window = dataset[points]
    x = window[:,:context_length].astype(np.int64)
    y = window[:,1:].astype(np.int64)
    return torch.from_numpy(x).long().to(device), torch.from_numpy(y).long().to(device)