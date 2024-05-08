import glob
import os
import re
from pathlib import Path

import torch

from .scaler import MinMaxScaler

def increment_path(path, exist_ok=False, sep="", mkdir=False):
    path = Path(path)
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = sorted(Path(path.parent).glob(f'{path.stem}{sep}*'))
        matches = [re.search(rf"{re.escape(path.stem)}{sep}(\d+)", d.name) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = Path(f"{path}{sep}{n}{suffix}")

    if mkdir and not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


class Normalizer:

    """
    Normalizes the data using Min-Max Scaling to the range (-1, 1).

    Attributes:
        scaler (MinMaxScaler): The scaler used for normalization and inverse transformation.
    """

    def __init__(self, data):
        flat = data.reshape(-1, data.shape[-1])
        self.scaler = MinMaxScaler((-1, 1), clip=True)
        self.scaler.fit(flat)

    def normalize(self, x):
        batch, seq, ch = x.shape
        x = x.reshape(-1, ch)
        return self.scaler.transform(x).reshape((batch, seq, ch))

    def unnormalize(self, x):
        batch, seq, ch = x.shape
        x = x.reshape(-1, ch)
        x = torch.clip(x, -1, 1)  # clip to force compatibility
        return self.scaler.inverse_transform(x).reshape((batch, seq, ch))


def vectorize_many(data):
    """
    Flattens and concatenates a list of tensors from shape (batch_size, seq_len, *, channels)
    to (batch_size, seq_len, -1) where -1 is the concatenated result of all flattened channels.

    Args:
        data (list of torch.Tensor): List of tensors to concatenate.

    Returns:
        torch.Tensor: Concatenated tensor.
    """
    batch_size, seq_len = data[0].shape[:2]
    out = [x.reshape(batch_size, seq_len, -1).contiguous() for x in data]
    return torch.cat(out, dim=2)

