import numpy as np
import torch


def get_batch(path, batch_size, seq_len, device, rng):
    """Random windows from a flat uint16 token file. Target = input shifted by one token."""
    data = np.memmap(path, dtype=np.uint16, mode='r')
    ix = rng.integers(0, len(data) - seq_len - 1, size=batch_size)
    x = torch.from_numpy(np.stack([data[i:i + seq_len].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + seq_len].astype(np.int64) for i in ix]))
    return x.to(device), y.to(device)
