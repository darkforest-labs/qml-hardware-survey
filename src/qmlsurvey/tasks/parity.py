"""n-bit parity. Deterministic, classical baseline (1-layer linear) cannot solve it."""
from __future__ import annotations

import numpy as np
import torch

NAME = "parity"
N_BITS = 4


def load(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(256, N_BITS)).astype(np.float32)
    y = (X.sum(axis=1) % 2).astype(np.int64)
    # 80/20 split
    idx = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    tr, te = idx[:cut], idx[cut:]
    meta = {
        "task": NAME,
        "in_features": N_BITS,
        "out_features": 2,
        "kind": "classification",
        "n_train": len(tr),
        "n_test": len(te),
    }
    return (
        torch.from_numpy(X[tr]),
        torch.from_numpy(y[tr]),
        torch.from_numpy(X[te]),
        torch.from_numpy(y[te]),
        meta,
    )
