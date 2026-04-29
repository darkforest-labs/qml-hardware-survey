"""sklearn make_moons. Cleanly nonlinear, classical baseline non-trivial."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NAME = "moons"


def load(seed: int = 0):
    X, y = make_moons(n_samples=300, noise=0.2, random_state=seed)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y.astype(np.int64), test_size=0.2, random_state=seed
    )
    meta = {
        "task": NAME,
        "in_features": 2,
        "out_features": 2,
        "kind": "classification",
        "n_train": len(Xtr),
        "n_test": len(Xte),
    }
    return (
        torch.from_numpy(Xtr),
        torch.from_numpy(ytr),
        torch.from_numpy(Xte),
        torch.from_numpy(yte),
        meta,
    )
