"""MNIST 0-vs-1, PCA-reduced to 4 features. Less trivial than Iris."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NAME = "mnist_pca"


def load(seed: int = 0, n_components: int = 4, n_per_class: int = 200):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_all, y_all = mnist.data, mnist.target.astype(int)
    mask = (y_all == 0) | (y_all == 1)
    X_all, y_all = X_all[mask], y_all[mask]

    rng = np.random.default_rng(seed)
    idx0 = np.where(y_all == 0)[0]
    idx1 = np.where(y_all == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    keep = np.concatenate([idx0[:n_per_class], idx1[:n_per_class]])
    rng.shuffle(keep)
    X, y = X_all[keep], y_all[keep]

    X = PCA(n_components=n_components, random_state=seed).fit_transform(X)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y.astype(np.int64), test_size=0.2, random_state=seed
    )
    meta = {
        "task": NAME,
        "in_features": n_components,
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
