"""Toy tasks. Each exposes `load() -> (X_train, y_train, X_test, y_test, meta)`."""

from . import moons, parity

TASKS = {
    "parity": parity,
    "moons": moons,
}

try:
    from . import mnist_pca  # noqa: F401  (optional; pulls sklearn datasets)

    TASKS["mnist_pca"] = mnist_pca
except Exception:  # pragma: no cover
    pass
