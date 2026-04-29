"""Determinism test: same seed -> bit-identical training stats on local sims."""
from __future__ import annotations

from pathlib import Path

import pytest

from qmlsurvey.runner import run


@pytest.mark.parametrize("backend", ["default.qubit", "lightning.qubit"])
def test_run_is_deterministic(backend: str, tmp_path: Path):
    kwargs = dict(
        backend=backend,
        task="parity",
        epochs=3,
        lr=0.05,
        n_qubits=4,
        n_layers=2,
        shots=None,
        seed=0,
        assume_yes=True,
        out_dir=tmp_path,
    )
    a = run(**kwargs)
    b = run(**kwargs)
    assert a.quantum.final_test_acc == b.quantum.final_test_acc
    assert a.quantum.final_loss == b.quantum.final_loss
    assert a.classical_baseline.final_test_acc == b.classical_baseline.final_test_acc
    assert a.classical_baseline.final_loss == b.classical_baseline.final_loss
