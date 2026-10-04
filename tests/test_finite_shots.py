"""Finite-shot training path: broadcast_expand sidesteps PennyLane #4462.

Local simulators only — no AWS calls. Also pins the task-count estimator to
what the device actually executes, since on Braket every execution is billed.
"""
from __future__ import annotations

import pennylane as qml
import pytest
import torch

from qmlsurvey.backends import get_device
from qmlsurvey.catalog import estimate_cost_usd, estimate_task_count
from qmlsurvey.model import HybridModel, n_shift_params
from qmlsurvey.runner import _train
from qmlsurvey.tasks import TASKS

N_TRAIN, N_TEST = 8, 4


def _data():
    torch.manual_seed(0)
    return (
        torch.rand(N_TRAIN, 4),
        torch.randint(0, 2, (N_TRAIN,)),
        torch.rand(N_TEST, 4),
        torch.randint(0, 2, (N_TEST,)),
    )


def _braket_local_available() -> bool:
    try:
        get_device("braket.local.qubit", wires=2, shots=10)
        return True
    except Exception:  # noqa: BLE001 — plugin optional in CI.
        return False


FINITE_SHOT_BACKENDS = ["default.qubit"] + (
    ["braket.local.qubit"] if _braket_local_available() else []
)


def test_analytic_default_qubit_is_not_expanded():
    dev = get_device("default.qubit", wires=4, shots=None)
    model = HybridModel(4, 2, n_qubits=4, n_layers=2, device=dev)
    assert model.broadcast_expanded is False


@pytest.mark.parametrize("backend", FINITE_SHOT_BACKENDS)
def test_finite_shot_training_step_succeeds(backend: str):
    Xtr, ytr, _, _ = _data()
    dev = get_device(backend, wires=4, shots=200)
    model = HybridModel(4, 2, n_qubits=4, n_layers=2, device=dev)
    assert model.broadcast_expanded is True
    loss = torch.nn.functional.cross_entropy(model(Xtr), ytr)
    loss.backward()  # raised NotImplementedError (#4462) before broadcast_expand
    assert model.quantum_weights.grad.abs().sum().item() > 0
    assert model.encoder.weight.grad.abs().sum().item() > 0


def test_task_count_matches_tracked_executions():
    """estimate_task_count must equal what one epoch of _train really executes."""
    Xtr, ytr, Xte, yte = _data()
    dev = get_device("default.qubit", wires=4, shots=200)
    model = HybridModel(4, 2, n_qubits=4, n_layers=2, device=dev)
    with qml.Tracker(dev) as tracker:
        _train(model, Xtr, ytr, Xte, yte, epochs=1, lr=0.05)
    executed = tracker.totals["executions"]
    # estimate_task_count includes the runner's final predictions pass (n_test),
    # which _train itself does not perform.
    assert executed == estimate_task_count(N_TRAIN, N_TEST, 4, 2, epochs=1) - N_TEST
    p = n_shift_params(4, 2)
    assert executed == N_TRAIN * (1 + 2 * p) + N_TRAIN + N_TEST


def test_reference_run_cost_is_large_on_hardware():
    # Parity reference config, 1 epoch, real split sizes from tasks/parity.py.
    # This is the number the roadmap's Phase-4 budget must be checked against.
    _, _, _, _, meta = TASKS["parity"].load(seed=0)
    n_tr, n_te = meta["n_train"], meta["n_test"]
    n_tasks = estimate_task_count(n_tr, n_te, n_qubits=4, n_layers=2, epochs=1)
    assert n_tasks == n_tr * (1 + 2 * 28) + n_tr + n_te + n_te
    # One epoch on the cheapest QPU at 200 shots is already thousands of dollars.
    assert estimate_cost_usd("rigetti_cepheus", shots=200, n_tasks=n_tasks) > 1000.0


def test_subset_option_shrinks_split_and_task_estimate(tmp_path):
    from qmlsurvey.runner import run

    rec = run(
        backend="default.qubit",
        task="parity",
        epochs=1,
        shots=None,
        seed=0,
        assume_yes=True,
        out_dir=tmp_path,
        n_train_subset=2,
        n_test_subset=2,
    )
    assert (rec.n_train, rec.n_test) == (2, 2)
    assert rec.estimated_tasks == estimate_task_count(2, 2, 4, 2, epochs=1) == 120
    assert len(rec.predictions["quantum"]["y_true"]) == 2
