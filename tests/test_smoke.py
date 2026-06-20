"""Smoke tests. Local simulators only — no AWS calls."""
from __future__ import annotations

import pytest
import torch

from qmlsurvey.backends import get_device
from qmlsurvey.baselines import MatchedMLP
from qmlsurvey.catalog import CATALOG, estimate_cost_usd
from qmlsurvey.model import HybridModel
from qmlsurvey.tasks import TASKS

LOCAL_SIM_BACKENDS = ["default.qubit", "lightning.qubit"]


def test_catalog_keys_present():
    for k in ["default.qubit", "sv1", "ionq_forte_1", "rigetti_cepheus"]:
        assert k in CATALOG


def test_cost_estimate_qpu_includes_task_fee():
    cost = estimate_cost_usd("ionq_forte_1", shots=100)
    assert cost == 100 * 0.08 + 0.30


def test_cost_estimate_local_is_zero():
    assert estimate_cost_usd("default.qubit", shots=1000) == 0.0


def test_cost_estimate_scales_with_task_count():
    # QPU: per-task cost (shots + task fee) scales linearly with n_tasks.
    one = estimate_cost_usd("ionq_forte_1", shots=100, n_tasks=1)
    assert estimate_cost_usd("ionq_forte_1", shots=100, n_tasks=4) == pytest.approx(one * 4)
    # Cloud sim: calibrated against the real SV1 run — 2 broadcast inputs
    # billed $0.0075 (2 tasks x 3 s minimum x $0.075/min). See sv1.md.
    assert estimate_cost_usd("sv1", shots=100, n_tasks=2) == pytest.approx(0.0075)


def test_hybrid_model_forward_and_backprop():
    dev = get_device("default.qubit", wires=4, shots=None)
    model = HybridModel(in_features=4, out_features=2, n_qubits=4, n_layers=2, device=dev)
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    logits = model(x)
    assert logits.shape == (8, 2)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    assert model.quantum_weights.grad is not None
    assert model.quantum_weights.grad.abs().sum().item() > 0


@pytest.mark.parametrize("backend", LOCAL_SIM_BACKENDS)
def test_local_sim_backends_run_forward(backend: str):
    # lightning.qubit has stricter requirements; this confirms it is wired up.
    dev = get_device(backend, wires=4, shots=None)
    model = HybridModel(in_features=4, out_features=2, n_qubits=4, n_layers=2, device=dev)
    x = torch.randn(4, 4)
    out = model(x)
    assert out.shape == (4, 2)


def test_parity_task_loads():
    Xtr, ytr, Xte, yte, meta = TASKS["parity"].load(seed=0)
    assert Xtr.shape[1] == meta["in_features"]
    assert ytr.dtype == torch.int64


def test_matched_mlp_param_count_close():
    m = MatchedMLP(4, 2, target_params=200)
    # Within a factor of 2 is good enough — it's a baseline, not a contract.
    assert 50 <= m.n_total_params <= 400
