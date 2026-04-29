"""Phase-1 trainability / barren-plateau sweep.

For each (n_qubits, n_layers) cell on the grid, draw `N_INITS` independent
random initialisations of `HybridModel.quantum_weights` and record the
gradient of a fixed scalar loss (cross-entropy on a fixed small batch) with
respect to those quantum weights. Summarise per cell:

- ``grad_var_mean``   — mean over parameter components of Var across inits
- ``grad_var_max``    — max  over parameter components of Var across inits
- ``grad_var_first``  — Var of the very first quantum weight (canonical
                        barren-plateau probe; some published curves use this)
- ``grad_abs_mean``   — mean |gradient| across inits and components

The encoder / decoder weights and the input batch are themselves seeded
once per cell so that variance measured here is genuinely parameter-init
variance, not data variance.

Output: one JSON per cell at
``results/phase1/trainability/nq{n_qubits}_nl{n_layers}.json`` plus a
combined ``results/phase1/trainability/summary.json``.

Backend: ``default.qubit`` analytic + ``diff_method="best"`` (which selects
backprop on `default.qubit`). This is the cheapest configuration that still
gives true gradients.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from qmlsurvey import RUNRECORD_SCHEMA_VERSION, __version__
from qmlsurvey.backends import get_device
from qmlsurvey.model import HybridModel
from qmlsurvey.runner import _collect_package_versions

# --- config -------------------------------------------------------------

N_QUBITS_GRID = (2, 4, 6, 8, 10, 12)
N_LAYERS_GRID = (1, 2, 4, 8)
N_INITS = 200
BATCH = 32
SEED = 0
BACKEND = "default.qubit"

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "phase1" / "trainability"


# --- helpers ------------------------------------------------------------

def _measure_cell(n_qubits: int, n_layers: int, n_inits: int) -> dict:
    """Return per-cell summary."""
    # Per-cell fixed batch and fixed encoder/decoder weights.
    cell_seed = SEED * 1000 + n_qubits * 31 + n_layers
    torch.manual_seed(cell_seed)
    np.random.seed(cell_seed)

    in_features = n_qubits  # square encoder shape across cells
    out_features = 2

    # Fixed batch.
    X = torch.randn(BATCH, in_features)
    y = torch.randint(0, 2, (BATCH,))
    crit = torch.nn.CrossEntropyLoss()

    # Build a "scaffold" model to lift the deterministic encoder/decoder
    # weights out, then reuse those across all inits.
    dev = get_device(BACKEND, wires=n_qubits, shots=None)
    scaffold = HybridModel(
        in_features=in_features,
        out_features=out_features,
        n_qubits=n_qubits,
        n_layers=n_layers,
        device=dev,
    )
    enc_state = {k: v.detach().clone() for k, v in scaffold.encoder.state_dict().items()}
    dec_state = {k: v.detach().clone() for k, v in scaffold.decoder.state_dict().items()}

    n_qparams = scaffold.n_quantum_params
    grads = np.empty((n_inits, n_qparams), dtype=np.float64)

    t0 = time.perf_counter()
    for i in range(n_inits):
        # Draw a fresh quantum-weight init. Using the standard
        # barren-plateau prescription: uniform on [-pi, pi]. The
        # constructor uses 0.1*randn; we override here for a more
        # discriminating probe.
        torch.manual_seed(cell_seed * 7919 + i)
        m = HybridModel(
            in_features=in_features,
            out_features=out_features,
            n_qubits=n_qubits,
            n_layers=n_layers,
            device=dev,
        )
        m.encoder.load_state_dict(enc_state)
        m.decoder.load_state_dict(dec_state)
        with torch.no_grad():
            m.quantum_weights.copy_(
                (torch.rand(n_layers, n_qubits, 3) * 2.0 - 1.0) * np.pi
            )
        m.zero_grad()
        out = m(X)
        loss = crit(out, y)
        loss.backward()
        g = m.quantum_weights.grad
        assert g is not None
        grads[i, :] = g.detach().cpu().numpy().reshape(-1)
    wall = time.perf_counter() - t0

    var_per_param = grads.var(axis=0, ddof=1)  # (n_qparams,)
    abs_grads = np.abs(grads)
    cell = {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_qparams": int(n_qparams),
        "n_inits": n_inits,
        "batch": BATCH,
        "backend": BACKEND,
        "grad_var_mean": float(var_per_param.mean()),
        "grad_var_max": float(var_per_param.max()),
        "grad_var_first": float(var_per_param[0]),
        "grad_abs_mean": float(abs_grads.mean()),
        "grad_abs_max": float(abs_grads.max()),
        "wall_time_s": wall,
        "seconds_per_init": wall / n_inits,
    }
    return cell


# --- main ---------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells: list[dict] = []
    grand_t0 = time.perf_counter()
    for n_qubits in N_QUBITS_GRID:
        for n_layers in N_LAYERS_GRID:
            print(f"--- n_qubits={n_qubits} n_layers={n_layers} ---", flush=True)
            cell = _measure_cell(n_qubits, n_layers, N_INITS)
            print(
                f"  var_mean={cell['grad_var_mean']:.3e} "
                f"var_max={cell['grad_var_max']:.3e} "
                f"|g|_mean={cell['grad_abs_mean']:.3e} "
                f"({cell['wall_time_s']:.1f}s, "
                f"{cell['seconds_per_init']*1000:.1f}ms/init)",
                flush=True,
            )
            cells.append(cell)
            cell_path = OUT_DIR / f"nq{n_qubits}_nl{n_layers}.json"
            cell_path.write_text(json.dumps(cell, indent=2))
    grand_wall = time.perf_counter() - grand_t0

    summary = {
        "schema_version": RUNRECORD_SCHEMA_VERSION,
        "qmlsurvey_version": __version__,
        "experiment_group": "phase1-trainability",
        "backend": BACKEND,
        "n_qubits_grid": list(N_QUBITS_GRID),
        "n_layers_grid": list(N_LAYERS_GRID),
        "n_inits_per_cell": N_INITS,
        "batch": BATCH,
        "seed": SEED,
        "init_distribution": "uniform[-pi, pi]",
        "loss": "cross_entropy(model(X_fixed), y_fixed)",
        "wall_time_s_total": grand_wall,
        "cells": cells,
        "environment": {"packages": _collect_package_versions()},
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")
    print(f"Total wall time: {grand_wall/60:.1f} min")

    # Pretty matrix.
    print("\n=== grad_var_mean (Var across inits, mean over qparams) ===")
    header = "n_qubits \\ n_layers  " + "  ".join(f"{nl:>10d}" for nl in N_LAYERS_GRID)
    print(header)
    by_cell = {(c["n_qubits"], c["n_layers"]): c for c in cells}
    for nq in N_QUBITS_GRID:
        row = [f"{nq:>2d}                   "]
        for nl in N_LAYERS_GRID:
            row.append(f"  {by_cell[(nq, nl)]['grad_var_mean']:.3e}")
        print("".join(row))


if __name__ == "__main__":
    main()
