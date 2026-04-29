"""Phase-1 shot-noise sweep on parity.

Strategy (option 1 from `docs/integration-notes/local-sims.md`):

1. Train `HybridModel` once on `default.qubit` in analytic mode (the same
   reference config used elsewhere). This gives a single trained
   `state_dict`.
2. Evaluate the trained weights on the parity test set at multiple shot
   counts on every local simulator that supports forward inference,
   including `braket.local.qubit` (for which we cannot do gradients but can
   do forward-only).
3. Repeat each (backend, shots) combination over `N_TRIALS` independent
   shot draws to estimate the spread caused by finite sampling.

Outputs a single JSON summary at
`results/phase1/shot_noise/parity_shot_noise.json`.

This intentionally does NOT write standard `RunRecord`s — the noise sweep
is not a training run. We keep the reference RunRecords in
`results/phase1/cross_sim/` and add a separate, smaller artifact here.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from qmlsurvey import RUNRECORD_SCHEMA_VERSION, __version__
from qmlsurvey.backends import get_device
from qmlsurvey.model import HybridModel
from qmlsurvey.runner import _collect_package_versions, _train
from qmlsurvey.tasks import TASKS

# --- config -------------------------------------------------------------

TASK = "parity"
SEED = 0
N_QUBITS = 4
N_LAYERS = 2
EPOCHS = 30
LR = 0.05
N_TRIALS = 10  # independent shot draws per (backend, shots) cell

# (backend, shots). `None` = analytic. braket.local.qubit only supports finite shots.
EVAL_CELLS: list[tuple[str, int | None]] = [
    ("default.qubit", None),
    ("default.qubit", 5000),
    ("default.qubit", 1000),
    ("default.qubit", 500),
    ("default.qubit", 100),
    ("braket.local.qubit", 5000),
    ("braket.local.qubit", 1000),
    ("braket.local.qubit", 500),
    ("braket.local.qubit", 100),
]

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "phase1" / "shot_noise"
OUT_PATH = OUT_DIR / "parity_shot_noise.json"


# --- helpers ------------------------------------------------------------

def _build_model(backend: str, shots: int | None, in_f: int, out_f: int) -> HybridModel:
    dev = get_device(backend, wires=N_QUBITS, shots=shots)
    return HybridModel(
        in_features=in_f,
        out_features=out_f,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        device=dev,
    )


def _accuracy(model: HybridModel, X: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=-1)
        return float((preds == y).float().mean().item())


# --- main ---------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte, meta = TASKS[TASK].load(seed=SEED)

    # 1. Train reference model on default.qubit analytic.
    print("Training reference model on default.qubit (analytic)...")
    torch.manual_seed(SEED)
    ref = _build_model("default.qubit", None, meta["in_features"], meta["out_features"])
    ref_stats = _train(ref, Xtr, ytr, Xte, yte, epochs=EPOCHS, lr=LR)
    print(
        f"  trained: test_acc={ref_stats.final_test_acc:.4f} "
        f"loss={ref_stats.final_loss:.6f} time={ref_stats.wall_time_s:.2f}s"
    )
    ref_state = copy.deepcopy(ref.state_dict())

    # 2. Evaluate at each (backend, shots) cell, N_TRIALS independent draws.
    cells_out = []
    for backend, shots in EVAL_CELLS:
        label = f"{backend}/shots={shots}"
        print(f"\nEvaluating {label} ({N_TRIALS} trials)...")
        accs: list[float] = []
        per_trial_seconds: list[float] = []
        try:
            for trial in range(N_TRIALS):
                # Re-seed PennyLane's sampler so the trials are independent
                # but the script as a whole is deterministic.
                np.random.seed(SEED + 1 + trial)
                torch.manual_seed(SEED + 1 + trial)
                eval_model = _build_model(
                    backend, shots, meta["in_features"], meta["out_features"]
                )
                eval_model.load_state_dict(ref_state)
                eval_model.eval()
                t0 = time.perf_counter()
                acc = _accuracy(eval_model, Xte, yte)
                per_trial_seconds.append(time.perf_counter() - t0)
                accs.append(acc)
                print(f"  trial {trial:2d}: acc={acc:.4f}  ({per_trial_seconds[-1]:.2f}s)")
        except Exception as exc:  # noqa: BLE001 — record the failure, keep going
            cells_out.append(
                {
                    "backend": backend,
                    "shots": shots,
                    "error": f"{type(exc).__name__}: {exc}",
                    "trials": accs,
                }
            )
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            continue

        cells_out.append(
            {
                "backend": backend,
                "shots": shots,
                "n_trials": N_TRIALS,
                "acc_mean": mean(accs),
                "acc_std": stdev(accs) if len(accs) > 1 else 0.0,
                "acc_min": min(accs),
                "acc_max": max(accs),
                "trials": accs,
                "trial_seconds_mean": mean(per_trial_seconds),
            }
        )

    # 3. Write summary.
    summary = {
        "schema_version": RUNRECORD_SCHEMA_VERSION,
        "qmlsurvey_version": __version__,
        "experiment_group": "phase1-shot-noise",
        "task": TASK,
        "seed": SEED,
        "n_qubits": N_QUBITS,
        "n_layers": N_LAYERS,
        "epochs": EPOCHS,
        "lr": LR,
        "n_trials": N_TRIALS,
        "reference": {
            "backend": "default.qubit",
            "shots": None,
            "final_test_acc": ref_stats.final_test_acc,
            "final_loss": ref_stats.final_loss,
            "wall_time_s": ref_stats.wall_time_s,
        },
        "cells": cells_out,
        "environment": {"packages": _collect_package_versions()},
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_PATH}")

    # Pretty table.
    print("\n=== shot-noise summary (parity, trained reference weights) ===")
    print(f"{'backend':22s} {'shots':>6s}  {'mean':>7s}  {'std':>7s}  {'min':>5s}  {'max':>5s}")
    for c in cells_out:
        if "error" in c:
            print(f"{c['backend']:22s} {str(c['shots']):>6s}  ERROR: {c['error']}")
        else:
            print(
                f"{c['backend']:22s} {str(c['shots']):>6s}  "
                f"{c['acc_mean']:.4f}  {c['acc_std']:.4f}  "
                f"{c['acc_min']:.3f}  {c['acc_max']:.3f}"
            )


if __name__ == "__main__":
    main()
