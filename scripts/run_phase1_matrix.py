"""Phase-1 cross-simulator matrix sweep.

Runs the Phase-0 reference config across every available local simulator and
every task. Free, local-only, no shots cap on default.qubit / lightning.qubit
(analytic), 1000 shots on braket.local.qubit (analytic mode unsupported).

Output: one JSON per (backend, task) under `results/phase1/cross_sim/`.
"""
from __future__ import annotations

from pathlib import Path

from qmlsurvey.runner import run
from qmlsurvey.tasks import TASKS

REFERENCE_BASE = dict(
    epochs=30,
    lr=0.05,
    n_qubits=4,
    n_layers=2,
    seed=0,
    assume_yes=True,
    experiment_group="phase1-cross-sim",
)

# (backend, shots) — analytic where supported, 1000 for braket.local.qubit.
BACKENDS: list[tuple[str, int | None]] = [
    ("default.qubit", None),
    ("lightning.qubit", None),
    ("braket.local.qubit", 1000),
]

TASKS_TO_RUN = ("parity", "moons", "mnist_pca")

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "phase1" / "cross_sim"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for backend, shots in BACKENDS:
        for task in TASKS_TO_RUN:
            if task not in TASKS:
                print(f"skip {backend}/{task}: task not available")
                continue
            print(f"\n=== {backend} / {task} (shots={shots}) ===")
            try:
                run(
                    backend=backend,
                    task=task,
                    shots=shots,
                    notes=f"Phase-1 cross-sim sweep on {backend}.",
                    out_dir=OUT_DIR,
                    **REFERENCE_BASE,
                )
            except Exception as exc:  # noqa: BLE001 — keep going across the matrix.
                print(f"skip {backend}/{task}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
