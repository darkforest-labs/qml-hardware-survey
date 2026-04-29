"""Generate Phase-0 reference RunRecords on default.qubit.

Writes one JSON per task to `results/phase0/`. Free, local-only.
"""
from __future__ import annotations

from pathlib import Path

from qmlsurvey.runner import run
from qmlsurvey.tasks import TASKS

REFERENCE_CONFIG = dict(
    backend="default.qubit",
    epochs=30,
    lr=0.05,
    n_qubits=4,
    n_layers=2,
    shots=None,
    seed=0,
    assume_yes=True,
    experiment_group="phase0-reference",
)

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "phase0"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for task in ("parity", "moons", "mnist_pca"):
        if task not in TASKS:
            print(f"skip {task}: not available (likely missing optional deps / network)")
            continue
        print(f"\n=== {task} ===")
        try:
            run(task=task, notes="Phase-0 reference run.", out_dir=OUT_DIR, **REFERENCE_CONFIG)
        except Exception as exc:  # noqa: BLE001 — phase-0 script: keep going on any failure.
            print(f"skip {task}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
