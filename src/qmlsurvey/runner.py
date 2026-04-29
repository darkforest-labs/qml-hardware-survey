"""
Runner: train the hybrid model on (backend, task), train a parameter-matched
classical baseline on the same data, and write a RunRecord JSON to results/.

Hard guards:
- QPU and cloud-sim runs require --max-cost-usd; estimated cost must fit and
  the user must type 'yes' interactively (skipped when --yes is passed).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import __version__
from .backends import describe, get_device
from .baselines import MatchedMLP
from .catalog import estimate_cost_usd
from .model import HybridModel
from .tasks import TASKS

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


@dataclass
class TrainStats:
    final_train_acc: float
    final_test_acc: float
    best_test_acc: float
    final_loss: float
    epochs: int
    wall_time_s: float


@dataclass
class RunRecord:
    qmlsurvey_version: str
    timestamp_utc: str
    backend: str
    backend_kind: str
    task: str
    n_qubits: int
    n_layers: int
    shots: int | None
    epochs: int
    learning_rate: float
    seed: int
    estimated_cost_usd: float
    quantum: TrainStats
    classical_baseline: TrainStats
    quantum_total_params: int
    classical_total_params: int
    notes: str = ""
    environment: dict[str, Any] = field(default_factory=dict)

    def write(self, out_dir: Path = RESULTS_DIR) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.timestamp_utc.replace(':', '-')}_{self.backend}_{self.task}"
        path = out_dir / f"{stem}.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def _train(
    model: nn.Module,
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xte: torch.Tensor,
    yte: torch.Tensor,
    epochs: int,
    lr: float,
) -> TrainStats:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_test = 0.0
    final_loss = float("nan")
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()
        opt.step()
        final_loss = float(loss.item())

        model.eval()
        with torch.no_grad():
            test_acc = (model(Xte).argmax(-1) == yte).float().mean().item()
            best_test = max(best_test, test_acc)
    wall = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        train_acc = (model(Xtr).argmax(-1) == ytr).float().mean().item()
        test_acc = (model(Xte).argmax(-1) == yte).float().mean().item()
    return TrainStats(
        final_train_acc=float(train_acc),
        final_test_acc=float(test_acc),
        best_test_acc=float(best_test),
        final_loss=final_loss,
        epochs=epochs,
        wall_time_s=wall,
    )


def _confirm_cost(backend: str, est_cost: float, max_cost: float, assume_yes: bool) -> None:
    info = describe(backend)
    if info.kind == "local_sim":
        return
    if est_cost > max_cost:
        sys.exit(
            f"Estimated cost ${est_cost:.4f} exceeds --max-cost-usd ${max_cost:.4f}. Aborting."
        )
    if assume_yes:
        return
    print(
        f"\nBackend       : {info.name} ({info.kind})\n"
        f"Estimated cost: ${est_cost:.4f}\n"
        f"Cost cap      : ${max_cost:.4f}\n"
    )
    reply = input("Type 'yes' to submit, anything else to abort: ").strip().lower()
    if reply != "yes":
        sys.exit("Aborted by user.")


def run(
    backend: str,
    task: str,
    epochs: int = 30,
    lr: float = 0.05,
    n_qubits: int = 4,
    n_layers: int = 2,
    shots: int | None = None,
    seed: int = 0,
    max_cost_usd: float = 0.0,
    assume_yes: bool = False,
    notes: str = "",
) -> RunRecord:
    if task not in TASKS:
        raise SystemExit(f"Unknown task {task!r}. Available: {sorted(TASKS)}")
    torch.manual_seed(seed)

    Xtr, ytr, Xte, yte, meta = TASKS[task].load(seed=seed)

    info = describe(backend)
    est_cost = estimate_cost_usd(backend, shots or 0)
    _confirm_cost(backend, est_cost, max_cost_usd, assume_yes)

    device = get_device(backend, wires=n_qubits, shots=shots)
    qmodel = HybridModel(
        in_features=meta["in_features"],
        out_features=meta["out_features"],
        n_qubits=n_qubits,
        n_layers=n_layers,
        device=device,
    )
    print(f"Quantum model parameters: {qmodel.n_total_params}")
    qstats = _train(qmodel, Xtr, ytr, Xte, yte, epochs=epochs, lr=lr)
    print(
        f"Quantum  : test={qstats.final_test_acc:.3f} best={qstats.best_test_acc:.3f} "
        f"time={qstats.wall_time_s:.2f}s"
    )

    cmodel = MatchedMLP(meta["in_features"], meta["out_features"], qmodel.n_total_params)
    cstats = _train(cmodel, Xtr, ytr, Xte, yte, epochs=epochs, lr=lr)
    print(
        f"Classical: test={cstats.final_test_acc:.3f} best={cstats.best_test_acc:.3f} "
        f"time={cstats.wall_time_s:.2f}s  (params={cmodel.n_total_params}, hidden={cmodel.hidden})"
    )

    record = RunRecord(
        qmlsurvey_version=__version__,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        backend=backend,
        backend_kind=info.kind,
        task=task,
        n_qubits=n_qubits,
        n_layers=n_layers,
        shots=shots,
        epochs=epochs,
        learning_rate=lr,
        seed=seed,
        estimated_cost_usd=est_cost,
        quantum=qstats,
        classical_baseline=cstats,
        quantum_total_params=qmodel.n_total_params,
        classical_total_params=cmodel.n_total_params,
        notes=notes,
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    )
    out = record.write()
    print(f"Wrote {out.relative_to(REPO_ROOT)}")
    return record


def main() -> None:
    p = argparse.ArgumentParser(prog="qmlsurvey")
    p.add_argument("--backend", required=True, help="Backend name; see catalog.CATALOG.")
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--shots", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-cost-usd", type=float, default=0.0)
    p.add_argument("--yes", action="store_true", help="Skip interactive confirmation.")
    p.add_argument("--notes", default="")
    args = p.parse_args()
    run(
        backend=args.backend,
        task=args.task,
        epochs=args.epochs,
        lr=args.lr,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        shots=args.shots,
        seed=args.seed,
        max_cost_usd=args.max_cost_usd,
        assume_yes=args.yes,
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
