"""
Runner: train the hybrid model on (backend, task), train a parameter-matched
classical baseline on the same data, and write a RunRecord JSON to results/.

Hard guards:
- QPU and cloud-sim runs require --max-cost-usd; estimated cost must fit and
  the user must type 'yes' interactively (skipped when --yes is passed).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pennylane as qml
import torch
from torch import nn

from . import RUNRECORD_SCHEMA_VERSION, __version__
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
    queue_wait_s: float = 0.0
    device_runtime_s: float = 0.0
    # v2: per-epoch trace. Each entry:
    #   {epoch, loss, train_acc, test_acc, grad_l2, param_l2, wall_s}
    # Empty list on v1 records and on legacy callers.
    epoch_trace: list[dict[str, float]] = field(default_factory=list)


_TRACKED_PACKAGES = (
    "pennylane",
    "pennylane-lightning",
    "torch",
    "numpy",
    "scikit-learn",
    "amazon-braket-sdk",
    "amazon-braket-pennylane-plugin",
)


def _collect_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


@dataclass
class RunRecord:
    schema_version: int
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
    experiment_group: str = ""
    notes: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    # v2 additive blocks. All default to empty so a legacy caller still works.
    git: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    circuit_fingerprint: dict[str, Any] = field(default_factory=dict)
    init_params_sha256: dict[str, str] = field(default_factory=dict)
    # predictions["quantum"] / ["classical"] = {y_true, y_pred, logits}
    predictions: dict[str, dict[str, Any]] = field(default_factory=dict)
    # billing populated only when backend_kind in {cloud_sim, qpu}; otherwise {}.
    billing: dict[str, Any] = field(default_factory=dict)

    def write(self, out_dir: Path = RESULTS_DIR) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.timestamp_utc.replace(':', '-')}_{self.backend}_{self.task}"
        path = out_dir / f"{stem}.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def _param_l2(model: nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        s += float(p.detach().pow(2).sum().item())
    return s ** 0.5


def _grad_l2(model: nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += float(p.grad.detach().pow(2).sum().item())
    return s ** 0.5


def _param_sha256(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        # Cast to float32 bytes for stable hashing across float64/float32 inits.
        h.update(p.detach().cpu().to(torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def _git_info() -> dict[str, Any]:
    """Best-effort git SHA + dirty flag. Returns {} if git is unavailable."""
    repo = REPO_ROOT
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, stderr=subprocess.DEVNULL
        ).decode()
        return {"sha": sha, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {}


def _hardware_info() -> dict[str, Any]:
    return {
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def _circuit_fingerprint(qmodel: HybridModel, sample_input: torch.Tensor) -> dict[str, Any]:
    """Capture gate counts + depth via qml.specs. Returns {} on failure."""
    try:
        specs_fn = qml.specs(qmodel._circuit)
        specs = specs_fn(sample_input, qmodel.quantum_weights)
        # The shape of `specs` varies across PennyLane releases. Be defensive
        # and only pull the fields we care about; coerce numerics to plain int.
        resources = specs.get("resources") if isinstance(specs, dict) else None
        if resources is not None:
            gate_types = getattr(resources, "gate_types", {}) or {}
            depth = int(getattr(resources, "depth", 0) or 0)
            num_gates = int(getattr(resources, "num_gates", 0) or 0)
            num_wires = int(getattr(resources, "num_wires", qmodel.n_qubits))
        else:
            gate_types = specs.get("gate_types", {}) if isinstance(specs, dict) else {}
            depth = int(specs.get("depth", 0)) if isinstance(specs, dict) else 0
            num_gates = sum(int(v) for v in gate_types.values())
            num_wires = qmodel.n_qubits
        return {
            "gate_counts": {str(k): int(v) for k, v in gate_types.items()},
            "depth": depth,
            "num_gates": num_gates,
            "num_wires": num_wires,
            "n_circuit_params": int(qmodel.quantum_weights.numel()),
            "interface": "torch",
            "diff_method": "best",
        }
    except Exception as e:  # noqa: BLE001 — fingerprint is best-effort.
        return {"error": f"{type(e).__name__}: {e}"}


def _predictions_block(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        y_pred = logits.argmax(-1)
    return {
        "y_true": y.detach().cpu().tolist(),
        "y_pred": y_pred.detach().cpu().tolist(),
        "logits": logits.detach().cpu().to(torch.float32).tolist(),
    }


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
    trace: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        epoch_t0 = time.perf_counter()
        model.train()
        opt.zero_grad()
        logits = model(Xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()
        grad_l2 = _grad_l2(model)
        opt.step()
        param_l2 = _param_l2(model)
        final_loss = float(loss.item())

        model.eval()
        with torch.no_grad():
            train_acc = (model(Xtr).argmax(-1) == ytr).float().mean().item()
            test_acc = (model(Xte).argmax(-1) == yte).float().mean().item()
            best_test = max(best_test, test_acc)
        trace.append(
            {
                "epoch": epoch,
                "loss": final_loss,
                "train_acc": float(train_acc),
                "test_acc": float(test_acc),
                "grad_l2": grad_l2,
                "param_l2": param_l2,
                "wall_s": time.perf_counter() - epoch_t0,
            }
        )
    wall = time.perf_counter() - t0

    final_train_acc = trace[-1]["train_acc"] if trace else 0.0
    final_test_acc = trace[-1]["test_acc"] if trace else 0.0
    return TrainStats(
        final_train_acc=float(final_train_acc),
        final_test_acc=float(final_test_acc),
        best_test_acc=float(best_test),
        final_loss=final_loss,
        epochs=epochs,
        wall_time_s=wall,
        epoch_trace=trace,
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
    experiment_group: str = "",
    out_dir: Path | None = None,
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
    qinit_hash = _param_sha256(qmodel)
    fingerprint = _circuit_fingerprint(qmodel, Xtr[:1])
    qstats = _train(qmodel, Xtr, ytr, Xte, yte, epochs=epochs, lr=lr)
    print(
        f"Quantum  : test={qstats.final_test_acc:.3f} best={qstats.best_test_acc:.3f} "
        f"time={qstats.wall_time_s:.2f}s"
    )
    qpreds = _predictions_block(qmodel, Xte, yte)

    cmodel = MatchedMLP(meta["in_features"], meta["out_features"], qmodel.n_total_params)
    cinit_hash = _param_sha256(cmodel)
    cstats = _train(cmodel, Xtr, ytr, Xte, yte, epochs=epochs, lr=lr)
    print(
        f"Classical: test={cstats.final_test_acc:.3f} best={cstats.best_test_acc:.3f} "
        f"time={cstats.wall_time_s:.2f}s  (params={cmodel.n_total_params}, hidden={cmodel.hidden})"
    )
    cpreds = _predictions_block(cmodel, Xte, yte)

    record = RunRecord(
        schema_version=RUNRECORD_SCHEMA_VERSION,
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
        experiment_group=experiment_group,
        notes=notes,
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _collect_package_versions(),
        },
        git=_git_info(),
        hardware=_hardware_info(),
        circuit_fingerprint=fingerprint,
        init_params_sha256={"quantum": qinit_hash, "classical": cinit_hash},
        predictions={"quantum": qpreds, "classical": cpreds},
        billing={},  # populated by AWS-aware callers post-run
    )
    out = record.write(out_dir=out_dir or RESULTS_DIR)
    try:
        print(f"Wrote {out.relative_to(REPO_ROOT)}")
    except ValueError:
        print(f"Wrote {out}")
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
    p.add_argument("--experiment-group", default="", help="Optional tag to group related runs.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output dir (default: results/).",
    )
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
        experiment_group=args.experiment_group,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
