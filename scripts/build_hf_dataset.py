"""Flatten ``results/**/*.json`` into Hugging Face-friendly JSONL splits.

Three outputs under ``datasets/qml-hardware-survey/data/``:

* ``runs.jsonl``        — one row per RunRecord (flat hyperparams + final metrics
  + git/hardware/circuit_fingerprint/billing).
* ``epochs.jsonl``      — long-format per-epoch trace, joined to a run via
  ``run_id`` and tagged with ``model_kind`` (``quantum``/``classical``).
* ``predictions.jsonl`` — long-format per-test-sample predictions with
  ``run_id``, ``model_kind``, ``sample_idx``, ``y_true``, ``y_pred``,
  ``logits``.

This script is **stdlib only** and accepts both schema v1 and v2 records;
v1 records contribute a runs row but emit no epochs/predictions (the new
fields default to empty and are simply skipped — the gap stays honest).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "datasets" / "qml-hardware-survey" / "data"


def _iter_records() -> Iterator[tuple[str, dict[str, Any]]]:
    for path in sorted(RESULTS_DIR.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or "schema_version" not in data:
            continue
        run_id = path.stem
        yield run_id, data


def _flat_run_row(run_id: str, rec: dict[str, Any]) -> dict[str, Any]:
    q = rec.get("quantum", {}) or {}
    c = rec.get("classical_baseline", {}) or {}
    fp = rec.get("circuit_fingerprint", {}) or {}
    git = rec.get("git", {}) or {}
    hw = rec.get("hardware", {}) or {}
    billing = rec.get("billing", {}) or {}
    return {
        "run_id": run_id,
        "schema_version": rec.get("schema_version"),
        "qmlsurvey_version": rec.get("qmlsurvey_version"),
        "timestamp_utc": rec.get("timestamp_utc"),
        "experiment_group": rec.get("experiment_group", ""),
        "task": rec.get("task"),
        "backend": rec.get("backend"),
        "backend_kind": rec.get("backend_kind"),
        "n_qubits": rec.get("n_qubits"),
        "n_layers": rec.get("n_layers"),
        "shots": rec.get("shots"),
        "epochs": rec.get("epochs"),
        "learning_rate": rec.get("learning_rate"),
        "seed": rec.get("seed"),
        "estimated_cost_usd": rec.get("estimated_cost_usd"),
        "quantum_total_params": rec.get("quantum_total_params"),
        "classical_total_params": rec.get("classical_total_params"),
        "quantum_final_train_acc": q.get("final_train_acc"),
        "quantum_final_test_acc": q.get("final_test_acc"),
        "quantum_best_test_acc": q.get("best_test_acc"),
        "quantum_final_loss": q.get("final_loss"),
        "quantum_wall_time_s": q.get("wall_time_s"),
        "quantum_queue_wait_s": q.get("queue_wait_s"),
        "quantum_device_runtime_s": q.get("device_runtime_s"),
        "classical_final_train_acc": c.get("final_train_acc"),
        "classical_final_test_acc": c.get("final_test_acc"),
        "classical_best_test_acc": c.get("best_test_acc"),
        "classical_final_loss": c.get("final_loss"),
        "classical_wall_time_s": c.get("wall_time_s"),
        "circuit_depth": fp.get("depth"),
        "circuit_num_gates": fp.get("num_gates"),
        "circuit_n_params": fp.get("n_circuit_params"),
        "circuit_gate_counts_json": (
            json.dumps(fp.get("gate_counts"), sort_keys=True) if fp.get("gate_counts") else None
        ),
        "git_sha": git.get("sha"),
        "git_dirty": git.get("dirty"),
        "hardware_processor": hw.get("processor"),
        "hardware_cpu_count": hw.get("cpu_count"),
        "hardware_torch_cuda_available": hw.get("torch_cuda_available"),
        "billing_task_arn": billing.get("task_arn"),
        "billing_billed_duration_ms": billing.get("billed_duration_ms"),
        "billing_billed_cost_usd": billing.get("billed_cost_usd"),
        "billing_s3_result_uri": billing.get("s3_result_uri"),
        "notes": rec.get("notes", ""),
    }


def _epoch_rows(run_id: str, rec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for kind in ("quantum", "classical_baseline"):
        block = rec.get(kind, {}) or {}
        for entry in block.get("epoch_trace", []) or []:
            yield {
                "run_id": run_id,
                "model_kind": "quantum" if kind == "quantum" else "classical",
                **entry,
            }


def _prediction_rows(run_id: str, rec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    preds = rec.get("predictions", {}) or {}
    for kind in ("quantum", "classical"):
        block = preds.get(kind) or {}
        y_true = block.get("y_true") or []
        y_pred = block.get("y_pred") or []
        logits = block.get("logits") or []
        for idx, (t, p) in enumerate(zip(y_true, y_pred)):
            row = {
                "run_id": run_id,
                "model_kind": kind,
                "sample_idx": idx,
                "y_true": t,
                "y_pred": p,
            }
            if idx < len(logits):
                row["logits"] = logits[idx]
            yield row


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs_path = OUT_DIR / "runs.jsonl"
    epochs_path = OUT_DIR / "epochs.jsonl"
    preds_path = OUT_DIR / "predictions.jsonl"

    n_runs = n_epoch_rows = n_pred_rows = 0
    with (
        runs_path.open("w", encoding="utf-8") as runs_f,
        epochs_path.open("w", encoding="utf-8") as epochs_f,
        preds_path.open("w", encoding="utf-8") as preds_f,
    ):
        for run_id, rec in _iter_records():
            runs_f.write(json.dumps(_flat_run_row(run_id, rec)) + "\n")
            n_runs += 1
            for row in _epoch_rows(run_id, rec):
                epochs_f.write(json.dumps(row) + "\n")
                n_epoch_rows += 1
            for row in _prediction_rows(run_id, rec):
                preds_f.write(json.dumps(row) + "\n")
                n_pred_rows += 1

    print(
        f"Wrote {runs_path.relative_to(REPO_ROOT)} ({n_runs} runs), "
        f"{epochs_path.relative_to(REPO_ROOT)} ({n_epoch_rows} epoch rows), "
        f"{preds_path.relative_to(REPO_ROOT)} ({n_pred_rows} prediction rows)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
