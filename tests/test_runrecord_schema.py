"""RunRecord schema round-trip + version-pin test."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from qmlsurvey import RUNRECORD_SCHEMA_VERSION
from qmlsurvey.runner import RunRecord, TrainStats


def _make_record() -> RunRecord:
    stats = TrainStats(
        final_train_acc=0.9,
        final_test_acc=0.85,
        best_test_acc=0.88,
        final_loss=0.21,
        epochs=5,
        wall_time_s=1.5,
    )
    return RunRecord(
        schema_version=RUNRECORD_SCHEMA_VERSION,
        qmlsurvey_version="0.0.0-test",
        timestamp_utc="2026-01-01T00:00:00+00:00",
        backend="default.qubit",
        backend_kind="local_sim",
        task="parity",
        n_qubits=4,
        n_layers=2,
        shots=None,
        epochs=5,
        learning_rate=0.05,
        seed=0,
        estimated_cost_usd=0.0,
        quantum=stats,
        classical_baseline=stats,
        quantum_total_params=42,
        classical_total_params=44,
        environment={"python": "3.11.0", "platform": "test"},
    )


def test_schema_version_pinned():
    assert RUNRECORD_SCHEMA_VERSION == 1


def test_record_round_trips_through_json(tmp_path: Path):
    record = _make_record()
    out = record.write(out_dir=tmp_path)
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["schema_version"] == RUNRECORD_SCHEMA_VERSION
    assert payload["quantum"]["queue_wait_s"] == 0.0
    assert payload["experiment_group"] == ""
    # Round-trip equality (modulo dataclass <-> dict).
    assert payload == asdict(record)


def test_record_required_fields_present(tmp_path: Path):
    record = _make_record()
    payload = json.loads(record.write(out_dir=tmp_path).read_text())
    required = {
        "schema_version",
        "qmlsurvey_version",
        "timestamp_utc",
        "backend",
        "backend_kind",
        "task",
        "n_qubits",
        "n_layers",
        "shots",
        "epochs",
        "learning_rate",
        "seed",
        "estimated_cost_usd",
        "quantum",
        "classical_baseline",
        "quantum_total_params",
        "classical_total_params",
        "experiment_group",
        "notes",
        "environment",
    }
    assert required.issubset(payload.keys())
