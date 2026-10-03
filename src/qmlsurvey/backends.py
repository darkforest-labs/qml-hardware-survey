"""
Thin wrapper that returns a PennyLane `Device` given a backend name from
`catalog.CATALOG`. Keeps the rest of the code base ignorant of which SDK it
is talking to.
"""
from __future__ import annotations

import pennylane as qml

from .catalog import CATALOG, BackendInfo


class BackendUnavailable(RuntimeError):
    pass


def get_device(backend: str, wires: int, shots: int | None = None):
    """
    Return a PennyLane device for `backend`.

    `shots=None` => analytic mode (only valid on local simulators).
    """
    if backend not in CATALOG:
        raise KeyError(f"Unknown backend {backend!r}. See catalog.CATALOG.")
    info = CATALOG[backend]

    if info.name == "default.qubit":
        return qml.device("default.qubit", wires=wires, shots=shots)
    if info.name == "lightning.qubit":
        return qml.device("lightning.qubit", wires=wires, shots=shots)
    if info.name == "braket.local.qubit":
        try:
            return qml.device("braket.local.qubit", wires=wires, shots=shots or 1000)
        except qml.DeviceError as e:
            raise BackendUnavailable(
                "amazon-braket-pennylane-plugin not installed. "
                "Install with: pip install qmlsurvey[braket]"
            ) from e

    # Cloud sims and QPUs all go through braket.aws.qubit.
    if info.kind in ("cloud_sim", "qpu"):
        if shots is None:
            raise ValueError(f"{backend} requires explicit shots > 0")
        try:
            return qml.device(
                "braket.aws.qubit",
                device_arn=info.arn,
                wires=wires,
                shots=shots,
            )
        except qml.DeviceError as e:
            raise BackendUnavailable(
                "amazon-braket-pennylane-plugin not installed. "
                "Install with: pip install qmlsurvey[braket]"
            ) from e

    raise BackendUnavailable(f"No device factory for {backend!r}")


def describe(backend: str) -> BackendInfo:
    return CATALOG[backend]


def device_snapshot(backend: str) -> dict:
    """Best-effort, free, read-only snapshot of a Braket device's current state.

    Returns ``{}`` for local simulators. For cloud sims and QPUs, returns
    status, ``is_available`` (inside an execution window right now), queue
    depth, qubit count, listed price and the UTC execution windows. Never
    raises: any failure is recorded under ``"error"`` so a RunRecord can still
    be written. Pattern borrowed from rydberg-playground's ``aquila_snapshot``.
    """
    info = CATALOG[backend]
    if info.kind == "local_sim" or not info.arn:
        return {}
    from datetime import datetime, timezone

    snap: dict = {
        "arn": info.arn,
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        from braket.aws import AwsDevice

        dev = AwsDevice(info.arn)
        snap["status"] = str(dev.status)
        snap["is_available"] = bool(dev.is_available)
        props = dev.properties
        svc = getattr(props, "service", None)
        if svc is not None:
            snap["execution_windows_utc"] = [
                {
                    "day": str(getattr(w.executionDay, "value", w.executionDay)),
                    "start": str(w.windowStartHour),
                    "end": str(w.windowEndHour),
                }
                for w in (svc.executionWindows or [])
            ]
            cost = getattr(svc, "deviceCost", None)
            if cost is not None:
                snap["listed_cost"] = {"price": float(cost.price), "unit": str(cost.unit)}
        paradigm = getattr(props, "paradigm", None)
        qc = getattr(paradigm, "qubitCount", None)
        if qc is not None:
            snap["qubit_count"] = int(qc)
        try:
            q = dev.queue_depth()
            snap["queue_depth"] = {
                str(getattr(k, "value", k)): v for k, v in q.quantum_tasks.items()
            }
        except Exception as e:  # noqa: BLE001 — queue depth is nice-to-have.
            snap["queue_error"] = repr(e)
    except Exception as e:  # noqa: BLE001 — snapshot must never block a run.
        snap["error"] = repr(e)
    return snap
