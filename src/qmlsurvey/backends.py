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
