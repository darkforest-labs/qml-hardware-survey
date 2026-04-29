"""
Static metadata for known Braket backends.

This is a hand-maintained snapshot of AWS Braket device info. AWS is the source
of truth at runtime (`AwsDevice.properties`); this module exists only so we can
estimate costs and gate operations without a network round-trip.

Update as AWS changes pricing or device availability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DeviceKind = Literal["local_sim", "cloud_sim", "qpu"]


@dataclass(frozen=True)
class BackendInfo:
    name: str
    kind: DeviceKind
    arn: str | None  # None for purely local
    qubits: int
    cost_per_shot_usd: float = 0.0
    cost_per_minute_usd: float = 0.0
    per_task_fee_usd: float = 0.0
    region: str = "local"
    notes: str = ""


CATALOG: dict[str, BackendInfo] = {
    # ---- Local (free) ----
    "default.qubit": BackendInfo(
        name="default.qubit",
        kind="local_sim",
        arn=None,
        qubits=24,
        notes="PennyLane reference simulator. Supports broadcasting + backprop.",
    ),
    "lightning.qubit": BackendInfo(
        name="lightning.qubit",
        kind="local_sim",
        arn=None,
        qubits=28,
        notes="C++ accelerated PennyLane simulator. adjoint diff supported.",
    ),
    "braket.local.qubit": BackendInfo(
        name="braket.local.qubit",
        kind="local_sim",
        arn=None,
        qubits=25,
        notes="Braket local SDK simulator. Useful as a Braket-fidelity sanity check.",
    ),
    # ---- Cloud simulators ----
    "sv1": BackendInfo(
        name="sv1",
        kind="cloud_sim",
        arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
        qubits=34,
        cost_per_minute_usd=0.075,
        region="all",
        notes="State vector simulator. Good first cloud step.",
    ),
    "dm1": BackendInfo(
        name="dm1",
        kind="cloud_sim",
        arn="arn:aws:braket:::device/quantum-simulator/amazon/dm1",
        qubits=17,
        cost_per_minute_usd=0.075,
        region="all",
        notes="Density matrix simulator (noise modelling).",
    ),
    "tn1": BackendInfo(
        name="tn1",
        kind="cloud_sim",
        arn="arn:aws:braket:::device/quantum-simulator/amazon/tn1",
        qubits=50,
        cost_per_minute_usd=0.275,
        region="all",
        notes="Tensor network simulator.",
    ),
    # ---- QPUs ----
    "ionq_aria_1": BackendInfo(
        name="ionq_aria_1",
        kind="qpu",
        arn="arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
        qubits=25,
        cost_per_shot_usd=0.03,
        per_task_fee_usd=0.30,
        region="us-east-1",
        notes="Trapped ion, all-to-all connectivity. Slow but high fidelity.",
    ),
    "rigetti_ankaa_3": BackendInfo(
        name="rigetti_ankaa_3",
        kind="qpu",
        arn="arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3",
        qubits=84,
        cost_per_shot_usd=0.0009,
        per_task_fee_usd=0.30,
        region="us-west-1",
        notes="Superconducting, limited connectivity. Cheapest gate-model QPU.",
    ),
    "iqm_garnet": BackendInfo(
        name="iqm_garnet",
        kind="qpu",
        arn="arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet",
        qubits=20,
        cost_per_shot_usd=0.00145,
        per_task_fee_usd=0.30,
        region="eu-north-1",
        notes="Superconducting, EU region.",
    ),
}


def estimate_cost_usd(
    backend: str, shots: int, estimated_runtime_minutes: float = 0.05
) -> float:
    info = CATALOG[backend]
    if info.kind == "qpu":
        return info.cost_per_shot_usd * shots + info.per_task_fee_usd
    if info.kind == "cloud_sim":
        return info.cost_per_minute_usd * max(0.05, estimated_runtime_minutes)
    return 0.0


def list_backends(kind: DeviceKind | None = None) -> list[BackendInfo]:
    if kind is None:
        return list(CATALOG.values())
    return [b for b in CATALOG.values() if b.kind == kind]
