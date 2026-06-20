"""
Static metadata for known Braket backends.

This is a hand-maintained snapshot of AWS Braket device info. AWS is the source
of truth at runtime (`AwsDevice.properties`); this module exists only so we can
estimate costs and gate operations without a network round-trip.

Update as AWS changes pricing or device availability.

Snapshot as of 2026-06-19. Sources: AWS Braket pricing page
(<https://aws.amazon.com/braket/pricing/>) and the supported-devices doc
(<https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html>).
Notable changes since the 2026-04 snapshot:
  - IonQ Aria-1 retired from Braket; current IonQ QPU is Forte-1
    (also Forte-Enterprise-1), $0.08/shot.
  - Rigetti Cepheus-1-108Q (108 qubits) launched 2026-04-07 and replaces
    Ankaa-3; at $0.000425/shot it is now the cheapest gate-model QPU. Ankaa-3
    is now RETIRED (confirmed live via `braket search-devices` on 2026-06-19)
    and is dropped from this catalog.
  - New: AQT Ibex-Q1 (12-qubit trapped ion, EU) and IQM Emerald (54 qubits).
  - IQM Garnet unchanged at $0.00145/shot.
QuEra Aquila (neutral-atom / Analog Hamiltonian Simulation) is intentionally
omitted: it is not gate-based and is out of scope per ROADMAP.md.
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
        notes="Tensor network simulator. Per-minute rate retained from the "
        "2026-04 snapshot; not separately tabulated on the 2026-06 pricing page.",
    ),
    # ---- QPUs (gate-based, in-scope) ----
    # Ordered cheapest -> most expensive per shot. All carry a $0.30 per-task fee.
    "rigetti_cepheus": BackendInfo(
        name="rigetti_cepheus",
        kind="qpu",
        arn="arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q",
        qubits=108,
        cost_per_shot_usd=0.000425,
        per_task_fee_usd=0.30,
        region="us-west-1",
        notes="Superconducting, 3x4 array of 9-qubit chiplets. Launched "
        "2026-04-07, replaces Ankaa-3. Cheapest gate-model QPU.",
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
    "iqm_emerald": BackendInfo(
        name="iqm_emerald",
        kind="qpu",
        arn="arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald",
        qubits=54,
        cost_per_shot_usd=0.0016,
        per_task_fee_usd=0.30,
        region="eu-north-1",
        notes="Superconducting, EU region. Larger successor to Garnet.",
    ),
    "aqt_ibex_q1": BackendInfo(
        name="aqt_ibex_q1",
        kind="qpu",
        arn="arn:aws:braket:eu-north-1::device/qpu/aqt/Ibex-Q1",
        qubits=12,
        cost_per_shot_usd=0.0235,
        per_task_fee_usd=0.30,
        region="eu-north-1",
        notes="Trapped ion (Ca-40), all-to-all connectivity, EU region "
        "(Innsbruck). First EU-hosted trapped-ion device on Braket.",
    ),
    "ionq_forte_1": BackendInfo(
        name="ionq_forte_1",
        kind="qpu",
        arn="arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
        qubits=36,
        cost_per_shot_usd=0.08,
        per_task_fee_usd=0.30,
        region="us-east-1",
        notes="Trapped ion, all-to-all connectivity. Replaces Aria-1 (retired). "
        "At the 2026-06-19 snapshot Forte-1 was OFFLINE and Forte-Enterprise-1 "
        "(.../qpu/ionq/Forte-Enterprise-1) was ONLINE at the same per-shot price.",
    ),
}


# Cloud simulators bill per-minute with a 3-second minimum *per task*.
CLOUD_SIM_MIN_TASK_MINUTES = 0.05  # 3 s


def estimate_cost_usd(
    backend: str,
    shots: int,
    n_tasks: int = 1,
    estimated_runtime_minutes: float = CLOUD_SIM_MIN_TASK_MINUTES,
) -> float:
    """Estimate the USD cost of a run.

    Cost scales with ``n_tasks``, not just ``shots``. Each Braket quantum task
    is billed independently, and a broadcasted forward over a batch of B inputs
    submits B tasks; parameter-shift gradients and multiple epochs multiply this
    further. Calibrated against a real SV1 run — 2 broadcast inputs billed
    $0.0075 = ``2 x max(3 s, runtime) x $0.075/min`` — see
    ``docs/integration-notes/sv1.md``.

    - ``qpu``:       ``n_tasks x (cost_per_shot x shots + per_task_fee)``
    - ``cloud_sim``: ``n_tasks x max(3 s, runtime) x cost_per_minute``

    Callers issuing multi-task runs (any batch > 1, any gradient, any epoch
    count) MUST pass the real ``n_tasks``; the default of 1 only covers a single
    single-input task and will under-estimate otherwise. Never raise a cost cap
    to make a run fit — fix the ``n_tasks`` estimate instead.
    """
    info = CATALOG[backend]
    if info.kind == "qpu":
        per_task = info.cost_per_shot_usd * shots + info.per_task_fee_usd
        return per_task * n_tasks
    if info.kind == "cloud_sim":
        per_task_minutes = max(CLOUD_SIM_MIN_TASK_MINUTES, estimated_runtime_minutes)
        return info.cost_per_minute_usd * per_task_minutes * n_tasks
    return 0.0


def list_backends(kind: DeviceKind | None = None) -> list[BackendInfo]:
    if kind is None:
        return list(CATALOG.values())
    return [b for b in CATALOG.values() if b.kind == kind]
