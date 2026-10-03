"""
The one hybrid model used across the entire survey.

Architecture (kept deliberately small):

    x  -> Linear(in -> n_qubits)  -> tanh                   (encoder)
       -> QNode(n_qubits, n_layers): RY(scaled) + variational layers
       -> Linear(n_qubits -> out)                            (decoder)

The QNode uses input broadcasting so a whole batch is one QNode call (no
per-sample Python loop in this code base). How that batch reaches the device
depends on the device:

- Analytic ``default.qubit`` / ``lightning.qubit``: the broadcasted tape is
  executed as-is with backprop / adjoint gradients. One device execution per
  forward.
- Any finite-shot device (``default.qubit`` with ``shots``, ``braket.local.qubit``,
  SV1 / DM1, every QPU): ``diff_method="best"`` resolves to parameter-shift,
  which cannot differentiate broadcasted parameters (PennyLane #4462, still
  open at 0.45.1). We therefore wrap the QNode in
  ``qml.transforms.broadcast_expand``, which splits the batch into one tape per
  input *before* execution. Training then works, at the price of
  ``B x (1 + 2 x n_shift_params)`` device executions per training step. On
  Braket each of those is a separately billed task; ``catalog.estimate_task_count``
  mirrors this formula and was verified against ``qml.Tracker``.
"""
from __future__ import annotations

import math

import pennylane as qml
import torch
from torch import nn


def _has_finite_shots(device) -> bool:
    """True if the device samples (finite shots) rather than returning exact expvals."""
    shots = getattr(device, "shots", None)
    if shots is None:
        return False
    total = getattr(shots, "total_shots", shots)
    return bool(total)


def n_shift_params(n_qubits: int, n_layers: int) -> int:
    """Trainable gate parameters per tape that parameter-shift must differentiate.

    ``n_qubits`` RY encoding angles (trainable because the encoder is trained)
    plus ``3 x n_layers x n_qubits`` Rot angles. Each costs two shifted executions.
    """
    return n_qubits + 3 * n_layers * n_qubits


def _make_circuit(n_qubits: int, n_layers: int, device):
    @qml.qnode(device, interface="torch", diff_method="best")
    def circuit(inputs, weights):
        # inputs shape: (batch, n_qubits) — angle encoding, scaled to [-pi, pi]
        for q in range(n_qubits):
            qml.RY(math.pi * inputs[..., q], wires=q)

        # Variational layers: per-qubit Rot + ring of CNOTs
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.Rot(
                    weights[layer, q, 0],
                    weights[layer, q, 1],
                    weights[layer, q, 2],
                    wires=q,
                )
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])

        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit


class HybridModel(nn.Module):
    """Small hybrid classifier / regressor. Same module across all backends."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_qubits: int = 4,
        n_layers: int = 2,
        device=None,
    ):
        super().__init__()
        if device is None:
            device = qml.device("default.qubit", wires=n_qubits)

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoder = nn.Linear(in_features, n_qubits)
        self.quantum_weights = nn.Parameter(
            0.1 * torch.randn(n_layers, n_qubits, 3)
        )
        self.decoder = nn.Linear(n_qubits, out_features)
        # `_qnode` is the raw QNode (used for qml.specs); `_circuit` is what
        # forward() calls, expanded per-input on finite-shot devices (see module doc).
        self._qnode = _make_circuit(n_qubits, n_layers, device)
        self.broadcast_expanded = _has_finite_shots(device)
        self._circuit = (
            qml.transforms.broadcast_expand(self._qnode)
            if self.broadcast_expanded
            else self._qnode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = torch.tanh(self.encoder(x))  # (batch, n_qubits) in [-1, 1]
        # PennyLane returns a list of length n_qubits; stack to (batch, n_qubits).
        out = self._circuit(encoded, self.quantum_weights)
        if isinstance(out, (list, tuple)):
            out = torch.stack(out, dim=-1)
        # PennyLane expvals come back as float64; align with the rest of the model.
        out = out.to(dtype=encoded.dtype)
        return self.decoder(out)

    @property
    def n_quantum_params(self) -> int:
        return int(self.quantum_weights.numel())

    @property
    def n_shift_params(self) -> int:
        return n_shift_params(self.n_qubits, self.n_layers)

    @property
    def n_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
