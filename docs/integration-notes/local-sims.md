# Local simulators — integration notes

Captures friction encountered while running the reference config across the
three local simulators currently wired into `qmlsurvey`. External register:
descriptions of what was observed, not interpretations.

## Backends covered

| Backend              | Provider              | Analytic mode | Adjoint diff | Notes |
|----------------------|-----------------------|---------------|--------------|-------|
| `default.qubit`      | PennyLane (built-in)  | Yes (`shots=None`) | Yes | Reference simulator. |
| `lightning.qubit`    | PennyLane (C++ ext.)  | Yes           | Yes          | Faster on large circuits, slower on small. |
| `braket.local.qubit` | Amazon Braket plugin  | No            | No           | Requires `pip install -e .[braket]`; defaults `shots=1000`. |

## Phase-1 cross-simulator matrix (reference config)

`n_qubits=4, n_layers=2, epochs=30, lr=0.05, seed=0`, three tasks
(`parity`, `moons`, `mnist_pca`). Recorded in `results/phase1/cross_sim/`.

| Backend            | Task       | Final test acc | Final loss | Wall time (s) |
|--------------------|------------|---------------:|-----------:|--------------:|
| `default.qubit`    | parity     | 1.0000         | 0.030083   |  0.96 |
| `default.qubit`    | moons      | 0.9000         | 0.220416   |  1.06 |
| `default.qubit`    | mnist_pca  | 1.0000         | 0.047948   |  0.94 |
| `lightning.qubit`  | parity     | 1.0000         | 0.030083   | 23.05 |
| `lightning.qubit`  | moons      | 0.9000         | 0.220416   | 26.85 |
| `lightning.qubit`  | mnist_pca  | 1.0000         | 0.047948   | 33.09 |
| `braket.local.qubit` | parity   | —              | —          | —     |
| `braket.local.qubit` | moons    | —              | —          | —     |
| `braket.local.qubit` | mnist_pca| —              | —          | —     |

### Agreement

`default.qubit` and `lightning.qubit` agree on the reference config to at
least 6 decimal places of final loss for all three tasks (and are bitwise
identical on the metrics PennyLane returns to PyTorch). Treating
`default.qubit` as ground truth, `lightning.qubit` reproduces it.

### Wall-time observation

At `n_qubits=4`, `lightning.qubit` is roughly 23–33× slower than
`default.qubit` end-to-end. This is consistent with PennyLane's published
guidance that the C++ backend pays a per-call overhead that only amortizes
on larger circuits; we have not measured the crossover point yet.

## `braket.local.qubit` — broadcasting / parameter-shift block

All three `braket.local.qubit` runs failed with:

> `NotImplementedError: Computing the gradient of broadcasted tapes with
> respect to the broadcasted parameters using the parameter-shift rule
> gradient transform is currently not supported.` (PennyLane issue #4462.)

`HybridModel.forward` deliberately uses input broadcasting (project hard
rule #4 — no Python `for` over batch). `braket.local.qubit` does not support
adjoint differentiation, so PennyLane falls back to parameter-shift, which
in turn does not support broadcasted parameters in the current versions
(`pennylane==0.42.3`, `amazon-braket-pennylane-plugin==1.33.7`).

This is a real integration constraint surfaced by the survey, not a project
bug. Options if we want `braket.local.qubit` coverage in later phases:

1. **Inference-only path.** Use it for forward / shot-noise characterisation
   on parameters that were already trained on `default.qubit`. The forward
   pass works under broadcasting; only `loss.backward()` blows up.
2. **Per-sample loop, `braket.local.qubit` only.** Would violate the
   broadcasting rule for the rest of the codebase, so worth a separate
   helper rather than bending `HybridModel`.
3. **Wait for upstream.** Track PL #4462; revisit.

For now the matrix is honest about the gap rather than working around it.

## Reproducing

```powershell
$env:PYTHONNOUSERSITE = "1"  # if a Python 3.13 user-site is on PATH
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev,braket]
python scripts/run_phase1_matrix.py
```
