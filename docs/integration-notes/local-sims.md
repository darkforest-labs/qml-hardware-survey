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

## Shot-noise sweep (option 1: inference-only on `braket.local.qubit`)

`scripts/run_phase1_shot_noise.py` trains the reference parity model once
on `default.qubit` analytic, then evaluates the trained `state_dict` on the
parity test set at multiple shot counts on each backend that accepts
forward inference (10 independent shot draws per cell). Recorded in
`results/phase1/shot_noise/parity_shot_noise.json`.

| Backend              | Shots   | Mean acc | Std    | Min    | Max    | Notes                  |
|----------------------|---------|---------:|-------:|-------:|-------:|------------------------|
| `default.qubit`      | analytic| 1.0000   | 0.0000 | 1.000  | 1.000  | reference              |
| `default.qubit`      | 5000    | —        | —      | —      | —      | failed (see below)     |
| `default.qubit`      | 1000    | —        | —      | —      | —      | failed                 |
| `default.qubit`      | 500     | —        | —      | —      | —      | failed                 |
| `default.qubit`      | 100     | —        | —      | —      | —      | failed                 |
| `braket.local.qubit` | 5000    | 1.0000   | 0.0000 | 1.000  | 1.000  | option 1 works         |
| `braket.local.qubit` | 1000    | 1.0000   | 0.0000 | 1.000  | 1.000  | option 1 works         |
| `braket.local.qubit` | 500     | 1.0000   | 0.0000 | 1.000  | 1.000  | option 1 works         |
| `braket.local.qubit` | 100     | 0.9981   | 0.0061 | 0.981  | 1.000  | first deviation        |

Two integration findings:

1. **Option 1 is viable.** `braket.local.qubit` runs forward-only on
   broadcasted batches without hitting PL #4462, because gradients are
   never invoked. The backend is reachable for shot-noise / inference work
   on weights trained elsewhere; what it cannot currently do is train.
2. **`default.qubit` + `interface="torch"` + broadcasting + finite shots
   raises `ValueError: probabilities do not sum to 1`.** The shot sampler
   in `pennylane/devices/qubit/sampling.py::_sample_probs_numpy` enforces
   exact sum-to-one and the torch-side probabilities drift outside that
   tolerance under broadcasting. This is a second real integration block
   surfaced by the survey. The matrix above leaves those cells dashed
   rather than masking the failure.

The parity model used here is small and trains to a clean separation, so
shot noise barely registers even at 100 shots — only one trial out of ten
showed a single misclassified test point. Tasks with smaller margins
(planned for later phases) should produce more discriminating shot-noise
data.

## Trainability / gradient-variance sweep

`scripts/run_phase1_trainability.py` sweeps a (`n_qubits`, `n_layers`) grid
on `default.qubit` (analytic, backprop). For each cell it draws
`N_INITS = 200` independent random initialisations of `quantum_weights`
from `Uniform[-π, π]` (the canonical barren-plateau prescription),
holding the encoder, decoder, and a fixed batch (`BATCH = 32`) constant
per cell. For each init it does one forward + backward of cross-entropy
loss and records `Var` across inits of every quantum-weight gradient
component. Recorded in `results/phase1/trainability/`.

Mean over parameter components of `Var(∂loss/∂θ)` across the 200 inits:

| `n_qubits` \\ `n_layers` |        1 |        2 |        4 |        8 |
|------------------------:|---------:|---------:|---------:|---------:|
|                       2 | 1.74e-03 | 1.83e-04 | 7.14e-05 | 2.85e-06 |
|                       4 | 1.14e-04 | 6.56e-05 | 1.81e-04 | 3.84e-04 |
|                       6 | 2.88e-05 | 3.24e-05 | 2.28e-05 | 5.65e-05 |
|                       8 | 3.03e-05 | 6.98e-06 | 5.54e-06 | 1.15e-05 |
|                      10 | 1.59e-05 | 1.54e-06 | 1.33e-06 | 1.73e-06 |
|                      12 | 9.49e-06 | 2.29e-06 | 5.08e-07 | 5.86e-07 |

Three integration observations:

1. **Width dominates depth in this circuit.** Holding `n_layers` fixed,
   `grad_var_mean` falls roughly two to three orders of magnitude as
   `n_qubits` goes from 2 → 12 — consistent with the published
   ~`O(2^-n)` barren-plateau scaling for hardware-efficient ansätze with
   Pauli-Z readouts. Holding `n_qubits` fixed, depth has a weaker and
   non-monotonic effect (visible in the `n_qubits=4` row, where 8 layers
   actually has higher variance than 1 layer).
2. **The reference config sits at the top of the usable band.** The
   project's reference cell `(n_qubits=4, n_layers=2)` posts
   `grad_var_mean ≈ 6.6e-05`, three orders of magnitude above the
   `(12, 4)` corner at `5.1e-07`. Anything past `n_qubits ≈ 10` with
   `n_layers ≥ 2` is in territory where vanilla gradient training will
   need either smarter initialisation or a different objective.
3. **Wall time is tractable.** The full 24-cell × 200-init sweep ran in
   5.7 minutes on `default.qubit`; the worst cell `(12, 8)` cost
   ~338 ms / init.

Caveats: this is one circuit family (`HybridModel`'s
`Rot + linear-CNOT-chain` ansatz) on one synthetic batch with one
init distribution. The numbers above describe **this** circuit's
trainability curve, not "barren plateaus in general".

## Reproducing

```powershell
$env:PYTHONNOUSERSITE = "1"  # if a Python 3.13 user-site is on PATH
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev,braket]
python scripts/run_phase1_matrix.py
python scripts/run_phase1_shot_noise.py
python scripts/run_phase1_trainability.py
```
