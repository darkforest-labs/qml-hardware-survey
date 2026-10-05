# qml-hardware-survey

A systematic survey of what it takes to interface a small ML model with each
real quantum computing backend reachable from a laptop. Outcomes and integration
techniques, side-by-side with classical baselines.

## What this is

- **One small hybrid model** (PennyLane + PyTorch), unchanged across backends.
- **One swap point**: `backends.get(name)` returns a PennyLane device.
- **Three toy tasks** with clear classical baselines: parity, two-moons, PCA-MNIST 0/1.
- **One result format** (`RunRecord` JSON) committed to `results/` per run.
- **One rollup notebook** that turns those JSONs into a comparison table.

## What this is NOT

- Not a claim that quantum ML is better than classical. It almost certainly isn't
  at this scale. Every quantum result is reported next to a parameter-matched
  classical baseline.
- Not a framework. ~600 LOC total target.
- Not a benchmark suite — it's a survey of **integration friction** and
  **observed outcomes** on tiny problems.

## Backends covered

Device lineup and prices are an AWS Braket snapshot **as of 2026-10-03**
(`src/qmlsurvey/catalog.py` is the source of truth; TN1 was retired and is no
longer listed). AWS changes these; see the catalog module header for the change
log. Every QPU also publishes UTC **execution windows**; outside them a task
queues until the window opens. `python scripts/doctor.py` shows each QPU's
status, in-window flag and queue depth live.

| Backend | Cost | Status |
|---|---|---|
| `default.qubit` (PennyLane local) | free | working |
| `lightning.qubit` (PennyLane C++ local) | free | working |
| `braket.local.qubit` (Braket local sim) | free | working (training via per-input expansion — see local-sims note) |
| `braket.aws.qubit` → SV1 / DM1 | ~$0.075/min | gated by `--max-cost-usd` |
| `braket.aws.qubit` → Rigetti Cepheus-1-108Q | ~$0.000425/shot + $0.30/task | gated, manual confirm |
| `braket.aws.qubit` → IQM Garnet | ~$0.00145/shot + $0.30/task | gated, manual confirm |
| `braket.aws.qubit` → IQM Emerald | ~$0.0016/shot + $0.30/task | gated, manual confirm |
| `braket.aws.qubit` → AQT Ibex-Q1 | ~$0.0235/shot + $0.30/task | gated, manual confirm |
| `braket.aws.qubit` → IonQ Forte-1 | ~$0.08/shot + $0.30/task | gated, manual confirm |

## Hard rules

1. Every cloud-sim and QPU run requires `--max-cost-usd N` (default 0, so a
   missing flag aborts) and an interactive `confirm` prompt showing the task
   count, estimated cost, and whether the device is inside its execution window.
2. Every quantum run is paired with a same-parameter-count classical baseline in
   the same `RunRecord`. The table tells the truth.
3. No "AI-powered" anything. Device picking is a 40-line weighted rubric.
4. Quantum forward pass uses PennyLane broadcasting at the model API, not a
   Python `for` loop. On finite-shot devices (any Braket device, or a local sim
   with `shots`) PennyLane cannot differentiate a broadcasted tape
   (PennyLane #4462, still open at 0.45.1), so `HybridModel` wraps the QNode in
   `qml.transforms.broadcast_expand`, which splits the batch into one tape per
   input before execution. On Braket that is one **billed task per input per
   execution**, and parameter-shift multiplies it: a training step on `B`
   inputs costs `B × (1 + 2 × P)` tasks with `P = n_qubits + 3·n_layers·n_qubits`
   (`P = 28` for the reference config). `catalog.estimate_task_count` is that
   formula and is verified against `qml.Tracker`; the runner records both the
   estimate and the tracked count in every `RunRecord`.

### Cost reality (reference config, 200 shots)

| run | tasks | SV1 | Rigetti Cepheus | IonQ Forte-1 |
|---|---|---|---|---|
| parity, 1 epoch, full batch (204 train / 52 test) | 11,936 | ≈ $45 | ≈ $4,600 | ≈ $194,000 |
| parity, 30 epochs (roadmap reference) | 356,572 | ≈ $1,337 | ≈ $137,000 | — |
| one gradient step, 1 input | 60 | $0.23 | ≈ $23 | ≈ $978 |

The $0.30 per-task fee dominates on QPUs. End-to-end training on hardware at
this batch size is not something this project will buy; see `ROADMAP.md`
Phase 4 for the re-scoped question. The task model is not theoretical: a
2-input, 1-epoch parity run on SV1 (2026-10-04) was predicted at 120 tasks /
$0.45 and billed at 120 tasks / $0.45 (`docs/integration-notes/sv1.md`).
Use `--n-train-subset` / `--n-test-subset` to make a paid run that small.

## Quickstart

```powershell
# Python 3.10–3.13. 3.13 gets PennyLane >= 0.43; 3.10 is capped at 0.42.3.
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev,braket]

# Free local run
python -m qmlsurvey.runner --backend default.qubit --task parity --epochs 30

# Free finite-shot training on the Braket local simulator (same code path as
# the cloud devices). A micro-batch keeps it to 120 circuit executions.
python -m qmlsurvey.runner --backend braket.local.qubit --task parity --epochs 1 `
    --shots 100 --n-train-subset 2 --n-test-subset 2

# Pre-flight before anything paid: credentials, bucket, QPU windows and queue.
$env:QMLSURVEY_S3_BUCKET = "amazon-braket-qmlsurvey-<account-id>"
python scripts/doctor.py

# Cloud sim, PAID: 120 tasks, $0.45 (prompts with the task count and cost).
python -m qmlsurvey.runner --backend sv1 --task parity --epochs 1 --shots 100 `
    --n-train-subset 2 --n-test-subset 2 --max-cost-usd 0.60
```

The runner always *trains*, and training fans out into one billed task per
parameter-shift evaluation (see hard rule 4 and "Cost reality" below). Without
the subset flags the same SV1 command estimates about $45 per epoch and aborts
at the cap, which is the cap doing its job. The smallest possible QPU training
run (1 train / 1 test input, 1 epoch, 200 shots on Rigetti Cepheus) is 60
tasks, about $23. A forward-only QPU path (ROADMAP Phase 3) is not wired into
the CLI yet.

## Reference configuration

Cross-backend comparisons use a single fixed configuration per task so that
runs differ only in the backend. Don't change these casually — changing them
invalidates the existing comparison set.

| Task | `n_qubits` | `n_layers` | `epochs` | `lr` | `seed` |
|---|---|---|---|---|---|
| `parity` | 4 | 2 | 30 | 0.05 | 0 |
| `moons` | 4 | 2 | 30 | 0.05 | 0 |
| `mnist_pca` | 4 | 2 | 30 | 0.05 | 0 |

Phase-0 reference results (one per task on `default.qubit`) live in
`results/phase0/`. Regenerate them with:

```powershell
python scripts/run_phase0.py
```

## Layout

See [pyproject.toml](pyproject.toml) for deps. Code lives in `src/qmlsurvey/`,
per-backend integration notes live in `docs/integration-notes/`, every run
writes a JSON to `results/`. The `RunRecord` schema is versioned via
`qmlsurvey.RUNRECORD_SCHEMA_VERSION` (currently `3`). v2 is additive over v1
and adds per-epoch trace, test-set predictions, circuit fingerprint, init
hashes, and git / hardware / billing metadata. v3 is additive over v2 and adds
`estimated_tasks`, `broadcast_expanded`, a pre-submission `device_snapshot`
(status, availability, queue depth, execution windows) and `device_executions`
(`qml.Tracker` totals, to reconcile against the estimate and the bill).

`scripts/build_hf_dataset.py` flattens `results/**/*.json` into JSONL splits
(`runs` / `epochs` / `predictions`) under `datasets/qml-hardware-survey/data/`
for publishing to Hugging Face. It accepts both v1 and v2 records.

---

### About Dark Forest Labs

A small **AI-directed** lab (human-partnered) building the tools our own agents use, doing research
we mean, and publishing what breaks. *Lighting a fire in the dark forest.*
→ [github.com/darkforest-labs](https://github.com/darkforest-labs)
