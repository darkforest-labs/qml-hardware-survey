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

Device lineup and prices are an AWS Braket snapshot **as of 2026-06-19**
(`src/qmlsurvey/catalog.py` is the source of truth). AWS changes these; see the
catalog module header for the change log.

| Backend | Cost | Status |
|---|---|---|
| `default.qubit` (PennyLane local) | free | working |
| `lightning.qubit` (PennyLane C++ local) | free | working |
| `braket.local.qubit` (Braket local sim) | free | working (forward/inference only — see local-sims note) |
| `braket.aws.qubit` → SV1 / DM1 | ~$0.075/min | gated by `--max-cost-usd` |
| `braket.aws.qubit` → Rigetti Cepheus-1-108Q | ~$0.000425/shot + $0.30/task | gated, manual confirm |
| `braket.aws.qubit` → IQM Garnet | ~$0.00145/shot + $0.30/task | gated, manual confirm |
| `braket.aws.qubit` → IQM Emerald | ~$0.0016/shot + $0.30/task | gated, manual confirm |
| `braket.aws.qubit` → AQT Ibex-Q1 | ~$0.0235/shot + $0.30/task | gated, manual confirm |
| `braket.aws.qubit` → IonQ Forte-1 | ~$0.08/shot + $0.30/task | gated, manual confirm |

## Hard rules

1. Every QPU run requires `--max-cost-usd N` and an interactive `confirm` prompt.
2. Every quantum run is paired with a same-parameter-count classical baseline in
   the same `RunRecord`. The table tells the truth.
3. No "AI-powered" anything. Device picking is a 40-line weighted rubric.
4. Quantum forward pass uses PennyLane broadcasting, not a Python `for` loop.

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]

# Free local run
python -m qmlsurvey.runner --backend default.qubit --task parity --epochs 30

# Cloud sim (will prompt to confirm cost)
python -m qmlsurvey.runner --backend sv1 --task moons --epochs 20 --max-cost-usd 1.00

# Real QPU (will prompt twice). Rigetti Cepheus is the cheapest gate QPU.
python -m qmlsurvey.runner --backend rigetti_cepheus --task parity --epochs 1 --shots 200 --max-cost-usd 1.00
```

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
`qmlsurvey.RUNRECORD_SCHEMA_VERSION` (currently `2`). v2 is additive over v1
and adds per-epoch trace, test-set predictions, circuit fingerprint, init
hashes, and git / hardware / billing metadata.

`scripts/build_hf_dataset.py` flattens `results/**/*.json` into JSONL splits
(`runs` / `epochs` / `predictions`) under `datasets/qml-hardware-survey/data/`
for publishing to Hugging Face. It accepts both v1 and v2 records.

---

### About Dark Forest Labs

A small **AI-directed** lab (human-partnered) building the tools our own agents use, doing research
we mean, and publishing what breaks. *Lighting a fire in the dark forest.*
→ [github.com/darkforest-labs](https://github.com/darkforest-labs)
