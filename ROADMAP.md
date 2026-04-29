# Roadmap: qml-hardware-survey

Framed per `epistemological_style_guide.md`: this project is a **capability
probe / integration survey**, not a research program. Deliverables are
integration notes, run records, and a comparison table — not claims about
quantum advantage. Each phase has a concrete exit condition and a cost ceiling.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[-]` skipped

---

## Phase 0 — Foundation (free, local only)

**Goal:** prove the harness end-to-end before spending a cent.

- [x] CI: GitHub Actions running `pytest` + `ruff check` on push (local sims only).
- [x] Add `lightning.qubit` to the smoke-test matrix alongside `default.qubit`.
- [x] Pin `RunRecord` schema with a `schema_version: 1` field; add `tests/test_runrecord_schema.py` round-trip test.
- [x] Capture `pip freeze` (or at least `pennylane`, `torch`, plugin versions) into `RunRecord.environment`. Currently records only Python + platform.
- [x] Seed audit: confirm `torch.manual_seed` + numpy/sklearn seeds are all controlled per task. Add `test_determinism.py` that runs the same task twice and asserts identical `final_test_acc`.
- [x] Decide on a fixed `(seed, n_qubits, n_layers, epochs, lr)` "reference config" per task and record it in `README.md`. Every cross-backend comparison uses this config.

**Exit:** `pytest -q` green, one reference-config `RunRecord` per task on `default.qubit` committed under `results/phase0/`. **✅ met.**

---

## Phase 1 — Local simulator characterization (free)

**Goal:** know what the model can and cannot do *before* hardware noise enters.

- [x] Run reference config on **all three local sims** (`default.qubit`, `lightning.qubit`, `braket.local.qubit`) × all three tasks = 9 records. Same seed. *(6/9 done; `braket.local.qubit` blocked by PL #4462 — broadcasted parameter-shift gradients unsupported. Documented in `docs/integration-notes/local-sims.md`.)*
- [x] Sanity check: `default.qubit` and `lightning.qubit` should agree to numerical noise; `braket.local.qubit` (with finite shots) should agree within shot-noise CIs. Document any disagreement. *(default.qubit ≡ lightning.qubit to 6 decimals on all three tasks; lightning ≈23–33× slower at 4 qubits; braket gap noted above.)*
- [x] **Shot-noise sweep** on one task (parity is cheapest): shots ∈ {100, 500, 1000, 5000, analytic}. Plot accuracy vs shots. This is the calibration curve we'll need to interpret QPU runs. *(`scripts/run_phase1_shot_noise.py` + option-1 inference path. `braket.local.qubit` reachable for forward-only at all shot counts; parity model saturates accuracy except 1/10 trials at 100 shots. `default.qubit` torch interface + broadcasting + finite shots blocks on `_sample_probs_numpy` "probabilities do not sum to 1" — second integration finding logged in `docs/integration-notes/local-sims.md`.)*
- [ ] **Trainability check**: gradient variance vs `n_qubits` ∈ {2, 4, 6, 8} and `n_layers` ∈ {1, 2, 4}. Flag barren-plateau territory before paying for it on a QPU.
- [ ] First rollup: `scripts/rollup.py` ingests `results/**/*.json` into a single dataframe and emits `results/SUMMARY.md`.

**Exit:** rollup table shows the three local sims agreeing on the reference config; shot-noise curve documented in `docs/integration-notes/local-sims.md`.

---

## Phase 2 — AWS plumbing dry-run (≤ $1 total)

**Goal:** de-risk every non-quantum failure mode (auth, region, quota, plugin install) on the cheapest possible cloud target.

- [ ] Document AWS setup in `docs/integration-notes/aws-setup.md`: account, region enable, S3 results bucket, IAM policy, `aws configure sso` or access keys, Braket service activation.
- [ ] Add a `qmlsurvey doctor` subcommand (or `scripts/doctor.py`) that checks: boto3 importable, plugin importable, Braket-visible devices, S3 bucket writable, current AWS region. Run before *any* paid call.
- [ ] **First paid call**: 1 epoch of parity on **`sv1`** (cheapest cloud sim, $0.075/min) with `--shots 100 --max-cost-usd 0.50`. Goal is "the call returned a JSON," not accuracy.
- [ ] One full reference-config run on `sv1` for each of the three tasks. Record actual billed cost vs estimate in the integration note.
- [ ] One run on `dm1` with a deliberately injected noise model — substrate for predicting what QPUs will do.

**Exit:** `docs/integration-notes/sv1.md` and `dm1.md` populated with billed-vs-estimated table; `doctor` script committed.

**Hard cap:** $1 cumulative across this phase. If exceeded, stop and inspect the cost model in `catalog.estimate_cost_usd` — its `estimated_runtime_minutes=0.05` default is almost certainly wrong for real workloads.

---

## Phase 3 — First QPU contact (≤ $5 total)

**Goal:** one small, well-characterized run on the cheapest QPU. Inference-only, no training.

- [ ] Pick **Rigetti Ankaa-3** ($0.0009/shot + $0.30/task) — cheapest per-shot.
- [ ] Workflow: train on `default.qubit` to convergence on parity, then run *forward pass only* on QPU with the trained weights. Compare QPU-Z-expectations to simulator-Z-expectations sample-by-sample. Isolates "circuit fidelity" from "training under noise."
- [ ] Shots: 200. Estimated cost ~$0.48. Cap: `--max-cost-usd 1.00`.
- [ ] Capture queue wait time separately from circuit wall-time in `RunRecord` (new fields: `queue_wait_s`, `device_runtime_s`).
- [ ] Repeat once on **IQM Garnet** ($0.00145/shot, EU region) and once on **IonQ Aria-1** ($0.03/shot, slow but high fidelity). Three integration notes.

**Exit:** three `docs/integration-notes/{rigetti_ankaa_3,iqm_garnet,ionq_aria_1}.md` files with billed cost, queue time, expectation-value agreement vs simulator, and any errors hit. SUMMARY.md updated.

**Hard cap:** $5 cumulative.

---

## Phase 4 — Training on hardware, scoped (≤ $25 total)

**Goal:** answer "can the hybrid model train end-to-end against a QPU at all?" — not "is it good."

- [ ] Pick the *single* (backend, task) pair with the best Phase-3 fidelity. Almost certainly `rigetti_ankaa_3 × parity`.
- [ ] Run **5 epochs** with shots=200, full QPU in the loop. Cost estimate: ~5 × ($0.18 + $0.30) ≈ $2.40 plus queue overhead.
- [ ] Compare against (a) same model trained on `default.qubit` with shots=200 (shot-noise-only), (b) same model trained noise-free, (c) parameter-matched MLP. Already automatic via `MatchedMLP` for (c); add (a) and (b) as additional records with a shared `experiment_group` field.
- [ ] If gradients are dominated by shot noise (very likely), document it. Don't grind epochs hoping it converges; that's burning money to confirm a known phenomenon.

**Exit:** one honest `docs/integration-notes/training-on-qpu.md` saying what worked, what didn't, and what it cost.

**Hard cap:** $25 cumulative across the project to date.

---

## Phase 5 — Synthesis

**Goal:** produce the artifact a stranger would actually read.

- [ ] `results/SUMMARY.md` regenerated from all `RunRecord` JSONs, with one row per (backend, task) showing: quantum acc, classical-matched acc, params, shots, wall time, queue time, billed cost.
- [ ] `docs/findings.md`: **what we learned about integration**, written in the external register. Defensible-claim examples: "Adding a Braket cloud-sim backend required N lines of code and one IAM policy change." "Wall-clock per epoch on Rigetti Ankaa-3 at 200 shots was X seconds, of which Y was queue time." "On parity at this scale, the classical baseline matches or exceeds quantum across all backends tested."
- [ ] What this is **not**: a benchmark paper, a claim about QML performance trends, or anything generalizable beyond "a small hybrid model on these QPUs as of \[date]."
- [ ] Tag a `v0.1.0` release pinning the schema and result set.

**Exit:** repo is in a state where the README, SUMMARY, and integration notes are mutually consistent and a domain reader (per the style guide's honest-description test) wouldn't bounce off it.

---

## Cross-cutting work (any phase)

- **Cost discipline.** Treat `--max-cost-usd` as the contract. Never raise the cap to make a run go through; inspect the estimator instead.
- **Schema stability.** Bump `schema_version` if `RunRecord` fields change; keep a migration path for old JSONs in the rollup.
- **LOC budget.** README targets ~600 LOC. New code (doctor script, rollup, schema test) puts pressure on this — prefer extending existing modules over adding new ones.
- **Backend additions.** New backend = new `BackendInfo` in `CATALOG` + one branch in `backends.get_device` + one integration note. Refuse to add anything else; that's the discipline that makes the swap-point claim true.
- **Style-guide check before any external write-up.** Run anything in `docs/findings.md` or a future webpage description through the patterns list in `epistemological_style_guide.md`.

---

## Out of scope (call it explicitly)

- IBM Quantum, Quantinuum, Pasqal, neutral-atom, photonic backends. They require different SDKs and break the "one swap point" claim. Mention as future work, don't start.
- Error mitigation, circuit cutting, ZNE, etc. Those are research projects, not integration surveys.
- Anything labeled "quantum advantage." Not the question being asked here.
