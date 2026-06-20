# SV1 — Amazon Braket on-demand state-vector simulator

Phase-2 first paid contact. Region `us-east-1`. Snapshot 2026-06-20.
External register: what was observed, not interpretation.

## Setup

- Versions: `pennylane==0.42.3`, `amazon-braket-pennylane-plugin==1.33.7`,
  `amazon-braket-sdk==1.110.1`, `boto3==1.42.97`.
- Device ARN: `arn:aws:braket:::device/quantum-simulator/amazon/sv1`.
- Credentials: IAM user access key (`shared-credentials-file`), region
  `us-east-1`.
- S3 results bucket: `amazon-braket-qmlsurvey-290318879194` (us-east-1),
  public access fully blocked. **Braket requires the bucket name to start with
  `amazon-braket-`** — see Findings.
- `python scripts/doctor.py` with `QMLSURVEY_S3_BUCKET` set: all required
  checks `[PASS]`, S3 put+delete probe `[PASS]`.

## First paid call — inference only

A *training* run via the current code path is **not expected to work** on SV1.
The model broadcasts the batch through the QNode (`model.py`, hard rule #4),
and with `diff_method="best"` a Braket device routes gradients through
PennyLane's parameter-shift transform, which does not support broadcasted
parameters (PL #4462) — `loss.backward()` raises `NotImplementedError`.

Scope of evidence: this failure is **confirmed empirically on
`braket.local.qubit`** (all three cross-sim training runs, this session). It is
**inferred, not paid-to-confirm, on SV1** — I ran only the forward call. The one
caveat specific to SV1: SV1 has a native server-side *adjoint gradient* that
`"best"` might select instead of parameter-shift, in which case a training
attempt would fail *differently* (that path has its own no-broadcast / `shots=0`
/ single-observable constraints) or, in a narrow config, might succeed. Settling
that costs a small SV1 spend we have not made. Either way, the first paid call
here is a **forward pass only**, mirroring the option-1 inference path used for
`braket.local.qubit`.

- Model: `HybridModel(n_qubits=4, n_layers=2)`, task `parity`, `shots=100`.
- Inputs: **2** parity test points (deliberately tiny to minimise cost).
- Outcome: returned logits + predictions; the call completed successfully.

## Observations — billed vs estimated

Captured with `braket.tracking.Tracker`:

| metric | value |
|---|---|
| tasks submitted | 2 (one per broadcast input), both `COMPLETED` |
| shots | 200 (100 × 2) |
| actual execution duration | 0.005 s (5 ms) |
| **billed execution duration** | **6 s** (2 tasks × 3 s minimum) |
| **billed cost (Tracker)** | **$0.0075** |
| `catalog.estimate_cost_usd("sv1", 100)` | $0.00375 |
| wall time incl. submit/poll | 7.5 s |

### Cost-model finding (calibrates the estimator)

`catalog.estimate_cost_usd` returns a flat $0.00375 for SV1 — it models a
single 3 s task at $0.075/min and **ignores how many tasks a run actually
submits**. The real driver is the **per-task 3 s minimum × task count**:
broadcasting over `N` inputs submits `N` tasks, each floored at 3 s, so real
cost ≈ `N × $0.00375`. For `N=2` that is exactly $0.0075 — double the estimate.
A training run would multiply this again by (forward + gradient task count) ×
epochs. Follow-up: make `estimate_cost_usd` take an explicit task count
(= broadcast width × steps) instead of one fixed runtime. Per the cost
discipline, do **not** raise caps to compensate — fix the estimator.

## Findings

1. **S3 results bucket name must start with `amazon-braket-`.** A bucket named
   `qmlsurvey-results-*` is rejected at `CreateQuantumTask` with
   `ValidationException: The bucket ... does not start with 'amazon-braket-'`.
   The Braket service-linked role only grants S3 access to `amazon-braket-*`
   buckets. `aws-setup.md` §3 has been corrected. (No task was created on the
   rejected attempt, so it cost nothing.)
2. **Training via the current broadcasted path is blocked** — confirmed on
   `braket.local.qubit` (PL #4462), inferred on SV1 (not paid-to-confirm; see
   the caveat above about SV1's native adjoint gradient). Forward/inference is
   the reachable path. Whether PennyLane ≥0.43 fixes #4462 could not be tested
   here — the environment's package index is frozen at 0.42.3 (see
   `local-sims.md`).
3. **The 3 s per-task minimum dominates tiny-circuit cost.** Actual compute was
   5 ms; billed was 6 s. Batching more shots into a single task is far cheaper
   than spreading work across many minimum-billed tasks.

## Spend

- This call: **$0.0075**. Phase-2 cumulative: **$0.0075** of the $1.00 cap.
