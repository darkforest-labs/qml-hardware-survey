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

- 2026-06-20 inference call: **$0.0075**.
- 2026-10-04 micro-training run: **$0.4500**.
- Phase-2 cumulative: **$0.4575** of the $1.00 cap.

## Update 2026-10-03 — training path now reachable, and what it would cost

- PL #4462 is **still open** at `pennylane==0.45.1` / plugin `1.35.2`. The
  "wait for a fix" option did not pan out.
- `HybridModel` now wraps the QNode in `qml.transforms.broadcast_expand` on any
  finite-shot device (see `local-sims.md`, 2026-10-03). Training through the
  Braket plugin is therefore expected to work on SV1 — **inferred from
  `braket.local.qubit`, still not paid-to-confirm here.**
- The task fan-out is now predicted exactly by `catalog.estimate_task_count`:
  `B × (1 + 2 × 28)` tasks per training step on `B` inputs, plus `B + B_test`
  for the eval pass. At SV1's 3 s minimum per task ($0.00375), the parity
  reference config (204 / 52, 30 epochs) is **356,572 tasks ≈ $1,337**, far
  above every cap in `ROADMAP.md`. One epoch is ≈ $45. A 2-input, 1-epoch
  micro-run is 120 tasks ≈ $0.45 and is the proposed next paid call.
- Open question that call would settle: whether SV1 bills each parameter-shift
  task at the 3 s minimum (expected) or batches them (the plugin submits them
  as separate tasks, so no). **Settled below: 3 s minimum per task, exactly.**

## Second paid call — 2026-10-04 — first gradient computed on a Braket device

Record: `results/phase2/sv1_micro/2026-10-04T02-04-51+00-00_sv1_parity.json`
(schema v3, `experiment_group=phase2-sv1-micro`). Reproduce with:

```powershell
$env:QMLSURVEY_S3_BUCKET = "amazon-braket-qmlsurvey-<account-id>"
python -m qmlsurvey.runner --backend sv1 --task parity --epochs 1 --shots 100 `
    --seed 0 --n-train-subset 2 --n-test-subset 2 --max-cost-usd 0.60 `
    --experiment-group phase2-sv1-micro --out-dir results/phase2/sv1_micro
```

Stack: `pennylane==0.45.1`, `amazon-braket-pennylane-plugin==1.35.2`,
`amazon-braket-sdk==1.127.3.post0`, Python 3.13. Device snapshot at submit:
SV1 `ONLINE`, `is_available=True`, queue 0/0.

What ran: `HybridModel(4, 2)` on parity, **2 train / 2 test inputs, 1 epoch**,
shots=100, `broadcast_expand` path, Adam lr=0.05. One forward + one
parameter-shift backward on the train pair, one no-grad eval pass over both
pairs, one predictions pass over the test pair. Purpose: reconcile the task
fan-out and the bill, not accuracy (2 inputs cannot say anything about
accuracy; it was 0.5).

### Predicted vs executed vs billed

| quantity | predicted | observed |
|---|---|---|
| device tasks | 120 (`estimate_task_count(2, 2, 4, 2, 1)`) | **120** (`qml.Tracker`), 120 `COMPLETED` (Braket Tracker) |
| shots | 12,000 | 12,000 |
| actual execution duration | — | **1.56 s** total (13 ms / task) |
| billed execution duration | 360 s (120 × 3 s minimum) | **360.0 s** |
| cost | $0.4500 | **$0.4500** (Braket Tracker), ratio 1.00 |
| wall time, submit → last result | — | 314 s (≈ 2.6 s / task, sequential) |
| gradient | non-zero | `grad_l2 = 0.49` over the 54 model params |

### Findings

1. **The task-count model is exact on the cloud, not just locally.** Every
   parameter-shift evaluation is its own Braket task. 120 predicted, 120
   submitted, 120 completed, 120 billed.
2. **Billing is the 3 s floor × task count, nothing else.** 1.56 s of real
   compute was billed as 360 s. The $0.075/min rate never mattered; this
   workload is 230× overhead. There is no batching of parameter-shift tasks by
   the plugin or the service.
3. **The June cost-model fix is confirmed.** `estimate_cost_usd` with the real
   `n_tasks` matched the Braket Tracker to the cent. The pre-run estimate in
   the confirm prompt is now a number you can plan a budget on.
4. **The hybrid training loop closes against a cloud device.** Gradient
   computed, optimizer stepped, parameters changed (`param_l2` recorded). This
   is the first time the project has trained — rather than only inferred —
   through `braket.aws.qubit`. Still inferred, not confirmed, for `dm1` and
   for QPUs; the code path is identical but the devices are not.
5. **Wall time is submit/poll bound.** 2.6 s per task at `poll_interval=1 s`,
   strictly sequential. The plugin accepts `parallel=True`; untested here and
   not needed for 120 tasks, but a Phase-3 forward pass over 52 test inputs
   would otherwise take ~2–3 min of polling.
6. **The S3 destination works as configured.** `QMLSURVEY_S3_BUCKET` →
   `s3://amazon-braket-qmlsurvey-<account>/qmlsurvey/sv1/`. No bucket-name
   rejection this time (finding #1 above still applies to the prefix).

What this does *not* say: anything about accuracy, convergence, or whether
SV1 training is useful. With the parity reference batch costing ≈ $45/epoch
at this fan-out, SV1 training is a correctness check, not a workflow.
