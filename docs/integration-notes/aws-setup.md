# AWS / Braket setup for Phase 2

> **Scope.** Everything needed to make `python scripts/doctor.py` pass, so
> Phase-2 cloud-sim runs (`sv1`, `dm1`) can be issued with confidence.
> Nothing in this note costs money on its own. The first paid call is
> tracked separately in `sv1.md` (created when Phase 2 actually runs).

## 0. What gets billed in this phase

- **Cloud simulators** (`sv1`, `dm1`): \$0.075 / minute, billed in
  whole-second granularity, 15 s minimum per task. There is **no** per-task
  fee on cloud sims.
- **Phase-2 cap**: \$1 cumulative across all calls.
- The cost estimator in `qmlsurvey.catalog.estimate_cost_usd` defaults to
  `estimated_runtime_minutes=0.05` (3 s). The first real billed run (`sv1.md`,
  2026-06-20) confirmed it under-estimates: it models a *single* 3 s task and
  ignores task count, but each broadcast input becomes its own task floored at
  the 3 s minimum, so real cost ≈ `N_tasks × $0.00375`. Fix the estimator to
  take an explicit task count; never raise the cap to compensate.

## 1. AWS account + region

- Use an existing AWS account or create one at <https://aws.amazon.com/>.
- Pick **one** active region for Phase 2. Recommended: `us-east-1` — it
  hosts both `sv1` and `dm1` and is the IonQ region (relevant later).
  - Rigetti devices (Cepheus-1-108Q) live in `us-west-1`.
  - IQM (Garnet, Emerald) and AQT Ibex-Q1 live in `eu-north-1`.
- Cloud simulators (`sv1`, `dm1`) are listed as `region="all"` in
  the catalog because Braket exposes them in every Braket-enabled region;
  pick whichever region matches your S3 bucket to avoid cross-region data
  charges.

## 2. Braket service activation

- Console → **Amazon Braket** → click through the one-time service
  activation. This writes a service-linked role
  (`AWSServiceRoleForAmazonBraket`) into IAM. No charge.
- Verify in the console that **Devices** lists `SV1`, `DM1`, etc. in the
  region you picked.

## 3. S3 results bucket

Braket writes task results to S3. Create one bucket dedicated to this
project so cleanup and cost attribution are trivial.

> **The bucket name MUST start with `amazon-braket-`.** The Braket
> service-linked role only grants S3 access to `amazon-braket-*` buckets; any
> other name is rejected at `CreateQuantumTask` with
> `ValidationException: ... does not start with 'amazon-braket-'`. Verified the
> hard way in `sv1.md` (2026-06-20).

```powershell
$bucket = "amazon-braket-qmlsurvey-<your-account-id>"
aws s3api create-bucket `
    --bucket $bucket `
    --region us-east-1
aws s3api put-public-access-block `
    --bucket $bucket `
    --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

(For regions other than `us-east-1`, add
`--create-bucket-configuration LocationConstraint=<region>`.)

Export the bucket name for `scripts/doctor.py` to probe:

```powershell
$env:QMLSURVEY_S3_BUCKET = $bucket
```

## 4. IAM identity

Two acceptable shapes; pick one.

### 4a. IAM user with access keys (simplest, fine for a personal account)

- IAM → Users → **Create user** → no console access needed.
- Attach the AWS-managed policy **`AmazonBraketFullAccess`** plus a
  minimal S3 policy scoped to the bucket from §3:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket", "s3:DeleteObject"],
      "Resource": [
        "arn:aws:s3:::amazon-braket-qmlsurvey-<account-id>",
        "arn:aws:s3:::amazon-braket-qmlsurvey-<account-id>/*"
      ]
    }
  ]
}
```

- Create an access key for the user, then locally:

```powershell
aws configure  # paste access key, secret, region (us-east-1), output (json)
```

### 4b. IAM Identity Center (SSO)

If you're on an org account with SSO:

```powershell
aws configure sso
aws sso login --profile <profile>
$env:AWS_PROFILE = "<profile>"
```

The same `AmazonBraketFullAccess` + scoped S3 permissions must be on the
permission set you assume.

## 5. Verify

```powershell
$env:PYTHONNOUSERSITE = "1"
python scripts/doctor.py
```

Expected: every required check `[PASS]`, S3 either `[PASS]` (bucket env
set) or `[SKIP]` (env unset). If `Braket devices visible` returns 0,
either the region is wrong, the IAM policy is missing
`braket:SearchDevices`, or Braket isn't activated in that region.

## 6. What `doctor` does **not** check

- It never creates a Braket task → never spends money.
- It does not validate that `sv1`/`dm1` are *online* right now; that's
  surfaced in the device status string from `search_devices`. A QPU may
  be `OFFLINE` for maintenance even when permissions are fine.
- It does not verify the cost estimator. That is calibrated by comparing
  estimated vs billed in the per-device integration notes after the
  first real run.

## 7. Pre-paid-call checklist

Before invoking anything against `sv1` or `dm1`:

1. `python scripts/doctor.py` returns `OK`.
2. `--max-cost-usd` is set on every call (Phase-2 contract: \$1 cap total).
3. The cumulative spend so far in this phase is recorded somewhere I can
   read (running tally in `sv1.md` once that note exists).
4. The call uses `shots <= 1000` and a single epoch unless explicitly
   noted otherwise.

## 8. Live preflight result — 2026-06-19

`python scripts/doctor.py` was run against a real account (read-only, no paid
call). All required checks `[PASS]`:

```
[PASS] boto3 import: boto3 1.42.97
[PASS] amazon-braket-sdk import: version 1.110.1
[PASS] amazon-braket-pennylane-plugin import: ok
[PASS] AWS credentials: resolved via shared-credentials-file
[PASS] AWS region: us-east-1
[PASS] Braket devices visible: 10 devices in this region
[SKIP] S3 bucket writable: skipped (set QMLSURVEY_S3_BUCKET to enable)
```

A direct `aws braket search-devices` across the three regions the catalog
targets returned the following live status — this is the authoritative
source that drove the 2026-06 catalog refresh:

| Region | Device | Provider | Status |
|--------|--------|----------|--------|
| us-east-1 | SV1 / DM1 | Amazon Braket | ONLINE |
| us-east-1 | TN1 | Amazon Braket | RETIRED (re-checked 2026-10-03; was listed ONLINE in June) |
| us-east-1 | Forte Enterprise 1 | IonQ | ONLINE |
| us-east-1 | Forte 1 | IonQ | OFFLINE in June; ONLINE on 2026-10-03 |
| us-east-1 | Aria 1 / Aria 2 / Harmony | IonQ | RETIRED |
| us-east-1 | Aquila | QuEra | ONLINE (neutral-atom; out of scope) |
| us-west-1 | Cepheus-1-108Q | Rigetti | ONLINE |
| us-west-1 | Ankaa-3 / Ankaa-2 / Aspen-* | Rigetti | RETIRED |
| eu-north-1 | Garnet / Emerald | IQM | ONLINE |
| eu-north-1 | IBEX Q1 | AQT | ONLINE |

Two integration findings worth carrying into Phase 2/3:

1. **IonQ Forte-1 was OFFLINE at snapshot time** while Forte-Enterprise-1 was
   ONLINE. If a Phase-3 IonQ run is wanted before Forte-1 returns, target
   `.../qpu/ionq/Forte-Enterprise-1` instead. QPU `OFFLINE` is a transient
   maintenance state, distinct from `RETIRED`.
2. **The S3 results bucket is not yet provisioned** (`QMLSURVEY_S3_BUCKET`
   unset → that check SKIPs). Braket QPU/cloud-sim tasks write results to S3,
   so before the first paid call create the bucket per §3 and export the var.

## 9. Re-check 2026-10-03 — devices, windows, queue

A live `search_devices` over all five Braket regions (`us-east-1`, `us-west-1`,
`us-west-2`, `eu-north-1`, `eu-west-2`) plus `AwsDevice.properties` for each
in-catalog QPU. Observed, not interpreted:

| Device | Status | API `qubitCount` | Price | Queue (Normal) | Execution windows (UTC) |
|---|---|---|---|---|---|
| Rigetti Cepheus-1-108Q | ONLINE | 107 | $0.000425/shot + $0.30/task | 0 | every day 00:00–06:59, 09:00–19:00, 21:00–23:59 |
| IQM Garnet | ONLINE | 20 | $0.00145/shot + $0.30/task | 0 | weekdays 00:00–01:29, 03:15–15:29, 17:15–23:59 |
| IQM Emerald | ONLINE | 54 | $0.0016/shot + $0.30/task | 4 | weekdays 00:00–03:29, 07:20–23:59 |
| AQT Ibex-Q1 | ONLINE | 12 | $0.0235/shot + $0.30/task | 0 | Mon 11–15, Tue/Wed/Fri 08–15 |
| IonQ Forte-1 | ONLINE | 36 | $0.08/shot + $0.30/task | 0 | daily, 30-min gap at 18:00 (14:00–20:00 Wed) |
| IonQ Forte-Enterprise-1 | ONLINE | 36 | $0.08/shot + $0.30/task | 0 | daily, gap Thu 08:00–14:00 |
| QuEra Aquila (out of scope) | ONLINE | 256 | $0.01/shot + $0.30/task | 0 | Fri 04:00 → Mon 03:59 with gaps |
| SV1 / DM1 | ONLINE | — | $0.075/min, 3 s minimum per task | — | always |
| TN1 | **RETIRED** | — | — | — | — |

Retired but still returned by the API: Ankaa-2, Ankaa-3, all Aspens, IonQ
Aria-1 / Aria-2 / Harmony, Xanadu Borealis, OQC Lucy, all D-Wave systems.

Findings:

1. **Execution windows, not queue depth, set the wait.** At 2026-10-03 ~19:30
   UTC (a Saturday) every in-catalog QPU except IonQ reported
   `is_available=False` with an empty queue. A task submitted then would sit
   until the next window. `scripts/doctor.py` now prints this per QPU, and
   `runner.run` stores a `device_snapshot` in each `RunRecord` and shows the
   in-window flag in the confirm prompt.
2. **Cepheus reports 107 qubits**, not the 108 in its name. Catalog updated.
3. **Prices unchanged** since June for every in-catalog device. The pricing
   page lists only SV1's per-minute rate; DM1 is billed the same in practice
   (see `sv1.md`), TN1 is gone.
4. **This AWS account is shared with `rydberg-playground`** (Aquila runs,
   ≈ $68.50 estimated spend to date there). Cost Explorer totals will mix the
   two projects unless filtered by device ARN or tag.
