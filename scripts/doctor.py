"""Pre-flight check for Phase-2 AWS work.

Read-only. Never makes a paid call. Useful before running anything that
hits Braket, so a missing IAM permission or wrong region surfaces here
rather than mid-experiment.

Checks (each independent; one failure does not abort the others):
  1. boto3 importable, version reported.
  2. amazon-braket-sdk importable, version reported.
  3. amazon-braket-pennylane-plugin importable (constructing the device
     factory without ARN, so no AWS call).
  4. AWS credentials resolvable via the default boto3 chain.
  5. Active region (env var, then session default).
  6. Braket-visible devices in the active region (search_devices: free).
  7. Optional S3 bucket writable: only runs if QMLSURVEY_S3_BUCKET is set.
     Writes + deletes a tiny probe key under ``probes/doctor-<ts>.txt``.

Exit code is 0 if every *required* check passes; non-zero otherwise.
S3 is treated as optional — missing bucket env => skipped, not failed.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    optional: bool = False


def _check_boto3() -> CheckResult:
    try:
        import boto3

        return CheckResult("boto3 import", True, f"boto3 {boto3.__version__}")
    except Exception as e:
        return CheckResult("boto3 import", False, repr(e))


def _check_braket_sdk() -> CheckResult:
    try:
        import braket._sdk as sdk

        return CheckResult("amazon-braket-sdk import", True, f"version {sdk.__version__}")
    except Exception:
        try:
            import braket  # noqa: F401

            return CheckResult("amazon-braket-sdk import", True, "version unknown")
        except Exception as e2:
            return CheckResult("amazon-braket-sdk import", False, repr(e2))


def _check_braket_plugin() -> CheckResult:
    # Don't construct an AWS device — that needs an ARN and credentials.
    # Just import the plugin module to confirm pennylane will resolve it.
    try:
        import importlib

        importlib.import_module("braket.pennylane_plugin")
        return CheckResult("amazon-braket-pennylane-plugin import", True, "ok")
    except Exception as e:
        return CheckResult("amazon-braket-pennylane-plugin import", False, repr(e))


def _check_aws_credentials() -> CheckResult:
    try:
        import boto3

        sess = boto3.Session()
        creds = sess.get_credentials()
        if creds is None:
            return CheckResult(
                "AWS credentials", False, "boto3 found no credentials in the default chain"
            )
        frozen = creds.get_frozen_credentials()
        method = getattr(creds, "method", "unknown")
        akid = (frozen.access_key or "")[:4] + "…"
        return CheckResult("AWS credentials", True, f"resolved via {method} (akid={akid})")
    except Exception as e:
        return CheckResult("AWS credentials", False, repr(e))


def _check_region() -> CheckResult:
    try:
        import boto3

        sess = boto3.Session()
        region = sess.region_name or os.environ.get("AWS_REGION") or os.environ.get(
            "AWS_DEFAULT_REGION"
        )
        if not region:
            return CheckResult(
                "AWS region",
                False,
                "no region in session, AWS_REGION, or AWS_DEFAULT_REGION",
            )
        return CheckResult("AWS region", True, region)
    except Exception as e:
        return CheckResult("AWS region", False, repr(e))


def _check_braket_devices() -> CheckResult:
    """List Braket devices via the control-plane API (free, no task creation)."""
    try:
        import boto3

        client = boto3.client("braket")
        # search_devices is paginated; cap to a few pages so we don't spin.
        names: list[str] = []
        statuses: dict[str, str] = {}
        next_token = None
        for _ in range(3):
            kwargs = {"filters": []}
            if next_token:
                kwargs["nextToken"] = next_token
            resp = client.search_devices(**kwargs)
            for d in resp.get("devices", []):
                n = d.get("deviceName", "?")
                names.append(n)
                statuses[n] = d.get("deviceStatus", "?")
            next_token = resp.get("nextToken")
            if not next_token:
                break
        if not names:
            return CheckResult(
                "Braket devices visible", False, "search_devices returned 0 devices"
            )
        sample = ", ".join(f"{n}({statuses[n]})" for n in names[:6])
        return CheckResult(
            "Braket devices visible",
            True,
            f"{len(names)} devices in this region; sample: {sample}",
        )
    except Exception as e:
        return CheckResult("Braket devices visible", False, repr(e))


def _check_s3_bucket() -> CheckResult:
    bucket = os.environ.get("QMLSURVEY_S3_BUCKET")
    if not bucket:
        return CheckResult(
            "S3 bucket writable",
            True,
            "skipped (set QMLSURVEY_S3_BUCKET to enable)",
            optional=True,
        )
    try:
        import boto3

        s3 = boto3.client("s3")
        key = f"probes/doctor-{int(time.time())}.txt"
        body = b"qmlsurvey doctor probe"
        s3.put_object(Bucket=bucket, Key=key, Body=body)
        s3.delete_object(Bucket=bucket, Key=key)
        return CheckResult("S3 bucket writable", True, f"put+delete on s3://{bucket}/{key}")
    except Exception as e:
        return CheckResult("S3 bucket writable", False, repr(e), optional=True)


CHECKS = [
    _check_boto3,
    _check_braket_sdk,
    _check_braket_plugin,
    _check_aws_credentials,
    _check_region,
    _check_braket_devices,
    _check_s3_bucket,
]


def main() -> int:
    print("qmlsurvey doctor — Phase-2 pre-flight")
    print("=" * 54)
    results: list[CheckResult] = []
    for fn in CHECKS:
        try:
            r = fn()
        except Exception:
            r = CheckResult(fn.__name__, False, traceback.format_exc(limit=1).strip())
        results.append(r)
        tag = "PASS" if r.ok else ("SKIP" if r.optional else "FAIL")
        print(f"  [{tag}] {r.name}: {r.detail}")
    print("=" * 54)
    required_failures = [r for r in results if not r.ok and not r.optional]
    if required_failures:
        print(f"FAIL: {len(required_failures)} required check(s) failed.")
        return 1
    print("OK: all required checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
