"""Roll up everything under ``results/`` into ``results/SUMMARY.md``.

Schema-aware: dispatches on ``experiment_group`` (with ``schema_version``
gating). Three groups are currently understood:

- ``phase1-cross-sim``    — one ``RunRecord`` per file (cross-sim matrix).
- ``phase1-shot-noise``   — one summary file with a ``cells`` list.
- ``phase1-trainability`` — one summary file with a ``cells`` list.
- ``phase2-*`` (any group starting with ``phase2``) — one ``RunRecord`` per
  file from a Braket cloud simulator or QPU; tabulated with estimated vs
  tracked tasks and estimated vs billed cost.

Anything else with a recognised ``schema_version`` but unknown group is
listed as "unrecognised" with its path and group, so new result kinds show
up rather than being silently dropped.

Pure-stdlib on purpose — keeps `qmlsurvey`'s footprint small.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_PATH = RESULTS_DIR / "SUMMARY.md"

SUPPORTED_SCHEMA = {1, 2, 3}


def _load_all() -> list[tuple[Path, dict]]:
    out: list[tuple[Path, dict]] = []
    for p in sorted(RESULTS_DIR.rglob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            out.append((p, data))
    return out


def _fmt(v: Any, kind: str = "g") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if kind == "acc":
            return f"{v:.4f}"
        if kind == "loss":
            return f"{v:.6f}"
        if kind == "wall":
            return f"{v:.2f}"
        if kind == "sci":
            return f"{v:.3e}"
        return f"{v:g}"
    return str(v)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _cross_sim_section(records: list[dict]) -> str:
    records_sorted = sorted(records, key=lambda r: (r.get("backend", ""), r.get("task", "")))
    headers = [
        "backend",
        "task",
        "n_qubits",
        "n_layers",
        "shots",
        "epochs",
        "q_test_acc",
        "q_loss",
        "q_wall_s",
        "cls_test_acc",
        "cls_wall_s",
    ]
    rows = []
    for r in records_sorted:
        q = r.get("quantum", {}) or {}
        c = r.get("classical_baseline", {}) or {}
        rows.append(
            [
                str(r.get("backend", "—")),
                str(r.get("task", "—")),
                _fmt(r.get("n_qubits")),
                _fmt(r.get("n_layers")),
                _fmt(r.get("shots")) if r.get("shots") is not None else "analytic",
                _fmt(r.get("epochs")),
                _fmt(q.get("final_test_acc"), "acc"),
                _fmt(q.get("final_loss"), "loss"),
                _fmt(q.get("wall_time_s"), "wall"),
                _fmt(c.get("final_test_acc"), "acc"),
                _fmt(c.get("wall_time_s"), "wall"),
            ]
        )
    return _md_table(headers, rows)


def _cloud_section(records: list[dict]) -> str:
    """Paid runs: what we predicted vs what the device executed and AWS billed."""
    records_sorted = sorted(records, key=lambda r: r.get("timestamp_utc", ""))
    headers = [
        "timestamp_utc",
        "backend",
        "task",
        "n_train/n_test",
        "epochs",
        "shots",
        "est_tasks",
        "executed",
        "est_cost_usd",
        "billed_usd",
        "billed_s",
        "q_test_acc",
        "cls_test_acc",
        "avail_at_submit",
    ]
    rows = []
    for r in records_sorted:
        q = r.get("quantum", {}) or {}
        c = r.get("classical_baseline", {}) or {}
        b = r.get("billing", {}) or {}
        ex = r.get("device_executions", {}) or {}
        snap = r.get("device_snapshot", {}) or {}
        stats = b.get("quantum_tasks_statistics", {}) or {}
        billed_s = None
        for st in stats.values():
            billed_s = (billed_s or 0.0) + float(st.get("billed_execution_duration_s", 0.0))
        rows.append(
            [
                str(r.get("timestamp_utc", "—")),
                str(r.get("backend", "—")),
                str(r.get("task", "—")),
                f"{r.get('n_train', '—')}/{r.get('n_test', '—')}",
                _fmt(r.get("epochs")),
                _fmt(r.get("shots")),
                _fmt(r.get("estimated_tasks")),
                _fmt(ex.get("executions")),
                _fmt(r.get("estimated_cost_usd"), "acc"),
                _fmt(b.get("billed_cost_usd"), "acc"),
                _fmt(billed_s, "wall"),
                _fmt(q.get("final_test_acc"), "acc"),
                _fmt(c.get("final_test_acc"), "acc"),
                str(snap.get("is_available", "—")),
            ]
        )
    return _md_table(headers, rows)


def _shot_noise_section(summary: dict) -> str:
    ref = summary.get("reference", {}) or {}
    head = (
        f"Reference: `{ref.get('backend')}` analytic, "
        f"final_test_acc={_fmt(ref.get('final_test_acc'), 'acc')}, "
        f"final_loss={_fmt(ref.get('final_loss'), 'loss')}, "
        f"n_trials={summary.get('n_trials')}.\n\n"
    )
    headers = ["backend", "shots", "acc_mean", "acc_std", "acc_min", "acc_max", "note"]
    rows = []
    for c in summary.get("cells", []):
        err = c.get("error")
        shots_s = _fmt(c.get("shots")) if c.get("shots") is not None else "analytic"
        if err:
            note = err.split("\n")[0][:80]
            rows.append(
                [str(c.get("backend", "—")), shots_s, "—", "—", "—", "—", note]
            )
        else:
            rows.append(
                [
                    str(c.get("backend", "—")),
                    shots_s,
                    _fmt(c.get("acc_mean"), "acc"),
                    _fmt(c.get("acc_std"), "acc"),
                    _fmt(c.get("acc_min"), "acc"),
                    _fmt(c.get("acc_max"), "acc"),
                    "",
                ]
            )
    return head + _md_table(headers, rows)


def _trainability_section(summary: dict) -> str:
    cells = summary.get("cells", [])
    if not cells:
        return "_(no cells)_"
    nq_set = sorted({c["n_qubits"] for c in cells})
    nl_set = sorted({c["n_layers"] for c in cells})
    by_cell = {(c["n_qubits"], c["n_layers"]): c for c in cells}
    head = (
        f"Backend: `{summary.get('backend')}`, "
        f"inits/cell={summary.get('n_inits_per_cell')}, "
        f"batch={summary.get('batch')}, "
        f"init={summary.get('init_distribution')}, "
        f"loss={summary.get('loss')}, "
        f"wall={_fmt(summary.get('wall_time_s_total'), 'wall')}s.\n\n"
        f"`grad_var_mean` (Var across inits, mean over qparams):\n\n"
    )
    headers = ["n_qubits \\ n_layers"] + [str(nl) for nl in nl_set]
    rows = []
    for nq in nq_set:
        row = [str(nq)]
        for nl in nl_set:
            cell = by_cell.get((nq, nl))
            row.append(_fmt(cell["grad_var_mean"], "sci") if cell else "—")
        rows.append(row)
    return head + _md_table(headers, rows)


def main() -> None:
    items = _load_all()
    cross_sim: list[dict] = []
    shot_noise: list[tuple[Path, dict]] = []
    trainability: list[tuple[Path, dict]] = []
    phase0: list[dict] = []
    cloud: list[dict] = []
    other: list[tuple[Path, dict]] = []

    for p, d in items:
        # Per-cell trainability files have no schema_version / group; skip.
        if {"n_qubits", "n_layers", "grad_var_mean"}.issubset(d.keys()) and (
            "cells" not in d
        ):
            continue
        sv = d.get("schema_version")
        if sv not in SUPPORTED_SCHEMA:
            other.append((p, d))
            continue
        group = d.get("experiment_group")
        if group == "phase1-cross-sim":
            cross_sim.append(d)
        elif group == "phase1-shot-noise":
            shot_noise.append((p, d))
        elif group == "phase1-trainability":
            trainability.append((p, d))
        elif group == "phase0-reference":
            phase0.append(d)
        elif isinstance(group, str) and group.startswith("phase2"):
            cloud.append(d)
        else:
            other.append((p, d))

    lines: list[str] = [
        "# Results summary",
        "",
        "Auto-generated by `scripts/rollup.py`. Do not edit by hand — re-run "
        "the script after adding new results.",
        "",
    ]

    if phase0:
        lines += ["## Phase 0 — reference runs", "", _cross_sim_section(phase0), ""]

    if cross_sim:
        lines += ["## Phase 1 — cross-simulator matrix", "", _cross_sim_section(cross_sim), ""]

    for p, summary in shot_noise:
        rel = p.relative_to(ROOT).as_posix()
        lines += [
            f"## Phase 1 — shot-noise sweep (`{summary.get('task')}`)",
            "",
            f"Source: `{rel}`",
            "",
            _shot_noise_section(summary),
            "",
        ]

    for p, summary in trainability:
        rel = p.relative_to(ROOT).as_posix()
        lines += [
            "## Phase 1 — trainability / gradient-variance sweep",
            "",
            f"Source: `{rel}`",
            "",
            _trainability_section(summary),
            "",
        ]

    if cloud:
        lines += [
            "## Phase 2 — Braket cloud runs (paid)",
            "",
            "Estimated vs executed task counts and estimated vs billed cost, per run. "
            "`billed_s` is AWS's billed execution duration (3 s minimum per task on "
            "the on-demand simulators). `avail_at_submit` is the device's "
            "`is_available` flag when the run started.",
            "",
            _cloud_section(cloud),
            "",
        ]

    if other:
        lines += ["## Unrecognised result files", ""]
        for p, d in other:
            rel = p.relative_to(ROOT).as_posix()
            lines.append(
                f"- `{rel}` (schema_version={d.get('schema_version')!r}, "
                f"experiment_group={d.get('experiment_group')!r})"
            )
        lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH.relative_to(ROOT).as_posix()}")
    print(
        f"  cloud records: {len(cloud)}  cross_sim records: {len(cross_sim)}  "
        f"shot_noise files: {len(shot_noise)}  "
        f"trainability files: {len(trainability)}  "
        f"phase0 records: {len(phase0)}  "
        f"other: {len(other)}"
    )


if __name__ == "__main__":
    main()
