---
license: cc-by-4.0
pretty_name: QML Hardware Survey
tags:
  - quantum-machine-learning
  - hybrid-models
  - benchmarks
  - pennylane
  - braket
task_categories:
  - tabular-classification
  - other
size_categories:
  - n<1K
configs:
  - config_name: runs
    data_files: data/runs.jsonl
  - config_name: epochs
    data_files: data/epochs.jsonl
  - config_name: predictions
    data_files: data/predictions.jsonl
---

# QML Hardware Survey

A capability-probe dataset comparing small hybrid quantum-classical models to
parameter-matched classical MLPs across simulators and (eventually) cloud
quantum hardware. Each row is one training run captured by the
[`qml-hardware-survey`](https://github.com/bshepp/qml-hardware-survey)
project.

This is a **hoarder-style** dataset: every run keeps full hyperparameters,
per-epoch loss/accuracy/grad-norm trace, test-set predictions, circuit
gate-counts, init-weight hashes, git SHA, and (where applicable) cloud
billing metadata. The intent is to enable secondary analysis (loss
landscapes, barren plateau scans, classical vs. quantum scaling) without
requiring a re-run.

## Splits / configs

* **runs** — one row per training run. Flat hyperparameters + final metrics
  + circuit fingerprint + billing.
* **epochs** — long format. One row per (`run_id`, `model_kind`, `epoch`)
  with `loss`, `train_acc`, `test_acc`, `grad_l2`, `param_l2`, `wall_s`.
* **predictions** — long format. One row per (`run_id`, `model_kind`,
  `sample_idx`) with `y_true`, `y_pred`, `logits`.

Join everything on `run_id` (plus `model_kind` for the long splits).

## Schema versioning

The upstream `RunRecord` schema is pinned via
`qmlsurvey.RUNRECORD_SCHEMA_VERSION`. Records currently live at v1
(Phase 0) and v2 (Phase 1 onward). v1 records contribute a row to `runs`
but emit no rows to `epochs` / `predictions` — those fields didn't exist
yet. The gap is intentional and visible.

## What this dataset is not

* Not an accuracy benchmark. Hybrid models here are tiny (4–12 qubits, 1–8
  layers) and toy tasks (parity, two-moons, MNIST-PCA-3) are picked for
  diagnosability, not for headline numbers.
* Not a barren-plateau dataset by itself, but the trainability sweep + the
  per-epoch `grad_l2` trace make it usable as input to one.
* Not a hardware leaderboard. Cloud QPU runs (when present) are single-seed
  capability probes, not statistically powered comparisons.

## License

Released under [CC-BY-4.0](LICENSE). Cite as:

```
@misc{qml-hardware-survey,
  author = {Sheppard, Brandon},
  title  = {QML Hardware Survey},
  year   = {2026},
  url    = {https://github.com/bshepp/qml-hardware-survey}
}
```
