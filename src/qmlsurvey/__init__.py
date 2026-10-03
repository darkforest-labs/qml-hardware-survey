"""qml-hardware-survey: small hybrid model, many backends, honest comparisons."""

__version__ = "0.1.0"

# Bump when RunRecord fields change in a non-additive way.
# v1 -> v2: additive only. New optional blocks (default empty/None):
#   epoch_trace (per-epoch loss/acc/grad_l2/param_l2 inside TrainStats),
#   git, circuit_fingerprint, init_params_sha256, predictions,
#   hardware, billing. Old v1 readers should ignore unknown keys.
# v2 -> v3: additive only. New optional blocks (default empty):
#   device_snapshot (status / availability / queue depth / execution windows
#   captured before submission), device_executions (qml.Tracker totals for the
#   quantum model vs. the pre-run task estimate), estimated_tasks, and
#   broadcast_expanded (whether the batch was split per input, see model.py).
RUNRECORD_SCHEMA_VERSION = 3
