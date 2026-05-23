"""qml-hardware-survey: small hybrid model, many backends, honest comparisons."""

__version__ = "0.1.0"

# Bump when RunRecord fields change in a non-additive way.
# v1 -> v2: additive only. New optional blocks (default empty/None):
#   epoch_trace (per-epoch loss/acc/grad_l2/param_l2 inside TrainStats),
#   git, circuit_fingerprint, init_params_sha256, predictions,
#   hardware, billing. Old v1 readers should ignore unknown keys.
RUNRECORD_SCHEMA_VERSION = 2
