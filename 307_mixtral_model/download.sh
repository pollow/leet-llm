#!/usr/bin/env bash
# Fetch + convert hf-internal-testing/tiny-random-MixtralForCausalLM for the
# real-weights parity test in 307.
#
# Downloads ONLY the files we need (config.json + model.safetensors, ~2 MB),
# converts HF weight names → MixtralParams with NO un-permute (rotate-half as-is),
# verifies our mixtral_forward matches HF logits, and writes:
#
#   307_mixtral_model/mixtral_tiny.npz                  weights (git-ignored)
#   307_mixtral_model/tests/fixtures/real_ref.npz       committed reference
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LEET_LLM_TARGET=solution uv run --group gen python "$HERE/convert.py"
echo "Done -> $HERE/mixtral_tiny.npz + $HERE/tests/fixtures/real_ref.npz"
