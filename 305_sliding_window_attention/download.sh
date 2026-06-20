#!/usr/bin/env bash
# Fetch + convert hf-internal-testing/tiny-random-MistralForCausalLM for the
# real-weights parity test in 305.
#
# Downloads ONLY the files we need (config.json + model.safetensors, ~2 MB),
# converts HF weight names → MistralParams with NO un-permute (rotate-half as-is),
# verifies our mistral_forward matches HF logits, and writes:
#
#   305_sliding_window_attention/mistral_tiny.npz         weights (git-ignored)
#   305_sliding_window_attention/tests/fixtures/real_ref.npz  committed reference
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
uv run --group gen python "$HERE/convert.py"
echo "Done -> $HERE/mistral_tiny.npz + $HERE/tests/fixtures/real_ref.npz"
