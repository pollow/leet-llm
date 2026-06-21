#!/usr/bin/env bash
# Fetch + convert llamafactory/tiny-random-Llama-3 for the real-weights parity
# test in 311 (Tier B — genuine-HF cross-check + loader cover).
#
# This ungated tiny checkpoint ships an ACTIVE rope_type=llama3 long-context
# schedule — exactly the delta 311 implements — so nothing is forced (unlike 310).
# Downloads ONLY config.json + model.safetensors, converts HF weight names →
# Llama31Params with NO un-permute (rotate-half as-is), verifies our llama31_forward
# matches genuine HF logits, and writes:
#
#   311_llama31_model/llama31_tiny.npz               weights (git-ignored)
#   311_llama31_model/tests/fixtures/real_ref.npz    committed reference
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LEET_LLM_TARGET=solution uv run --group gen python "$HERE/convert.py"
echo "Done -> $HERE/llama31_tiny.npz + $HERE/tests/fixtures/real_ref.npz"
