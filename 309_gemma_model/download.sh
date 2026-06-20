#!/usr/bin/env bash
# Fetch + convert hf-internal-testing/tiny-random-Gemma2ForCausalLM for the
# real-weights parity test in 309.
#
# Downloads ONLY config.json + model.safetensors, converts HF weight names →
# GemmaParams with NO un-permute (rotate-half as-is), verifies our gemma_forward
# matches genuine HF logits, and writes:
#
#   309_gemma_model/gemma_tiny.npz                   weights (git-ignored)
#   309_gemma_model/tests/fixtures/real_ref.npz      committed reference
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LEET_LLM_TARGET=solution uv run --group gen python "$HERE/convert.py"
echo "Done -> $HERE/gemma_tiny.npz + $HERE/tests/fixtures/real_ref.npz"
