#!/usr/bin/env bash
# Fetch + convert Qwen/Qwen3-0.6B for the real-weights parity test in 306.
#
# Downloads ONLY the files we need (config.json + model.safetensors, ~1.2 GB),
# converts HF weight names → Qwen3Params with NO un-permute (rotate-half as-is),
# verifies our qwen3_forward matches HF logits, and writes:
#
#   306_qk_norm/qwen3_0_6b.npz                     weights (git-ignored)
#   306_qk_norm/tests/fixtures/real_ref.npz         committed reference
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LEET_LLM_TARGET=solution uv run --group gen python "$HERE/convert.py"
echo "Done -> $HERE/qwen3_0_6b.npz + $HERE/tests/fixtures/real_ref.npz"
