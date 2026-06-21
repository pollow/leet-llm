#!/usr/bin/env bash
# Fetch + convert hf-internal-testing/tiny-random-GptOssForCausalLM for the
# real-weights parity test in 310 (Tier B — genuine-HF cross-check + loader cover).
#
# Downloads ONLY config.json + model.safetensors, converts HF weight names →
# GptOssParams with NO un-permute (rotate-half as-is), forces eager attention +
# default RoPE (the checkpoint's YaRN scaling is deferred to 311), verifies our
# gptoss_forward matches genuine HF logits, and writes:
#
#   310_gptoss_model/gptoss_tiny.npz                 weights (git-ignored)
#   310_gptoss_model/tests/fixtures/real_ref.npz     committed reference
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LEET_LLM_TARGET=solution uv run --group gen python "$HERE/convert.py"
echo "Done -> $HERE/gptoss_tiny.npz + $HERE/tests/fixtures/real_ref.npz"
