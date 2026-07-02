#!/usr/bin/env bash
# Fetch + convert Qwen/Qwen3-0.6B for the OPTIONAL real-weights demo of 401.
#
# The graded path (`uv run grade 401`) is fully hermetic — it runs against the tiny
# frozen float64 oracle in tests/fixtures/kv_cache.npz and needs no download. This
# script only populates real weights so you can watch your own `kv_generate` produce
# actual Qwen3 text (token-equal to the stateless 304/306 run, much faster).
#
# It reuses 306's converter, which downloads ONLY config.json + model.safetensors
# (~1.2 GB), converts HF weight names → Qwen3Params (no un-permute), and writes:
#
#   306_qk_norm/qwen3_0_6b.npz     full weights (git-ignored, shared with 306)
#
# Load those weights with 306's `load_qwen3`, build a `KVCache`, and call `kv_generate`.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
LEET_LLM_TARGET=solution uv run --group gen python "$REPO/306_qk_norm/convert.py"
echo "Done -> $REPO/306_qk_norm/qwen3_0_6b.npz"
echo
echo "Now watch your own KVCache generate real text:"
echo "  uv run python $REPO/401_kv_cache/kv_cache.py --prompt 'Once upon a time' --max-new-tokens 40"
