#!/usr/bin/env bash
# 308 — download real DeepSeek-V3 weights for the parity test.
#
# NOTE: Both currently available public tiny DeepSeek-V3 checkpoints use Yarn RoPE
# with rope_interleave=True, which requires an implementation beyond the scope of
# this task. The real-weights parity test (B) is therefore not included.
#
# When a tiny checkpoint with default half-rotate RoPE becomes available, run:
#   uv run --group gen python 308_deepseek_model/convert.py
# to download, convert, and commit real_ref.npz.
#
# See README.md for details.
echo "No suitable public tiny DeepSeek-V3 checkpoint available yet."
echo "See 308_deepseek_model/README.md for details."
