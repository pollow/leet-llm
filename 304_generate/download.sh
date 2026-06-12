#!/usr/bin/env bash
# Fetch + convert Karpathy's stories15M (the checkpoint llama3.np runs) for the real-weight
# story-generation capstone in 304 (also used by 303's parity test).
#
# If you already have ../../llama3.np/{stories15M.model.npz,tokenizer.model}, this just
# symlinks them. Otherwise it runs convert.py, which downloads ONLY the 3 needed files
# (config.json + model.safetensors + tokenizer.model, ~60 MB) from Xenova/llama2.c-stories15M
# and converts them to our .npz (un-permuting q/k from HF rotate-half to interleaved RoPE).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SIB="$HERE/../../llama3.np"
if [ -f "$SIB/stories15M.model.npz" ]; then
  ln -sf "$SIB/stories15M.model.npz" "$HERE/stories15M.model.npz"
  [ -f "$SIB/tokenizer.model" ] && ln -sf "$SIB/tokenizer.model" "$HERE/tokenizer.model"
  echo "linked sibling llama3.np weights"
  exit 0
fi
uv run --group gen python "$HERE/convert.py"
echo "Done -> $HERE/stories15M.model.npz + $HERE/tokenizer.model"
