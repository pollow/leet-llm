#!/usr/bin/env bash
# Fetch the real opus-mt-en-zh checkpoint, then convert it to our .npz layout.
# CC-BY-4.0 (Helsinki-NLP / OPUS-MT). Weights are NOT committed; this is opt-in.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
uv run --group gen python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Helsinki-NLP/opus-mt-en-zh")
print("downloaded to", p)
PY
uv run --group gen python "$HERE/convert.py"
echo "Done -> $HERE/opus_mt_en_zh.npz"
