#!/usr/bin/env bash
# Fetch Karpathy's stories15M (MIT) as our .npz, for the real-weight parity test.
# If you already have ../llama3.np/stories15M.model.npz, this just symlinks it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SIB="$HERE/../../llama3.np/stories15M.model.npz"
if [ -f "$SIB" ]; then ln -sf "$SIB" "$HERE/stories15M.model.npz"; echo "linked sibling"; exit 0; fi
uv run --group gen python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, pathlib
# stories15M is widely mirrored; adjust repo_id if needed.
p = hf_hub_download("Aananda-Giri/stories15M", "stories15M.model.npz")  # example mirror
shutil.copy(p, pathlib.Path(__file__).resolve().parent / "stories15M.model.npz")
PY
echo "Done -> $HERE/stories15M.model.npz"
