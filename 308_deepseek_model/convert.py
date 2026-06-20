"""308 — fetch + convert real DeepSeek-V3 weights for the parity test.
(AUTHORING/DEMO, gen group)

NOTE: Both currently available public tiny DeepSeek-V3 checkpoints use Yarn RoPE
with rope_interleave=True, which requires an implementation beyond the scope of
this task (task 308 implements standard half-rotate RoPE only). The real-weights
parity test (B) is therefore not included in the shipped version.

When a tiny checkpoint with default half-rotate RoPE becomes available, this
script should:
1. Download config.json + model.safetensors (or pytorch_model.bin)
2. Map HF weights → our layout and save as deepseek_tiny.npz (git-ignored)
3. Run genuine DeepseekV3ForCausalLM → get reference logits
4. Assert our deepseek_forward matches at documented tolerance
5. Write tests/fixtures/real_ref.npz with the reference logits

Run with:
    LEET_LLM_TARGET=solution uv run --group gen python 308_deepseek_model/convert.py
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).parent

# The available public tiny DeepSeek-V3 checkpoints:
# - bzantium/tiny-deepseek-v3: full-size dims (hidden=7168) with yarn+interleave
# - hf-internal-testing/tiny-random-DeepseekV3ForCausalLM: degenerate (kv_lora=512 > d=8)
#
# Neither is suitable for a basic student implementation that uses standard RoPE.
# This convert.py is a placeholder for when a suitable checkpoint becomes available.

print("No suitable public tiny DeepSeek-V3 checkpoint is available for download.")
print("Both known checkpoints use Yarn RoPE + rope_interleave=True which requires")
print("an extended implementation beyond task 308's scope.")
print("See 308_deepseek_model/README.md for details.")
sys.exit(0)
