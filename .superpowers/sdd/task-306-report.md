# Task 306 — Qwen3 Whole-Model Forward: Report (2026-06-20 retrofit)

## File Layout

```
306_qk_norm/
  qk_norm.py               stub (NotImplementedError) — qk_norm op + Qwen3Config/Params/load/forward
  solution.py              byte-identical to qk_norm.py (diff = empty)
  convert.py               download Qwen/Qwen3-0.6B (bfloat16 via safetensors.torch), write qwen3_0_6b.npz + real_ref.npz
  download.sh              LEET_LLM_TARGET=solution uv run --group gen python convert.py
  tests/
    gen_fixtures.py        regenerates qknorm.npz + tiny_qwen3.npz; runs HF-anchor
    test_qk_norm.py        qk-norm invariants (6 categories) + model parity A + skippable B (17 total)
    fixtures/
      qknorm.npz           qk_norm operator fixture (unchanged from original)
      tiny_qwen3.npz       hermetic whole-model fixture: composed float64 oracle + HF-named weights
      real_ref.npz         genuine Qwen3ForCausalLM (float64) logits on Qwen3-0.6B weights
leet_llm/_registry.py      +Qwen3Config, Qwen3Params, load_qwen3, qwen3_forward
.gitignore                  +306_qk_norm/qwen3_0_6b.npz
```

## gen_fixtures.py HF-Anchor Max-Diff

Composed float64 torch oracle vs genuine `Qwen3ForCausalLM` (float32 HF) on the same random weights:

**max-abs-diff = 8.02e-06** (threshold: rtol=1e-3, atol=1e-3) ✓

## convert.py: Our Forward vs Genuine HF on Real Qwen3-0.6B

`qwen3_forward` (float64, weights from bfloat16 via safetensors.torch→float32→float64) vs genuine `Qwen3ForCausalLM` (float64, from_pretrained):

**max-abs-diff = 9.11e-06** (threshold: rtol=1e-5, atol=1e-4) ✓

## grade -s 306 Pass Output

```
.................                   [100%]
17 passed in 6.05s
```
(16 always-on + 1 real-weights B with Qwen3-0.6B downloaded)

## grade 306 (Stub) Clean Fail

```
FFFFFFFFFFFFFFFFF                   [100%]
17 failed
```
All 17 fail with `NotImplementedError` only — no collection errors, no KeyError, no import errors.

## Byte-Identical Confirmation

```
diff 306_qk_norm/qk_norm.py 306_qk_norm/solution.py
(empty — IDENTICAL)
```

## Key Implementation Notes

- **bfloat16 fix**: `safetensors.numpy.load_file` does not support bfloat16 (Qwen3-0.6B weights). Fixed by using `safetensors.torch.load_file` then `.float().numpy()`.
- **group_last_axis convention**: takes `n_groups` (number of groups, NOT group size); already returns `(B, n_groups, L, head_dim)`. Pass `n_heads`/`n_kv_heads` — no additional transpose.
- **Tied embeddings**: Qwen3-0.6B has `tie_word_embeddings=True`; `lm_head.weight` absent from checkpoint. `load_qwen3` uses `weights.get("lm_head.weight", tok_embed)`.
- **download.sh**: Passes `LEET_LLM_TARGET=solution` so `convert.py` uses the working forward (stub would raise NotImplementedError).
- **No activation override needed**: Qwen3 uses SiLU by default — unlike Mistral's random-init checkpoint which has gelu in config.

## Commit

`5986368` — `306: add Qwen3 whole-model forward (qk-norm, rotate-half) + real-weights download`
