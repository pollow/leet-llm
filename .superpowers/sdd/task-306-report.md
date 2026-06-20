# Task 306 Report — per-head Q/K RMSNorm (Qwen3) scaffold

## What was built

Task 306 `qk_norm` scaffold. All files committed on master.

### Files created / modified
- `306_qk_norm/qk_norm.py` — learner stub (raises NotImplementedError)
- `306_qk_norm/solution.py` — byte-identical to stub (raises NotImplementedError); real solution was temporarily written and then reverted
- `306_qk_norm/README.md` — Description · The Math · Function Signature · Read More · How to Test
- `306_qk_norm/tests/gen_fixtures.py` — authoring-only fixture generator (transformers 5.9.0, Qwen3ForCausalLM, float64)
- `306_qk_norm/tests/test_qk_norm.py` — 13 tests (fixture parity + property/invariant)
- `306_qk_norm/tests/fixtures/qknorm.npz` — committed fixture
- `leet_llm/_registry.py` — added `"qk_norm": ("306_qk_norm", "qk_norm")` to L3 Track C section

## Key implementation decision: float32 cast in HF RMSNorm

`Qwen3RMSNorm.forward` internally casts hidden_states to float32 for variance computation, then casts back to the input dtype. With float64 inputs, this breaks parity at `rtol=1e-9` (gap ~1e-7). The fixture therefore captures the pre-norm Q/K tensors from a genuine Qwen3 forward (guaranteeing realistic inputs), and computes the expected output via pure float64 numpy RMSNorm — not from HF's post-norm tensors. This keeps the test hermetic and the math pure.

## Fixture layout

- `q_pre`: `(n_q_heads=4, L=5, head_dim=4)` — pre-norm Q from Qwen3 config `hidden_size=16, num_attention_heads=4, num_key_value_heads=2, head_dim=4`
- `k_pre`: `(n_kv_heads=2, L=5, head_dim=4)` — pre-norm K
- `q_post`, `k_post`: expected outputs (float64 rms_norm reference)
- `q_weight`, `k_weight`: randomised (seed 99) to avoid trivial all-ones weight path
- `eps`: 1e-6 (Qwen3 default)

Axis layout: `(..., n_heads, L, head_dim)` — norm acts over last axis (`head_dim`). This matches the post-transpose convention in Qwen3's forward (`q.transpose(1,2)` produces `(batch, n_heads, seq, head_dim)`).

## Validation

### `uv run grade -s 306` (temporary real implementation)

```
13 passed in 0.07s
```

Real solution used: `rms_norm(q, q_weight, eps=eps)` + `rms_norm(k, k_weight, eps=eps)` — two lines, reusing 212.

### `uv run grade 306` (unsolved stub)

```
13 failed in 0.13s
```

All failures: `NotImplementedError: Implement qk_norm — see 306_qk_norm/README.md`. No collection errors, no import errors, no KeyError.

## Tests (13 total)

| Test | Category |
|---|---|
| `test_matches_qwen3_fixture` | Real fixture parity (`rtol=1e-9`) |
| `test_shape_preserved` (×4 param) | Shape preserved for GQA configs |
| `test_dtype_preserved` | Output is float64 |
| `test_identity_weight_matches_rms_norm_q/k` | weight=1 reduces to `rms_norm(212)` |
| `test_weight_scaling_applied_correctly` | Uniform weight `w` scales output by `w` |
| `test_per_head_independence_q` | Perturbing one head doesn't affect others |
| `test_per_position_independence_q` | Perturbing one position doesn't affect others |
| `test_q_k_independent` | Q and K are normalised independently |
| `test_batch_dim_passthrough` | Leading batch dim works correctly |

## Self-review

- Stub and solution are byte-identical ✓
- No reference implementation committed ✓
- Registry entry added ✓
- README follows 305 shape, no numpy recipe, no `grade -s` mention ✓
- No `→ L4` line (qk_norm has no cache interaction) ✓
- Fixture uses genuine Qwen3ForCausalLM class (not torch composition) ✓
- `eps` correctly read from `attn.q_norm.variance_epsilon` (= 1e-6) ✓
- Weights randomised to avoid trivial all-ones path ✓
- Float32-cast issue documented in gen_fixtures.py header ✓

---

## Review-fix report (2026-06-20)

Applied four review fixes (Fix 1–3 + validation + commit).

### Fix 1 — HF-anchor in gen_fixtures.py

`_extract_qk_fixtures()` now applies the genuine `attn.q_norm` / `attn.k_norm` HF
modules to the captured float64 pre-norm tensors and returns `q_post_hf` / `k_post_hf`.
`main()` runs two sanity checks before writing the fixture:

1. float64 self-consistency (Q and K) — `rtol=1e-12, atol=0`
2. HF-anchor (Q and K) — `rtol=1e-4, atol=1e-5` (tolerates HF's internal f32 cast)

The graded `qknorm.npz` still stores the pure float64 numpy oracle, not the
float32-tainted HF output. `q_post_hf`/`k_post_hf` are excluded from the npz.

**gen_fixtures output (2026-06-20):**
```
  wrote qknorm.npz
  q_pre  shape=(4, 5, 4)  dtype=float64
  q_post shape=(4, 5, 4)
  k_pre  shape=(2, 5, 4)
  k_post shape=(2, 5, 4)
  q_weight=[ 0.61268586 -1.17535369 -0.76464929 -0.66656567]
  k_weight=[ 0.74436599 -0.64531736 -1.38902779 -0.2729676 ]
  eps=1e-06
  sanity check 1 passed (float64 self-consistency for Q and K)
  sanity check 2 passed (HF-anchor: numpy oracle matches Qwen3 q_norm/k_norm)
```

### Fix 2 — Lean stub docstring

`306_qk_norm/qk_norm.py` and `306_qk_norm/solution.py` module docstrings now match
the lean hint style of `215_gqa/gqa.py` — no closed-form numpy expression, no formula
leakage. Both files confirmed byte-identical (`diff` returns empty).

### Fix 3 — Fixture parity tolerance tightened

`test_qk_norm.py::test_matches_qwen3_fixture` now uses `atol=1e-12` (down from
`atol=1e-9`) so near-zero mismatches are not masked.

### Validation

**`uv run --group gen python 306_qk_norm/tests/gen_fixtures.py`** — both HF-anchor
asserts pass (sanity check 2).

**`uv run grade -s 306`** (temporary real solution):
```
.............                                                            [100%]
13 passed in 0.06s
```

**`uv run grade 306`** (unsolved stub):
```
FFFFFFFFFFFFF                                                            [100%]
13 failed — all raise: NotImplementedError: Implement qk_norm — see 306_qk_norm/README.md
```

**Byte-identity:**
```
diff 306_qk_norm/qk_norm.py 306_qk_norm/solution.py  →  (empty, BYTE-IDENTICAL)
```
