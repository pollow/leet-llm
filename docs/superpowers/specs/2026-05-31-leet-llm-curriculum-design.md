# leet-llm — Curriculum Design

> A LeetCode-style course that builds a full-featured LLM from the ground up, one
> testable mini-function at a time. Inspired by [`llama3.np`](https://github.com/likejazz/llama3.np)
> but broken into bite-sized, gradeable steps for learners without deep ML background.

- **Status:** design approved (ladder + L0); L1–L6 task lists authored just-in-time.
- **Date:** 2026-05-31

---

## 1. Goal & Philosophy

Learners assemble a complete LLM stack from primitives, each step a small function
with tests and a reference solution — the way LeetCode teaches algorithms. The course
mirrors the real evolution of the field:

1. **Build the forward pass by hand in NumPy** — understand every tensor, no magic.
   This is the spirit of `llama3.np`: pretrained weights in, generated text out, zero
   framework autograd.
2. **Graduate to PyTorch** exactly when we need gradients — nobody hand-rolls a
   backward pass. We lean on **LoRA + RL** as the home-friendly training story rather
   than full pretraining.

**Pedagogical spine:** classic transformer first, then the Llama-3 upgrade as
*contrasting* tasks (LayerNorm→RMSNorm, sinusoidal PE→RoPE, MHA→GQA, GELU-MLP→SwiGLU),
so learners understand *why* modern models made each choice. Llama is treated as "a
clean GPT," with optional BERT/GPT asides on the encoder-vs-decoder split.

### Principles

- **Baby steps, small bites.** Every task is one focused, independently testable unit
  (one or two tightly-related functions). When in doubt, split it smaller.
- **Progressive disclosure — never leak future hard concepts into earlier tasks.** A
  task's learner-facing README frames the work *only* with concepts already introduced.
  Foreshadowing belongs in the **designer notes** of this spec, never in the task README
  — forward-references to scary terms drive beginners away. *Example:* task `001`
  presents reshape/transpose as foundational array manipulation; it must **not** mention
  multi-head attention, even though that's where the skill is later used. Concretely, L0
  function names stay concept-neutral (`group_last_axis`, not `split_heads`); the LLM
  meaning is introduced in L2 when attention is actually taught.
- **Reuse compounds.** Later tasks build on the learner's *own* earlier implementations
  through the `leet_llm` facade (see §5). L3's Llama runs on the RMSNorm *they* wrote in
  L2 — not a fresh copy. Single source of truth = each task's stub file.
- **Deltas, not rewrites.** Later models are framed as diffs against the baseline already
  built.
- **PoC over performance.** We demo the *algorithm* (e.g., FlashAttention's online
  softmax), not a GPU kernel.

---

## 2. The Level Ladder (L0–L6)

The first digit of a task id is its level; the last two are the task number within the
level (`001` = level 0 task 01, `402` = level 4 task 02). Up to 99 tasks per level.

### 🟦 NumPy Phase — build & serve the forward pass by hand

| Lv | Title | Theme | Representative tasks |
|----|-------|-------|----------------------|
| **L0** | NumPy Foundations | the math substrate | reshape/transpose, broadcasting, batched matmul, einsum, stable softmax/logsumexp, top-k, masking, sampling RNG *(fully spec'd in §6)* |
| **L1** | Tokenization & Batching | text ⇄ tensors | char tokenizer, BPE (vocab/merges/encode/decode), special tokens, padding, attention masks, position ids |
| **L2** | Operators & Layers | small computation fns | **classic:** Linear, Embedding, GELU/SiLU, LayerNorm, sinusoidal/learned PE, scaled-dot-product attention, MHA, FFN, residual+norm, GPT block. **Llama upgrade:** RMSNorm, RoPE, SwiGLU, GQA, KV-cache, Llama block. *(BERT/GPT asides)* |
| **L3** | Whole-Model & Inference | entire-model work | assemble GPT→Llama stack, weight loading, sampling (greedy/temp/top-k/top-p), generation loop (**= `llama3.np`**), then OSS-zoo deltas: MoE router+experts (Mixtral/Qwen/DeepSeek), MLA (DeepSeek), sliding-window+sinks (Mistral), RoPE scaling (YaRN/NTK), QK-norm |
| **L4** | Inference Systems & Serving | systems & scale-out | **single-node:** FlashAttention PoC (online softmax + tiling), PagedAttention / KV block-table manager, continuous-batching scheduler (vLLM/SGLang), prefix/radix caching. **multi-host:** tensor parallelism (shard a linear + all-reduce), pipeline parallelism, collectives from scratch (all-gather/all-reduce), disaggregated prefill/decode |

### 🟧 PyTorch Phase — hand the backward pass to a framework

| Lv | Title | Theme | Representative tasks |
|----|-------|-------|----------------------|
| **L5** | PyTorch & LoRA Fine-Tuning | training dynamics | tensors/autograd/`nn.Module`, **parity-check: re-implement a Llama block in torch and assert outputs match your L2 NumPy version**, optimizer step, LoRA adapters (low-rank A·B injection), supervised fine-tune on a toy corpus |
| **L6** | RL Infrastructure | alignment | reward-model scoring, advantage estimation, **PPO** & **GRPO** objectives, KL-to-reference penalty, a **PoC trainer loop** (rollout → score → update) |

**Phase boundary** sits at L4→L5. The L5 parity-check task is the keystone bridging the
two phases: it proves the learner's NumPy mental model and their torch model are the
same computation.

---

## 3. Repository & Task Layout

Flat, self-contained task folders at the top level, plus one importable `leet_llm`
package that ties the learner's work together (§5).

```
leet-llm/
├── README.md                  # the ladder + progress tracker + how-to-run
├── pyproject.toml             # uv-managed project + the `grade` console script
├── leet_llm/                  # the facade package (import surface + tooling)
│   ├── __init__.py            # lazy re-exports driven by the registry
│   ├── _registry.py           # friendly name -> (task folder, attribute)
│   ├── _loader.py             # path-based module loader (stub vs solution)
│   ├── grader.py              # load(__file__) helper used by every test
│   └── cli.py                 # the `grade` command
├── 001_numpy_array_basics/
│   ├── README.md              # description · the math · signature · links · how to test
│   ├── array_basics.py        # stub: type-hinted signatures + docstrings, learner fills in
│   ├── solution.py            # reference implementation
│   └── tests/
│       └── test_array_basics.py   # pytest; uses leet_llm.grader.load
├── 002_broadcasting/
│   └── ...
└── ...
```

**Conventions**
- Folder name = `NNN_slug`; the 3-digit prefix *is* the `LXY` id (`L`=level, `XY`=task #).
- The stub file and `solution.py` expose the **same function names**, so the same test
  grades either one.
- Each task `README.md` has fixed sections: **Description · The Math · Function Signature ·
  Read More (links) · How to Test** — and obeys progressive disclosure (§1).

**Tooling:** `uv` for env/deps. `numpy>=2.0` for L0–L4; `torch` added on demand for L5–L6
(`uv add torch`). `pytest` (dev group) is the grader.

---

## 4. The `grade` Command

A console script (`[project.scripts] grade = "leet_llm.cli:main"`) keeps the test loop
short:

```bash
uv run grade            # from inside a task folder: grade that task
uv run grade 001        # grade task 001 from anywhere
uv run grade 2          # grade every Level-2 task (folders starting with "2")
uv run grade all        # grade the whole course
uv run grade -s 001     # grade the REFERENCE solution (sets LEET_LLM_TARGET=solution)
```

Plain `uv run pytest 001_numpy_array_basics` still works; `grade` is just the shortcut.

---

## 5. Reuse & Import Mechanism (the `leet_llm` facade)

Later tasks must build on the learner's earlier work, but `NNN_slug` folders aren't
importable and cross-folder imports would be ad-hoc. Solution: a thin facade package.

- **`leet_llm/_registry.py`** is the single map from a friendly public name to the task
  that owns it:
  ```python
  REGISTRY = {
      "group_last_axis": ("001_numpy_array_basics", "group_last_axis"),
      "softmax":         ("005_softmax",            "softmax"),
      "rms_norm":        ("204_rms_norm",           "rms_norm"),
  }
  ```
- **`leet_llm/__init__.py`** uses module `__getattr__` to lazily resolve a name: it loads
  the owning task's file via `_loader` and returns the attribute. So a later task writes:
  ```python
  from leet_llm import rms_norm, softmax, group_last_axis
  ```
- **Stub vs solution** is governed by the `LEET_LLM_TARGET` env var (default = the
  learner's stub; `solution` = reference). The facade respects it, so a learner's L3
  Llama genuinely runs on *their own* L2 RMSNorm; CI grades reference-on-reference.
- **Friendly gating:** importing a name whose task isn't implemented yet raises a clear
  *"implement task NNN first"* — the correct nudge, since the course is done in order.
- **Single source of truth** stays the task stub; the facade only re-exports. No
  duplication.

`leet_llm/_loader.py` loads each task module by **file path under a unique name**
(`<folder>_<stem>`) so a full-course run never collides two different `solution.py`
files in `sys.modules`.

---

## 6. Level 0 — NumPy Foundations (fully spec'd)

**Level goal:** fluency with the array operations that all of ML rests on — reshaping,
broadcasting, matrix products, reductions, indexing, and sampling. 10 tasks.

**Progressive-disclosure note:** every public name and README below is concept-neutral.
The "Designer note" column records where each skill is *later* reused — this is for the
curriculum author only and must never appear in the task's README.

> Notation in tasks: generic axes only — `(B, L, F)` = batch, length, features, etc.
> No "heads", "tokens", or "attention" vocabulary at this level.

| Id | Slug | Primary signature | Learner-facing goal | Designer note (NOT in README) |
|----|------|-------------------|---------------------|-------------------------------|
| **001** | `numpy_array_basics` | `group_last_axis(x, n_groups)`; `ungroup_last_axis(x)` | reshape + transpose round-trip: split the last axis into `n_groups` and bring the group axis forward `(B,L,F) ↔ (B,G,L,F/G)`; undo it exactly | head split/merge for MHA |
| **002** | `broadcasting` | `add_bias(x, b)`; `standardize(x)` | broadcast a vector across leading axes; standardize over the last axis (subtract mean, divide std) — no Python loops | LayerNorm/RMSNorm |
| **003** | `affine` | `affine(x, W, b)` | the affine map `y = x · Wᵀ + b` via batched matmul; reason about `(B,L,F)·(F_out,F)` shapes | every Linear layer |
| **004** | `batched_matmul` | `batched_matmul(a, b)` (with the einsum variants) | batched matrix product over leading axes via `einsum`; also outer product & trace | attention scores `QKᵀ` |
| **005** | `softmax` | `softmax(x, axis=-1)` | numerically stable softmax (subtract max); rows sum to 1, no overflow on large inputs | attention weights + sampling |
| **006** | `logsumexp` | `logsumexp(x, axis=-1)`; `log_softmax(x, axis=-1)` | the stable `log Σ exp` trick and log-softmax built on it | loss / perplexity |
| **007** | `topk` | `top_k(x, k)`; `argmax(x, axis)` | top-k values + indices along an axis (`argpartition`) | greedy & top-k decoding |
| **008** | `gather_onehot` | `gather_rows(table, idx)`; `one_hot(idx, n)` | gather rows of a `(N, F)` table by integer indices; one-hot encode | embedding lookup |
| **009** | `masking` | `masked_fill(x, mask, value)`; `triangular_mask(n)` | set positions where a boolean mask is true to a fill value; build a triangular boolean mask | causal/padding attention masks |
| **010** | `rng_sampling` | `sample_categorical(probs, rng)` | seeded, reproducible sampling from a categorical distribution | temperature/top-p sampling |

**Each L0 task README links** to relevant NumPy docs (`reshape`, `transpose`, `einsum`,
broadcasting rules, `random.Generator`) plus one neutral conceptual reference where
useful (e.g., the log-sum-exp trick).

**Testing pattern (all levels):** each `test_*.py` checks (a) correctness against a
hand-computed or reference value, (b) shape/dtype, (c) numerical-stability edge cases
(large/negative inputs, masked rows), and (d) that no disallowed shortcut is used where
relevant. Tests use fixed seeds for reproducibility and load the target via
`leet_llm.grader.load(__file__)`.

---

## 7. Open Items (deferred, not blocking)

- Exact task lists for **L1–L6** — authored per-level, just-in-time, following the L0
  template above and the progressive-disclosure rule.
- Which concrete tiny checkpoint to target in L3 (e.g., a `stories15M`-class Llama) for
  the `llama3.np` capstone, including a weight-conversion helper.
- A scaffold script to generate new `NNN_slug/` task skeletons + register them.
- CI (GitHub Actions) running `LEET_LLM_TARGET=solution uv run grade all` on push.

---

## 8. Out of Scope

- Hand-rolled autograd / manual backward passes (we graduate to PyTorch for gradients).
- Full pretraining from scratch (LoRA + RL only).
- Production-grade kernels or real distributed deployment (PoC/simulation only).
- A web UI or hosted grader (local `grade`/`pytest` is the grader for v1).
