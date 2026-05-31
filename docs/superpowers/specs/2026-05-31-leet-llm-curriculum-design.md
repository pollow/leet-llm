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

**Principles**
- Every task is one focused, independently testable function.
- Each task foreshadows a real LLM component — even L0 NumPy drills use LLM-shaped data.
- Later models are framed as *deltas* against the baseline you already built.
- PoC over performance: we demo the *algorithm* (e.g., FlashAttention's online softmax),
  not a GPU kernel.

---

## 2. The Level Ladder (L0–L6)

The first digit of a task id is its level; the last two are the task number within the
level (`001` = level 0 task 01, `402` = level 4 task 02). Up to 99 tasks per level.

### 🟦 NumPy Phase — build & serve the forward pass by hand

| Lv | Title | Theme | Representative tasks |
|----|-------|-------|----------------------|
| **L0** | NumPy Foundations | the math substrate | batched matmul, broadcasting, einsum, stable softmax/logsumexp, top-k, masking, sampling RNG *(fully spec'd in §4)* |
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

Flat, self-contained task folders at the top level — each task owns everything it needs.

```
leet-llm/
├── README.md                  # the ladder + progress tracker + how-to-run
├── conftest.py                # shared pytest config (optional fixtures, seeds)
├── 001_numpy_array_basics/
│   ├── README.md              # description · the math · signature · paper links · how to test
│   ├── array_basics.py        # stub: type-hinted signatures + docstrings, learner fills in
│   ├── solution.py            # reference implementation
│   └── tests/
│       └── test_array_basics.py   # pytest; imported against the stub (and solution in CI)
├── 002_broadcasting/
│   └── ...
├── 101_char_tokenizer/        # 1 = level 1, 01 = first task
│   └── ...
└── ...
```

**Conventions**
- Folder name = `NNN_slug` where `NNN` is the 3-digit task id and `slug` is a short name.
- The 3-digit prefix *is* the level/task id (`LXY`): `L`=level, `XY`=task number.
- Each task's `README.md` has fixed sections: **Description · The Math · Function Signature · Read More (links) · How to Test**.
- The stub file and the solution file share the same module name (e.g. `array_basics.py`)
  so the same test file can target either via an import path / env switch.
- Tests run against the learner's stub by default; CI additionally runs the suite against
  `solution.py` to guarantee the reference passes.

**Tooling (proposed):** `pytest` for tests, `numpy` for L0–L4, `torch` for L5–L6.
A top-level `requirements.txt` (or `pyproject.toml`) pins versions. A small test runner
(`make test 001` / `pytest 001_numpy_array_basics`) keeps the loop tight.

---

## 4. Level 0 — NumPy Foundations (fully spec'd)

**Level goal:** never be confused by an array shape again. Every drill uses LLM-shaped
tensors `(B=batch, T=sequence, D=model dim, H=heads)` so the mechanics transfer directly
to later levels. 10 tasks.

> Notation: `B` batch, `T` tokens, `D` model dim, `H` heads, `V` vocab, `d = D/H` head dim.

| Id | Slug | Primary signature | Goal | Foreshadows |
|----|------|-------------------|------|-------------|
| **001** | `numpy_array_basics` | `split_heads(x: np.ndarray, n_heads: int) -> np.ndarray`; `merge_heads(x: np.ndarray) -> np.ndarray` | reshape + transpose round-trip: `(B,T,D) ↔ (B,H,T,d)` | multi-head attention |
| **002** | `broadcasting` | `add_bias(x: np.ndarray, b: np.ndarray) -> np.ndarray`; `row_normalize(x: np.ndarray) -> np.ndarray` | broadcast a bias over batch/seq; normalize over last axis (mean/var) — no Python loops | LayerNorm/RMSNorm |
| **003** | `matmul` | `linear(x: np.ndarray, W: np.ndarray, b: np.ndarray \| None) -> np.ndarray` | `y = x · Wᵀ + b` with batched matmul; reason about `(B,T,D)·(D_out,D)` shapes | every Linear layer |
| **004** | `einsum` | `attention_scores(q: np.ndarray, k: np.ndarray) -> np.ndarray` | compute `QKᵀ` via `einsum` for `(B,H,T,d)` → `(B,H,T,T)`; also express outer product & trace | attention scores |
| **005** | `softmax` | `softmax(x: np.ndarray, axis: int = -1) -> np.ndarray` | numerically stable softmax (subtract max); verify rows sum to 1 and large inputs don't overflow | attention + sampling |
| **006** | `logsumexp` | `logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray`; `cross_entropy(logits, targets) -> np.ndarray` | stable `log Σ exp`; build CE loss on top | loss / perplexity |
| **007** | `topk` | `top_k(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]`; `argmax(x, axis)` | return top-k values + indices along last axis (`argpartition`) | top-k sampling, greedy decode |
| **008** | `gather_onehot` | `embedding_lookup(table: np.ndarray, ids: np.ndarray) -> np.ndarray`; `one_hot(ids, vocab: int)` | gather rows by integer ids (`(V,D)`,`(B,T)`→`(B,T,D)`); one-hot encode | embedding layer |
| **009** | `masking` | `apply_causal_mask(scores: np.ndarray) -> np.ndarray`; `apply_padding_mask(scores, mask) -> np.ndarray` | set masked positions to `-inf` before softmax (causal upper-triangle + padding) | causal attention |
| **010** | `rng_sampling` | `sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray` | seeded, reproducible sampling from a categorical distribution | temperature/top-p sampling |

**Each L0 task README links** to relevant NumPy docs (`np.matmul`, `np.einsum`,
broadcasting rules, `np.random.Generator`) plus one conceptual reference where useful
(e.g., the log-sum-exp trick, the softmax temperature explainer).

**Testing pattern (all levels):** each `test_*.py` checks (a) correctness against a
hand-computed or `scipy`/reference value, (b) shape/dtype, (c) numerical-stability edge
cases (large/negative inputs, masked rows), and (d) that no disallowed shortcut is used
where relevant (e.g., must not call `scipy.special.softmax`). Tests use fixed seeds for
reproducibility.

---

## 5. Open Items (deferred, not blocking)

- Exact task lists for **L1–L6** — authored per-level, just-in-time, following the L0
  template above.
- Which concrete tiny checkpoint to target in L3 (e.g., a `stories15M`-class Llama) for
  the `llama3.np` capstone, including a weight-conversion helper.
- Whether to ship a `cookiecutter`/scaffold script to generate new `NNN_slug/` task
  skeletons consistently.
- CI configuration (GitHub Actions) to run all `solution.py` suites on push.

---

## 6. Out of Scope

- Hand-rolled autograd / manual backward passes (we graduate to PyTorch for gradients).
- Full pretraining from scratch (LoRA + RL only).
- Production-grade kernels or real distributed deployment (PoC/simulation only).
- A web UI or hosted grader (local `pytest` is the grader for v1).
