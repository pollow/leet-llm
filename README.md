# leet-llm

> Build a full-featured LLM from the ground up — one small, testable function at a time.

`leet-llm` is a LeetCode-style course for building a large language model from scratch.
Instead of reading thousands of lines of model code at once, you implement each piece as
a tiny, gradeable function with its own tests and reference solution — then watch them
assemble into a complete, working LLM.

It's inspired by [`llama3.np`](https://github.com/likejazz/llama3.np) (a pure-NumPy Llama-3
inference engine), but broken into bite-sized steps for people **without** a deep ML
background.

---

## Philosophy

The course follows the real arc of the field, in two phases:

### 🟦 Phase 1 — Build the forward pass by hand, in NumPy
Pretrained weights go in, generated text comes out — with **zero framework autograd**.
You implement every matmul, every softmax, every attention head yourself, so there is no
magic left. This is the spirit of `llama3.np`.

### 🟧 Phase 2 — Graduate to PyTorch
We switch to PyTorch *exactly* when we need gradients — nobody hand-rolls a backward pass
here. Training is taught through the home-friendly path of **LoRA fine-tuning** and
**reinforcement learning**, not full pretraining.

**The teaching spine:** classic transformer first, then the Llama-3 upgrade as
*contrasting* tasks — so you learn not just *what* modern models do, but *why* they
changed:

| Classic | → | Llama-3 |
|---------|---|---------|
| LayerNorm | → | RMSNorm |
| Sinusoidal/learned positions | → | RoPE |
| Multi-head attention | → | Grouped-query attention |
| GELU MLP | → | SwiGLU |

Llama is, in the end, "a clean GPT" — so once you've built it, the rest of the modern
open-source zoo (Mixtral, Qwen, DeepSeek, Mistral, Gemma…) is just a set of *deltas* you
already have the tools to understand.

---

## The Levels

Each task has a 3-digit id: the **first digit is the level**, the **last two are the task
number** within that level (`001` = level 0, task 01 · `402` = level 4, task 02).

### 🟦 NumPy Phase — build & serve the forward pass

| Lv | Title | What you'll build |
|----|-------|-------------------|
| **L0** | **NumPy Foundations** | The math substrate: batched matmul, broadcasting, `einsum`, numerically-stable softmax & logsumexp, top-k, masking, seeded sampling. |
| **L1** | **Tokenization & Batching** | Text ⇄ tensors: a char tokenizer, then BPE (vocab, merges, encode/decode), special tokens, padding, attention masks, position ids. |
| **L2** | **Operators & Layers** | The reusable layers as small functions. *Classic:* Linear, Embedding, GELU/SiLU, LayerNorm, positional encoding, attention, MHA, FFN, the GPT block. *Llama upgrade:* RMSNorm, RoPE, SwiGLU, GQA, KV-cache, the Llama block. |
| **L3** | **Whole-Model & Inference** | Assemble the full stack, load real pretrained weights, sample (greedy / temperature / top-k / top-p), and run the generation loop — **rebuilding `llama3.np`**. Then morph it into the OSS zoo: Mixture-of-Experts, DeepSeek MLA, sliding-window attention, RoPE scaling, QK-norm. |
| **L4** | **Inference Systems & Serving** | Make it fast and scalable. *Single node:* FlashAttention (online softmax + tiling), PagedAttention (KV block manager), a continuous-batching scheduler (vLLM/SGLang-style), prefix caching. *Multi-host:* tensor parallelism, pipeline parallelism, collectives from scratch, disaggregated prefill/decode. |

### 🟧 PyTorch Phase — hand the backward pass to a framework

| Lv | Title | What you'll build |
|----|-------|-------------------|
| **L5** | **PyTorch & LoRA Fine-Tuning** | Tensors, autograd, `nn.Module`. **Re-implement a Llama block in PyTorch and prove it matches your L2 NumPy version** — the keystone that bridges both phases. Then an optimizer step, LoRA adapters, and a supervised fine-tune on a toy corpus. |
| **L6** | **RL Infrastructure** | Alignment from scratch: reward-model scoring, advantage estimation, the **PPO** and **GRPO** objectives, a KL-to-reference penalty, and a proof-of-concept trainer loop (rollout → score → update). |

---

## How a Task Is Structured

Every task is a self-contained folder. You edit one stub file; the tests tell you when
you're done.

```
001_numpy_array_basics/
├── README.md            # description · the math · function signature · links to read · how to test
├── array_basics.py      # ← you fill in the stubbed functions here
├── solution.py          # the reference implementation (peek only when stuck!)
└── tests/
    └── test_array_basics.py   # pytest cases that grade your work
```

Each task `README.md` follows the same shape:

1. **Description** — what you're building and why it matters.
2. **The Math** — the formula/algorithm, in plain terms.
3. **Function Signature** — the exact function(s) to implement.
4. **Read More** — links to the relevant papers, docs, or blog posts.
5. **How to Test** — the command to run.

---

## Getting Started

```bash
# 1. Install dependencies (NumPy for L0–L4, PyTorch for L5–L6)
pip install -r requirements.txt

# 2. Pick a task, open its README, and fill in the stub
$EDITOR 001_numpy_array_basics/array_basics.py

# 3. Run that task's tests until they're green
pytest 001_numpy_array_basics

# 4. Stuck? Read the task README's "Read More" links first.
#    Still stuck? Compare with solution.py.
```

Work the levels in order — each one builds on the last.

---

## Progress

| Level | Tasks | Status |
|-------|-------|--------|
| L0 — NumPy Foundations | 10 | 🚧 authoring |
| L1 — Tokenization & Batching | — | ⬜ planned |
| L2 — Operators & Layers | — | ⬜ planned |
| L3 — Whole-Model & Inference | — | ⬜ planned |
| L4 — Inference Systems & Serving | — | ⬜ planned |
| L5 — PyTorch & LoRA Fine-Tuning | — | ⬜ planned |
| L6 — RL Infrastructure | — | ⬜ planned |

---

## Design

The full curriculum design lives in
[`docs/superpowers/specs/2026-05-31-leet-llm-curriculum-design.md`](docs/superpowers/specs/2026-05-31-leet-llm-curriculum-design.md),
including the complete Level 0 task specification.
