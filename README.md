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
| **L1** | **Tokenization & Batching** | Text ⇄ tensors: char & byte tokenizers, then BPE (train, save/load, encode/decode), special tokens, a load-only tiktoken path, padding, padding masks, position ids. |
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

This project uses [`uv`](https://docs.astral.sh/uv/). NumPy covers L0–L4; PyTorch is
added on demand for L5–L6 (`uv add torch`).

```bash
# 1. Set up the environment (NumPy + pytest)
uv sync

# 2. Pick a task, open its README, and fill in the stub
$EDITOR 001_numpy_array_basics/array_basics.py

# 3. Grade it until it's green
uv run grade 001            # or, from inside the folder, just: uv run grade
                            # uv run grade 2  → all of Level 2 · uv run grade all → everything

# 4. Stuck? Read the task README's "Read More" links first.
#    Still stuck? `uv run grade -s 001` runs the reference solution.
```

**Reuse compounds.** Once you've implemented a building block, later tasks import it as a
library — e.g. a Level-2 attention task does `from leet_llm import group_last_axis` and
runs on *your own* code. Work the levels in order; each builds on the last.

---

## Progress

| Level | Tasks | Status |
|-------|-------|--------|
| L0 — NumPy Foundations | 11 | ✅ scaffolded |
| L1 — Tokenization & Batching | 13 | ✅ scaffolded |
| L2 — Operators & Layers | 16 | ✅ scaffolded |
| L3 — Whole-Model & Inference | 11 | ✅ scaffolded |
| L4 — Inference Systems & Serving | — | ⬜ planned |
| L5 — PyTorch & LoRA Fine-Tuning | — | ⬜ planned |
| L6 — RL Infrastructure | — | ⬜ planned |
