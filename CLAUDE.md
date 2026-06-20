# CLAUDE.md

Guidance for working in this repo. Student-facing docs live in `README.md` and per-task
`README.md`s; this file is for authoring/maintenance and is **not** shown to students.

## What this project is

`leet-llm` is a LeetCode-style course for building a full LLM from scratch, one small,
testable function at a time. Pretrained weights go in and generated text comes out, with
the whole forward pass hand-written — inspired by [`llama3.np`](https://github.com/likejazz/llama3.np)
but broken into bite-sized, gradeable steps for people without a deep ML background.

## Direction — the ladder

A locked 7-level ladder, in two phases. Task ids are 3 digits: first digit = level, last
two = task within the level (`213` = L2 task 13).

- **NumPy phase (L0–L4)** — build & serve the forward pass with zero framework autograd.
  - **L0** NumPy Foundations · **L1** Tokenization & Batching · **L2** Operators & Layers ·
    **L3** Whole-Model & Inference (rebuild `llama3.np`, then the OSS-zoo deltas) ·
    **L4** Inference Systems & Serving (FlashAttention, PagedAttention, parallelism).
- **PyTorch phase (L5–L6)** — hand the backward pass to a framework.
  - **L5** PyTorch & LoRA Fine-Tuning · **L6** RL Infrastructure (PPO/GRPO).

**Teaching spine:** classic transformer first, then the Llama-3 upgrade as *contrasting*
tasks (LayerNorm→RMSNorm, sinusoidal→RoPE, MHA→GQA, GELU-MLP→SwiGLU) so students learn
both *what* modern models do and *why* they changed.

Current status: L0–L3 scaffolded; 301–304 (the two L3 capstone tracks: opus-mt translate
and stories15M Llama) have reference solutions. L4–L6 not yet authored.

## Design principles

- **One function, one task.** Each task is a self-contained folder; the student edits a
  single stub file, and `tests/` grades it.
- **Stub vs. solution.** Each task ships exactly one student stub (the lone non-`solution`,
  non-`conftest`, non-`convert` `.py`) plus a `solution.py` reference. The stub's functions
  **raise `NotImplementedError`** — a stub must never contain a working implementation, or
  it leaks the answer. `solution.py` is the only place the full implementation lives.
- **READMEs don't leak.** Per-task READMEs follow a fixed shape (Description · the Math ·
  Function Signature · Read More · How to Test) and state framework facts, never the
  solution. Author-only tricks (e.g. `grade -s`) stay out of them.
- **Reuse compounds.** Implemented building blocks are imported via the `leet_llm` facade
  (`from leet_llm import softmax`). `leet_llm/__init__.py` + `_registry.py` resolve each
  name lazily to the task that owns it, loading the **stub by default** or the
  **`solution.py`** when `LEET_LLM_TARGET=solution`. So a task's tests exercise the
  student's own earlier code, and `grade -s` runs an all-solutions stack end-to-end.
- **Tests encode invariants**, not just oracle values, so a partial/wrong stub fails loudly.

## Grading

- `uv run grade <task>` — grade the student stub (e.g. `llama_model.py`). Also
  `uv run grade <level-digit>` for a whole level, `uv run grade all` for everything.
- `uv run grade -s <task>` — grade the reference `solution.py` instead (sets
  `LEET_LLM_TARGET=solution`). Use this to verify a task's solution passes end-to-end.
  Keep this flag **out of per-task READMEs**; it must not be exposed to students.

## Design docs (now and future)

Specs and implementation plans live under `docs/superpowers/`, dated and kebab-cased:

- `docs/superpowers/specs/` — curriculum & per-level **design specs** (the source of
  truth, e.g. `2026-05-31-leet-llm-curriculum-design.md`).
- `docs/superpowers/plans/` — per-track **implementation plans** and memos.

Put new design work there following the same `YYYY-MM-DD-<slug>.md` convention.
