# AGENTS.md

Guidance for working in this repo, shared by **all** coding agents. This is the canonical
authoring/maintenance doc; `CLAUDE.md` is a symlink to it, so edit `AGENTS.md` only.
Student-facing docs live in `README.md` and per-task `README.md`s; this file is for
authoring/maintenance and is **not** shown to students.

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

Current status: L0–L3 scaffolded (including the full L3 301–311 track); L4–L6 not yet
authored.

## Design principles

- **One function, one task.** Each task is a self-contained folder; the student edits a
  single stub file, and `tests/` grades it.
- **Stub vs. solution.** Each task ships exactly one student stub (the lone non-`solution`,
  non-`conftest`, non-`convert` `.py`) plus a `solution.py` reference. The stub's functions
  **raise `NotImplementedError`** — a stub must never contain a working implementation, or
  it leaks the answer. `solution.py` is the only place the full implementation lives.
- **READMEs are tutorials, not specs.** Per-task READMEs follow a fixed shape
  (Description · the Math · Function Signature · Read More · How to Test) and state
  framework facts, never the solution (author-only tricks like `grade -s` stay out). But
  accurate-as-a-spec is not the bar: **a prepared student who does not know the answer
  must be able to implement from a blank stub without guessing hidden design intent.**
  Teach mechanics *and* design intent ("why this choice, when to use it again, what it
  costs"); naming a helper is not teaching it. The full bar — delta map, ordered
  Why/Purpose/What/How/Check steps, verification ladder, debug playbook — is the
  **README Tutorial Standard**: `docs/superpowers/specs/2026-06-26-readme-tutorial-standard.md`.
  Follow it for every L3+ README.
- **Masks are boolean.** The project-wide mask contract is a **bool** array: `True` means
  masked/hidden/forbidden, `False` means visible/allowed. This holds in code *and* docs.
  Do **not** introduce or describe "additive masks" (`-inf`/`-1e9` score bias) as the API
  contract — additive form is at most an internal implementation detail of an operator,
  described as an equivalence, never the primary interface. (Reference: §D of the README
  Tutorial Standard, and `205`/`009` signatures.)
- **Reuse is mandatory, not aspirational.** Before writing any logic in a task, check
  whether the `leet_llm` facade already exposes it (`from leet_llm import softmax`) and
  import it instead of re-inlining. `leet_llm/__init__.py` + `_registry.py` resolve each
  name lazily to the task that owns it, loading the **stub by default** or the
  **`solution.py`** when `LEET_LLM_TARGET=solution` — so a task's tests exercise the
  student's own earlier code, and `grade -s` runs an all-solutions stack end-to-end.
  Re-implementing an existing primitive is a defect to fix, not a style preference. (L0–L3
  predate strict enforcement and still carry un-saturated reuse to audit later.)
- **Tests encode invariants**, not just oracle values, so a partial/wrong stub fails loudly.

## Grading

- `uv run grade <task>` — grade the student stub (e.g. `llama_model.py`). Also
  `uv run grade <level-digit>` for a whole level, `uv run grade all` for everything.
- `uv run grade -s <task>` — grade the reference `solution.py` instead (sets
  `LEET_LLM_TARGET=solution`). Use this to verify a task's solution passes end-to-end.
  Keep this flag **out of per-task READMEs**; it must not be exposed to students.

## Test weight tiers (how a task sources weights for grading)

Every task grades against a committed golden. There are three tiers for where the
weights — and the golden — come from. Pick the **highest tier the architecture allows**.

- **Tier A — local-random → our oracle (every task).** `gen_fixtures.py` makes tiny
  seeded random weights and freezes the logits of **our own float64 numpy oracle**. The
  always-on grade-time check compares the student against this oracle at `rtol≈1e-9`. To
  prove the oracle faithful (not self-circular) it is anchored against the genuine HF
  class at **authoring time only** (decision 2 in the L3 plan) — that anchor does *not*
  run at grade time.
- **Tier B — tiny-random HF checkpoint (skippable).** `download.sh`/`convert.py` fetch a
  `hf-internal-testing/tiny-random-*` checkpoint and commit a golden produced by the
  **genuine HF class**. Weights are *random*, so the output is meaningless — there's no
  demo — but it adds the only **grade-time** genuine-HF cross-check and exercises the real
  HF weight-name layout through `load_*`. For random weights it partly overlaps Tier A's
  authoring anchor; its marginal value is the independent grade-time oracle + loader cover.
- **Tier C — real pretrained (skippable).** Download an actual trained model so the
  forward *does something* (302 translates, 304 tells a story, 306 runs Qwen3-0.6B). This
  is the strongest validation and the only one with a satisfying end-to-end demo.

**Decision rule.** Prefer **C** when a *small, ungated* real checkpoint exists
(≈≤1 GB, no license gate — e.g. stories15M, Qwen3-0.6B). Fall back to **B** when the
family has no small real checkpoint but a tiny-random one loads (e.g. Mistral 305,
Mixtral 308, Gemma-2 310 — Gemma ships only 2B/9B/27B, all large + gated). **Omit B/C**
(Tier A only) when no public checkpoint loads under the task's math (e.g. DeepSeek 311:
the only tiny checkpoints use yarn+interleaved RoPE, out of scope) — and **say so in the
README**. Before claiming a checkpoint is absent/unusable, *verify it* (`list_repo_files`,
read its `config.json`) — don't assume from the name or guess its size.

## Design docs (now and future)

Specs and implementation plans live under `docs/superpowers/`, dated and kebab-cased:

- `docs/superpowers/specs/` — curriculum & per-level **design specs** (the source of
  truth, e.g. `2026-05-31-leet-llm-curriculum-design.md`).
- `docs/superpowers/plans/` — per-track **implementation plans** and memos.

Put new design work there following the same `YYYY-MM-DD-<slug>.md` convention.
