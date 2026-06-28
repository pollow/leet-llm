# README Tutorial Standard (v1)

## Why this exists

Advanced task READMEs can be accurate as specs but still fail as tutorials.
This standard defines what a student-facing README must contain so a prepared student can
implement from a blank stub without guessing hidden design intent.

This file replaces the earlier draft version and is the source of truth for future rewrites.

---

## Scope and assumptions

- This is an authoring standard for README rewrites (especially L3+ tasks with multiple deltas).
- Student-facing section shape remains fixed:
  - Description
  - The Math
  - Function Signature
  - Read More
  - How to Test
- We improve implementation guidance without leaking copy-paste solutions.

---

## Acceptance bar (must pass)

A README is acceptable only if a prepared student can complete implementation linearly:

1. Start from the correct baseline.
2. Implement each delta in order.
3. Verify each local step before moving on.
4. Assemble final wiring in deterministic order.
5. Diagnose common failures quickly.

If a student can "follow words" but cannot derive the actual implementation steps, the README fails.

---

## Core design rule: implementable over descriptive

For every non-trivial delta, the README must teach both:

- **Mechanics**: what to write.
- **Design intent**: why this choice exists, when to use it again, and what tradeoff it introduces.

Do not ship "API wiring only" explanations for math that students are expected to implement.

---

## Standard section template (L3+)

Use the following section order.

### 1) Orientation (short)

- One paragraph: model family, baseline, and what changed.
- One sentence: what is explicitly out of scope.

### 2) Baseline and delta map (mandatory)

Add a table:

| Component | Baseline behavior | Task delta | Where wired |
|---|---|---|---|

Goal: students can see what is unchanged and where each change lands.

### 3) Prerequisite checklist (mandatory)

List exact prerequisite operators and expected behavior (not only task IDs).

### 4) Step-by-step implementation path (mandatory)

For each delta step, always include:

- **Why add this?** (mandatory depth)
  - what baseline weakness this targets,
  - what practical failure mode it prevents or improves,
  - when to use this design in future models,
  - what tradeoff/cost it introduces.
- **Purpose**: behavior target after adding the delta.
- **What**: shape-level contract and interfaces.
- **How**: formulas or pseudocode (no full solution code).
- **Check**: one fast invariant/test before proceeding.

### 5) Cross-task dependency contract (mandatory when applicable)

When the task depends on other tasks/operators:

- state where the primitive is implemented,
- state where it is consumed/wired,
- state compatibility constraints.

If prerequisite docs do not yet fully teach the required primitive, this README must include
implementable math/pseudocode for that primitive. "Call function X" is not sufficient.

### 6) Integration assembly path (mandatory)

Provide one deterministic forward wiring order, including:

- norm/projection/position steps,
- mask policy,
- residual order,
- final projection path.

### 7) Verification ladder (mandatory)

Strict order:

1. Unit tests for new operators.
2. Wiring checks for cross-task dependencies.
3. Whole-model parity/invariants.
4. Optional real-weight parity.

### 8) Debug playbook (mandatory)

Map:

- symptom -> likely cause -> first check to run.

### 9) Out-of-scope boundary (mandatory)

Explicitly list deferred topics so students do not chase future-level concerns.

---

## Required depth rules for common failure patterns

### A) Non-obvious arithmetic knobs (bias, clamp, saturation, scaling)

Do not present these as "just mirror checkpoint behavior".
Document them as design levers:

- why they exist,
- what problem they solve,
- when they are appropriate,
- what they cost.

### B) Long-context schedule math (RoPE variants, YaRN, etc.)

If students must implement schedule math, include exact computation steps:

- branch definitions,
- transition/ramp boundaries,
- blending equation,
- edge-case handling.

Naming a helper function alone does not meet the bar.

### C) Cross-task wiring

Always separate:

- primitive math implementation responsibility,
- model-forward wiring responsibility.

Students should be able to implement each side independently, then integrate.

---

## Rewrite checklist (for authors)

Before marking a README rewrite complete, verify:

1. Baseline-vs-delta map exists and is accurate.
2. Every delta step has full "Why/Purpose/What/How/Check".
3. Non-obvious knobs include rationale + applicability + tradeoff.
4. Dependency math is implementable from docs even if prerequisite task is pending.
5. Integration order is explicit and deterministic.
6. Verification ladder is executable in order.
7. Debug playbook covers likely mistakes.
8. Out-of-scope boundaries are explicit.

---

## Suggested author workflow

1. Draft delta map and ordered implementation steps first.
2. Add "Why add this?" only after you can state baseline weakness + tradeoff.
3. Add checks immediately after each step (not at the end).
4. Add dependency section and verify "implementable without guessing".
5. Finish with verification ladder + debug playbook + out-of-scope.

---

## README + docstring co-maintenance standard

When updating a task README, update the task stub/solution docstrings in the same changeset.

### A) Simplification logic (what to keep vs remove)

- **README owns rationale**: model deltas, design intent, tradeoffs, and step-by-step teaching.
- **Docstrings own contracts**: input/output shapes, required wiring invariants, and high-risk gotchas.
- **Remove duplication**: do not repeat long derivations, full delta catalogs, or full HF key listings in docstrings if README already contains them.
- **Keep one-line anchors**: preserve critical formulas and behavior invariants where they directly affect implementation correctness.

### B) Execution path (how to apply)

Apply this order whenever README changes meaningfully:

1. Update README first (source of truth for tutorial depth and rationale).
2. Trim/refresh module-level docstrings to brief task scope + README pointer.
3. Trim/refresh function/class docstrings to API contract + key gotchas.
4. Verify no behavior changes were introduced accidentally.

This keeps docs coherent while preventing drift between teaching text and code-local guidance.

### C) Stub/solution consistency contract (mandatory)

For each task pair (stub file and `solution.py`):

- Keep **function signatures identical** (name, args, defaults, return annotation).
- Keep **docstrings semantically aligned** (same contract and caveats; wording may differ slightly).
- Keep **data structure definitions aligned** (e.g., dataclass fields and defaults).
- Do **not** overwrite or simplify away reference implementation logic in `solution.py`.

Rule of thumb: sync documentation surfaces and interfaces, never replace solved code with stubs.

---

## Rollout usage

Use this standard to rewrite older README tasks incrementally.
Each rewrite should be reviewable as a small, testable doc changeset.
