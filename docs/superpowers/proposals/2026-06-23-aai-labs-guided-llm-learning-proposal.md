# AAI Labs — Founding Engineers Proposal

## 1/ Your Proposal

### Project Name

**leet-llm — Guided Learning on All LLM Models & Techniques**

### One-Line Description

An interactive learning platform where you learn every modern LLM architecture and inference technique by implementing each piece as a small, locally-graded function — with interactive visualizations of the core components.

### The Problem

**What problem does this solve?**
Understanding modern LLMs is gated behind two bad options: (1) read thousands of lines of model code at once, or (2) take a math-heavy course that never touches a running model. Neither builds *operational* understanding — the kind where you can look at DeepSeek's MLA or Mixtral's MoE router and know exactly what it does and why it replaced the thing before it. And none of the existing resources teach *serving* — the KV-cache, FlashAttention, paged attention, continuous batching, and parallelism that turn a model into a production system.

**Who has this problem?**
Every engineer ramping into AI — new grads, transfers, the broader industry. Internally: AAI/ADO engineers who must reason about model internals to author good tasks.

**Why does it matter?**
Operational LLM literacy is the bottleneck. Externally it's a credible, ambitious open-learning product in a category (LeetCode) that has no real equivalent for LLMs. Internally it's the fastest on-ramp for anyone joining AAI/ADO.

### Your Solution

A web companion + a local-first coding curriculum — "leet-llm":

**The repo is the product.** Learners clone it, implement stubbed functions in their own editor, and grade locally with `uv run grade <task>`. A locked, ordered ladder of 51+ tasks (L0 NumPy → L1 tokenization → L2 operators → L3 whole-model → L4 inference engine → L5–L6 training/RL, to be designed). Reuse compounds — by L3, your model runs entirely on functions *you* wrote. Curriculum is hand-designed with AI help; a human owns the difficulty curve.

**The website is the companion.** It adds what a repo can't:

1. **Rich task pages** — the per-task READMEs rendered with better formatting, expandable math sections, and linked references. A polished reading experience that can be further expanded and refined over time.
2. **Interactive visualizations (headline differentiator).** See the math work: softmax (temperature slider, overflow toggle), RMSNorm vs LayerNorm, RoPE (animated rotations by position), attention (live Q·Kᵀ heatmap → mask → softmax → V), GQA/MoE/sliding-window, and inference-engine concepts (KV-cache filling, paged block tables, FlashAttention tiling, ring all-reduce).
3. **Discussion.** Per-task Disqus threads (public) or a linked Google Chat group (internal).
4. **Progress tracking.** Self-confirmed completion: learners mark tasks done as they pass locally.
5. **AI Q&A (stretch goal).** An assistant that answers conceptual questions and gives Socratic hints — without leaking the answer.

The platform teaches **both halves**: the model *and* the inference engine that serves it. L4 tasks shift from operator-filling to **system design** — build a mini-vLLM behind a contract API, graded by behavioral invariants.

### Comparable Products in Meta / in the World

| Product | Usage / Scale | What's Missing? |
| :---- | :---- | :---- |
| **leetllm.com** | New site (launched Jun 2026); 158 lessons, 31 practice problems | Lesson-oriented reading site with light practice. No "implement this function" coding tasks, no code reuse across tasks, no interactive visualizations, no inference-engine coverage. Teaches *about* LLMs, not *building* them. |
| **nanoGPT** (Karpathy) | 60k GitHub stars; now deprecated → nanochat | A single training script for GPT-2 only. No curriculum, no grading, no multi-architecture coverage (no Llama/Mixtral/DeepSeek/Gemma), no inference systems, no interactive viz. A monolithic codebase to read, not a structured learning path. |
| **llm.c** (Karpathy) | 30k GitHub stars | GPT-2/3 pretraining in raw C/CUDA — teaches GPU kernels, not model architectures. Single architecture, no curriculum structure, no grading, no web platform. Targets kernel engineers, not the broad AI ramp-up audience. |
| **LLMs-from-scratch** (Raschka) | 98k GitHub stars; Manning book | Excellent book + Jupyter notebooks; builds one GPT model. Bonus chapters now cover Llama/Qwen/Gemma, but it's a linear read-along, not a graded practice platform. No code-reuse ladder, no inference-engine coverage, no interactive visualizations. |
| **llama3.np** | 1k GitHub stars | Our direct inspiration. Pure-NumPy Llama-3 inference — but one monolithic file, one architecture, no curriculum, no grading. leet-llm is llama3.np *decomposed into 51+ graded steps* that extend to the whole OSS zoo and inference systems. |
| **LeetCode** | 50M+ users | The gold standard for algorithmic interview prep. Zero LLM/ML content — no model architectures, no inference systems, no ML-specific grading (numerical tolerance, weight-loading, reuse). leet-llm brings the LeetCode *format* to the LLM domain. |

**What makes this exciting and unique?** No existing product combines (a) graded, hands-on implementation of *all* major LLM architectures + inference systems, (b) interactive visualizations of the math, and (c) a code-reuse ladder where your earlier work is the library for later tasks — in a single platform.

### Product Category

| [x] Education / Learning Product — Interactive learning, doc generators | [ ] AI Agent / Assistant — Autonomous AI agent for a specific domain |
| :---- | :---- |
| [ ] Developer Tool / CLI | [ ] DevOps / Infrastructure Tooling |
| [ ] Library / SDK / Framework | [ ] Data Tool / Pipeline |
| [ ] Internal Productivity Tool | [x] Research Prototype / Benchmark — Novel research implementation or eval framework |
| [ ] API / Platform Service | [ ] Security / Privacy Tool |
| [ ] Consumer App | [ ] Creative / Media Tool |
| [ ] Communication / Collaboration | [ ] Hardware / Embedded / IoT |

### Technical Approach

**Stack Orientation:** **External / Open Stack** (preferred). The only Meta dependency is Avocado for the bonus AI tutor and for data generation — intrinsic to the data-return mandate. Everything else is standard open-source, keeping curriculum content publicly releasable.

**Primary Tech Stack**

| Area | Choice |
| :---- | :---- |
| Curriculum core | Python 3.11, NumPy, pytest, `uv` (already built) |
| Web companion | Astro + React islands, TypeScript, Tailwind |
| API (dynamic) | Hono (TypeScript) — auth, user management, progress |
| Visualizations | React + D3.js / Canvas / WebGL (hydrated as Astro islands) |
| Discussion | Disqus (public) / Google Chat group (internal) |
| Auth | Meta SSO (internal) / Neon Auth (public) |
| Hosting (internal) | Meta Nest (static + API) — Meta's new internal deployment platform |
| Hosting (public) | Cloudflare Pages (static) + Cloudflare Workers (API) |

**Meta Tech Stack:** Nest for internal deployment; Avocado/Metacode for AI assistant and data generation. The app itself is built on an open stack (Astro, Hono, React) so the same codebase deploys to both Nest (internal) and Cloudflare (public) with minimal adapter changes.

**Key Architectural Decisions**

1. **Clone-and-grade locally** — learners clone the repo and run `uv run grade <task>` in their own terminal, using their preferred editor and debugging tools. No server-side execution. This is how engineers actually work.
2. **Code-reuse ladder via a lazy facade** — the `leet_llm` registry resolves each import to the learner's own earlier code, so L3 runs on L2 code *you wrote*. This compounds learning and makes every task a dependency-rich RL task.
3. **Website = companion, not host** — Astro renders task pages as static HTML from the repo's markdown; interactive visualizations hydrate as React islands only where needed; discussion via Disqus (public) or Google Chat (internal). Hono handles auth (Meta SSO internally, Neon Auth publicly) and user progress. Static and API deploy independently.

## 2/ Launch Intent

- [x] **Both** — Internal first, then external

Internal first to Meta engineers (AAI/ADO on-ramp); curriculum and visualizations are natural OSS candidates via the Labs open-source path. Avocado integration stays internal.

## 3/ Team Composition

### Founding Team

| Name | Role | Key Strength |
| :---- | :---- | :---- |
| Chang Xing | Tech Lead / Founding Engineer | Built the existing 51-task L0–L3 curriculum, grading harness, and `leet_llm` reuse facade end-to-end |
| [TBD — recruiting] | Co-Founder | Frontend / data-viz / product preferred |

### Engineers Needed (3–4 additional)

| Profile | Count | FT/Fractional? | Notes |
| :---- | :---- | :---- | :---- |
| Web / Frontend | 1 | Full-Time | Astro + React islands + interactive visualizations (D3/Canvas) |
| Product Generalist (Full-Stack) | 1 | Full-Time | End-to-end feature work, progress tracking, web companion |
| ML Generalist | 1 | Full-Time | Curriculum expansion (L4–L6), AI tutor integration |
| Design / PM | 1 | Fractional | Viz design, UX for the learning flow |
| **Total** | **3–4** | | |

## 4/ Data Generation Plan

### Data Types You Will Generate

- [x] SWE-Bench style tasks and environments
- [x] Trace corrections / annotations
- [x] Development artifacts (PRs, code reviews, design docs)
- [x] Evaluation benchmarks (evals)

### Domain Depth

**Primary domain:** ML/AI — model architectures, numerical computing, inference systems.

**Additional domains with depth:** Web platform engineering (Astro/React/Hono/TypeScript), data visualization, DevOps/containerization (Dockerized task environments for SWE/T-Bench).

### Training Data from Usage

Once the platform has learners, the web companion produces natural training data:

- **Tutor traces** — if the AI Q&A assistant lands, tutoring sessions (hint requests, conceptual Q&A) generate SFT-quality dialogue grounded in concrete code and test output.
- **Progress telemetry** — which tasks learners attempt, completion order, and where they get stuck — informs curriculum difficulty calibration and surfaces weak spots in model reasoning.
- **Discussion-sourced anti-patterns** — common wrong implementations (e.g., softmax without max-subtraction, wrong RoPE rotation order) surfaced in per-task discussion threads — valuable negative examples for training.

### The Exchange

The product *is* the task bank. Every curriculum task is a ready-made SWE/T-Bench RL task with a crisp, non-gameable reward signal (the committed pytest invariants). The L4 system-design tasks (paged KV allocator, continuous-batching scheduler, ring collectives) are the highest-value: open-ended, contract-graded problems that exercise systems reasoning, not function recall. These are original, hand-authored — not scraped from open-source bugs — sidestepping future-contamination concerns.

## 5/ Rough Timeline

| Phase | Duration | Key Milestones |
| :---- | :---- | :---- |
| **Planning, Infra & Dev Env Setup** | Week 1 | Repo Dockerized for SWE/T-Bench; first L0–L1 tasks emitted. Team onboarded, architecture finalized. *Data flowing immediately.* |
| **MVP** | ~1 month | Web companion live (rich task pages, interactive visualizations for softmax/RMSNorm/RoPE/attention, per-task discussion, self-confirmed progress). Learners clone repo and grade locally. Internal launch to Meta engineers. |
| **v1** | ~2 months | Full visualization suite incl. inference-engine concepts; L4 tasks authored; AI Q&A exploration; stable usage metrics; first trace-correction batch shipped. |

### Key Milestones & Deliverables

**End of Week 1:** Dev env Dockerized, first SWE/T-Bench tasks emitted. Team onboarded, architecture finalized.

**End of Month 1 (MVP):** Learners clone the repo, work through L0–L3 locally with `uv run grade`. Web companion shows rich task pages with interactive visualizations, per-task discussion threads, and self-confirmed progress tracking. First trace corrections captured.

**End of Month 2 (v1):** L4 inference-engine tasks live; full visualization suite; AI Q&A assistant evaluated; measurable internal adoption; steady data pipeline (SWE/T-Bench tasks + trace corrections).

## 6/ Biggest Risks And Dependencies

- **Grading determinism** — float64 NumPy must reproduce across hardware. Mitigated by pinned seeds/threads/BLAS; already proven in the existing 51-task harness.
- **Visualization quality bar** — interactive viz is the headline differentiator; needs dedicated frontend + design investment to land well.
- **Nest onboarding** — the team hasn't used Nest yet; need early access and support to deploy the internal version.
- **What we need from Labs:** Nest access for internal deployment; Cloudflare account for public deployment; fractional design support for the visualization UX.

## 7/ Anything Else?

- **Working prototype:** the `leet-llm` repo already has 51 scaffolded, self-grading tasks (L0–L3), the `leet_llm` reuse facade, and the CLI grading harness. This is not a concept — it's a proven engine being productized.
- **Design docs:** full curriculum design, per-level specs, and the L4 inference-systems design are committed under `docs/superpowers/`.
- **Prior art:** inspired by [`llama3.np`](https://github.com/likejazz/llama3.np) (pure-NumPy Llama-3 inference), decomposed into 51+ graded steps and extended to the full OSS zoo + inference systems.
