# \[Template\] AAI Labs \- Founding Engineers Proposal

## About AAI Labs

Meta [Applied AI Labs](https://fb.workplace.com/groups/1276596511213031) is **Applied AI's incubator** for entrepreneurial engineers. We back small teams who want to build real products, tools, and libraries. **Projects span every domain and tech stack**: consumer products, developer tools, open-source libraries, internal platforms. Some will ship internally to Meta employees; and some may be released publicly.

We draw inspiration from labs divisions at other top companies \- dedicated spaces where small teams build breakthrough products unconstrained by existing roadmaps.

**POCs:** [Apurva Sinha](mailto:apsinha@meta.com), [Himanshu Verma](mailto:himanshuv@meta.com), [Brad Bulkley](mailto:blb@meta.com)

## Who We Are Looking For

We are looking for **Founding Engineers** \- solo or groups of up to 2 \- who demonstrate:

* **Vision & Conviction**: You have a clear, compelling idea for something you want to build and ship. You can articulate why it matters.  
* **Entrepreneurial Drive**: You will **recruit 3-4 engineers** within AAI, rally them behind your vision, and lead with urgency. The team can grow over time.  
* **Excitement & Leadership**: You inspire others. People want to work with you because of your energy.  
* **Balance Between Vision** and **Data Return**: You can build your vision while ensuring strong data generation for model improvement. **This dual focus is non-negotiable**.  
* **Bias Toward Action**: You get things done. You don't over-plan. MVP in under a month.  
* **Technical Range**: Comfortable operating across the stack or deep in a domain.

**Tech Stack Preference**: We **prefer non-Meta (open/standard) tech stacks** \- standard open-source frameworks, common languages, industry-standard infrastructure platforms. Some Meta-internal tech stack usage is okay where it provides a clear advantage, but default to open.

**You do not have to build an app** or **an AI-focused product.** Developer tools, libraries, Enterprise tools, infra \- anything goes, and will be the majority of projects.  
---

## 1/ Your Proposal

### Project Name

\[Your project name\]

### One-Line Description

\[One sentence: what is it?\]

### The Problem

**What problem does this solve or what aspiration does this address?**

**Who has this problem?** 

**Why does it matter?**

### Your Solution

**What are you building? How does it work at a high level?**

### Comparable Products in Meta / in the World

\[What existing products are similar? What is their usage/traction? What's missing or broken about them? What gap do you fill?\]

| Product | Usage / Scale | What's Missing? |
| :---- | :---- | :---- |
| \[Product 1\] | \[e.g., 10M users, widely adopted\] | \[Gap you address\] |
| \[Product 2\] | \[e.g., Popular in X community\] | \[Gap you address\] |
| \[Product 3\] | \[e.g., Enterprise standard\] | \[Gap you address\] |

**What makes your idea exciting or unique? (1-2 sentences)**

### Product Category

**Check all that apply:**

| \[  \] Consumer App — End-user mobile or web application | \[  \] AI Agent / Assistant — Autonomous AI agent for a specific domain |
| :---- | :---- |
| \[  \] Developer Tool / CLI — IDE plugins, linters, debuggers, code generators | \[  \] DevOps / Infrastructure Tooling — CI/CD, deployment, monitoring, IaC, observability |
| \[  \] Library / SDK / Framework — Reusable packages other engineers build on | \[  \] Data Tool / Pipeline — Data processing, visualization, annotation, ETL |
| \[  \] Internal Productivity Tool — Workflow automation, dashboards, admin tools | \[  \] Creative / Media Tool — Image, video, audio, content generation/editing |
| \[  \] API / Platform Service — Backend service (DB, Hosting etc.) or API | \[  \] Security / Privacy Tool — Scanning, vulnerability detection, compliance |
| \[  \] Education / Learning Product — Interactive learning, doc generators | \[  \] Communication / Collaboration — Chat, knowledge sharing |
| \[  \] Hardware / Embedded / IoT — Software for devices, wearables, embedded | \[  \] Research Prototype / Benchmark — Novel research implementation or eval framework |
| \[  \] Other: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |  |

### Technical Approach

**Stack Orientation**  
Start here — is your project primarily built on an external/open stack or Meta-internal stack?

* **External / Open Stack** (preferred) — Standard open-source tools, cloud infra, public frameworks. This is our default expectation. Open stacks produce more transferable data, easier to isolate, and faster dev cycles.  
* **Meta-Internal Stack** — Buck, Hack, internal services, Meta-specific infra. Acceptable when Meta tooling provides unique value.  
* **Hybrid** — Mostly open, with targeted Meta dependencies where justified.

Which orientation and why?

**Primary Tech Stack**  
All reasonable costs of development will be funded. Example stacks by area:

| Area | Example Technologies |
| :---- | :---- |
| Web | React, Next.js, Vercel, TypeScript, Node.js, Tailwind |
| Mobile | Swift/iOS, Kotlin/Android, React Native, Flutter |
| Infra / Hosting / Backend | AWS, GCP, Vercel, PostgreSQL, Redis, Kubernetes, Docker |
| AI/ML Frameworks | PyTorch, HuggingFace, LangChain, vLLM, ONNX, LlamaIndex |
| DevOps / Infra-as-Code | Terraform, Pulumi, GitHub Actions, ArgoCD, Ansible |
| Hardware / Embedded | Rust, C/C++, RTOS, Arduino, embedded Linux, Zephyr |
| Data / Analytics | Spark, dbt, Airflow, Presto, Pandas, DuckDB |
| Meta-Internal | Buck, Hack/PHP, Configerator, Tupperware, FBLearner, Internal LLM serving |

Describe your core technology choices across all layers of your system:

* 

**Meta Tech Stack (if any)**  
If you depend on Meta-internal technology, list each dependency with justification. We prefer open stacks. Meta dependencies should be the exception, not the rule.

**Key Architectural Direction**  
What are the 2–3 most important technical bets you're making? Why this architecture over alternatives?

1. **Decision 1**: \[e.g., "Monorepo with shared types across frontend/backend" — reduces integration bugs, speeds iteration for a small team\]  
2. **Decision 2**: \[e.g., "Local-first with sync" — Why: enables offline use, reduces server costs, better UX\]  
3. **Decision 3**: \[e.g., "Plugin architecture from day 1" — Why: allows community extensions, generates diverse integration data\]

## 2/ Launch Intent

How do you envision launching this?

- [ ] **Internal only** \- Built for Meta employees, solving an internal need  
- [ ] **External** \- Built for the public, intended for open release  
- [ ] **Both** \- Internal first, then external  
- [ ] **Technology Demonstrator** \- Primarily a technical/research challenge

**Brief explanation of your launch reasoning:**

## 3/ Team Composition

### Founding Team (1 or 2 people)

The founding team are the leader(s) of the project and accountable for execution and results.

| Name | Role | Key Strength |
| :---- | :---- | :---- |
| \[Founder 1\] | \[e.g., Tech Lead\] | \[e.g., Full-stack, shipped X\] |
| \[Founder 2\] |  |  |

### Engineers Needed (3-4 additional, fractional okay)

| Profile | Count Needed | Full-Time or Fractional? | Notes (e.g. key skills needed) |
| :---- | :---- | :---- | :---- |
| iOS |  |  |  |
| Android |  |  |  |
| Web / Frontend |  |  |  |
| Product Generalist (Full-Stack) |  |  |  |
| Systems Generalist (Infra/Backend) |  |  |  |
| ML Generalist |  |  |  |
| Other Specialized Eng (specify) |  |  |  |
| Other Functions (Design, PM, etc.) \- *Fractional* |  |  |  |
| **Total** |  |  |  |

As the project expands and demonstrates good results, the team can expand.

## 4/ Data Generation Plan

### The Exchange

We expect high-quality, real-world data for training our models. This is the return the incubator demands.

* You will **exclusively use Metacode** and **Meta’s models** (Avocado for now) for entire project  
* You **will create a RL environment** (Docker likely) **early** for your full product code and testing, so RL tasks run in a full-fidelity environment.  
* Expect to spend a **significant portion** of your time generating these artifacts.

### Data Types You Will Generate

- [ ] SWE-Bench style tasks and environments  
- [ ] Trace corrections / annotations  
- [ ] Development artifacts (PRs, code reviews, design docs)  
- [ ] Bug Reports \-\> Tasks / Annotations  
- [ ] Evaluation benchmarks (evals)  
- [ ] Other (specify): \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### Domain Depth

While we accept broad data, we would like projects to also go **deep into one or more domains to** generate rich, domain-specific training data.

**Your primary domain**: \[e.g., DevOps, Infrastructure-as-Code, Consumer AI, Mobile Development, Data Engineering, Security, etc.\]

**Additional Domains with depth**:

### Training Data from Usage

**Once your product has users, what training data could you collect from their usage?**  
*Example: Our IaC tool will produce hundreds of Terraform plans → users will apply sequence corrections, generating high-quality infra automation training data. Each user interaction produces annotated traces of intent → code → deployment → fix cycles.*

## 5/ Rough Timeline

| Phase | Duration | Key Milestones |
| :---- | :---- | :---- |
| **Planning, Infra & Dev Env Setup** | Max 1 week | Repo created, CI/CD set up, team onboarded, architecture finalized |
| **MVP** | \~1 month | Core functionality working, demo-able, first users or internal dogfooding |
| **v1** | \~2 months | Feature-complete first version, stable, measurable usage/impact |

### Key Milestones & Deliverables

**End of Week 1**: \[What's done?\]

**End of Month 1 (MVP)**: \[What can users do?\]

**End of Month 2 (v1)**: \[What does success look like?\]

## 6/ Biggest Risks And Dependencies

Just a few bullet points or sentences of biggest risks to execution. Think of it also as, what do you need most from us?

## 7/ Anything Else?

\[Optional: Links to prototypes, design mocks, prior art, relevant experience, or anything that strengthens your proposal.\]  