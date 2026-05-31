# leet-llm — Level 1: Tokenization & Batching (Design)

> The text ⇄ tensors layer. Learners build a SentencePiece-style BPE tokenizer from
> scratch (train → save/load → encode/decode), add a load-only tiktoken path for future
> real models, then the batching plumbing (padding, masks, position ids) that L2/L3 feed on.

- **Status:** design approved 2026-05-31. Extends the locked ladder in
  `2026-05-31-leet-llm-curriculum-design.md` (§2, §6).
- **Level goal:** turn raw text into the exact tensors a transformer consumes, and back —
  understanding every merge, every special token, every pad position by hand.

---

## 1. Anchor: what the capstone actually uses

The L3 capstone (`llama3.np` in the sibling workspace) does **not** run a real Llama-3
tokenizer. It runs Karpathy's **stories15M** checkpoint (TinyStories, Llama-2 arch) with a
**SentencePiece BPE, 32,000 vocab**, shipped as plain JSON `{"tokens": [...], "scores": [...]}`:

- Vocab layout: `<unk>`, `<s>`(BOS=1), `</s>`(EOS=2), then the **256 raw bytes** as fallback,
  then merged pieces (`' t'`, `'er'`, `' th'` — note leading spaces are ordinary symbols).
- `scores` are **negative ranks** (`-1, -2, …`); earlier/higher-priority merges score higher.
- Reference encoder is the clone's 62-line `tokenizer.py` — our **golden oracle**.

So L1's primary path is **SentencePiece-lineage** (score-greedy merge, no regex, no
byte→unicode remap). The tiktoken path is added **load-only** for future OSS models.

### The two lineages (for the spec's memory)

| | SentencePiece (stories15M, **primary**) | tiktoken / byte-level (Llama-3.x, Qwen, GPT-4) |
|---|---|---|
| Artifact | `{tokens, scores}` JSON | raw-byte **rank table** `dict[bytes,int]` |
| Pre-tokenize | none (flat stream, spaces are symbols) | **regex** split; merges never cross chunks |
| Byte handling | 256-byte fallback baked into vocab | raw bytes throughout (no GPT-2 unicode remap) |
| Encode | greedy **highest-score** adjacent merge | rank-greedy merge **within each chunk** |
| In L1 | **trained + loaded** (full path) | **load-only** (109–110) |

**Real Llama-3.2-1B** (and Qwen, GPT-4) all use the raw-byte tiktoken format, vocab
128,256 for Llama-3.x. Adding any of them later is just pointing task 110 at their rank
table — no new code. We commit to the **raw-byte tiktoken format**, so the GPT-2
`merges.txt` + printable-unicode remap is deliberately **not** built (designer note only).

---

## 2. The settled BPE algorithm

One algorithm, used everywhere; matches the stories15M reference so golden tests are exact.

**Train** (`bpe_train`):
1. Start from the flat character/byte sequence of the corpus (spaces are ordinary symbols).
2. Count every adjacent pair. Pick **highest count**; **tie-break = lexicographically
   smallest pair** (compare left symbol, then right).
3. Merge that pair everywhere, append the new piece to vocab, assign it the next
   **descending score** (`-1, -2, …`). The score column *is* the merge order.
4. Repeat to target vocab size.

**Encode** (`bpe_encode`) — exactly `tokenizer.py`: map text → char ids, then loop: among
all adjacent pairs find the one whose concatenation is in vocab with the **highest score**
(strict `>`, so ties keep the earliest position); merge that single position; repeat until
no adjacent pair is mergeable.

**Decode:** concatenate piece strings, strip specials.

**Worked micro-example** (corpus `"ab ab"`, base symbols `a b ⎵`):

| Round | Pair counts | Winner (tie-break) | New piece | Score | Sequence |
|---|---|---|---|---|---|
| 1 | `(a,b)=2 (b,⎵)=1 (⎵,a)=1` | `(a,b)` count 2 | `"ab"` | −1 | `[ab, ⎵, ab]` |
| 2 | `(ab,⎵)=1 (⎵,ab)=1` | `(⎵,ab)` — `⎵`<`a` | `"⎵ab"` | −2 | `[ab, ⎵ab]` |

Encoding `"ab ab"` then greedily reconstructs `[ab, ⎵ab]`. The leading-space token emerges
naturally, mirroring the real 32k vocab.

---

## 3. The task list (13 tasks, ids 101–113)

Level theme allows tokenizer vocabulary now; still **no "attention"/"heads"** vocabulary —
the padding mask is a "padding mask" (the attention link is L2; designer-note only).

| Id | Slug | Builds | Reuses | Lineage |
|----|------|--------|--------|---------|
| **101** | `char_vocab` | `stoi`/`itos` from a corpus; char encode/decode; round-trip | — | — |
| **102** | `byte_tokenizer` | text ↔ utf-8 bytes ↔ ids; 256-byte fallback idea | — | — |
| **103** | `bpe_step` | count adjacent pairs **+** apply one merge (the two BPE primitives) | — | SP |
| **104** | `bpe_train` | training loop → `vocab + scores`; tie-break specified | 103 | SP |
| **105** | `tokenizer_io` | save/load the `{tokens, scores}` JSON; train→save→load round-trip | 104 | SP |
| **106** | `bpe_encode` | score-greedy encode from a **loaded** artifact; golden vs stories15M | 105 | SP |
| **107** | `bpe_decode` | ids→text, byte-fallback pieces, BOS/EOS strip; `decode(encode(x))==x` | 106 | SP |
| **108** | `special_tokens` | add/strip BOS/EOS, handle `<unk>` (generic over both lineages) | 107 | — |
| **109** | `regex_pretokenize` | GPT-4/Llama-3 regex split into chunks (merges won't cross) | — | tiktoken |
| **110** | `tiktoken_load_encode` | load raw-byte **rank table**; rank-greedy merge within chunks; encode/decode; golden vs GPT-2/Llama-3.2 | 109 | tiktoken |
| **111** | `padding_and_mask` | pad variable-length id lists to `(B, Lmax)`; build 0/1 **padding mask** | L0 `masked_fill` | — |
| **112** | `position_indices` | per-sequence positions `0..L-1`, padding-aware | — | — |
| **113** | `build_batch` | assemble `{input_ids, pad_mask, position_ids}` — the exact L2/L3 tensors | 111,112 | — |

**Design principle that ties it together:** `encode`/`decode` are parameterized by a
*loaded artifact*, not coupled to the trainer. So the same encoder runs the learner's toy
tokenizer **and** the real `tokenizer.model.np` — "loading the real one" is a grading
milestone, not new code. Batching (111–113) is lineage-agnostic, so it sits after both paths.

**Designer notes (NOT in learner READMEs):**
- 103/104 → BPE used by every modern tokenizer; 105's JSON = the real artifact format.
- 111's "padding mask" → becomes the attention mask in L2.
- The deliberately-skipped **GPT-2 `merges.txt` + `bytes_to_unicode` remap** and the
  **tiktoken trainer** are future-L3 OSS-zoo deltas; L1 only *loads* the raw-byte tiktoken
  format, which already unlocks Llama-3.2 / Qwen / GPT-4 for inference.

---

## 4. Testing strategy

Tokenization adds three things L0 didn't have: algorithmic determinism (tie-breaks), an
external oracle, and round-trip invariants. Five layers, plus a fixture plan.

1. **Golden-oracle parity (gold standard).** Freeze `(string → [ids])` pairs generated
   *once* at authoring time by the reference encoder, then hard-code them into the test.
   The suite asserts the learner reproduces the frozen numbers with **zero runtime
   dependency** on `tiktoken` or the clone.
2. **Round-trip / invariant tests.** `decode(encode(x))==x` for in-vocab text;
   `train→save→load→encode` ≡ pre-save encode. Property-style over random strings.
3. **Hand-computed micro tests** for training (103/104): a tiny embedded string where the
   first 2–3 merges are derivable by hand, asserting exact `(pair, score)` incl. tie-break.
4. **Shape/dtype/structure tests** for batching (111–113): `(B, Lmax)`, mask dtype, dict
   keys present.
5. **Edge cases:** empty string, unknown char (→`<unk>`/byte fallback), single token,
   all-pad row, truncation past `max_seq_len`.

### Corpora — no corpus files; embed strings in the test

- **Tier 1 (exact-merge):** tens of chars chosen to exercise ties, e.g. `"ab ab ab cab"`
  → hand-derive and assert the first 2–3 merges.
- **Tier 2 (train-to-target):** a few-hundred-char fixed paragraph; assert **invariants**
  (terminates, `len(vocab)==target`, all base symbols present, scores strictly descending,
  round-trip) rather than the exact merge list.
- Cite the Sennrich `{low,lower,newest,widest}` example in the README as the textbook
  reference, but derive our own trace (it uses `</w>` word-boundary BPE; our convention is
  flat/space-inclusive).

### Fixture artifacts

| Path | Bulk fixture | Real-parity fixture |
|---|---|---|
| SentencePiece (106/107) | tiny self-trained `{tokens,scores}` from the Tier-2 corpus | vendored **`tokenizer.model.np`** (32k); ids frozen from clone's `tokenizer.py` |
| tiktoken (110) | tiny **self-trained** byte-level-BPE+regex rank table (~512 ranks, ~30 KB, license-clean) | **GPT-2** via `tiktoken.get_encoding("gpt2")` (MIT); vendor `_mergeable_ranks` (already `dict[bytes,int]`) or freeze ids only |

- **Self-trained tiny tables are primary** — deterministic, tiny, no large blobs in tests.
- **One real-parity test per path** proves production-tokenizer compatibility.
- **Skip Llama-3.2 as a CI dependency** (gated, Llama-licensed); leave an opt-in local test
  (`@pytest.mark.skipif` on a missing path/env) for anyone who has accepted the license.
- **License note:** `tokenizer.model.np` (Llama-2) and any vendored Llama tokenizer are
  Llama-licensed — fine for personal use; flag before publishing the repo publicly. GPT-2
  ranks and `tiktoken` are MIT.

---

## 5. Reuse registrations (`leet_llm/_registry.py`)

Add as each task lands (single source of truth = the task stub):
`char_encode/char_decode` (101), `text_to_bytes/bytes_to_text` (102), `count_pairs/apply_merge`
(103), `bpe_train` (104), `save_tokenizer/load_tokenizer` (105), `bpe_encode` (106),
`bpe_decode` (107), `add_special_tokens/strip_special_tokens` (108), `regex_split` (109),
`tiktoken_encode/tiktoken_decode` (110), `pad_batch/padding_mask` (111), `position_ids`
(112), `build_batch` (113). (Exact public names finalized per-task at scaffold time.)

---

## 6. Out of scope (L1)

- The tiktoken **trainer** (byte-level merge learning over regex chunks) — L3 delta.
- GPT-2 `merges.txt` format + `bytes_to_unicode` remap — L3 delta.
- SentencePiece **unigram** algorithm, normalization/`▁` rules — we model the BPE variant
  the capstone uses; mention unigram in a README "Read More" aside only.
- Truncation/packing strategies beyond simple right-pad + `max_seq_len` cap.
