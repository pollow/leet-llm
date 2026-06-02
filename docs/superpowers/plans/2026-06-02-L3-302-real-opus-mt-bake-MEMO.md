# MEMO — Bake the real `opus-mt-en-zh` capstone reference (L3 task 302, option 3)

> Pick-up-cold notes for a fresh session. Goal: fetch the real `Helsinki-NLP/opus-mt-en-zh`
> checkpoint **once**, produce the committed reference fixture the gated capstone test needs,
> and eyeball a real English→Chinese translation. ~300 MB download, network + `gen` group.

## What already exists (task 302 scaffold, already merged)
- `302_translate/download.sh` — `snapshot_download` the checkpoint, then runs `convert.py`.
- `302_translate/convert.py` — loads the real `MarianMTModel` (float64), writes:
  - `302_translate/opus_mt_en_zh.npz` — full HF state_dict (**gitignored**, ~300 MB, never commit).
  - `302_translate/tests/fixtures/real_ref.npz` — tiny: `{src_ids, expected_ids (HF greedy), config scalars}`. **Commit this.**
- `302_translate/tests/test_translate.py::test_real_en_zh_matches_hf_greedy` — `@skipif` until both files exist; prefix-matches the learner's `translate` greedy output to `expected_ids`.

## Steps
1. `cd /home/deus/workspace/leet-llm`
2. `! bash 302_translate/download.sh`  ← run via the `!` prefix so output lands in-session. Expect it to print a Chinese translation of *"I have a dream that one day this nation will rise up."* and `wrote opus_mt_en_zh.npz (gitignored) + tests/fixtures/real_ref.npz`.
3. **Sanity-check our assembly on the real weights** (don't rely on the scaffold alone). Reuse the controller-proven Marian recipe (`/home/deus/.claude/jobs/.../tmp/prove_301.py` pattern) but point it at `opus_mt_en_zh.npz` + the real config, and assert it reproduces HF `MarianMTModel` logits to ~1e-9 and that greedy argmax matches `real_ref.npz["expected_ids"]` (prefix). If it matches, our `load_marian`/`encoder`/`decoder`/`transformer_logits` recipe is correct on the real model.
4. Commit ONLY the reference: `git add 302_translate/tests/fixtures/real_ref.npz && git commit -m "L3 302: bake real opus-mt-en-zh greedy reference fixture"`. Confirm `git status` does NOT show `opus_mt_en_zh.npz` (it's gitignored).

## Caveats / things to verify against the real `config.json` (spec §8 open items)
- **Activation:** the tiny hermetic fixture used `activation_function="gelu"`, and **L2 207 `ffn` hardcodes exact-erf GELU**. If the real config says `"swish"`/`"silu"` (some OPUS-MT models do), the FFN won't match → small follow-up: parameterize the activation in `ffn` (207) or in the Marian assembly. `convert.py` already stores `activation` in `real_ref.npz` — check it first.
- **`scale_embedding`:** read from config; `load_marian`/`transformer_logits` already honor `cfg.scale_embedding` (×√d). Confirm the real value.
- **Shared vs separate vocab:** en→zh may set `share_encoder_decoder_embeddings` / a distinct `decoder_vocab_size`. `load_marian` falls back to `model.shared.weight` when `decoder.embed_tokens.weight` is absent — verify the keys present in `opus_mt_en_zh.npz`.
- **Generation config drift:** real Marian `generate` defaults can include `num_beams>1`, `forced_eos_token_id`, and pad-suppression. `convert.py` forces `num_beams=1, do_sample=False`; the test is a **prefix** match, tolerating a forced-eos tail. If divergence appears mid-sequence, pass `forced_eos_token_id=None` (and check `bad_words_ids`) in `convert.py`'s `generate(...)`.
- **The gated test only PASSES once the learner has solved 301+302 (+ the L2 classic operators).** Baking `real_ref.npz` just creates the target; in the unsolved scaffold the test still raises `NotImplementedError` until solved. Baking is an authoring step, independent of solving.
- **License:** weights are CC-BY-4.0 (Helsinki-NLP / OPUS-MT). We commit only the derived token-id reference (`real_ref.npz`) with attribution already in `302_translate/README.md`; never the weights.

## Done when
`302_translate/tests/fixtures/real_ref.npz` is committed, `opus_mt_en_zh.npz` stays gitignored, the printed zh translation is sensible, and (if 301/302 are solved) `uv run grade 302` runs `test_real_en_zh_matches_hf_greedy` green instead of skipping.
