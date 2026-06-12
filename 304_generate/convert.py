"""304 — fetch + convert stories15M to our .npz (AUTHORING/DEMO, gen group).

    uv run --group gen python 304_generate/convert.py

stories15M is Karpathy's llama2.c TinyStories model — the exact checkpoint `llama3.np`
runs (Llama-2 arch == Llama-3 arch: RMSNorm + RoPE + SwiGLU; here n_kv==n_heads, so GQA
degenerates to MHA). We pull it from the canonical HF port `Xenova/llama2.c-stories15M`,
downloading ONLY the three files we need (no `.bin` duplicate, no `onnx/` dir):

    config.json  (~600 B) · model.safetensors (~60 MB) · tokenizer.model (~500 KB)

HF stores q/k **permuted** for its rotate-half RoPE; stories15M / `llama3.np` (and our
task 213/216) use **interleaved** RoPE, so we un-permute q/k back to the original layout.
We then verify our `llama_forward` reproduces HF's logits before writing anything.

Outputs (next to this file):
  - stories15M.model.npz          full HF-named weights, float32 (git-ignored)
  - tokenizer.model               Llama-2 SentencePiece, for the demo/benchmark (git-ignored)
  - tests/fixtures/real_ref.npz   tiny committed fixture: prompt ids + HF greedy story ids + cfg
"""

from __future__ import annotations

import json
import pathlib
import shutil

import numpy as np

HERE = pathlib.Path(__file__).parent
NAME = "Xenova/llama2.c-stories15M"
PROMPT = "Once upon a time"
GREEDY_NEW_TOKENS = 48  # length of the committed greedy reference story


def _unpermute(w: np.ndarray, n_heads: int, d1: int, d2: int) -> np.ndarray:
    """Inverse of HF's q/k permute: HF rotate-half layout -> interleaved (Meta) layout."""
    return w.reshape(n_heads, 2, d1 // n_heads // 2, d2).transpose(0, 2, 1, 3).reshape(d1, d2)


def _build_weights(sd: dict, cfg: dict) -> dict:
    H, KV, d, L = (cfg["num_attention_heads"], cfg["num_key_value_heads"],
                   cfg["hidden_size"], cfg["num_hidden_layers"])
    emb = sd["lm_head.weight"].astype(np.float32)  # tied: embed_tokens == lm_head here
    W = {"model.embed_tokens.weight": emb, "lm_head.weight": emb,
         "model.norm.weight": sd["model.norm.weight"].astype(np.float32)}
    for i in range(L):
        p = f"model.layers.{i}."
        W[p + "self_attn.q_proj.weight"] = _unpermute(sd[p + "self_attn.q_proj.weight"].astype(np.float32), H, d, d)
        W[p + "self_attn.k_proj.weight"] = _unpermute(sd[p + "self_attn.k_proj.weight"].astype(np.float32), KV, d, d)
        for nm in ("self_attn.v_proj.weight", "self_attn.o_proj.weight",
                   "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                   "input_layernorm.weight", "post_attention_layernorm.weight"):
            W[p + nm] = sd[p + nm].astype(np.float32)
    return W


def main() -> None:
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    # --- download only the three files we need (avoid .bin dupe + onnx/) ---
    cfg = json.load(open(hf_hub_download(NAME, "config.json")))
    sd = load_file(hf_hub_download(NAME, "model.safetensors"))
    tok_src = hf_hub_download(NAME, "tokenizer.model")
    shutil.copy(tok_src, HERE / "tokenizer.model")

    W = _build_weights(sd, cfg)
    np.savez(HERE / "stories15M.model.npz", **W)

    # --- verify our forward matches HF before trusting the conversion ---
    import torch
    from transformers import AutoModelForCausalLM
    from leet_llm import LlamaConfig, load_llama, llama_forward

    lcfg = LlamaConfig(dim=cfg["hidden_size"], n_layers=cfg["num_hidden_layers"],
                       n_heads=cfg["num_attention_heads"], n_kv_heads=cfg["num_key_value_heads"],
                       vocab_size=cfg["vocab_size"], max_seq_len=cfg["max_position_embeddings"],
                       norm_eps=cfg["rms_norm_eps"], rope_base=cfg["rope_theta"])
    params = load_llama(W, lcfg)
    hf = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(HERE / "tokenizer.model"))
    prompt_ids = np.array([[sp.bos_id()] + sp.encode(PROMPT)])

    probe = llama_forward(prompt_ids, params, lcfg)[0, -1]
    with torch.no_grad():
        ref = hf(torch.tensor(prompt_ids)).logits[0, -1].numpy()
    np.testing.assert_allclose(probe, ref, rtol=1e-4, atol=1e-3)
    print(f"[verify] our llama_forward matches HF logits (max abs diff "
          f"{np.max(np.abs(probe - ref)):.2e}) ✓")

    # --- bake the greedy reference story from the genuine HF model ---
    with torch.no_grad():
        gen = hf.generate(torch.tensor(prompt_ids), do_sample=False,
                          max_new_tokens=GREEDY_NEW_TOKENS, num_beams=1)
    expected_ids = gen[0].numpy()

    (HERE / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    np.savez(HERE / "tests" / "fixtures" / "real_ref.npz",
             prompt_ids=prompt_ids, expected_ids=expected_ids,
             max_new_tokens=np.array(GREEDY_NEW_TOKENS),
             dim=np.array(lcfg.dim), n_layers=np.array(lcfg.n_layers),
             n_heads=np.array(lcfg.n_heads), n_kv_heads=np.array(lcfg.n_kv_heads),
             vocab_size=np.array(lcfg.vocab_size), max_seq_len=np.array(lcfg.max_seq_len),
             norm_eps=np.array(lcfg.norm_eps), rope_base=np.array(lcfg.rope_base))

    print("story:", sp.decode(expected_ids.tolist()))
    print("wrote stories15M.model.npz + tokenizer.model (gitignored) + tests/fixtures/real_ref.npz")


if __name__ == "__main__":
    main()
