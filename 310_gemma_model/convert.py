"""310 — fetch + convert hf-internal-testing/tiny-random-Gemma2ForCausalLM
to our .npz (AUTHORING/DEMO, gen group).

    uv run --group gen python 310_gemma_model/convert.py

Downloads ONLY config.json + model.safetensors — no un-permute needed because
Gemma-2 uses rotate-half RoPE, same layout as HF.  Verifies our ``gemma_forward``
reproduces genuine HF logits (eager attention, float32), then commits ``real_ref.npz``.

Outputs (next to this file):
  gemma_tiny.npz                      full weights (git-ignored)
  tests/fixtures/real_ref.npz         committed reference logits + cfg

We force ``attn_implementation='eager'`` so HF uses the explicit-softmax path that
our float64 forward mirrors (PyTorch SDPA diverges numerically).  Gemma-2 ties
embeddings, so the checkpoint has no ``lm_head.weight`` — ``load_gemma`` reuses
``model.embed_tokens.weight`` as the output projection.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
NAME = "hf-internal-testing/tiny-random-Gemma2ForCausalLM"
# Fixed prompt for the committed reference
INPUT_IDS = np.array([[1, 2, 3, 4]], dtype=np.int32)


def main() -> None:
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files

    cfg_path = hf_hub_download(NAME, "config.json")
    cfg = json.load(open(cfg_path))

    repo_files = set(list_repo_files(NAME))
    if "model.safetensors" in repo_files:
        from safetensors.torch import load_file as torch_load_file
        sd_torch = torch_load_file(hf_hub_download(NAME, "model.safetensors"))
        sd = {k: v.float().numpy() for k, v in sd_torch.items()}
    elif "pytorch_model.bin" in repo_files:
        bin_path = hf_hub_download(NAME, "pytorch_model.bin")
        sd_torch = torch.load(bin_path, map_location="cpu", weights_only=True)
        sd = {k: v.float().numpy() for k, v in sd_torch.items()}
    else:
        raise FileNotFoundError(f"No weight file found in {NAME}")

    H = cfg["num_attention_heads"]
    KV = cfg["num_key_value_heads"]
    d = cfg["hidden_size"]
    head_dim = cfg["head_dim"]
    L_layers = cfg["num_hidden_layers"]

    W: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": sd["model.embed_tokens.weight"].astype(np.float32),
        "model.norm.weight": sd["model.norm.weight"].astype(np.float32),
    }
    for i in range(L_layers):
        p = f"model.layers.{i}"
        for nm in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ):
            W[f"{p}.{nm}"] = sd[f"{p}.{nm}"].astype(np.float32)

    np.savez(HERE / "gemma_tiny.npz", **W)
    print(f"Wrote gemma_tiny.npz ({len(W)} arrays)")

    # Genuine HF logits (eager attention to match our explicit-softmax path).
    from transformers import AutoConfig, Gemma2ForCausalLM
    from leet_llm import GemmaConfig, gemma_forward, load_gemma

    hf_config = AutoConfig.from_pretrained(NAME)
    hf_config.attn_implementation = "eager"
    hf = Gemma2ForCausalLM.from_pretrained(
        NAME, config=hf_config, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        hf_logits = hf(torch.tensor(INPUT_IDS, dtype=torch.long)).logits.float().numpy()
    print(f"[hf] genuine Gemma2ForCausalLM (eager, float32) logits: shape={hf_logits.shape}")

    rope_params = cfg.get("rope_parameters") or {}
    rope_base = rope_params.get("rope_theta", cfg.get("rope_theta", 10000.0))

    gcfg = GemmaConfig(
        dim=d,
        n_layers=L_layers,
        n_heads=H,
        n_kv_heads=KV,
        head_dim=head_dim,
        vocab_size=cfg["vocab_size"],
        intermediate_size=cfg["intermediate_size"],
        norm_eps=cfg.get("rms_norm_eps", 1e-6),
        rope_base=float(rope_base),
        query_pre_attn_scalar=cfg.get("query_pre_attn_scalar", head_dim),
        final_logit_softcapping=cfg.get("final_logit_softcapping", 30.0),
        attn_logit_softcapping=cfg.get("attn_logit_softcapping", 50.0),
        sliding_window=cfg.get("sliding_window", 4096),
        max_seq_len=cfg.get("max_position_embeddings", 8192),
    )
    params = load_gemma(W, gcfg)
    out_logits = gemma_forward(INPUT_IDS, params, gcfg)

    max_diff = np.max(np.abs(out_logits - hf_logits))
    np.testing.assert_allclose(
        out_logits, hf_logits, rtol=1e-2, atol=1e-2,
        err_msg=(
            f"BLOCKED: gemma_forward vs genuine HF max_abs_diff={max_diff:.3e}. "
            "Fix gemma_forward before regenerating real_ref.npz."
        ),
    )
    print(f"[verify] gemma_forward vs genuine Gemma2ForCausalLM float32 (max abs diff {max_diff:.2e}) ✓")

    (HERE / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    np.savez(
        HERE / "tests" / "fixtures" / "real_ref.npz",
        input_ids=INPUT_IDS,
        logits=hf_logits.astype(np.float32),
        dim=np.array(gcfg.dim),
        n_layers=np.array(gcfg.n_layers),
        n_heads=np.array(gcfg.n_heads),
        n_kv_heads=np.array(gcfg.n_kv_heads),
        head_dim=np.array(gcfg.head_dim),
        vocab_size=np.array(gcfg.vocab_size),
        intermediate_size=np.array(gcfg.intermediate_size),
        norm_eps=np.array(gcfg.norm_eps),
        rope_base=np.array(gcfg.rope_base),
        query_pre_attn_scalar=np.array(gcfg.query_pre_attn_scalar),
        final_logit_softcapping=np.array(gcfg.final_logit_softcapping),
        attn_logit_softcapping=np.array(gcfg.attn_logit_softcapping),
        sliding_window=np.array(gcfg.sliding_window),
        max_seq_len=np.array(gcfg.max_seq_len),
    )
    print("Wrote tests/fixtures/real_ref.npz")


if __name__ == "__main__":
    main()
