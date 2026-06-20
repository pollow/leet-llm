"""305 — fetch + convert hf-internal-testing/tiny-random-MistralForCausalLM
to our .npz (AUTHORING/DEMO, gen group).

    uv run --group gen python 305_sliding_window_attention/convert.py

Downloads ONLY config.json + model.safetensors (~2 MB) — no un-permute needed
because Mistral uses rotate-half RoPE, same layout as HF. Verifies our
``mistral_forward`` reproduces genuine HF logits, then commits ``real_ref.npz``.

Outputs (next to this file):
  mistral_tiny.npz                   full weights (git-ignored)
  tests/fixtures/real_ref.npz        committed reference logits + cfg

Note: hf-internal-testing/tiny-random-MistralForCausalLM has sliding_window=4096,
so the band does NOT activate at short L — but the band is already exercised by the
operator invariant tests + the hermetic fixture (tiny_mistral.npz with window=3).

The random-init checkpoint has hidden_act="gelu" in its config, but real Mistral
always uses SiLU. We force hidden_act="silu" before instantiating the HF model so
that both sides (HF and our mistral_forward / swiglu_ffn) use the same activation.
The weights are unchanged — only the config field differs. This makes test B a
genuine parity check: our forward vs real MistralForCausalLM (SiLU-forced, float64)
on the downloaded weights, at rtol=1e-5 / atol=1e-4.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
NAME = "hf-internal-testing/tiny-random-MistralForCausalLM"
# Fixed prompt for the committed reference
INPUT_IDS = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)


def main() -> None:
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files

    # Download config
    cfg_path = hf_hub_download(NAME, "config.json")
    cfg = json.load(open(cfg_path))

    # Download weights — prefer safetensors, fall back to pytorch_model.bin
    repo_files = set(list_repo_files(NAME))
    if "model.safetensors" in repo_files:
        from safetensors.numpy import load_file
        sd = load_file(hf_hub_download(NAME, "model.safetensors"))
    elif "pytorch_model.bin" in repo_files:
        bin_path = hf_hub_download(NAME, "pytorch_model.bin")
        sd_torch = torch.load(bin_path, map_location="cpu", weights_only=True)
        sd = {k: v.numpy() for k, v in sd_torch.items()}
    else:
        raise FileNotFoundError(f"No weight file found in {NAME}")

    # Build our weight dict — no un-permute (rotate-half, HF layout as-is)
    H = cfg["num_attention_heads"]
    KV = cfg["num_key_value_heads"]
    d = cfg["hidden_size"]
    L_layers = cfg["num_hidden_layers"]

    tok_embed = sd["model.embed_tokens.weight"].astype(np.float32)
    W: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": tok_embed,
        "model.norm.weight": sd["model.norm.weight"].astype(np.float32),
        # Mistral has untied lm_head; if not present in checkpoint, use embed
        "lm_head.weight": sd.get("lm_head.weight", tok_embed).astype(np.float32),
    }
    for i in range(L_layers):
        p = f"model.layers.{i}"
        for nm in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ):
            W[f"{p}.{nm}"] = sd[f"{p}.{nm}"].astype(np.float32)

    np.savez(HERE / "mistral_tiny.npz", **W)
    print(f"Wrote mistral_tiny.npz ({len(W)} arrays)")

    # Get genuine HF logits with hidden_act forced to "silu" (real Mistral activation).
    # The random-init checkpoint config has hidden_act="gelu"; we override it so the HF
    # model uses the same activation as our mistral_forward (swiglu_ffn uses SiLU).
    # The weights are unchanged — only the config field is overridden.
    from transformers import AutoConfig, MistralForCausalLM
    from leet_llm import MistralConfig, load_mistral, mistral_forward

    hf_config = AutoConfig.from_pretrained(NAME)
    hf_config.hidden_act = "silu"  # force SiLU to match real Mistral + our forward
    hf = MistralForCausalLM.from_pretrained(
        NAME, config=hf_config, torch_dtype=torch.float64
    ).eval()
    with torch.no_grad():
        hf_logits = hf(torch.tensor(INPUT_IDS, dtype=torch.long)).logits.numpy()  # (1, L, V)
    print(f"[hf] genuine MistralForCausalLM (SiLU-forced, float64) logits: shape={hf_logits.shape}")

    sliding_window = cfg.get("sliding_window", None) or cfg.get("sliding_window_size", 4096)
    mcfg = MistralConfig(
        dim=d,
        n_layers=L_layers,
        n_heads=H,
        n_kv_heads=KV,
        vocab_size=cfg["vocab_size"],
        sliding_window=int(sliding_window),
        max_seq_len=cfg.get("max_position_embeddings", 4096),
        norm_eps=cfg.get("rms_norm_eps", 1e-5),
        rope_base=cfg.get("rope_theta", 10000.0),
    )
    params = load_mistral(W, mcfg)

    # Our forward (weights cast to float64 inside the forward)
    out_logits = mistral_forward(INPUT_IDS, params, mcfg)  # (1, L, V)

    # Genuine parity check: our forward vs real HF (SiLU-forced, float64).
    # Both sides use SiLU on the same downloaded weights so this MUST pass.
    # If it does not, there is a real forward bug — STOP and report.
    max_diff = np.max(np.abs(out_logits - hf_logits))
    np.testing.assert_allclose(
        out_logits, hf_logits, rtol=1e-5, atol=1e-4,
        err_msg=(
            f"BLOCKED: mistral_forward vs genuine HF (SiLU-forced) max_abs_diff={max_diff:.3e}. "
            "Fix mistral_forward before regenerating real_ref.npz."
        ),
    )
    print(f"[verify] mistral_forward vs genuine HF SiLU (max abs diff {max_diff:.2e}) ✓")

    # Write the committed reference — logits are from the genuine HF model (SiLU, float64)
    (HERE / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    np.savez(
        HERE / "tests" / "fixtures" / "real_ref.npz",
        input_ids=INPUT_IDS,
        logits=hf_logits,  # genuine HF MistralForCausalLM (SiLU-forced, float64)
        dim=np.array(mcfg.dim),
        n_layers=np.array(mcfg.n_layers),
        n_heads=np.array(mcfg.n_heads),
        n_kv_heads=np.array(mcfg.n_kv_heads),
        vocab_size=np.array(mcfg.vocab_size),
        sliding_window=np.array(mcfg.sliding_window),
        max_seq_len=np.array(mcfg.max_seq_len),
        norm_eps=np.array(mcfg.norm_eps),
        rope_base=np.array(mcfg.rope_base),
    )
    print("Wrote tests/fixtures/real_ref.npz")


if __name__ == "__main__":
    main()
