"""307 — fetch + convert hf-internal-testing/tiny-random-MixtralForCausalLM
to our .npz (AUTHORING/DEMO, gen group).

    uv run --group gen python 307_mixtral_model/convert.py

Downloads ONLY config.json + model.safetensors (~2 MB) — no un-permute needed
because Mixtral uses rotate-half RoPE, same layout as HF.  Verifies our
``mixtral_forward`` reproduces genuine HF logits, then commits ``real_ref.npz``.

Outputs (next to this file):
  mixtral_tiny.npz                    full weights (git-ignored)
  tests/fixtures/real_ref.npz         committed reference logits + cfg

Weight layout in checkpoint:
  model.layers.{i}.mlp.gate.weight          (num_experts, d)        router
  model.layers.{i}.mlp.experts.gate_up_proj (num_experts, 2*Fd, d)  [gate;up]
  model.layers.{i}.mlp.experts.down_proj    (num_experts, d, Fd)    down

Note: the random-init checkpoint has hidden_act="silu" — no override needed.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
NAME = "hf-internal-testing/tiny-random-MixtralForCausalLM"
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
        from safetensors.torch import load_file as torch_load_file
        sd_torch = torch_load_file(hf_hub_download(NAME, "model.safetensors"))
        # Convert bfloat16 → float32
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
    L_layers = cfg["num_hidden_layers"]
    NE = cfg["num_local_experts"]
    NK = cfg["num_experts_per_tok"]
    tie_embed = cfg.get("tie_word_embeddings", False)

    tok_embed = sd["model.embed_tokens.weight"].astype(np.float32)
    lm_head_w = sd.get("lm_head.weight", tok_embed).astype(np.float32)

    W: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": tok_embed,
        "model.norm.weight": sd["model.norm.weight"].astype(np.float32),
        "lm_head.weight": lm_head_w,
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
            "mlp.gate.weight",
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
        ):
            W[f"{p}.{nm}"] = sd[f"{p}.{nm}"].astype(np.float32)

    np.savez(HERE / "mixtral_tiny.npz", **W)
    print(f"Wrote mixtral_tiny.npz ({len(W)} arrays)")

    # Get genuine HF logits with hidden_act forced to "silu" (real Mixtral activation).
    # Note: HF MixtralForCausalLM uses grouped_mm which requires float32 or bfloat16
    # (float64 is not supported). We run HF in float32 and compare at rtol=1e-3.
    from transformers import AutoConfig, MixtralForCausalLM
    from leet_llm import MixtralConfig, load_mixtral, mixtral_forward

    hf_config = AutoConfig.from_pretrained(NAME)
    hf_config.hidden_act = "silu"  # force SiLU (matches real Mixtral + our swiglu_ffn)
    hf = MixtralForCausalLM.from_pretrained(
        NAME, config=hf_config, torch_dtype=torch.float32
    ).eval()
    with torch.no_grad():
        hf_logits = hf(torch.tensor(INPUT_IDS, dtype=torch.long)).logits.float().numpy()
    print(f"[hf] genuine MixtralForCausalLM (SiLU-forced, float32) logits: shape={hf_logits.shape}")

    rope_params = cfg.get("rope_parameters") or {}
    rope_base = rope_params.get("rope_theta", cfg.get("rope_theta", 10000.0))

    mcfg = MixtralConfig(
        dim=d,
        n_layers=L_layers,
        n_heads=H,
        n_kv_heads=KV,
        vocab_size=cfg["vocab_size"],
        num_local_experts=NE,
        num_experts_per_tok=NK,
        max_seq_len=cfg.get("max_position_embeddings", 4096),
        norm_eps=cfg.get("rms_norm_eps", 1e-5),
        rope_base=float(rope_base),
    )
    params = load_mixtral(W, mcfg)

    # Our forward (weights cast to float64 inside forward)
    out_logits = mixtral_forward(INPUT_IDS, params, mcfg)

    # Genuine parity check
    max_diff = np.max(np.abs(out_logits - hf_logits))
    # HF uses float32; our forward is float64 — tolerance reflects float32 precision gap
    np.testing.assert_allclose(
        out_logits, hf_logits, rtol=1e-3, atol=1e-3,
        err_msg=(
            f"BLOCKED: mixtral_forward vs genuine HF max_abs_diff={max_diff:.3e}. "
            "Fix mixtral_forward before regenerating real_ref.npz."
        ),
    )
    print(f"[verify] mixtral_forward vs genuine MixtralForCausalLM float32 (max abs diff {max_diff:.2e}) ✓")

    # Write committed reference — logits from genuine HF model (float64)
    (HERE / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    np.savez(
        HERE / "tests" / "fixtures" / "real_ref.npz",
        input_ids=INPUT_IDS,
        logits=hf_logits,
        dim=np.array(mcfg.dim),
        n_layers=np.array(mcfg.n_layers),
        n_heads=np.array(mcfg.n_heads),
        n_kv_heads=np.array(mcfg.n_kv_heads),
        vocab_size=np.array(mcfg.vocab_size),
        num_local_experts=np.array(mcfg.num_local_experts),
        num_experts_per_tok=np.array(mcfg.num_experts_per_tok),
        max_seq_len=np.array(mcfg.max_seq_len),
        norm_eps=np.array(mcfg.norm_eps),
        rope_base=np.array(mcfg.rope_base),
    )
    print("Wrote tests/fixtures/real_ref.npz")


if __name__ == "__main__":
    main()
