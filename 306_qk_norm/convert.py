"""306 — fetch + convert Qwen/Qwen3-0.6B to our .npz (AUTHORING/DEMO, gen group).

    uv run --group gen python 306_qk_norm/convert.py

Downloads ONLY config.json + model.safetensors (~1.2 GB) — no un-permute needed
because Qwen3 uses rotate-half RoPE, same layout as HF.  Verifies our
``qwen3_forward`` reproduces genuine HF logits, then commits ``real_ref.npz``.

Outputs (next to this file):
  qwen3_0_6b.npz                      full weights (git-ignored)
  tests/fixtures/real_ref.npz         committed reference logits + cfg

Note: Qwen3-0.6B has ``tie_word_embeddings=True`` — lm_head.weight is absent in the
checkpoint; ``load_qwen3`` must fall back to ``model.embed_tokens.weight``.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

# safetensors.numpy does not support bfloat16; safetensors.torch is used instead.

HERE = pathlib.Path(__file__).parent
NAME = "Qwen/Qwen3-0.6B"
# Fixed prompt for the committed reference
INPUT_IDS = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)


def main() -> None:
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.numpy import load_file

    # Download config
    cfg_path = hf_hub_download(NAME, "config.json")
    cfg = json.load(open(cfg_path))

    # Download weights (safetensors only — Qwen3-0.6B is a single shard)
    # Qwen3 weights are bfloat16; use safetensors.torch (not .numpy) since
    # safetensors.numpy doesn't support bfloat16.
    repo_files = set(list_repo_files(NAME))
    if "model.safetensors" in repo_files:
        from safetensors.torch import load_file as torch_load_file
        sd_torch = torch_load_file(hf_hub_download(NAME, "model.safetensors"))
        # Convert torch tensors → numpy float32
        sd = {k: v.float().numpy() for k, v in sd_torch.items()}
    else:
        raise FileNotFoundError(
            f"Expected model.safetensors in {NAME}. "
            "Check HF repo for sharded safetensors."
        )

    H = cfg["num_attention_heads"]
    KV = cfg["num_key_value_heads"]
    d = cfg["hidden_size"]
    head_dim = cfg.get("head_dim", d // H)
    L_layers = cfg["num_hidden_layers"]
    tie_embed = cfg.get("tie_word_embeddings", False)

    tok_embed = sd["model.embed_tokens.weight"].astype(np.float32)
    # lm_head absent when tie_word_embeddings=True — use embed
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
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ):
            W[f"{p}.{nm}"] = sd[f"{p}.{nm}"].astype(np.float32)

    np.savez(HERE / "qwen3_0_6b.npz", **W)
    print(f"Wrote qwen3_0_6b.npz ({len(W)} arrays)")

    # Get genuine HF logits (float64) from Qwen3ForCausalLM
    from transformers import AutoConfig, Qwen3ForCausalLM
    from leet_llm import Qwen3Config, load_qwen3, qwen3_forward

    hf_config = AutoConfig.from_pretrained(NAME)
    hf = Qwen3ForCausalLM.from_pretrained(
        NAME, config=hf_config, torch_dtype=torch.float64
    ).eval()
    with torch.no_grad():
        hf_logits = hf(torch.tensor(INPUT_IDS, dtype=torch.long)).logits.numpy()
    print(f"[hf] genuine Qwen3ForCausalLM (float64) logits: shape={hf_logits.shape}")

    sliding_window = cfg.get("sliding_window", None)
    mcfg = Qwen3Config(
        dim=d,
        n_layers=L_layers,
        n_heads=H,
        n_kv_heads=KV,
        head_dim=head_dim,
        vocab_size=cfg["vocab_size"],
        max_seq_len=cfg.get("max_position_embeddings", 4096),
        norm_eps=cfg.get("rms_norm_eps", 1e-6),
        qk_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        rope_base=cfg.get("rope_theta", 10000.0),
    )
    params = load_qwen3(W, mcfg)

    # Our forward (weights cast to float64 inside forward)
    out_logits = qwen3_forward(INPUT_IDS, params, mcfg)

    # Genuine parity check
    max_diff = np.max(np.abs(out_logits - hf_logits))
    np.testing.assert_allclose(
        out_logits, hf_logits, rtol=1e-5, atol=1e-4,
        err_msg=(
            f"BLOCKED: qwen3_forward vs genuine HF max_abs_diff={max_diff:.3e}. "
            "Fix qwen3_forward before regenerating real_ref.npz."
        ),
    )
    print(f"[verify] qwen3_forward vs genuine Qwen3ForCausalLM (max abs diff {max_diff:.2e}) ✓")

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
        head_dim=np.array(mcfg.head_dim),
        vocab_size=np.array(mcfg.vocab_size),
        max_seq_len=np.array(mcfg.max_seq_len),
        norm_eps=np.array(mcfg.norm_eps),
        qk_norm_eps=np.array(mcfg.qk_norm_eps),
        rope_base=np.array(mcfg.rope_base),
    )
    print("Wrote tests/fixtures/real_ref.npz")


if __name__ == "__main__":
    main()
