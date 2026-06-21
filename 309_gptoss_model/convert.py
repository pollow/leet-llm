"""309 — fetch + convert hf-internal-testing/tiny-random-GptOssForCausalLM
to our .npz (AUTHORING/DEMO, gen group).

    uv run --group gen python 309_gptoss_model/convert.py

Downloads ONLY config.json + model.safetensors — no un-permute needed because
GPT-OSS uses rotate-half RoPE, same layout as HF.  Verifies our ``gptoss_forward``
reproduces genuine HF logits (eager attention, float32), then commits ``real_ref.npz``.

Outputs (next to this file):
  gptoss_tiny.npz                     full weights (git-ignored)
  tests/fixtures/real_ref.npz         committed reference logits + cfg

Two forced settings (analogous to 308 forcing SiLU):
  * ``attn_implementation='eager'`` — the explicit-softmax-with-sink path our float64
    forward mirrors (GPT-OSS has no SDPA path; flash needs the sink kernel).
  * ``rope_type='default'`` — the checkpoint declares YaRN long-context scaling, which
    is deferred to 307 / L4.  Our forward uses default rotate-half RoPE, so we force the
    genuine model to the same RoPE for an apples-to-apples cross-check.  The weights are
    random (no demo); this layer exists only as the grade-time genuine-HF anchor + loader
    coverage (Tier B, see README).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
NAME = "hf-internal-testing/tiny-random-GptOssForCausalLM"
INPUT_IDS = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32)

# Every per-layer weight (names match HF exactly — no un-permute).
_LAYER_KEYS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.q_proj.bias",
    "self_attn.k_proj.weight", "self_attn.k_proj.bias",
    "self_attn.v_proj.weight", "self_attn.v_proj.bias",
    "self_attn.o_proj.weight", "self_attn.o_proj.bias",
    "self_attn.sinks",
    "mlp.router.weight", "mlp.router.bias",
    "mlp.experts.gate_up_proj", "mlp.experts.gate_up_proj_bias",
    "mlp.experts.down_proj", "mlp.experts.down_proj_bias",
)


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
    NE = cfg["num_local_experts"]
    NK = cfg["num_experts_per_tok"]

    W: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": sd["model.embed_tokens.weight"].astype(np.float32),
        "model.norm.weight": sd["model.norm.weight"].astype(np.float32),
        "lm_head.weight": sd.get("lm_head.weight", sd["model.embed_tokens.weight"]).astype(np.float32),
    }
    for i in range(L_layers):
        p = f"model.layers.{i}"
        for nm in _LAYER_KEYS:
            W[f"{p}.{nm}"] = sd[f"{p}.{nm}"].astype(np.float32)

    np.savez(HERE / "gptoss_tiny.npz", **W)
    print(f"Wrote gptoss_tiny.npz ({len(W)} arrays)")

    # Genuine HF logits — eager attention + default RoPE (see module docstring).
    from transformers import AutoConfig, GptOssForCausalLM
    from leet_llm import GptOssConfig, gptoss_forward, load_gptoss

    rope_params = cfg.get("rope_parameters") or {}
    rope_base = float(rope_params.get("rope_theta", cfg.get("rope_theta", 150000.0)))

    hf_config = AutoConfig.from_pretrained(NAME)
    hf_config.attn_implementation = "eager"
    hf_config.rope_parameters = {"rope_type": "default", "rope_theta": rope_base}
    hf = GptOssForCausalLM.from_pretrained(
        NAME, config=hf_config, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        hf_logits = hf(torch.tensor(INPUT_IDS, dtype=torch.long)).logits.float().numpy()
    print(f"[hf] genuine GptOssForCausalLM (eager, default-rope, float32) logits: {hf_logits.shape}")

    gcfg = GptOssConfig(
        dim=d,
        n_layers=L_layers,
        n_heads=H,
        n_kv_heads=KV,
        head_dim=head_dim,
        vocab_size=cfg["vocab_size"],
        intermediate_size=cfg["intermediate_size"],
        num_local_experts=NE,
        num_experts_per_tok=NK,
        sliding_window=cfg.get("sliding_window", 128),
        norm_eps=cfg.get("rms_norm_eps", 1e-5),
        rope_base=rope_base,
        max_seq_len=cfg.get("max_position_embeddings", 4096),
    )
    params = load_gptoss(W, gcfg)
    out_logits = gptoss_forward(INPUT_IDS, params, gcfg)

    max_diff = np.max(np.abs(out_logits - hf_logits))
    np.testing.assert_allclose(
        out_logits, hf_logits, rtol=1e-2, atol=1e-2,
        err_msg=(
            f"BLOCKED: gptoss_forward vs genuine HF max_abs_diff={max_diff:.3e}. "
            "Fix gptoss_forward before regenerating real_ref.npz."
        ),
    )
    print(f"[verify] gptoss_forward vs genuine GptOssForCausalLM float32 (max abs diff {max_diff:.2e}) ✓")

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
        num_local_experts=np.array(gcfg.num_local_experts),
        num_experts_per_tok=np.array(gcfg.num_experts_per_tok),
        sliding_window=np.array(gcfg.sliding_window),
        norm_eps=np.array(gcfg.norm_eps),
        rope_base=np.array(gcfg.rope_base),
        max_seq_len=np.array(gcfg.max_seq_len),
    )
    print("Wrote tests/fixtures/real_ref.npz")


if __name__ == "__main__":
    main()
