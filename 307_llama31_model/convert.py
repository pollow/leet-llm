"""307 — fetch + convert llamafactory/tiny-random-Llama-3 to our .npz
(AUTHORING/DEMO, gen group).

    uv run --group gen python 307_llama31_model/convert.py

Downloads ONLY config.json + model.safetensors — no un-permute needed because
Llama-3.1 uses rotate-half RoPE, same layout as HF.  Unlike 309 (which forced the
checkpoint's RoPE to default) this checkpoint ships an **active** ``rope_type=llama3``
long-context schedule — exactly the delta 307 implements — so nothing is forced.
Verifies our ``llama31_forward`` reproduces genuine HF logits (eager attention,
float32), then commits ``real_ref.npz``.

Outputs (next to this file):
  llama31_tiny.npz                    full weights (git-ignored)
  tests/fixtures/real_ref.npz         committed reference logits + cfg

No small *trained* (ungated) Llama-3.1 checkpoint exists (the real 1B+ models are
license-gated), so there is no Tier-C end-to-end demo; the random weights here exist
only as the grade-time genuine-HF anchor + loader coverage (Tier B, see README).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
NAME = "llamafactory/tiny-random-Llama-3"
INPUT_IDS = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32)

_LAYER_KEYS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def main() -> None:
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files

    cfg = json.load(open(hf_hub_download(NAME, "config.json")))

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
    L_layers = cfg["num_hidden_layers"]

    W: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": sd["model.embed_tokens.weight"].astype(np.float32),
        "model.norm.weight": sd["model.norm.weight"].astype(np.float32),
        "lm_head.weight": sd.get("lm_head.weight", sd["model.embed_tokens.weight"]).astype(np.float32),
    }
    for i in range(L_layers):
        p = f"model.layers.{i}"
        for nm in _LAYER_KEYS:
            W[f"{p}.{nm}"] = sd[f"{p}.{nm}"].astype(np.float32)

    np.savez(HERE / "llama31_tiny.npz", **W)
    print(f"Wrote llama31_tiny.npz ({len(W)} arrays)")

    # Genuine HF logits — eager attention, native llama3 RoPE schedule.
    from transformers import AutoConfig, LlamaForCausalLM
    from leet_llm import Llama31Config, llama31_forward, load_llama31

    rope_scaling = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
    rope_base = float(rope_scaling.get("rope_theta", cfg.get("rope_theta", 500000.0)))

    hf_config = AutoConfig.from_pretrained(NAME)
    hf_config.attn_implementation = "eager"
    hf = LlamaForCausalLM.from_pretrained(
        NAME, config=hf_config, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        hf_logits = hf(torch.tensor(INPUT_IDS, dtype=torch.long)).logits.float().numpy()
    print(f"[hf] genuine LlamaForCausalLM (eager, llama3-rope, float32) logits: {hf_logits.shape}")

    lcfg = Llama31Config(
        dim=d,
        n_layers=L_layers,
        n_heads=H,
        n_kv_heads=KV,
        vocab_size=cfg["vocab_size"],
        max_seq_len=cfg.get("max_position_embeddings", 131072),
        norm_eps=cfg.get("rms_norm_eps", 1e-5),
        rope_base=rope_base,
        rope_scaling={k: v for k, v in rope_scaling.items() if k != "rope_theta"} or None,
    )
    params = load_llama31(W, lcfg)
    out_logits = llama31_forward(INPUT_IDS, params, lcfg)

    max_diff = np.max(np.abs(out_logits - hf_logits))
    np.testing.assert_allclose(
        out_logits, hf_logits, rtol=1e-2, atol=1e-2,
        err_msg=(
            f"BLOCKED: llama31_forward vs genuine HF max_abs_diff={max_diff:.3e}. "
            "Fix llama31_forward before regenerating real_ref.npz."
        ),
    )
    print(f"[verify] llama31_forward vs genuine LlamaForCausalLM float32 (max abs diff {max_diff:.2e}) ✓")

    (HERE / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    np.savez(
        HERE / "tests" / "fixtures" / "real_ref.npz",
        input_ids=INPUT_IDS,
        logits=hf_logits.astype(np.float32),
        dim=np.array(lcfg.dim),
        n_layers=np.array(lcfg.n_layers),
        n_heads=np.array(lcfg.n_heads),
        n_kv_heads=np.array(lcfg.n_kv_heads),
        vocab_size=np.array(lcfg.vocab_size),
        max_seq_len=np.array(lcfg.max_seq_len),
        norm_eps=np.array(lcfg.norm_eps),
        rope_base=np.array(lcfg.rope_base),
        rope_scaling=np.array(json.dumps(lcfg.rope_scaling)),
    )
    print("Wrote tests/fixtures/real_ref.npz")


if __name__ == "__main__":
    main()
