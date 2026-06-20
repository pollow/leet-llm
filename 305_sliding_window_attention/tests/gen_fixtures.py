"""305 — generate frozen golden fixtures for the sliding-window causal mask
AND the Mistral whole-model forward.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 305_sliding_window_attention/tests/gen_fixtures.py

Two fixtures are written:

1. ``band.npz`` — the HF Mistral additive sliding-window mask (L=6, W=3).
   Oracle: genuine ``MistralForCausalLM`` via forward hook.

2. ``tiny_mistral.npz`` — whole-model logits at a tiny seeded config (float64
   composed oracle + frozen HF-named weights). The composed oracle uses genuine
   torch primitives (F.linear, F.rms_norm, F.silu, F.scaled_dot_product_attention)
   with rotate-half RoPE and the band mask, NEVER the genuine MistralForCausalLM,
   so float64 accumulation is preserved → whole-model parity at rtol=1e-9.
   As an authoring sanity check we also assert the composed oracle matches a genuine
   ``MistralForCausalLM`` at rtol≈1e-3 (the float32 HF class is close, not exact).
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from transformers import MistralConfig as HFMistralConfig
from transformers import MistralForCausalLM

FIX = pathlib.Path(__file__).parent / "fixtures"

# ─── tiny model config ───────────────────────────────────────────────────────
V, d, NL, H, KV, Fd = 64, 16, 2, 4, 2, 32
EPS = 1e-5
BASE = 10000.0
# Small sliding window so the band *activates* at short L
WINDOW = 3
L = 5  # sequence length for the whole-model fixture


# ─── fixture 1: band mask (unchanged from the original) ──────────────────────

def _extract_hf_band(seq_len: int, window: int) -> np.ndarray:
    """Run a forward pass through a tiny MistralForCausalLM and capture the
    sliding-window attention mask produced internally by HF.

    Returns the ``(seq_len, seq_len)`` additive float64 mask:
      0.0  where attended  (i - window < j <= i)
      -inf where masked    (j > i or j <= i - window)
    """
    cfg = HFMistralConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        vocab_size=64,
        sliding_window=window,
        torch_dtype=torch.float64,
    )
    torch.manual_seed(42)
    model = MistralForCausalLM(cfg)
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def hook_fn(module, args, kwargs, output):  # noqa: ARG001
        bool_mask = kwargs.get("attention_mask")
        if bool_mask is not None and "mask" not in captured:
            # Shape is (batch=1, 1, L, L); True = attended, False = masked
            captured["mask"] = bool_mask[0, 0].clone()
        return output

    hook = model.model.layers[0].self_attn.register_forward_hook(
        hook_fn, with_kwargs=True
    )
    input_ids = torch.ones(1, seq_len, dtype=torch.long)
    with torch.no_grad():
        model(input_ids)
    hook.remove()

    bool_mask = captured["mask"].numpy()  # (L, L), bool, True=attended
    # Convert to additive float64: 0.0 where attended, -inf where masked
    additive = np.where(bool_mask, 0.0, -np.inf).astype(np.float64)
    return additive


# ─── rotate-half RoPE (HF convention, float64) ───────────────────────────────

def _rope_half_torch(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    """Rotate-half RoPE in float64 torch.

    x: (B, n_heads, L, head_dim)
    positions: (L,)
    Returns same shape.
    """
    head_dim = x.shape[-1]
    idx = torch.arange(0, head_dim, 2, dtype=torch.float64)
    inv_freq = 1.0 / (base ** (idx / head_dim))                   # (head_dim/2,)
    angle = torch.outer(positions.to(torch.float64), inv_freq)    # (L, head_dim/2)
    # Expand to (1, 1, L, head_dim) for broadcasting
    cos = torch.cat([angle.cos(), angle.cos()], dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat([angle.sin(), angle.sin()], dim=-1).unsqueeze(0).unsqueeze(0)
    # rotate_half: split into first/second half, negate second, concat
    x1, x2 = x[..., :head_dim // 2], x[..., head_dim // 2:]
    rotate = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotate * sin


# ─── sliding-window band mask (float64, additive) ────────────────────────────

def _band_mask_torch(seq_len: int, window: int) -> torch.Tensor:
    """Additive causal band mask (0.0 attended, -inf masked), float64."""
    rows = torch.arange(seq_len)[:, None]
    cols = torch.arange(seq_len)[None, :]
    causal = rows >= cols                        # lower triangle (including diag)
    in_band = cols > rows - window               # not too old
    attended = causal & in_band
    mask = torch.where(attended, torch.tensor(0.0, dtype=torch.float64),
                       torch.tensor(-float("inf"), dtype=torch.float64))
    return mask


# ─── composed float64 oracle ──────────────────────────────────────────────────

def _composed_oracle(W: dict, T: dict, ids: np.ndarray) -> np.ndarray:
    """Run the composed float64 Mistral forward.

    W: numpy weight dict (HF names)
    T: torch tensor dict (same keys)
    ids: (1, L) int array
    Returns logits (1, L, V) numpy float64.
    """
    head_dim = d // H
    pos = torch.arange(L, dtype=torch.long)
    mask = _band_mask_torch(L, WINDOW)           # (L, L), additive float64

    h = T["model.embed_tokens.weight"][torch.from_numpy(ids[0])].unsqueeze(0)  # (1, L, d)

    for i in range(NL):
        p = f"model.layers.{i}"

        # Pre-attention RMSNorm
        a = F.rms_norm(h, (d,), weight=T[f"{p}.input_layernorm.weight"], eps=EPS)

        # QKV projections
        q = F.linear(a, T[f"{p}.self_attn.q_proj.weight"])   # (1, L, d)
        k = F.linear(a, T[f"{p}.self_attn.k_proj.weight"])   # (1, L, KV*head_dim)
        v = F.linear(a, T[f"{p}.self_attn.v_proj.weight"])   # (1, L, KV*head_dim)

        # Split into heads: (1, L, n_heads, head_dim) → (1, n_heads, L, head_dim)
        q = q.reshape(1, L, H, head_dim).transpose(1, 2)
        k = k.reshape(1, L, KV, head_dim).transpose(1, 2)
        v = v.reshape(1, L, KV, head_dim).transpose(1, 2)

        # Apply rotate-half RoPE
        q = _rope_half_torch(q, pos, BASE)
        k = _rope_half_torch(k, pos, BASE)

        # GQA: repeat k/v
        reps = H // KV
        k = k.repeat_interleave(reps, dim=1)
        v = v.repeat_interleave(reps, dim=1)

        # Scaled dot-product attention with additive band mask
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # (1, H, L, head_dim)

        # Merge heads: (1, H, L, head_dim) → (1, L, d)
        o = o.transpose(1, 2).reshape(1, L, d)
        o = F.linear(o, T[f"{p}.self_attn.o_proj.weight"])

        # Residual + pre-FFN norm
        h = h + o
        f = F.rms_norm(h, (d,), weight=T[f"{p}.post_attention_layernorm.weight"], eps=EPS)

        # SwiGLU FFN
        swi = F.linear(
            F.silu(F.linear(f, T[f"{p}.mlp.gate_proj.weight"]))
            * F.linear(f, T[f"{p}.mlp.up_proj.weight"]),
            T[f"{p}.mlp.down_proj.weight"],
        )
        h = h + swi

    h = F.rms_norm(h, (d,), weight=T["model.norm.weight"], eps=EPS)
    logits = (h @ T["lm_head.weight"].T).detach().numpy()
    return logits


def main() -> None:
    FIX.mkdir(exist_ok=True)

    # ── fixture 1: band mask ──────────────────────────────────────────────────
    seq_len = 6
    window = 3
    mask = _extract_hf_band(seq_len, window)
    np.savez(FIX / "band.npz", mask=mask, seq_len=np.array(seq_len),
             window=np.array(window))
    print(f"  wrote band.npz  seq_len={seq_len} window={window}")
    rows = []
    for row in mask:
        rows.append("  " + " ".join("  0" if v == 0.0 else "-inf" for v in row))
    print("\n".join(rows))

    # ── fixture 2: tiny_mistral.npz ───────────────────────────────────────────
    rng = np.random.default_rng(42)
    head_dim = d // H
    ids = rng.integers(0, V, size=(1, L))

    W: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": rng.standard_normal((V, d)),
        "model.norm.weight":         rng.standard_normal((d,)),
        "lm_head.weight":            rng.standard_normal((V, d)),
    }
    for i in range(NL):
        p = f"model.layers.{i}"
        W[f"{p}.input_layernorm.weight"]          = rng.standard_normal((d,))
        W[f"{p}.post_attention_layernorm.weight"] = rng.standard_normal((d,))
        W[f"{p}.self_attn.q_proj.weight"]         = rng.standard_normal((d, d))
        W[f"{p}.self_attn.k_proj.weight"]         = rng.standard_normal((KV * head_dim, d))
        W[f"{p}.self_attn.v_proj.weight"]         = rng.standard_normal((KV * head_dim, d))
        W[f"{p}.self_attn.o_proj.weight"]         = rng.standard_normal((d, d))
        W[f"{p}.mlp.gate_proj.weight"]            = rng.standard_normal((Fd, d))
        W[f"{p}.mlp.up_proj.weight"]              = rng.standard_normal((Fd, d))
        W[f"{p}.mlp.down_proj.weight"]            = rng.standard_normal((d, Fd))

    # Convert to float64 torch tensors
    T = {k: torch.from_numpy(v.astype(np.float64)) for k, v in W.items()}

    # Run composed float64 oracle
    oracle_logits = _composed_oracle(W, T, ids)
    print(f"  composed oracle logits shape: {oracle_logits.shape}")

    # ── HF-anchor: verify the composed oracle matches genuine MistralForCausalLM ──
    # Build a genuine HF model with the SAME weights and SAME tiny config.
    # HF uses float32 internally (RMSNorm upcast etc.) so tolerance is ~1e-3.
    hf_cfg = HFMistralConfig(
        hidden_size=d,
        num_hidden_layers=NL,
        num_attention_heads=H,
        num_key_value_heads=KV,
        intermediate_size=Fd,
        vocab_size=V,
        sliding_window=WINDOW,
        max_position_embeddings=128,
        rms_norm_eps=EPS,
        rope_theta=BASE,
        hidden_act="silu",
        torch_dtype=torch.float32,
    )
    hf_model = MistralForCausalLM(hf_cfg)
    hf_model.eval()

    # Load our random weights into the HF model (convert back to float32)
    sd = hf_model.state_dict()
    with torch.no_grad():
        sd["model.embed_tokens.weight"].copy_(T["model.embed_tokens.weight"].float())
        sd["model.norm.weight"].copy_(T["model.norm.weight"].float())
        sd["lm_head.weight"].copy_(T["lm_head.weight"].float())
        for i in range(NL):
            p = f"model.layers.{i}"
            sd[f"{p}.input_layernorm.weight"].copy_(
                T[f"{p}.input_layernorm.weight"].float())
            sd[f"{p}.post_attention_layernorm.weight"].copy_(
                T[f"{p}.post_attention_layernorm.weight"].float())
            sd[f"{p}.self_attn.q_proj.weight"].copy_(
                T[f"{p}.self_attn.q_proj.weight"].float())
            sd[f"{p}.self_attn.k_proj.weight"].copy_(
                T[f"{p}.self_attn.k_proj.weight"].float())
            sd[f"{p}.self_attn.v_proj.weight"].copy_(
                T[f"{p}.self_attn.v_proj.weight"].float())
            sd[f"{p}.self_attn.o_proj.weight"].copy_(
                T[f"{p}.self_attn.o_proj.weight"].float())
            sd[f"{p}.mlp.gate_proj.weight"].copy_(
                T[f"{p}.mlp.gate_proj.weight"].float())
            sd[f"{p}.mlp.up_proj.weight"].copy_(
                T[f"{p}.mlp.up_proj.weight"].float())
            sd[f"{p}.mlp.down_proj.weight"].copy_(
                T[f"{p}.mlp.down_proj.weight"].float())
    hf_model.load_state_dict(sd)

    with torch.no_grad():
        hf_logits = hf_model(torch.tensor(ids, dtype=torch.long)).logits.numpy()

    np.testing.assert_allclose(oracle_logits, hf_logits, rtol=1e-3, atol=1e-3)
    max_diff = np.max(np.abs(oracle_logits - hf_logits))
    print(f"  HF-anchor: composed oracle vs MistralForCausalLM max-abs-diff = {max_diff:.2e} ✓")

    # ── write fixture ─────────────────────────────────────────────────────────
    np.savez(
        FIX / "tiny_mistral.npz",
        input_ids=ids,
        logits=oracle_logits,
        dim=np.array(d),
        n_layers=np.array(NL),
        n_heads=np.array(H),
        n_kv_heads=np.array(KV),
        vocab_size=np.array(V),
        sliding_window=np.array(WINDOW),
        max_seq_len=np.array(128),
        norm_eps=np.array(EPS),
        rope_base=np.array(BASE),
        **W,
    )
    print(f"  wrote tiny_mistral.npz  logits{oracle_logits.shape}")


if __name__ == "__main__":
    main()
