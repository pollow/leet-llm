"""306 — generate frozen golden fixtures for per-head Q/K RMSNorm
AND the Qwen3 whole-model forward.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 306_qk_norm/tests/gen_fixtures.py

Two fixtures are written:

1. ``qknorm.npz`` — the float64 reference for the qk_norm operator (unchanged from
   the original fixture generator).

2. ``tiny_qwen3.npz`` — whole-model logits at a tiny seeded config (float64
   composed oracle + frozen HF-named weights).  The composed oracle uses genuine
   torch primitives (F.linear, F.rms_norm, F.silu, F.scaled_dot_product_attention)
   with rotate-half RoPE and per-head qk-norm, NEVER the genuine Qwen3ForCausalLM,
   so float64 accumulation is preserved → whole-model parity at rtol=1e-9.
   As an authoring sanity check we also assert the composed oracle matches a genuine
   ``Qwen3ForCausalLM`` at rtol≈1e-3 (the float32 HF class is close, not exact).
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Qwen3Config as HFQwen3Config
from transformers import Qwen3ForCausalLM

FIX = pathlib.Path(__file__).parent / "fixtures"

# ─── tiny model config ───────────────────────────────────────────────────────
V, d, NL, H, KV, Fd = 64, 16, 2, 4, 2, 32
HEAD_DIM = 4   # explicit Qwen3 head_dim (NOT d // H = 4 here, kept equal for tiny)
EPS = 1e-6
QK_EPS = 1e-6
BASE = 10000.0
L = 5  # sequence length for the whole-model fixture


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
    cos = torch.cat([angle.cos(), angle.cos()], dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat([angle.sin(), angle.sin()], dim=-1).unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :head_dim // 2], x[..., head_dim // 2:]
    rotate = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotate * sin


# ─── per-head qk-norm (float64 torch) ────────────────────────────────────────

def _qk_norm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Per-head RMSNorm over the last axis.

    x: (..., n_heads, L, head_dim)
    weight: (head_dim,)
    Returns same shape.
    """
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return (x / rms) * weight


# ─── causal mask (float64, additive) ─────────────────────────────────────────

def _causal_mask_torch(seq_len: int) -> torch.Tensor:
    """Additive causal mask (0.0 attended, -inf masked), float64."""
    rows = torch.arange(seq_len)[:, None]
    cols = torch.arange(seq_len)[None, :]
    attended = rows >= cols
    return torch.where(attended, torch.tensor(0.0, dtype=torch.float64),
                       torch.tensor(-float("inf"), dtype=torch.float64))


# ─── composed float64 oracle ──────────────────────────────────────────────────

def _composed_oracle(W: dict, T: dict, ids: np.ndarray) -> np.ndarray:
    """Run the composed float64 Qwen3 forward.

    W: numpy weight dict (HF names)
    T: torch tensor dict (same keys)
    ids: (1, L) int array
    Returns logits (1, L, V) numpy float64.
    """
    pos = torch.arange(L, dtype=torch.long)
    mask = _causal_mask_torch(L)   # (L, L), additive float64

    h = T["model.embed_tokens.weight"][torch.from_numpy(ids[0])].unsqueeze(0)  # (1, L, d)

    for i in range(NL):
        p = f"model.layers.{i}"

        # Pre-attention RMSNorm
        a = F.rms_norm(h, (d,), weight=T[f"{p}.input_layernorm.weight"], eps=EPS)

        # QKV projections (no bias)
        q = F.linear(a, T[f"{p}.self_attn.q_proj.weight"])   # (1, L, H*head_dim)
        k = F.linear(a, T[f"{p}.self_attn.k_proj.weight"])   # (1, L, KV*head_dim)
        v = F.linear(a, T[f"{p}.self_attn.v_proj.weight"])   # (1, L, KV*head_dim)

        # Split into heads: (1, L, n_heads, head_dim) → (1, n_heads, L, head_dim)
        q = q.reshape(1, L, H, HEAD_DIM).transpose(1, 2)
        k = k.reshape(1, L, KV, HEAD_DIM).transpose(1, 2)
        v = v.reshape(1, L, KV, HEAD_DIM).transpose(1, 2)

        # Per-head qk-norm BEFORE RoPE
        q = _qk_norm_torch(q, T[f"{p}.self_attn.q_norm.weight"], QK_EPS)
        k = _qk_norm_torch(k, T[f"{p}.self_attn.k_norm.weight"], QK_EPS)

        # Apply rotate-half RoPE
        q = _rope_half_torch(q, pos, BASE)
        k = _rope_half_torch(k, pos, BASE)

        # GQA: repeat k/v
        reps = H // KV
        k = k.repeat_interleave(reps, dim=1)
        v = v.repeat_interleave(reps, dim=1)

        # Scaled dot-product attention with causal mask
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # (1, H, L, head_dim)

        # Merge heads: (1, H, L, head_dim) → (1, L, d)
        o = o.transpose(1, 2).reshape(1, L, H * HEAD_DIM)
        o = F.linear(o, T[f"{p}.self_attn.o_proj.weight"])

        # Residual + pre-FFN norm
        h = h + o
        f = F.rms_norm(h, (d,), weight=T[f"{p}.post_attention_layernorm.weight"], eps=EPS)

        # SwiGLU FFN (SiLU gate)
        swi = F.linear(
            F.silu(F.linear(f, T[f"{p}.mlp.gate_proj.weight"]))
            * F.linear(f, T[f"{p}.mlp.up_proj.weight"]),
            T[f"{p}.mlp.down_proj.weight"],
        )
        h = h + swi

    h = F.rms_norm(h, (d,), weight=T["model.norm.weight"], eps=EPS)
    logits = (h @ T["lm_head.weight"].T).detach().numpy()
    return logits


# ─── qknorm fixture (unchanged from original) ────────────────────────────────

def _rms_norm_f64(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """Pure float64 RMSNorm over the last axis — the reference computation."""
    rms = np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def _extract_qk_fixtures(seq_len: int = 5) -> dict[str, np.ndarray]:
    """Run a forward pass through a tiny Qwen3ForCausalLM and capture pre-norm
    Q and K tensors, plus the norm weights."""
    cfg = HFQwen3Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        vocab_size=64,
        torch_dtype=torch.float64,
    )
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(cfg).to(torch.float64)

    rng_w = torch.Generator()
    rng_w.manual_seed(99)
    attn_init = model.model.layers[0].self_attn
    with torch.no_grad():
        attn_init.q_norm.weight.data = torch.randn(
            attn_init.q_norm.weight.shape, generator=rng_w, dtype=torch.float64
        )
        attn_init.k_norm.weight.data = torch.randn(
            attn_init.k_norm.weight.shape, generator=rng_w, dtype=torch.float64
        )

    model.eval()

    attn = model.model.layers[0].self_attn
    captured: dict[str, torch.Tensor] = {}

    orig_forward = attn.forward

    def patched_forward(
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        q_pre = attn.q_proj(hidden_states).view(hidden_shape)
        k_pre = attn.k_proj(hidden_states).view(hidden_shape)

        captured["q_pre"] = q_pre[0].permute(1, 0, 2).detach().clone()
        captured["k_pre"] = k_pre[0].permute(1, 0, 2).detach().clone()
        captured["q_weight"] = attn.q_norm.weight.detach().clone()
        captured["k_weight"] = attn.k_norm.weight.detach().clone()
        captured["eps"] = torch.tensor(attn.q_norm.variance_epsilon, dtype=torch.float64)

        return orig_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            **kwargs,
        )

    attn.forward = patched_forward

    torch.manual_seed(7)
    input_ids = torch.randint(0, 64, (1, seq_len))
    with torch.no_grad():
        model(input_ids)

    attn.forward = orig_forward

    q_pre = captured["q_pre"].numpy()
    k_pre = captured["k_pre"].numpy()
    q_weight = captured["q_weight"].numpy()
    k_weight = captured["k_weight"].numpy()
    eps = float(captured["eps"])

    q_post = _rms_norm_f64(q_pre, q_weight, eps)
    k_post = _rms_norm_f64(k_pre, k_weight, eps)

    with torch.no_grad():
        q_pre_t = torch.from_numpy(q_pre).to(torch.float64)
        k_pre_t = torch.from_numpy(k_pre).to(torch.float64)
        q_post_hf = attn.q_norm(q_pre_t.reshape(-1, q_pre.shape[-1])).reshape(q_pre.shape)
        k_post_hf = attn.k_norm(k_pre_t.reshape(-1, k_pre.shape[-1])).reshape(k_pre.shape)

    return {
        "q_pre": q_pre,
        "k_pre": k_pre,
        "q_post": q_post,
        "k_post": k_post,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "eps": np.array(eps),
        "q_post_hf": q_post_hf.numpy(),
        "k_post_hf": k_post_hf.numpy(),
    }


def main() -> None:
    FIX.mkdir(exist_ok=True)

    # ── fixture 1: qknorm.npz (qk_norm operator fixture) ─────────────────────
    data = _extract_qk_fixtures(seq_len=5)
    fixture_keys = {"q_pre", "k_pre", "q_post", "k_post", "q_weight", "k_weight", "eps"}
    np.savez(FIX / "qknorm.npz", **{k: data[k] for k in fixture_keys})
    print(f"  wrote qknorm.npz")

    # Sanity check 1: float64 self-consistency
    q_pre = data["q_pre"]
    q_post = data["q_post"]
    eps = float(data["eps"])
    rms_q = np.sqrt((q_pre**2).mean(axis=-1, keepdims=True) + eps)
    q_check = (q_pre / rms_q) * data["q_weight"]
    assert np.allclose(q_post, q_check, rtol=1e-12, atol=0), "Q self-consistency check failed"
    rms_k = np.sqrt((data["k_pre"] ** 2).mean(axis=-1, keepdims=True) + eps)
    k_check = (data["k_pre"] / rms_k) * data["k_weight"]
    assert np.allclose(data["k_post"], k_check, rtol=1e-12, atol=0), "K self-consistency check failed"
    print("  sanity check 1 passed (float64 self-consistency)")

    # Sanity check 2: HF-anchor for qk_norm
    q_post_hf = data["q_post_hf"]
    k_post_hf = data["k_post_hf"]
    np.testing.assert_allclose(q_post, q_post_hf, rtol=1e-4, atol=1e-5,
        err_msg="Q oracle diverges from genuine Qwen3 q_norm output")
    np.testing.assert_allclose(data["k_post"], k_post_hf, rtol=1e-4, atol=1e-5,
        err_msg="K oracle diverges from genuine Qwen3 k_norm output")
    print("  sanity check 2 passed (HF-anchor: qk_norm oracle matches Qwen3 q_norm/k_norm)")

    # ── fixture 2: tiny_qwen3.npz ─────────────────────────────────────────────
    rng = np.random.default_rng(42)
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
        W[f"{p}.self_attn.q_proj.weight"]         = rng.standard_normal((H * HEAD_DIM, d))
        W[f"{p}.self_attn.k_proj.weight"]         = rng.standard_normal((KV * HEAD_DIM, d))
        W[f"{p}.self_attn.v_proj.weight"]         = rng.standard_normal((KV * HEAD_DIM, d))
        W[f"{p}.self_attn.o_proj.weight"]         = rng.standard_normal((d, H * HEAD_DIM))
        # Randomise q_norm/k_norm so qk-norm is exercised (not trivially ones)
        W[f"{p}.self_attn.q_norm.weight"]         = rng.standard_normal((HEAD_DIM,))
        W[f"{p}.self_attn.k_norm.weight"]         = rng.standard_normal((HEAD_DIM,))
        W[f"{p}.mlp.gate_proj.weight"]            = rng.standard_normal((Fd, d))
        W[f"{p}.mlp.up_proj.weight"]              = rng.standard_normal((Fd, d))
        W[f"{p}.mlp.down_proj.weight"]            = rng.standard_normal((d, Fd))

    # Convert to float64 torch tensors
    T = {k: torch.from_numpy(v.astype(np.float64)) for k, v in W.items()}

    # Run composed float64 oracle
    oracle_logits = _composed_oracle(W, T, ids)
    print(f"  composed oracle logits shape: {oracle_logits.shape}")

    # ── HF-anchor: verify the composed oracle matches genuine Qwen3ForCausalLM ──
    # Qwen3 uses SiLU by default — no hidden_act override needed.
    hf_cfg = HFQwen3Config(
        hidden_size=d,
        num_hidden_layers=NL,
        num_attention_heads=H,
        num_key_value_heads=KV,
        head_dim=HEAD_DIM,
        intermediate_size=Fd,
        vocab_size=V,
        max_position_embeddings=128,
        rms_norm_eps=EPS,
        rope_theta=BASE,
        torch_dtype=torch.float32,
        tie_word_embeddings=False,
    )
    hf_model = Qwen3ForCausalLM(hf_cfg)
    hf_model.eval()

    # Load our random weights into the HF model (convert to float32)
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
            sd[f"{p}.self_attn.q_norm.weight"].copy_(
                T[f"{p}.self_attn.q_norm.weight"].float())
            sd[f"{p}.self_attn.k_norm.weight"].copy_(
                T[f"{p}.self_attn.k_norm.weight"].float())
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
    print(f"  HF-anchor: composed oracle vs Qwen3ForCausalLM max-abs-diff = {max_diff:.2e} ✓")

    # ── write fixture ─────────────────────────────────────────────────────────
    np.savez(
        FIX / "tiny_qwen3.npz",
        input_ids=ids,
        logits=oracle_logits,
        dim=np.array(d),
        n_layers=np.array(NL),
        n_heads=np.array(H),
        n_kv_heads=np.array(KV),
        head_dim=np.array(HEAD_DIM),
        vocab_size=np.array(V),
        max_seq_len=np.array(128),
        norm_eps=np.array(EPS),
        qk_norm_eps=np.array(QK_EPS),
        rope_base=np.array(BASE),
        **W,
    )
    print(f"  wrote tiny_qwen3.npz  logits{oracle_logits.shape}")


if __name__ == "__main__":
    main()
