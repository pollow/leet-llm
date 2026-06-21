"""308 — generate frozen golden fixtures for the Mixtral whole-model forward.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 308_mixtral_model/tests/gen_fixtures.py

One fixture is written:

``tiny_mixtral.npz`` — whole-model logits at a tiny seeded config (float64
composed oracle + frozen HF-named weights). The composed oracle uses genuine
torch primitives (F.linear, F.rms_norm, F.silu, F.scaled_dot_product_attention)
with rotate-half RoPE and the MoE top-k routing, NEVER the genuine MixtralForCausalLM,
so float64 accumulation is preserved → whole-model parity at rtol=1e-9.
As an authoring sanity check we also assert the composed oracle matches a genuine
``MixtralForCausalLM`` at rtol≈1e-3 (the float32 HF class is close, not exact).

Weight layout:
  model.layers.{i}.mlp.gate.weight          (num_experts, d)       router
  model.layers.{i}.mlp.experts.gate_up_proj (num_experts, 2*Fd, d) [gate;up] per expert
  model.layers.{i}.mlp.experts.down_proj    (num_experts, d, Fd)   down per expert
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from transformers import MixtralConfig as HFMixtralConfig
from transformers import MixtralForCausalLM

FIX = pathlib.Path(__file__).parent / "fixtures"

# ─── tiny model config ───────────────────────────────────────────────────────
V, d, NL, H, KV, Fd = 64, 16, 2, 4, 2, 32
NE = 4   # num_local_experts
NK = 2   # num_experts_per_tok
EPS = 1e-5
BASE = 10000.0
L = 5    # sequence length


# ─── rotate-half RoPE (HF convention, float64) ───────────────────────────────

def _rope_half_torch(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    """Rotate-half RoPE in float64 torch.

    x: (B, n_heads, L, head_dim)
    positions: (L,)
    Returns same shape.
    """
    head_dim = x.shape[-1]
    idx = torch.arange(0, head_dim, 2, dtype=torch.float64)
    inv_freq = 1.0 / (base ** (idx / head_dim))                    # (head_dim/2,)
    angle = torch.outer(positions.to(torch.float64), inv_freq)     # (L, head_dim/2)
    cos = torch.cat([angle.cos(), angle.cos()], dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat([angle.sin(), angle.sin()], dim=-1).unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :head_dim // 2], x[..., head_dim // 2:]
    rotate = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotate * sin


# ─── causal mask (float64, additive) ─────────────────────────────────────────

def _causal_mask_torch(seq_len: int) -> torch.Tensor:
    """Additive causal mask (0.0 attended, -inf masked), float64."""
    rows = torch.arange(seq_len)[:, None]
    cols = torch.arange(seq_len)[None, :]
    attended = rows >= cols
    return torch.where(attended, torch.tensor(0.0, dtype=torch.float64),
                       torch.tensor(-float("inf"), dtype=torch.float64))


# ─── MoE FFN (float64 torch) ─────────────────────────────────────────────────

def _moe_ffn_torch(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Sparse MoE FFN matching the genuine MixtralSparseMoeBlock convention.

    x: (T, d) where T = B*L (tokens)
    router_weight: (num_experts, d)   — mlp.gate.weight
    gate_up_proj: (num_experts, 2*Fd, d)
    down_proj: (num_experts, d, Fd)
    Returns: (T, d)
    """
    # Router: softmax over ALL experts first (float64 throughout to match our numpy forward)
    router_logits = F.linear(x, router_weight)                    # (T, num_experts)
    routing_weights = torch.softmax(router_logits, dim=-1)        # keep float64

    # Top-k selection
    weights, idx = torch.topk(routing_weights, top_k, dim=-1)    # (T, top_k)
    # Renormalise selected weights to sum to 1
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # Dispatch to experts
    out = torch.zeros_like(x)
    for k in range(top_k):
        expert_idx = idx[:, k]     # (T,) which expert for each token at slot k
        w_k = weights[:, k]        # (T,) weight for that expert
        for e in range(gate_up_proj.shape[0]):
            token_mask = (expert_idx == e)
            if not token_mask.any():
                continue
            x_e = x[token_mask]                                   # (T_e, d)
            gu = F.linear(x_e, gate_up_proj[e])                   # (T_e, 2*Fd)
            gate, up = gu.chunk(2, dim=-1)                        # each (T_e, Fd)
            h_e = F.silu(gate) * up                               # (T_e, Fd)
            h_e = F.linear(h_e, down_proj[e])                     # (T_e, d)
            w_e = w_k[token_mask].unsqueeze(-1)                   # (T_e, 1)
            out[token_mask] += h_e * w_e

    return out


# ─── composed float64 oracle ──────────────────────────────────────────────────

def _composed_oracle(W: dict, T: dict, ids: np.ndarray) -> np.ndarray:
    """Run the composed float64 Mixtral forward.

    W: numpy weight dict (HF names)
    T: torch tensor dict (same keys)
    ids: (1, L) int array
    Returns logits (1, L, V) numpy float64.
    """
    head_dim = d // H
    pos = torch.arange(L, dtype=torch.long)
    mask = _causal_mask_torch(L)    # (L, L), additive float64

    h = T["model.embed_tokens.weight"][torch.from_numpy(ids[0])].unsqueeze(0)  # (1, L, d)

    for i in range(NL):
        p = f"model.layers.{i}"

        # Pre-attention RMSNorm
        a = F.rms_norm(h, (d,), weight=T[f"{p}.input_layernorm.weight"], eps=EPS)

        # QKV projections
        q = F.linear(a, T[f"{p}.self_attn.q_proj.weight"])    # (1, L, d)
        k = F.linear(a, T[f"{p}.self_attn.k_proj.weight"])    # (1, L, KV*head_dim)
        v = F.linear(a, T[f"{p}.self_attn.v_proj.weight"])    # (1, L, KV*head_dim)

        # Split into heads: (1, n_heads, L, head_dim)
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

        # Scaled dot-product attention with causal mask
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # (1, H, L, head_dim)

        # Merge heads: (1, L, d)
        o = o.transpose(1, 2).reshape(1, L, d)
        o = F.linear(o, T[f"{p}.self_attn.o_proj.weight"])

        # Residual + pre-FFN norm
        h = h + o
        f_norm = F.rms_norm(h, (d,), weight=T[f"{p}.post_attention_layernorm.weight"], eps=EPS)

        # Sparse MoE FFN
        # Flatten to (T, d) for token-level routing
        f_flat = f_norm.reshape(-1, d)    # (L, d) for B=1
        moe_out = _moe_ffn_torch(
            f_flat,
            T[f"{p}.mlp.gate.weight"],
            T[f"{p}.mlp.experts.gate_up_proj"],
            T[f"{p}.mlp.experts.down_proj"],
            NK,
        )
        h = h + moe_out.reshape(1, L, d)

    h = F.rms_norm(h, (d,), weight=T["model.norm.weight"], eps=EPS)
    logits = (h @ T["lm_head.weight"].T).detach().numpy()
    return logits


def main() -> None:
    FIX.mkdir(exist_ok=True)

    # ── build seeded tiny weights ─────────────────────────────────────────────
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
        # MoE router
        W[f"{p}.mlp.gate.weight"]                 = rng.standard_normal((NE, d))
        # Expert weights: gate_up_proj (n_experts, 2*Fd, d), down_proj (n_experts, d, Fd)
        W[f"{p}.mlp.experts.gate_up_proj"]        = rng.standard_normal((NE, 2 * Fd, d))
        W[f"{p}.mlp.experts.down_proj"]           = rng.standard_normal((NE, d, Fd))

    # Convert to float64 torch tensors
    T = {k: torch.from_numpy(v.astype(np.float64)) for k, v in W.items()}

    # Run composed float64 oracle
    oracle_logits = _composed_oracle(W, T, ids)
    print(f"  composed oracle logits shape: {oracle_logits.shape}")

    # ── HF-anchor: verify the composed oracle matches genuine MixtralForCausalLM ──
    hf_cfg = HFMixtralConfig(
        hidden_size=d,
        num_hidden_layers=NL,
        num_attention_heads=H,
        num_key_value_heads=KV,
        intermediate_size=Fd,
        vocab_size=V,
        num_local_experts=NE,
        num_experts_per_tok=NK,
        max_position_embeddings=128,
        rms_norm_eps=EPS,
        rope_theta=BASE,
        hidden_act="silu",
        torch_dtype=torch.float32,
        tie_word_embeddings=False,
    )
    hf_model = MixtralForCausalLM(hf_cfg)
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
            sd[f"{p}.mlp.gate.weight"].copy_(
                T[f"{p}.mlp.gate.weight"].float())
            sd[f"{p}.mlp.experts.gate_up_proj"].copy_(
                T[f"{p}.mlp.experts.gate_up_proj"].float())
            sd[f"{p}.mlp.experts.down_proj"].copy_(
                T[f"{p}.mlp.experts.down_proj"].float())
    hf_model.load_state_dict(sd)

    with torch.no_grad():
        hf_logits = hf_model(torch.tensor(ids, dtype=torch.long)).logits.numpy()

    np.testing.assert_allclose(oracle_logits, hf_logits, rtol=1e-3, atol=1e-3)
    max_diff = np.max(np.abs(oracle_logits - hf_logits))
    print(f"  HF-anchor: composed oracle vs MixtralForCausalLM max-abs-diff = {max_diff:.2e} ✓")

    # ── write fixture ─────────────────────────────────────────────────────────
    np.savez(
        FIX / "tiny_mixtral.npz",
        input_ids=ids,
        logits=oracle_logits,
        dim=np.array(d),
        n_layers=np.array(NL),
        n_heads=np.array(H),
        n_kv_heads=np.array(KV),
        vocab_size=np.array(V),
        num_local_experts=np.array(NE),
        num_experts_per_tok=np.array(NK),
        max_seq_len=np.array(128),
        norm_eps=np.array(EPS),
        rope_base=np.array(BASE),
        **W,
    )
    print(f"  wrote tiny_mixtral.npz  logits{oracle_logits.shape}")


if __name__ == "__main__":
    main()
