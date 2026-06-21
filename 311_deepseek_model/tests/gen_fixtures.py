"""311 — generate frozen golden fixtures for the DeepSeek-V3 whole-model forward.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 311_deepseek_model/tests/gen_fixtures.py

One fixture is written:

``tiny_deepseek.npz`` — whole-model logits at a tiny seeded config (float64
composed numpy oracle + frozen HF-named weights).

The composed oracle uses numpy primitives (matmul, np.exp, np.concatenate) with the
same operations as the student forward, so float64 accumulation is bit-identical →
whole-model parity at rtol=1e-9.

As an authoring sanity check we also assert the numpy oracle matches a genuine
``DeepseekV3ForCausalLM`` (float32) at rtol=1e-3/atol=1e-3. DeepSeek's MoE kernel
routes through float32 gate weights (``router_logits = F.linear(hidden.float(), w.float())``)
so we allow a wider float32 vs float64 gap there. Max observed diff is documented below.

Weight layout in fixture:
  model.layers.{i} for dense layers (index < first_k_dense_replace):
    self_attn.kv_a_proj_with_mqa.weight    (kv_lora_rank+qk_rope_head_dim, d)
    self_attn.kv_a_layernorm.weight         (kv_lora_rank,)
    self_attn.kv_b_proj.weight              (n_heads*(qk_nope_head_dim+v_head_dim), kv_lora_rank)
    self_attn.q_proj.weight                 (n_heads*qk_head_dim, d)  [no q_lora in tiny]
    self_attn.o_proj.weight                 (d, n_heads*v_head_dim)
    mlp.gate_proj.weight                    (intermediate_size, d)
    mlp.up_proj.weight                      (intermediate_size, d)
    mlp.down_proj.weight                    (d, intermediate_size)
  for MoE layers (index >= first_k_dense_replace):
    (same MLA attn keys)
    mlp.gate.weight                         (n_routed_experts, d)
    mlp.gate.e_score_correction_bias        (n_routed_experts,)
    mlp.experts.gate_up_proj                (n_routed_experts, 2*moe_intermediate_size, d)
    mlp.experts.down_proj                   (n_routed_experts, d, moe_intermediate_size)
    mlp.shared_experts.gate_proj.weight     (n_shared_experts*moe_intermediate_size, d)
    mlp.shared_experts.up_proj.weight       (n_shared_experts*moe_intermediate_size, d)
    mlp.shared_experts.down_proj.weight     (d, n_shared_experts*moe_intermediate_size)
"""

from __future__ import annotations

import pathlib

import numpy as np

FIX = pathlib.Path(__file__).parent / "fixtures"

# ─── tiny model config ───────────────────────────────────────────────────────
# Tiny but realistic enough to exercise both dense and MoE layers.
# first_k_dense_replace=1 → layer 0 dense MLP, layer 1 MoE.
V = 64      # vocab_size
d = 32      # hidden_size
NL = 2      # num_hidden_layers
H = 4       # num_attention_heads
KV_LORA = 8   # kv_lora_rank
QK_NOPE = 8   # qk_nope_head_dim
QK_ROPE = 4   # qk_rope_head_dim
QK_HEAD = QK_NOPE + QK_ROPE   # qk_head_dim = 12
V_HEAD = 8    # v_head_dim
NE = 4      # n_routed_experts
NK = 2      # num_experts_per_tok
NS = 1      # n_shared_experts
NG = 2      # n_group
TKG = 1     # topk_group
FIRST_K = 1  # first_k_dense_replace
MOE_FFND = 8  # moe_intermediate_size
DENSE_FFND = 16  # intermediate_size (dense FFN)
EPS = 1e-5
BASE = 10000.0
L = 5       # sequence length
SCALING = QK_HEAD ** (-0.5)  # default rope scaling (no mscale for default rope type)


# ─── numpy float64 primitives ────────────────────────────────────────────────

def _rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return w * (x / rms)


def _rope_half(x: np.ndarray, positions: np.ndarray, base: float) -> np.ndarray:
    """Rotate-half RoPE, numpy float64. x: (..., head_dim), positions: (L,)"""
    head_dim = x.shape[-1]
    idx = np.arange(0, head_dim, 2, dtype=np.float64)
    inv_freq = 1.0 / (base ** (idx / head_dim))
    angle = np.outer(positions.astype(np.float64), inv_freq)   # (L, head_dim/2)
    cos = np.concatenate([np.cos(angle), np.cos(angle)], axis=-1)   # (L, head_dim)
    sin = np.concatenate([np.sin(angle), np.sin(angle)], axis=-1)   # (L, head_dim)
    cos = cos[np.newaxis, np.newaxis, :, :]   # (1, 1, L, head_dim)
    sin = sin[np.newaxis, np.newaxis, :, :]
    x1, x2 = x[..., :head_dim // 2], x[..., head_dim // 2:]
    return x * cos + np.concatenate([-x2, x1], axis=-1) * sin


def _softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - np.where(np.isfinite(m), m, 0.0))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-45)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


# ─── MLA attention (numpy float64) ───────────────────────────────────────────

def _mla_np(
    x: np.ndarray,       # (B, L, d)
    kv_a_proj: np.ndarray,       # (kv_lora_rank+qk_rope_head_dim, d)
    kv_a_layernorm_w: np.ndarray, # (kv_lora_rank,)
    kv_b_proj: np.ndarray,        # (n_heads*(qk_nope_head_dim+v_head_dim), kv_lora_rank)
    q_proj: np.ndarray,           # (n_heads*qk_head_dim, d)
    o_proj: np.ndarray,           # (d, n_heads*v_head_dim)
    pos: np.ndarray,              # (L,)
) -> np.ndarray:
    """Composed float64 MLA forward (no q_lora_rank path)."""
    B, Lseq, _ = x.shape

    # Q: direct proj
    q = x @ q_proj.T   # (B, L, n_heads*qk_head_dim)
    q = q.reshape(B, Lseq, H, QK_HEAD).transpose(0, 2, 1, 3)   # (B, H, L, QK_HEAD)
    q_nope = q[..., :QK_NOPE]
    q_rope = q[..., QK_NOPE:]

    # KV: down-proj → split latent + shared rope
    compressed = x @ kv_a_proj.T   # (B, L, kv_lora_rank+qk_rope_head_dim)
    c_kv = compressed[..., :KV_LORA]
    k_rope = compressed[..., KV_LORA:]

    # c_kv → layernorm → kv_b_proj → per-head [k_nope, v]
    c_kv_norm = _rms_norm(c_kv, kv_a_layernorm_w, EPS)
    kv = c_kv_norm @ kv_b_proj.T   # (B, L, n_heads*(qk_nope+v_head))
    kv = kv.reshape(B, Lseq, H, QK_NOPE + V_HEAD).transpose(0, 2, 1, 3)
    k_nope = kv[..., :QK_NOPE]
    v = kv[..., QK_NOPE:]

    # k_rope: MQA, (B, L, qk_rope) → (B, 1, L, qk_rope)
    k_rope = k_rope[:, np.newaxis, :, :]

    # Decoupled RoPE on q_rope and k_rope only
    q_rope = _rope_half(q_rope, pos, BASE)   # (B, H, L, QK_ROPE)
    k_rope = _rope_half(k_rope, pos, BASE)   # (B, 1, L, QK_ROPE)

    # Broadcast k_rope to all heads
    k_rope = np.broadcast_to(k_rope, (B, H, Lseq, QK_ROPE)).copy()

    # Full q and k
    q_full = np.concatenate([q_nope, q_rope], axis=-1)   # (B, H, L, QK_HEAD)
    k_full = np.concatenate([k_nope, k_rope], axis=-1)   # (B, H, L, QK_HEAD)

    # Causal additive mask
    rows = np.arange(Lseq)[:, None]
    cols = np.arange(Lseq)[None, :]
    mask = np.where(rows >= cols, 0.0, -np.inf).astype(np.float64)   # (L, L)

    # SDPA
    scores = (q_full @ k_full.transpose(0, 1, 3, 2)) * SCALING + mask   # (B, H, L, L)
    probs = _softmax(scores)
    attn_out = probs @ v   # (B, H, L, V_HEAD)

    # Merge heads
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, Lseq, H * V_HEAD)
    return attn_out @ o_proj.T


# ─── Dense SwiGLU FFN (numpy float64) ────────────────────────────────────────

def _dense_ffn_np(
    x: np.ndarray,
    gate_proj: np.ndarray,
    up_proj: np.ndarray,
    down_proj: np.ndarray,
) -> np.ndarray:
    gate = _silu(x @ gate_proj.T)
    up = x @ up_proj.T
    return (gate * up) @ down_proj.T


# ─── DeepSeek MoE FFN (numpy float64) ────────────────────────────────────────

def _deepseek_moe_np(
    x: np.ndarray,           # (B, L, d)
    router_weight: np.ndarray,       # (n_routed_experts, d)
    e_bias: np.ndarray,              # (n_routed_experts,)
    gate_up_proj: np.ndarray,        # (n_routed_experts, 2*moe_intermediate_size, d)
    down_proj: np.ndarray,           # (n_routed_experts, d, moe_intermediate_size)
    shared_gate: np.ndarray,         # (ns*moe_ffnd, d)
    shared_up: np.ndarray,           # (ns*moe_ffnd, d)
    shared_down: np.ndarray,         # (d, ns*moe_ffnd)
) -> np.ndarray:
    """DeepSeek MoE: sigmoid gating + group top-k + shared experts, numpy float64."""
    orig_shape = x.shape
    T = orig_shape[0] * orig_shape[1]
    x_flat = x.reshape(T, d)

    # Router logits + sigmoid
    router_logits = x_flat @ router_weight.T   # (T, n_routed_experts)
    scores = 1.0 / (1.0 + np.exp(-router_logits))   # sigmoid

    # Biased scores for selection (NOT used for final weights)
    scores_biased = scores + e_bias

    # Group top-k: reshape to (T, n_group, experts_per_group), take top-2 sum per group
    experts_per_group = NE // NG
    group_view = scores_biased.reshape(T, NG, experts_per_group)
    idx_g = np.argsort(group_view, axis=-1)[..., ::-1][..., :2]
    g_top2 = np.take_along_axis(group_view, idx_g, axis=-1).sum(axis=-1)   # (T, n_group)

    # Select topk_group groups
    group_idx = np.argsort(g_top2, axis=-1)[:, ::-1][:, :TKG]
    group_mask = np.zeros((T, NG), dtype=np.float64)
    np.put_along_axis(group_mask, group_idx, 1.0, axis=1)
    score_mask = (
        np.broadcast_to(group_mask[:, :, np.newaxis], (T, NG, experts_per_group))
        .reshape(T, NE)
    )

    # Mask out non-selected groups, select top-k experts
    masked_scores = np.where(score_mask.astype(bool), scores_biased, -np.inf)
    topk_indices = np.argsort(masked_scores, axis=-1)[:, ::-1][:, :NK]

    # Gather actual (unbiased) sigmoid scores for weighting
    topk_weights = scores[np.arange(T)[:, None], topk_indices]
    denom = topk_weights.sum(axis=-1, keepdims=True) + 1e-20
    topk_weights = topk_weights / denom   # norm_topk_prob=True, routed_scaling_factor=1.0

    # Dispatch to routed experts
    out_routed = np.zeros_like(x_flat)
    for k in range(NK):
        expert_idx = topk_indices[:, k]
        w_k = topk_weights[:, k]
        for e in range(NE):
            tok_mask = (expert_idx == e)
            if not tok_mask.any():
                continue
            x_e = x_flat[tok_mask]                            # (T_e, d)
            gu = x_e @ gate_up_proj[e].T                     # (T_e, 2*moe_ffnd)
            gate, up = gu[:, :MOE_FFND], gu[:, MOE_FFND:]
            h_e = _silu(gate) * up
            h_e = h_e @ down_proj[e].T                       # (T_e, d)
            out_routed[tok_mask] += h_e * w_k[tok_mask, np.newaxis]

    # Shared experts (always-on dense SwiGLU)
    out_shared = _dense_ffn_np(x, shared_gate, shared_up, shared_down)

    # Combine
    out = out_routed.reshape(orig_shape) + out_shared
    return out


# ─── composed float64 numpy oracle ───────────────────────────────────────────

def _composed_oracle_np(W: dict, ids: np.ndarray) -> np.ndarray:
    """Run the composed float64 numpy DeepSeek-V3 forward.

    W: numpy weight dict (HF names)
    ids: (1, L) int array
    Returns logits (1, L, V) numpy float64.
    """
    pos = np.arange(L, dtype=np.int64)
    h = W["model.embed_tokens.weight"].astype(np.float64)[ids[0]][np.newaxis]   # (1, L, d)

    for i in range(NL):
        p = f"model.layers.{i}"

        # Pre-attention RMSNorm
        h_norm = _rms_norm(h, W[f"{p}.input_layernorm.weight"].astype(np.float64), EPS)

        # MLA attention
        o = _mla_np(
            h_norm,
            kv_a_proj=W[f"{p}.self_attn.kv_a_proj_with_mqa.weight"].astype(np.float64),
            kv_a_layernorm_w=W[f"{p}.self_attn.kv_a_layernorm.weight"].astype(np.float64),
            kv_b_proj=W[f"{p}.self_attn.kv_b_proj.weight"].astype(np.float64),
            q_proj=W[f"{p}.self_attn.q_proj.weight"].astype(np.float64),
            o_proj=W[f"{p}.self_attn.o_proj.weight"].astype(np.float64),
            pos=pos,
        )
        h = h + o

        # Post-attention RMSNorm
        f_norm = _rms_norm(h, W[f"{p}.post_attention_layernorm.weight"].astype(np.float64), EPS)

        if i < FIRST_K:
            # Dense SwiGLU MLP
            ffn_out = _dense_ffn_np(
                f_norm,
                W[f"{p}.mlp.gate_proj.weight"].astype(np.float64),
                W[f"{p}.mlp.up_proj.weight"].astype(np.float64),
                W[f"{p}.mlp.down_proj.weight"].astype(np.float64),
            )
        else:
            # DeepSeek MoE
            ffn_out = _deepseek_moe_np(
                f_norm,
                router_weight=W[f"{p}.mlp.gate.weight"].astype(np.float64),
                e_bias=W[f"{p}.mlp.gate.e_score_correction_bias"].astype(np.float64),
                gate_up_proj=W[f"{p}.mlp.experts.gate_up_proj"].astype(np.float64),
                down_proj=W[f"{p}.mlp.experts.down_proj"].astype(np.float64),
                shared_gate=W[f"{p}.mlp.shared_experts.gate_proj.weight"].astype(np.float64),
                shared_up=W[f"{p}.mlp.shared_experts.up_proj.weight"].astype(np.float64),
                shared_down=W[f"{p}.mlp.shared_experts.down_proj.weight"].astype(np.float64),
            )
        h = h + ffn_out

    h = _rms_norm(h, W["model.norm.weight"].astype(np.float64), EPS)
    logits = h @ W["lm_head.weight"].astype(np.float64).T
    return logits


def main() -> None:
    FIX.mkdir(exist_ok=True)

    # ── build seeded tiny weights ─────────────────────────────────────────────
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
        # MLA attention weights
        W[f"{p}.self_attn.kv_a_proj_with_mqa.weight"] = rng.standard_normal((KV_LORA + QK_ROPE, d))
        W[f"{p}.self_attn.kv_a_layernorm.weight"]      = rng.standard_normal((KV_LORA,))
        W[f"{p}.self_attn.kv_b_proj.weight"]           = rng.standard_normal((H * (QK_NOPE + V_HEAD), KV_LORA))
        W[f"{p}.self_attn.q_proj.weight"]              = rng.standard_normal((H * QK_HEAD, d))
        W[f"{p}.self_attn.o_proj.weight"]              = rng.standard_normal((d, H * V_HEAD))

        if i < FIRST_K:
            W[f"{p}.mlp.gate_proj.weight"] = rng.standard_normal((DENSE_FFND, d))
            W[f"{p}.mlp.up_proj.weight"]   = rng.standard_normal((DENSE_FFND, d))
            W[f"{p}.mlp.down_proj.weight"] = rng.standard_normal((d, DENSE_FFND))
        else:
            W[f"{p}.mlp.gate.weight"]                       = rng.standard_normal((NE, d))
            W[f"{p}.mlp.gate.e_score_correction_bias"]      = rng.standard_normal((NE,))
            W[f"{p}.mlp.experts.gate_up_proj"]              = rng.standard_normal((NE, 2 * MOE_FFND, d))
            W[f"{p}.mlp.experts.down_proj"]                 = rng.standard_normal((NE, d, MOE_FFND))
            shared_ffnd = NS * MOE_FFND
            W[f"{p}.mlp.shared_experts.gate_proj.weight"]   = rng.standard_normal((shared_ffnd, d))
            W[f"{p}.mlp.shared_experts.up_proj.weight"]     = rng.standard_normal((shared_ffnd, d))
            W[f"{p}.mlp.shared_experts.down_proj.weight"]   = rng.standard_normal((d, shared_ffnd))

    # Run composed float64 numpy oracle
    oracle_logits = _composed_oracle_np(W, ids)
    print(f"  composed numpy oracle logits shape: {oracle_logits.shape}")

    # ── HF-anchor: verify the numpy oracle matches genuine DeepseekV3ForCausalLM ──
    # (float32 HF model ≈ float64 oracle at rtol=1e-3/atol=1e-3)
    # Use rope_interleave=False so HF uses the same half-rotate convention as our oracle.
    try:
        import torch
        from transformers import DeepseekV3Config as HFDeepseekConfig
        from transformers import DeepseekV3ForCausalLM

        hf_cfg = HFDeepseekConfig(
            hidden_size=d,
            num_hidden_layers=NL,
            num_attention_heads=H,
            num_key_value_heads=H,
            kv_lora_rank=KV_LORA,
            q_lora_rank=None,
            qk_rope_head_dim=QK_ROPE,
            qk_nope_head_dim=QK_NOPE,
            v_head_dim=V_HEAD,
            n_routed_experts=NE,
            num_experts_per_tok=NK,
            n_shared_experts=NS,
            n_group=NG,
            topk_group=TKG,
            first_k_dense_replace=FIRST_K,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
            moe_intermediate_size=MOE_FFND,
            intermediate_size=DENSE_FFND,
            vocab_size=V,
            rms_norm_eps=EPS,
            max_position_embeddings=128,
            hidden_act="silu",
            tie_word_embeddings=False,
            rope_interleave=False,  # use rotate-half convention (same as our oracle)
            attention_bias=False,
        )
        hf_model = DeepseekV3ForCausalLM(hf_cfg)
        hf_model.eval()

        sd = hf_model.state_dict()
        with torch.no_grad():
            for key in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
                sd[key].copy_(torch.from_numpy(W[key].astype(np.float32)))
            for i in range(NL):
                p = f"model.layers.{i}"
                for nm in (
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                    "self_attn.kv_a_proj_with_mqa.weight",
                    "self_attn.kv_a_layernorm.weight",
                    "self_attn.kv_b_proj.weight",
                    "self_attn.q_proj.weight",
                    "self_attn.o_proj.weight",
                ):
                    sd[f"{p}.{nm}"].copy_(torch.from_numpy(W[f"{p}.{nm}"].astype(np.float32)))
                if i < FIRST_K:
                    for nm in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
                        sd[f"{p}.{nm}"].copy_(torch.from_numpy(W[f"{p}.{nm}"].astype(np.float32)))
                else:
                    for nm in (
                        "mlp.gate.weight",
                        "mlp.gate.e_score_correction_bias",
                        "mlp.experts.gate_up_proj",
                        "mlp.experts.down_proj",
                        "mlp.shared_experts.gate_proj.weight",
                        "mlp.shared_experts.up_proj.weight",
                        "mlp.shared_experts.down_proj.weight",
                    ):
                        sd[f"{p}.{nm}"].copy_(torch.from_numpy(W[f"{p}.{nm}"].astype(np.float32)))
        hf_model.load_state_dict(sd)

        with torch.no_grad():
            hf_logits = hf_model(torch.tensor(ids, dtype=torch.long)).logits.float().numpy()

        # HF uses float32; our oracle is float64 — tolerance reflects float32 precision gap.
        # Note: DeepSeek's TopkRouter casts to float32 internally, contributing to the gap.
        max_diff = np.max(np.abs(oracle_logits - hf_logits))
        # Hard-fail (not swallowed): a divergence here means the committed oracle is
        # NOT faithful to genuine DeepSeek and must not be shipped.
        np.testing.assert_allclose(oracle_logits, hf_logits, rtol=1e-3, atol=1e-3)
        print(f"  HF-anchor: numpy oracle vs DeepseekV3ForCausalLM (float32) max-abs-diff = {max_diff:.2e} ✓")
    except ImportError as e:
        # Only an unavailable transformers/torch may skip the anchor — never a mismatch.
        print(f"  HF-anchor skipped (transformers/torch unavailable): {e}")

    # ── write fixture ─────────────────────────────────────────────────────────
    np.savez(
        FIX / "tiny_deepseek.npz",
        input_ids=ids,
        logits=oracle_logits,
        dim=np.array(d),
        n_layers=np.array(NL),
        n_heads=np.array(H),
        vocab_size=np.array(V),
        kv_lora_rank=np.array(KV_LORA),
        qk_nope_head_dim=np.array(QK_NOPE),
        qk_rope_head_dim=np.array(QK_ROPE),
        v_head_dim=np.array(V_HEAD),
        n_routed_experts=np.array(NE),
        num_experts_per_tok=np.array(NK),
        n_shared_experts=np.array(NS),
        n_group=np.array(NG),
        topk_group=np.array(TKG),
        first_k_dense_replace=np.array(FIRST_K),
        moe_intermediate_size=np.array(MOE_FFND),
        intermediate_size=np.array(DENSE_FFND),
        norm_topk_prob=np.array(True),
        routed_scaling_factor=np.array(1.0),
        max_seq_len=np.array(128),
        norm_eps=np.array(EPS),
        rope_base=np.array(BASE),
        **W,
    )
    print(f"  wrote tiny_deepseek.npz  logits{oracle_logits.shape}")


if __name__ == "__main__":
    main()
