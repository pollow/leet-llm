"""309 — generate frozen golden fixtures for the GPT-OSS whole-model forward.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 309_gptoss_model/tests/gen_fixtures.py

One fixture is written:

``tests/fixtures/tiny_gptoss.npz`` — whole-model logits at a tiny seeded config
(float64 composed numpy oracle + frozen HF-named weights).

The composed oracle uses numpy primitives (matmul, np.exp) with the same operations
as the student forward, so float64 accumulation is bit-identical → whole-model parity
at rtol=1e-9.

As an authoring sanity check we also assert the numpy oracle matches a genuine
``GptOssForCausalLM`` (float32, eager attention) at rtol=1e-3/atol=1e-3.  We force
``attn_implementation='eager'`` (the explicit-softmax-with-sink path our oracle
mirrors) and ``rope_type='default'`` (the real checkpoint uses YaRN long-context
scaling, deferred to 307 / L4 — see README).  Max observed diff is documented below.

Architecture exercised in the tiny config (2 layers):
  layer 0: sliding_attention (even-indexed) — band mask active at L=6, window=3
  layer 1: full_attention   (odd-indexed)
"""

from __future__ import annotations

import pathlib

import numpy as np

FIX = pathlib.Path(__file__).parent / "fixtures"

# ─── tiny model config ───────────────────────────────────────────────────────
V = 64           # vocab_size
d = 32           # hidden_size
NL = 2           # num_hidden_layers (one sliding + one full)
H = 4            # num_attention_heads
KVH = 2          # num_key_value_heads
HEAD_DIM = 8     # head_dim (GPT-OSS sets this explicitly)
INT_SIZE = 32    # intermediate_size (per-expert FFN)
NE = 4           # num_local_experts
NK = 2           # num_experts_per_tok
EPS = 1e-5       # rms_norm_eps
ROPE_BASE = 150000.0
SWA_WINDOW = 3   # sliding_window
ALPHA = 1.702
LIMIT = 7.0
L = 6            # sequence length
N_GROUPS = H // KVH


# ─── numpy float64 primitives ────────────────────────────────────────────────

def _rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    x64 = x.astype(np.float64)
    rms = np.sqrt((x64 ** 2).mean(axis=-1, keepdims=True) + eps)
    return (x64 / rms) * w.astype(np.float64)


def _rope_half(x: np.ndarray, positions: np.ndarray, base: float, head_dim: int) -> np.ndarray:
    """Rotate-half RoPE. x: (B, H, L, head_dim), positions: (L,)."""
    x64 = x.astype(np.float64)
    idx = np.arange(0, head_dim, 2, dtype=np.float64)
    inv_freq = 1.0 / (base ** (idx / head_dim))
    angle = np.outer(positions.astype(np.float64), inv_freq)        # (L, head_dim/2)
    cos = np.concatenate([np.cos(angle), np.cos(angle)], axis=-1)   # (L, head_dim)
    sin = np.concatenate([np.sin(angle), np.sin(angle)], axis=-1)
    cos = cos[np.newaxis, np.newaxis, :, :]
    sin = sin[np.newaxis, np.newaxis, :, :]
    x1, x2 = x64[..., :head_dim // 2], x64[..., head_dim // 2:]
    return x64 * cos + np.concatenate([-x2, x1], axis=-1) * sin


def _softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - np.where(np.isfinite(m), m, 0.0))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-300)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _moe(x, rw, rb, gup, gub, dp, db):
    """GPT-OSS MoE on x (T, d), float64."""
    x = x.astype(np.float64)
    T = x.shape[0]
    logits = x @ rw.T + rb            # (T, E)
    # top-k (descending), then softmax over the selected logits
    idx = np.argsort(logits, axis=-1)[:, ::-1][:, :NK]          # (T, NK)
    top_val = np.take_along_axis(logits, idx, axis=-1)          # (T, NK)
    scores = _softmax(top_val)                                  # (T, NK)
    out = np.zeros_like(x)
    for t in range(T):
        for j in range(NK):
            e = idx[t, j]
            gate_up = x[t] @ gup[e] + gub[e]                   # (2F,)
            gate = gate_up[::2]
            up = gate_up[1::2]
            gate = np.minimum(gate, LIMIT)
            up = np.clip(up, -LIMIT, LIMIT)
            glu = gate * _sigmoid(gate * ALPHA)
            gated = (up + 1.0) * glu                            # (F,)
            y = gated @ dp[e] + db[e]                           # (d,)
            out[t] += scores[t, j] * y
    return out


# ─── composed float64 numpy oracle ───────────────────────────────────────────

def _composed_oracle_np(W: dict, ids: np.ndarray) -> np.ndarray:
    pos = np.arange(L, dtype=np.int64)
    h = W["model.embed_tokens.weight"].astype(np.float64)[ids[0]][np.newaxis]  # (1, L, d)
    scale = float(HEAD_DIM) ** -0.5

    rows = np.arange(L)[:, None]
    cols = np.arange(L)[None, :]
    full_mask = np.where(rows >= cols, 0.0, -np.inf)
    swa_mask = np.where((rows >= cols) & (rows - cols < SWA_WINDOW), 0.0, -np.inf)

    for i in range(NL):
        p = f"model.layers.{i}"
        B, Lseq, _ = h.shape

        a = _rms_norm(h, W[f"{p}.input_layernorm.weight"], EPS)

        q = a @ W[f"{p}.self_attn.q_proj.weight"].astype(np.float64).T + W[f"{p}.self_attn.q_proj.bias"].astype(np.float64)
        k = a @ W[f"{p}.self_attn.k_proj.weight"].astype(np.float64).T + W[f"{p}.self_attn.k_proj.bias"].astype(np.float64)
        v = a @ W[f"{p}.self_attn.v_proj.weight"].astype(np.float64).T + W[f"{p}.self_attn.v_proj.bias"].astype(np.float64)

        q = q.reshape(B, Lseq, H, HEAD_DIM).transpose(0, 2, 1, 3)
        k = k.reshape(B, Lseq, KVH, HEAD_DIM).transpose(0, 2, 1, 3)
        v = v.reshape(B, Lseq, KVH, HEAD_DIM).transpose(0, 2, 1, 3)

        q = _rope_half(q, pos, ROPE_BASE, HEAD_DIM)
        k = _rope_half(k, pos, ROPE_BASE, HEAD_DIM)

        k = np.repeat(k, N_GROUPS, axis=1)
        v = np.repeat(v, N_GROUPS, axis=1)

        scores = (q @ k.transpose(0, 1, 3, 2)) * scale            # (B, H, L, L)
        mask = swa_mask if i % 2 == 0 else full_mask
        scores = scores + mask

        # attention sinks: append one per-head sink logit column, softmax, drop it
        sinks = W[f"{p}.self_attn.sinks"].astype(np.float64).reshape(1, H, 1, 1)
        sinks = np.broadcast_to(sinks, (B, H, Lseq, 1))
        combined = np.concatenate([scores, sinks], axis=-1)       # (B, H, L, L+1)
        probs = _softmax(combined)[..., :-1]                      # drop sink

        attn = probs @ v                                          # (B, H, L, HD)
        attn = attn.transpose(0, 2, 1, 3).reshape(B, Lseq, H * HEAD_DIM)
        attn = attn @ W[f"{p}.self_attn.o_proj.weight"].astype(np.float64).T + W[f"{p}.self_attn.o_proj.bias"].astype(np.float64)

        h = h + attn

        f_in = _rms_norm(h, W[f"{p}.post_attention_layernorm.weight"], EPS)
        moe = _moe(
            f_in.reshape(-1, d),
            W[f"{p}.mlp.router.weight"].astype(np.float64),
            W[f"{p}.mlp.router.bias"].astype(np.float64),
            W[f"{p}.mlp.experts.gate_up_proj"].astype(np.float64),
            W[f"{p}.mlp.experts.gate_up_proj_bias"].astype(np.float64),
            W[f"{p}.mlp.experts.down_proj"].astype(np.float64),
            W[f"{p}.mlp.experts.down_proj_bias"].astype(np.float64),
        ).reshape(B, Lseq, d)
        h = h + moe

    h = _rms_norm(h, W["model.norm.weight"], EPS)
    logits = h @ W["lm_head.weight"].astype(np.float64).T
    return logits


def main() -> None:
    FIX.mkdir(exist_ok=True)

    rng = np.random.default_rng(42)
    ids = rng.integers(0, V, size=(1, L))

    W: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": rng.standard_normal((V, d)).astype(np.float32),
        "model.norm.weight":         rng.standard_normal((d,)).astype(np.float32),
        "lm_head.weight":            rng.standard_normal((V, d)).astype(np.float32),
    }
    for i in range(NL):
        p = f"model.layers.{i}"
        W[f"{p}.input_layernorm.weight"]          = rng.standard_normal((d,)).astype(np.float32)
        W[f"{p}.post_attention_layernorm.weight"] = rng.standard_normal((d,)).astype(np.float32)
        W[f"{p}.self_attn.q_proj.weight"]         = rng.standard_normal((H * HEAD_DIM, d)).astype(np.float32)
        W[f"{p}.self_attn.q_proj.bias"]           = rng.standard_normal((H * HEAD_DIM,)).astype(np.float32)
        W[f"{p}.self_attn.k_proj.weight"]         = rng.standard_normal((KVH * HEAD_DIM, d)).astype(np.float32)
        W[f"{p}.self_attn.k_proj.bias"]           = rng.standard_normal((KVH * HEAD_DIM,)).astype(np.float32)
        W[f"{p}.self_attn.v_proj.weight"]         = rng.standard_normal((KVH * HEAD_DIM, d)).astype(np.float32)
        W[f"{p}.self_attn.v_proj.bias"]           = rng.standard_normal((KVH * HEAD_DIM,)).astype(np.float32)
        W[f"{p}.self_attn.o_proj.weight"]         = rng.standard_normal((d, H * HEAD_DIM)).astype(np.float32)
        W[f"{p}.self_attn.o_proj.bias"]           = rng.standard_normal((d,)).astype(np.float32)
        W[f"{p}.self_attn.sinks"]                 = rng.standard_normal((H,)).astype(np.float32)
        W[f"{p}.mlp.router.weight"]               = rng.standard_normal((NE, d)).astype(np.float32)
        W[f"{p}.mlp.router.bias"]                 = rng.standard_normal((NE,)).astype(np.float32)
        W[f"{p}.mlp.experts.gate_up_proj"]        = rng.standard_normal((NE, d, 2 * INT_SIZE)).astype(np.float32)
        W[f"{p}.mlp.experts.gate_up_proj_bias"]   = rng.standard_normal((NE, 2 * INT_SIZE)).astype(np.float32)
        W[f"{p}.mlp.experts.down_proj"]           = rng.standard_normal((NE, INT_SIZE, d)).astype(np.float32)
        W[f"{p}.mlp.experts.down_proj_bias"]      = rng.standard_normal((NE, d)).astype(np.float32)

    oracle_logits = _composed_oracle_np(W, ids)
    print(f"  composed numpy oracle logits shape: {oracle_logits.shape}")

    # ── HF-anchor: verify oracle matches genuine GptOssForCausalLM (float32, eager) ──
    try:
        import torch
        from transformers import GptOssConfig as HFGptOssConfig
        from transformers import GptOssForCausalLM

        hf_cfg = HFGptOssConfig(
            hidden_size=d,
            num_hidden_layers=NL,
            num_attention_heads=H,
            num_key_value_heads=KVH,
            head_dim=HEAD_DIM,
            intermediate_size=INT_SIZE,
            num_local_experts=NE,
            num_experts_per_tok=NK,
            vocab_size=V,
            rms_norm_eps=EPS,
            max_position_embeddings=128,
            sliding_window=SWA_WINDOW,
            tie_word_embeddings=False,
            rope_parameters={"rope_type": "default", "rope_theta": ROPE_BASE},
            attn_implementation="eager",
        )
        hf_model = GptOssForCausalLM(hf_cfg)
        hf_model.eval()

        sd = hf_model.state_dict()
        with torch.no_grad():
            for name, arr in W.items():
                sd[name].copy_(torch.from_numpy(arr))
        hf_model.load_state_dict(sd)

        with torch.no_grad():
            hf_logits = hf_model(torch.tensor(ids, dtype=torch.long)).logits.float().numpy()

        max_diff = float(np.max(np.abs(oracle_logits - hf_logits)))
        np.testing.assert_allclose(oracle_logits, hf_logits, rtol=1e-3, atol=1e-3)
        print(
            f"  HF-anchor (eager): numpy oracle vs GptOssForCausalLM (float32) "
            f"max-abs-diff = {max_diff:.2e} ✓"
        )
    except ImportError as e:
        print(f"  HF-anchor skipped (transformers/torch unavailable): {e}")

    np.savez(
        FIX / "tiny_gptoss.npz",
        input_ids=ids,
        logits=oracle_logits,
        dim=np.array(d),
        n_layers=np.array(NL),
        n_heads=np.array(H),
        n_kv_heads=np.array(KVH),
        head_dim=np.array(HEAD_DIM),
        vocab_size=np.array(V),
        intermediate_size=np.array(INT_SIZE),
        num_local_experts=np.array(NE),
        num_experts_per_tok=np.array(NK),
        sliding_window=np.array(SWA_WINDOW),
        norm_eps=np.array(EPS),
        rope_base=np.array(ROPE_BASE),
        max_seq_len=np.array(128),
        **W,
    )
    print(f"  wrote tiny_gptoss.npz  logits{oracle_logits.shape}")


if __name__ == "__main__":
    main()
