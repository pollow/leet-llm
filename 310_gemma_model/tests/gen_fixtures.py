"""310 — generate frozen golden fixtures for the Gemma-2 whole-model forward.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 310_gemma_model/tests/gen_fixtures.py

One fixture is written:

``tests/fixtures/tiny_gemma.npz`` — whole-model logits at a tiny seeded config (float64
composed numpy oracle + frozen HF-named weights).

The composed oracle uses numpy primitives (matmul, np.exp, np.tanh) with the
same operations as the student forward, so float64 accumulation is bit-identical →
whole-model parity at rtol=1e-9.

As an authoring sanity check we also assert the numpy oracle matches a genuine
``Gemma2ForCausalLM`` (float32, eager attention) at rtol=1e-3/atol=1e-3.
We force ``attn_implementation='eager'`` so that PyTorch SDPA does not
diverge from the eager (explicit softmax) path our oracle follows.
Max observed diff is documented below.

Architecture exercised in the tiny config (2 layers):
  layer 0: sliding_attention (even-indexed)
  layer 1: full_attention (odd-indexed)
"""

from __future__ import annotations

import pathlib

import numpy as np

FIX = pathlib.Path(__file__).parent / "fixtures"

# ─── tiny model config ───────────────────────────────────────────────────────
V = 64          # vocab_size
d = 32          # hidden_size
NL = 2          # num_hidden_layers  (one sliding + one full)
H = 4           # num_attention_heads
KVH = 2         # num_key_value_heads
HEAD_DIM = 8    # head_dim (Gemma-2 sets this explicitly)
INT_SIZE = 64   # intermediate_size (FFN)
EPS = 1e-6      # rms_norm_eps
ROPE_BASE = 10000.0
QUERY_SCALAR = 8         # query_pre_attn_scalar (int); scale = QUERY_SCALAR**-0.5
FINAL_SOFTCAP = 30.0     # final_logit_softcapping
ATTN_SOFTCAP = 50.0      # attn_logit_softcapping
SWA_WINDOW = 8           # sliding_window
L = 5                    # sequence length
N_GROUPS = H // KVH      # KV repeat groups


# ─── numpy float64 primitives ────────────────────────────────────────────────

def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    """GELU tanh approximation (gelu_pytorch_tanh). NOT SiLU."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _gemma_rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """Gemma-2 (1+w) RMSNorm: ``(1 + weight) * (x / rms)`` in float64."""
    x64 = x.astype(np.float64)
    rms = np.sqrt((x64 ** 2).mean(axis=-1, keepdims=True) + eps)
    return (1.0 + w.astype(np.float64)) * (x64 / rms)


def _rope_half(x: np.ndarray, positions: np.ndarray, base: float, head_dim: int) -> np.ndarray:
    """Rotate-half RoPE. x: (B, H, L, head_dim), positions: (L,)."""
    x64 = x.astype(np.float64)
    idx = np.arange(0, head_dim, 2, dtype=np.float64)
    inv_freq = 1.0 / (base ** (idx / head_dim))
    angle = np.outer(positions.astype(np.float64), inv_freq)  # (L, head_dim/2)
    cos = np.concatenate([np.cos(angle), np.cos(angle)], axis=-1)  # (L, head_dim)
    sin = np.concatenate([np.sin(angle), np.sin(angle)], axis=-1)
    cos = cos[np.newaxis, np.newaxis, :, :]  # (1, 1, L, head_dim)
    sin = sin[np.newaxis, np.newaxis, :, :]
    x1, x2 = x64[..., :head_dim // 2], x64[..., head_dim // 2:]
    return x64 * cos + np.concatenate([-x2, x1], axis=-1) * sin


def _softcap(x: np.ndarray, cap: float) -> np.ndarray:
    return cap * np.tanh(x / cap)


def _softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - np.where(np.isfinite(m), m, 0.0))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-45)


# ─── composed float64 numpy oracle ───────────────────────────────────────────

def _composed_oracle_np(W: dict, ids: np.ndarray) -> np.ndarray:
    """Run the composed float64 numpy Gemma-2 forward.

    W: numpy weight dict (HF names)
    ids: (1, L) int array
    Returns logits (1, L, V) numpy float64.
    """
    pos = np.arange(L, dtype=np.int64)

    # Embed + sqrt(d) scale
    embed_scale = np.sqrt(float(d))
    h = W["model.embed_tokens.weight"].astype(np.float64)[ids[0]][np.newaxis]  # (1, L, d)
    h = h * embed_scale

    for i in range(NL):
        p = f"model.layers.{i}"

        # ── Attention sub-block ───────────────────────────────────────────
        # 1. input_layernorm (1+w)
        h_norm = _gemma_rms_norm(h, W[f"{p}.input_layernorm.weight"].astype(np.float64), EPS)

        # 2. GQA attention with RoPE and attn logit softcap
        B, Lseq, _ = h_norm.shape
        q = h_norm @ W[f"{p}.self_attn.q_proj.weight"].astype(np.float64).T   # (B, L, H*HD)
        k = h_norm @ W[f"{p}.self_attn.k_proj.weight"].astype(np.float64).T   # (B, L, KVH*HD)
        v = h_norm @ W[f"{p}.self_attn.v_proj.weight"].astype(np.float64).T   # (B, L, KVH*HD)

        q = q.reshape(B, Lseq, H, HEAD_DIM).transpose(0, 2, 1, 3)    # (B, H, L, HD)
        k = k.reshape(B, Lseq, KVH, HEAD_DIM).transpose(0, 2, 1, 3)  # (B, KVH, L, HD)
        v = v.reshape(B, Lseq, KVH, HEAD_DIM).transpose(0, 2, 1, 3)  # (B, KVH, L, HD)

        q = _rope_half(q, pos, ROPE_BASE, HEAD_DIM)
        k = _rope_half(k, pos, ROPE_BASE, HEAD_DIM)

        # Repeat KV to H heads (GQA)
        k = np.repeat(k, N_GROUPS, axis=1)  # (B, H, L, HD)
        v = np.repeat(v, N_GROUPS, axis=1)  # (B, H, L, HD)

        scale = float(QUERY_SCALAR) ** -0.5
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # (B, H, L, L)

        # Attention logit softcap (before softmax)
        scores = _softcap(scores, ATTN_SOFTCAP)

        # Causal mask: even layers → sliding window; odd layers → full causal
        rows = np.arange(Lseq)[:, None]
        cols = np.arange(Lseq)[None, :]
        if i % 2 == 0:  # sliding_attention (layer 0)
            causal_mask = np.where(
                (rows >= cols) & (rows - cols < SWA_WINDOW), 0.0, -np.inf
            )
        else:  # full_attention (layer 1)
            causal_mask = np.where(rows >= cols, 0.0, -np.inf)

        scores = scores + causal_mask
        probs = _softmax(scores)

        attn_out = probs @ v  # (B, H, L, HD)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, Lseq, H * HEAD_DIM)
        attn_out = attn_out @ W[f"{p}.self_attn.o_proj.weight"].astype(np.float64).T

        # 3. post_attention_layernorm (sandwich)
        attn_normed = _gemma_rms_norm(
            attn_out, W[f"{p}.post_attention_layernorm.weight"].astype(np.float64), EPS
        )

        # 4. First residual
        h = h + attn_normed

        # ── FFN sub-block ─────────────────────────────────────────────────
        # 5. pre_feedforward_layernorm (sandwich)
        ffn_in = _gemma_rms_norm(
            h, W[f"{p}.pre_feedforward_layernorm.weight"].astype(np.float64), EPS
        )

        # 6. GeGLU FFN with GELU(tanh)
        gate = _gelu_tanh(ffn_in @ W[f"{p}.mlp.gate_proj.weight"].astype(np.float64).T)
        up = ffn_in @ W[f"{p}.mlp.up_proj.weight"].astype(np.float64).T
        ffn_out = (gate * up) @ W[f"{p}.mlp.down_proj.weight"].astype(np.float64).T

        # 7. post_feedforward_layernorm (sandwich)
        ffn_normed = _gemma_rms_norm(
            ffn_out, W[f"{p}.post_feedforward_layernorm.weight"].astype(np.float64), EPS
        )

        # 8. Second residual
        h = h + ffn_normed

    # Final (1+w) RMSNorm
    h = _gemma_rms_norm(h, W["model.norm.weight"].astype(np.float64), EPS)

    # lm_head (tied to embed_tokens) + final logit softcap
    logits = h @ W["model.embed_tokens.weight"].astype(np.float64).T
    logits = _softcap(logits, FINAL_SOFTCAP)
    return logits


def main() -> None:
    FIX.mkdir(exist_ok=True)

    # ── build seeded tiny weights ─────────────────────────────────────────────
    rng = np.random.default_rng(42)
    ids = rng.integers(0, V, size=(1, L))

    W: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": rng.standard_normal((V, d)).astype(np.float32),
        "model.norm.weight":         rng.standard_normal((d,)).astype(np.float32),
    }
    # No separate lm_head.weight — Gemma-2 ties embeddings.
    for i in range(NL):
        p = f"model.layers.{i}"
        for nm in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ):
            W[f"{p}.{nm}"] = rng.standard_normal((d,)).astype(np.float32)
        W[f"{p}.self_attn.q_proj.weight"] = rng.standard_normal((H * HEAD_DIM, d)).astype(np.float32)
        W[f"{p}.self_attn.k_proj.weight"] = rng.standard_normal((KVH * HEAD_DIM, d)).astype(np.float32)
        W[f"{p}.self_attn.v_proj.weight"] = rng.standard_normal((KVH * HEAD_DIM, d)).astype(np.float32)
        W[f"{p}.self_attn.o_proj.weight"] = rng.standard_normal((d, H * HEAD_DIM)).astype(np.float32)
        W[f"{p}.mlp.gate_proj.weight"] = rng.standard_normal((INT_SIZE, d)).astype(np.float32)
        W[f"{p}.mlp.up_proj.weight"]   = rng.standard_normal((INT_SIZE, d)).astype(np.float32)
        W[f"{p}.mlp.down_proj.weight"] = rng.standard_normal((d, INT_SIZE)).astype(np.float32)

    # Run composed float64 numpy oracle
    oracle_logits = _composed_oracle_np(W, ids)
    print(f"  composed numpy oracle logits shape: {oracle_logits.shape}")

    # ── HF-anchor: verify oracle matches genuine Gemma2ForCausalLM (float32) ──
    # We force attn_implementation='eager' so HF uses explicit softmax (same path as our
    # oracle) rather than PyTorch SDPA, which produces numerically different results.
    # Tolerance: rtol=1e-3/atol=1e-3 (float32 vs float64 gap; max observed diff ~1.7e-5).
    try:
        import torch
        from transformers import Gemma2Config as HFGemma2Config
        from transformers import Gemma2ForCausalLM

        hf_cfg = HFGemma2Config(
            hidden_size=d,
            num_hidden_layers=NL,
            num_attention_heads=H,
            num_key_value_heads=KVH,
            head_dim=HEAD_DIM,
            intermediate_size=INT_SIZE,
            vocab_size=V,
            rms_norm_eps=EPS,
            max_position_embeddings=128,
            sliding_window=SWA_WINDOW,
            query_pre_attn_scalar=QUERY_SCALAR,
            final_logit_softcapping=FINAL_SOFTCAP,
            attn_logit_softcapping=ATTN_SOFTCAP,
            tie_word_embeddings=True,
            rope_parameters={"rope_theta": ROPE_BASE, "rope_type": "default"},
            attn_implementation="eager",  # must match our oracle's explicit softmax path
        )
        hf_model = Gemma2ForCausalLM(hf_cfg)
        hf_model.eval()

        sd = hf_model.state_dict()
        with torch.no_grad():
            sd["model.embed_tokens.weight"].copy_(
                torch.from_numpy(W["model.embed_tokens.weight"])
            )
            sd["model.norm.weight"].copy_(
                torch.from_numpy(W["model.norm.weight"])
            )
            for i in range(NL):
                p = f"model.layers.{i}"
                for nm in (
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                    "pre_feedforward_layernorm.weight",
                    "post_feedforward_layernorm.weight",
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                ):
                    sd[f"{p}.{nm}"].copy_(torch.from_numpy(W[f"{p}.{nm}"]))
        hf_model.load_state_dict(sd)

        with torch.no_grad():
            hf_logits = hf_model(
                torch.tensor(ids, dtype=torch.long)
            ).logits.float().numpy()

        max_diff = float(np.max(np.abs(oracle_logits - hf_logits)))
        # Hard-fail: AssertionError must propagate — do NOT wrap in except Exception.
        # Only ImportError (missing torch/transformers) is allowed to skip the anchor.
        np.testing.assert_allclose(oracle_logits, hf_logits, rtol=1e-3, atol=1e-3)
        print(
            f"  HF-anchor (eager): numpy oracle vs Gemma2ForCausalLM (float32) "
            f"max-abs-diff = {max_diff:.2e} ✓"
        )
    except ImportError as e:
        # Only a missing dependency may skip the anchor — never a numerical mismatch.
        print(f"  HF-anchor skipped (transformers/torch unavailable): {e}")

    # ── write fixture ─────────────────────────────────────────────────────────
    np.savez(
        FIX / "tiny_gemma.npz",
        input_ids=ids,
        logits=oracle_logits,
        # config scalars
        dim=np.array(d),
        n_layers=np.array(NL),
        n_heads=np.array(H),
        n_kv_heads=np.array(KVH),
        head_dim=np.array(HEAD_DIM),
        vocab_size=np.array(V),
        intermediate_size=np.array(INT_SIZE),
        norm_eps=np.array(EPS),
        rope_base=np.array(ROPE_BASE),
        query_pre_attn_scalar=np.array(QUERY_SCALAR),
        final_logit_softcapping=np.array(FINAL_SOFTCAP),
        attn_logit_softcapping=np.array(ATTN_SOFTCAP),
        sliding_window=np.array(SWA_WINDOW),
        max_seq_len=np.array(128),
        # weights
        **W,
    )
    print(f"  wrote tiny_gemma.npz  logits{oracle_logits.shape}")


if __name__ == "__main__":
    main()
