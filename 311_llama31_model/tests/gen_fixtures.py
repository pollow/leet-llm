"""311 — generate frozen golden fixtures for the Llama-3.1 whole-model forward.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 311_llama31_model/tests/gen_fixtures.py

Two fixtures are written:

``tests/fixtures/tiny_llama31.npz`` — whole-model logits at a tiny seeded config
(float64 composed numpy oracle + frozen HF-named weights), RoPE schedule
``rope_type="llama3"`` (the real Llama-3.1 long-context schedule).

``tests/fixtures/rope_freqs.npz`` — per-``rope_type`` ``inv_freq`` goldens computed
by the genuine HF ``ROPE_INIT_FUNCTIONS`` (``linear`` / ``dynamic`` / ``llama3`` /
``yarn``) plus the matching ``scaling`` dicts (as JSON) the student passes to
``rope_scaled_freqs``.  This is the grade-time genuine-HF anchor for the operator.

The composed oracle uses numpy primitives (matmul, np.exp) with the same operations
as the student forward, so float64 accumulation is bit-identical → whole-model parity
at rtol=1e-9.

As an authoring sanity check we also assert the numpy oracle matches a genuine
``LlamaForCausalLM`` (float32, eager attention, ``rope_scaling=llama3``) at
rtol=1e-3/atol=1e-3.  Max observed diff is documented below
(numpy-oracle vs LlamaForCausalLM float32 ≈ 3e-5).
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

FIX = pathlib.Path(__file__).parent / "fixtures"

# ─── tiny model config ───────────────────────────────────────────────────────
V = 64            # vocab_size
d = 32            # hidden_size
NL = 2            # num_hidden_layers
H = 4             # num_attention_heads
KVH = 2           # num_key_value_heads
HEAD_DIM = d // H  # 8
INT_SIZE = 37     # intermediate_size (SwiGLU FFN)
EPS = 1e-5        # rms_norm_eps
ROPE_BASE = 500000.0
MAX_POS = 512     # max_position_embeddings
L = 6             # sequence length
N_GROUPS = H // KVH

# Llama-3.1's long-context RoPE schedule (the whole-model fixture uses this).
ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 64,
}


# ─── numpy float64 primitives ────────────────────────────────────────────────

def _rope_scaled_freqs(head_dim: int, base: float, scaling: dict | None) -> np.ndarray:
    """float64 reference for rope_scaled_freqs (mirrors HF ROPE_INIT_FUNCTIONS)."""
    dim = head_dim
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    if scaling is None:
        return inv_freq
    rope_type = scaling.get("rope_type", "default")
    if rope_type == "default":
        return inv_freq
    if rope_type == "linear":
        return inv_freq / scaling["factor"]
    if rope_type == "dynamic":
        factor = scaling["factor"]
        max_pos = scaling["max_position_embeddings"]
        seq_len = max(scaling.get("seq_len", max_pos), max_pos)
        new_base = base * ((factor * seq_len / max_pos) - (factor - 1)) ** (dim / (dim - 2))
        return 1.0 / (new_base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    if rope_type == "llama3":
        factor = scaling["factor"]
        lo = scaling["low_freq_factor"]
        hi = scaling["high_freq_factor"]
        old = scaling["original_max_position_embeddings"]
        low_wl, high_wl = old / lo, old / hi
        wl = 2 * np.pi / inv_freq
        inv_llama = np.where(wl > low_wl, inv_freq / factor, inv_freq)
        smooth = (old / wl - lo) / (hi - lo)
        smoothed = (1 - smooth) * inv_llama / factor + smooth * inv_llama
        is_med = ~(wl < high_wl) & ~(wl > low_wl)
        return np.where(is_med, smoothed, inv_llama)
    if rope_type == "yarn":
        factor = scaling["factor"]
        old = scaling["original_max_position_embeddings"]
        beta_fast = scaling.get("beta_fast", 32)
        beta_slow = scaling.get("beta_slow", 1)
        truncate = scaling.get("truncate", True)
        pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
        extrap = 1.0 / pos_freqs
        interp = 1.0 / (factor * pos_freqs)

        def fcd(num_rot):
            return (dim * math.log(old / (num_rot * 2 * math.pi))) / (2 * math.log(base))

        low, high = fcd(beta_fast), fcd(beta_slow)
        if truncate:
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, dim - 1)
        if low == high:
            high += 0.001
        ramp = np.clip((np.arange(dim // 2, dtype=np.float64) - low) / (high - low), 0, 1)
        extrap_factor = 1 - ramp
        return interp * (1 - extrap_factor) + extrap * extrap_factor
    raise ValueError(f"unknown rope_type {rope_type!r}")


def _rms_norm(x, w, eps):
    x64 = x.astype(np.float64)
    rms = np.sqrt((x64 ** 2).mean(axis=-1, keepdims=True) + eps)
    return (x64 / rms) * w.astype(np.float64)


def _rope_from_freqs(x, positions, inv_freq):
    x64 = x.astype(np.float64)
    angle = positions.astype(np.float64)[:, None] * inv_freq        # (L, d/2)
    angle = np.concatenate([angle, angle], axis=-1)                 # (L, d)
    cos = np.cos(angle)[None, None, :, :]
    sin = np.sin(angle)[None, None, :, :]
    hd = x64.shape[-1]
    x1, x2 = x64[..., : hd // 2], x64[..., hd // 2:]
    rot = np.concatenate([-x2, x1], axis=-1)
    return x64 * cos + rot * sin


def _softmax(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - np.where(np.isfinite(m), m, 0.0))
    return e / e.sum(axis=-1, keepdims=True)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _composed_oracle_np(W: dict, ids: np.ndarray) -> np.ndarray:
    pos = np.arange(L, dtype=np.int64)
    inv_freq = _rope_scaled_freqs(HEAD_DIM, ROPE_BASE, ROPE_SCALING)
    h = W["model.embed_tokens.weight"].astype(np.float64)[ids[0]][np.newaxis]  # (1, L, d)
    scale = float(HEAD_DIM) ** -0.5

    rows = np.arange(L)[:, None]
    cols = np.arange(L)[None, :]
    causal = np.where(rows >= cols, 0.0, -np.inf)

    for i in range(NL):
        p = f"model.layers.{i}"
        B, Lseq, _ = h.shape
        a = _rms_norm(h, W[f"{p}.input_layernorm.weight"], EPS)

        q = a @ W[f"{p}.self_attn.q_proj.weight"].astype(np.float64).T
        k = a @ W[f"{p}.self_attn.k_proj.weight"].astype(np.float64).T
        v = a @ W[f"{p}.self_attn.v_proj.weight"].astype(np.float64).T

        q = q.reshape(B, Lseq, H, HEAD_DIM).transpose(0, 2, 1, 3)
        k = k.reshape(B, Lseq, KVH, HEAD_DIM).transpose(0, 2, 1, 3)
        v = v.reshape(B, Lseq, KVH, HEAD_DIM).transpose(0, 2, 1, 3)

        q = _rope_from_freqs(q, pos, inv_freq)
        k = _rope_from_freqs(k, pos, inv_freq)

        k = np.repeat(k, N_GROUPS, axis=1)
        v = np.repeat(v, N_GROUPS, axis=1)

        scores = (q @ k.transpose(0, 1, 3, 2)) * scale + causal
        probs = _softmax(scores)
        attn = probs @ v
        attn = attn.transpose(0, 2, 1, 3).reshape(B, Lseq, H * HEAD_DIM)
        attn = attn @ W[f"{p}.self_attn.o_proj.weight"].astype(np.float64).T
        h = h + attn

        f_in = _rms_norm(h, W[f"{p}.post_attention_layernorm.weight"], EPS)
        gate = f_in @ W[f"{p}.mlp.gate_proj.weight"].astype(np.float64).T
        up = f_in @ W[f"{p}.mlp.up_proj.weight"].astype(np.float64).T
        ff = (_silu(gate) * up) @ W[f"{p}.mlp.down_proj.weight"].astype(np.float64).T
        h = h + ff

    h = _rms_norm(h, W["model.norm.weight"], EPS)
    return h @ W["lm_head.weight"].astype(np.float64).T


# ─── per-rope_type inv_freq goldens (operator anchor) ─────────────────────────

def _rope_freq_goldens():
    """Freeze genuine HF inv_freq for each rope_type + the student scaling dict."""
    import torch
    from transformers import LlamaConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    # (student-facing scaling dict, HF config max_position_embeddings, seq_len arg)
    cases = {
        "default": ({"rope_type": "default"}, MAX_POS, None),
        "linear": ({"rope_type": "linear", "factor": 4.0}, MAX_POS, None),
        "dynamic": (
            {"rope_type": "dynamic", "factor": 4.0, "max_position_embeddings": 64, "seq_len": 256},
            64, 256,
        ),
        "llama3": (
            {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
             "high_freq_factor": 4.0, "original_max_position_embeddings": 64},
            MAX_POS, None,
        ),
        "yarn": (
            {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 64},
            256, None,
        ),
    }
    out = {"head_dim": np.array(HEAD_DIM), "rope_base": np.array(ROPE_BASE)}
    for name, (scaling, maxp, seq_len) in cases.items():
        if name == "default":
            inv = 1.0 / (ROPE_BASE ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM))
        else:
            hf_scaling = {k: v for k, v in scaling.items()
                          if k not in ("seq_len", "max_position_embeddings")}
            hf_scaling.setdefault("rope_theta", ROPE_BASE)
            cfg = LlamaConfig(
                hidden_size=d, num_hidden_layers=1, num_attention_heads=H,
                num_key_value_heads=KVH, intermediate_size=INT_SIZE, vocab_size=V,
                max_position_embeddings=maxp, rope_theta=ROPE_BASE, rope_scaling=hf_scaling,
            )
            inv_t, _ = ROPE_INIT_FUNCTIONS[scaling["rope_type"]](
                cfg, torch.device("cpu"), seq_len=seq_len
            )
            inv = inv_t.numpy().astype(np.float64)
        out[f"{name}_inv_freq"] = inv
        out[f"{name}_scaling"] = np.array(json.dumps(scaling))
    return out


def main() -> None:
    FIX.mkdir(exist_ok=True)

    rng = np.random.default_rng(311)
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
        W[f"{p}.self_attn.k_proj.weight"]         = rng.standard_normal((KVH * HEAD_DIM, d)).astype(np.float32)
        W[f"{p}.self_attn.v_proj.weight"]         = rng.standard_normal((KVH * HEAD_DIM, d)).astype(np.float32)
        W[f"{p}.self_attn.o_proj.weight"]         = rng.standard_normal((d, H * HEAD_DIM)).astype(np.float32)
        W[f"{p}.mlp.gate_proj.weight"]            = rng.standard_normal((INT_SIZE, d)).astype(np.float32)
        W[f"{p}.mlp.up_proj.weight"]              = rng.standard_normal((INT_SIZE, d)).astype(np.float32)
        W[f"{p}.mlp.down_proj.weight"]            = rng.standard_normal((d, INT_SIZE)).astype(np.float32)

    oracle_logits = _composed_oracle_np(W, ids)
    print(f"  composed numpy oracle logits shape: {oracle_logits.shape}")

    # ── HF-anchor: verify oracle matches genuine LlamaForCausalLM (float32, eager) ──
    try:
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        hf_cfg = LlamaConfig(
            hidden_size=d,
            num_hidden_layers=NL,
            num_attention_heads=H,
            num_key_value_heads=KVH,
            intermediate_size=INT_SIZE,
            vocab_size=V,
            rms_norm_eps=EPS,
            max_position_embeddings=MAX_POS,
            tie_word_embeddings=False,
            rope_theta=ROPE_BASE,
            rope_scaling={**ROPE_SCALING, "rope_theta": ROPE_BASE},
            attn_implementation="eager",
        )
        hf_model = LlamaForCausalLM(hf_cfg).eval()
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
            f"  HF-anchor (eager, rope=llama3): numpy oracle vs LlamaForCausalLM "
            f"(float32) max-abs-diff = {max_diff:.2e} ✓"
        )
    except ImportError as e:
        print(f"  HF-anchor skipped (transformers/torch unavailable): {e}")

    # ── operator goldens (per-rope_type inv_freq vs genuine HF) ──
    rope_goldens = _rope_freq_goldens()
    np.savez(FIX / "rope_freqs.npz", **rope_goldens)
    print(f"  wrote rope_freqs.npz  ({sorted(k for k in rope_goldens if k.endswith('inv_freq'))})")

    np.savez(
        FIX / "tiny_llama31.npz",
        input_ids=ids,
        logits=oracle_logits,
        dim=np.array(d),
        n_layers=np.array(NL),
        n_heads=np.array(H),
        n_kv_heads=np.array(KVH),
        vocab_size=np.array(V),
        max_seq_len=np.array(MAX_POS),
        norm_eps=np.array(EPS),
        rope_base=np.array(ROPE_BASE),
        rope_scaling=np.array(json.dumps(ROPE_SCALING)),
        **W,
    )
    print(f"  wrote tiny_llama31.npz  logits{oracle_logits.shape}")


if __name__ == "__main__":
    main()
