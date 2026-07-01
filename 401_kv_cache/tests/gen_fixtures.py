"""401 — generate the frozen golden fixture for stateful prefill/decode.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 401_kv_cache/tests/gen_fixtures.py

Reuses 306's composed float64 Qwen3 oracle (F.linear / F.rms_norm / F.silu /
F.scaled_dot_product_attention with rotate-half RoPE + per-head qk-norm — NEVER the
genuine ``Qwen3ForCausalLM``, so float64 accumulation is preserved → whole-model parity
at rtol=1e-9). The only extension over 306 is a *variable-length* forward so we can
greedy-decode 8 steps and freeze the full per-position logits.

Writes ``fixtures/kv_cache.npz``:
  input_ids  (1, 5)   306's seeded prompt
  token_ids  (13,)    prompt + 8 greedy tokens (Test B reads this)
  logits     (1,13,V) per-position logits of the full 13-token sequence (Test A reads this)
  <HF weights>        all HF-named arrays (loaded via 306's load_qwen3 at grade time)
  <config scalars>    dim, n_layers, ... max_seq_len (=64 for cache preallocation)

Authoring sanity: assert the composed oracle matches a genuine ``Qwen3ForCausalLM`` on the
full sequence at rtol≈1e-3 (float32 HF is close, not exact) — proves the oracle faithful.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Qwen3Config as HFQwen3Config
from transformers import Qwen3ForCausalLM

FIX = pathlib.Path(__file__).parent / "fixtures"

# ─── tiny model config (the exact 306 tiny Qwen3, per L4 Global Constraint 4) ──
V, d, NL, H, KV, Fd = 64, 16, 2, 4, 2, 32
HEAD_DIM = 4
EPS = 1e-6
QK_EPS = 1e-6
BASE = 10000.0
PROMPT_LEN = 5          # 306's seeded prompt length
N_NEW = 8               # greedy tokens to generate → total length 13
MAX_SEQ_LEN = 64        # cache preallocation size


# ─── rotate-half RoPE (HF convention, float64) ───────────────────────────────
def _rope_half_torch(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    head_dim = x.shape[-1]
    idx = torch.arange(0, head_dim, 2, dtype=torch.float64)
    inv_freq = 1.0 / (base ** (idx / head_dim))
    angle = torch.outer(positions.to(torch.float64), inv_freq)
    cos = torch.cat([angle.cos(), angle.cos()], dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat([angle.sin(), angle.sin()], dim=-1).unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    rotate = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotate * sin


def _qk_norm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return (x / rms) * weight


def _causal_mask_torch(seq_len: int) -> torch.Tensor:
    rows = torch.arange(seq_len)[:, None]
    cols = torch.arange(seq_len)[None, :]
    attended = rows >= cols
    return torch.where(
        attended,
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(-float("inf"), dtype=torch.float64),
    )


def _composed_oracle(T: dict, ids: np.ndarray) -> np.ndarray:
    """Composed float64 Qwen3 forward over a sequence of *arbitrary* length.

    T: torch tensor dict (HF names); ids: (1, L) int array. Returns logits (1, L, V).
    """
    L = ids.shape[-1]
    pos = torch.arange(L, dtype=torch.long)
    mask = _causal_mask_torch(L)

    h = T["model.embed_tokens.weight"][torch.from_numpy(ids[0])].unsqueeze(0)

    for i in range(NL):
        p = f"model.layers.{i}"
        a = F.rms_norm(h, (d,), weight=T[f"{p}.input_layernorm.weight"], eps=EPS)

        q = F.linear(a, T[f"{p}.self_attn.q_proj.weight"])
        k = F.linear(a, T[f"{p}.self_attn.k_proj.weight"])
        v = F.linear(a, T[f"{p}.self_attn.v_proj.weight"])

        q = q.reshape(1, L, H, HEAD_DIM).transpose(1, 2)
        k = k.reshape(1, L, KV, HEAD_DIM).transpose(1, 2)
        v = v.reshape(1, L, KV, HEAD_DIM).transpose(1, 2)

        q = _qk_norm_torch(q, T[f"{p}.self_attn.q_norm.weight"], QK_EPS)
        k = _qk_norm_torch(k, T[f"{p}.self_attn.k_norm.weight"], QK_EPS)

        q = _rope_half_torch(q, pos, BASE)
        k = _rope_half_torch(k, pos, BASE)

        reps = H // KV
        k = k.repeat_interleave(reps, dim=1)
        v = v.repeat_interleave(reps, dim=1)

        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = o.transpose(1, 2).reshape(1, L, H * HEAD_DIM)
        o = F.linear(o, T[f"{p}.self_attn.o_proj.weight"])

        h = h + o
        f = F.rms_norm(h, (d,), weight=T[f"{p}.post_attention_layernorm.weight"], eps=EPS)
        swi = F.linear(
            F.silu(F.linear(f, T[f"{p}.mlp.gate_proj.weight"]))
            * F.linear(f, T[f"{p}.mlp.up_proj.weight"]),
            T[f"{p}.mlp.down_proj.weight"],
        )
        h = h + swi

    h = F.rms_norm(h, (d,), weight=T["model.norm.weight"], eps=EPS)
    return (h @ T["lm_head.weight"].T).detach().numpy()


def _build_weights() -> dict[str, np.ndarray]:
    """306's seeded tiny weights (seed 42) — so the prompt matches 306's input_ids."""
    rng = np.random.default_rng(42)
    ids = rng.integers(0, V, size=(1, PROMPT_LEN))  # consume ids draw first (as 306 does)

    W: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": rng.standard_normal((V, d)),
        "model.norm.weight": rng.standard_normal((d,)),
        "lm_head.weight": rng.standard_normal((V, d)),
    }
    for i in range(NL):
        p = f"model.layers.{i}"
        W[f"{p}.input_layernorm.weight"] = rng.standard_normal((d,))
        W[f"{p}.post_attention_layernorm.weight"] = rng.standard_normal((d,))
        W[f"{p}.self_attn.q_proj.weight"] = rng.standard_normal((H * HEAD_DIM, d))
        W[f"{p}.self_attn.k_proj.weight"] = rng.standard_normal((KV * HEAD_DIM, d))
        W[f"{p}.self_attn.v_proj.weight"] = rng.standard_normal((KV * HEAD_DIM, d))
        W[f"{p}.self_attn.o_proj.weight"] = rng.standard_normal((d, H * HEAD_DIM))
        W[f"{p}.self_attn.q_norm.weight"] = rng.standard_normal((HEAD_DIM,))
        W[f"{p}.self_attn.k_norm.weight"] = rng.standard_normal((HEAD_DIM,))
        W[f"{p}.mlp.gate_proj.weight"] = rng.standard_normal((Fd, d))
        W[f"{p}.mlp.up_proj.weight"] = rng.standard_normal((Fd, d))
        W[f"{p}.mlp.down_proj.weight"] = rng.standard_normal((d, Fd))
    return W, ids


def _hf_anchor(T: dict, ids: np.ndarray, oracle_logits: np.ndarray) -> None:
    """Assert the composed oracle matches genuine Qwen3ForCausalLM (rtol≈1e-3)."""
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
    sd = hf_model.state_dict()
    with torch.no_grad():
        for name in sd:
            if name in T:
                sd[name].copy_(T[name].float())
    hf_model.load_state_dict(sd)
    with torch.no_grad():
        hf_logits = hf_model(torch.tensor(ids, dtype=torch.long)).logits.numpy()
    np.testing.assert_allclose(oracle_logits, hf_logits, rtol=1e-3, atol=1e-3)
    max_diff = float(np.max(np.abs(oracle_logits - hf_logits)))
    print(f"  HF-anchor: composed oracle vs Qwen3ForCausalLM max-abs-diff = {max_diff:.2e} OK")


def main() -> None:
    FIX.mkdir(exist_ok=True)

    W, prompt_ids = _build_weights()
    T = {k: torch.from_numpy(v.astype(np.float64)) for k, v in W.items()}

    # ── greedy-decode the composed oracle 8 steps ────────────────────────────
    ids = [int(t) for t in prompt_ids[0]]
    for _ in range(N_NEW):
        logits = _composed_oracle(T, np.array(ids)[None, :])  # (1, len, V)
        ids.append(int(np.argmax(logits[0, -1])))
    token_ids = np.array(ids, dtype=np.int64)                 # (13,)
    assert token_ids.shape[0] == PROMPT_LEN + N_NEW

    # ── freeze the full-sequence per-position logits ─────────────────────────
    full_logits = _composed_oracle(T, token_ids[None, :])     # (1, 13, V)
    print(f"  composed oracle full logits shape: {full_logits.shape}")
    print(f"  token_ids: {token_ids.tolist()}")

    # Self-consistency: greedy argmax of positions [4:12] reproduces tokens [5:13].
    greedy = np.argmax(full_logits[0], axis=-1)               # (13,)
    assert np.array_equal(greedy[PROMPT_LEN - 1 : -1], token_ids[PROMPT_LEN:]), \
        "greedy self-consistency failed"
    print("  self-consistency OK (causal teacher-forcing == incremental greedy)")

    # ── HF-anchor on the full sequence ───────────────────────────────────────
    _hf_anchor(T, token_ids[None, :], full_logits)

    np.savez(
        FIX / "kv_cache.npz",
        input_ids=prompt_ids.astype(np.int64),
        token_ids=token_ids,
        logits=full_logits,
        dim=np.array(d),
        n_layers=np.array(NL),
        n_heads=np.array(H),
        n_kv_heads=np.array(KV),
        head_dim=np.array(HEAD_DIM),
        vocab_size=np.array(V),
        max_seq_len=np.array(MAX_SEQ_LEN),
        norm_eps=np.array(EPS),
        qk_norm_eps=np.array(QK_EPS),
        rope_base=np.array(BASE),
        **W,
    )
    print(f"  wrote kv_cache.npz  logits{full_logits.shape}  tokens{token_ids.shape}")


if __name__ == "__main__":
    main()
