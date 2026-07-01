"""402 — generate the frozen golden fixture for the continuous-batching engine.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 402_continuous_batching/tests/gen_fixtures.py

Reuses 401/306's composed float64 Qwen3 oracle (the exact tiny config, Global
Constraint 4) and 306's seeded weights (so the model is byte-identical to 401's).
The only extension is that we freeze the greedy sequence for **several prompts of
different lengths** — each generated until its total sequence length reaches the
engine's length budget (``max_seq_len``) — so the engine's *raggedness*,
*mid-batch retirement*, and *slot reuse* are exercised: with prompts of length
``4/6/3`` and a length budget of ``12`` the three requests generate ``8/6/9``
tokens, so they finish at *different* steps.

Retirement model (matches ``Engine``): a request retires when it emits ``eos`` or
when its total length hits the length budget ``max_seq_len``. The graded scenario is
budget-driven (deterministic); a single frozen ``eos_probe`` token exercises the EOS
branch of ``is_finished`` too.

Writes ``fixtures/continuous_batching.npz``:
  n_req            ()      number of requests
  slots            ()      max concurrent requests (a GIVEN systems fact = 2)
  max_seq_len      ()      cache preallocation AND the length budget (= 12 here)
  prompt_len_i     ()      prompt length of request i         (i = 0 .. n_req-1)
  n_new_i          ()      generated-token count of request i (== budget - prompt_len)
  seq_i            (Li,)   full token ids of request i (prompt + generated)
  eos_probe_id     ()      a token req 0 first emits at eos_probe_nnew (EOS-branch test)
  eos_probe_nnew   ()      generated-token count at which eos_probe first appears
  <HF weights>             all HF-named arrays (loaded via 306's load_qwen3 at grade time)
  <config scalars>         dim, n_layers, ... max_seq_len

Authoring sanity: assert the composed oracle matches a genuine ``Qwen3ForCausalLM``
on the longest sequence at rtol≈1e-3 (proves the oracle faithful, non-circular).
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

# ─── serving scenario (GIVENs) ────────────────────────────────────────────────
# Length budget doubles as the cache-preallocation size. Kept small (12, vs 401's
# 64) so the ragged multi-request scenario stays a fast, readable fixture — a scale
# choice, exactly like 403's block_size=4. Prompts of different lengths hit the
# budget after different numbers of generated tokens → staggered retirement.
SLOTS = 2
MAX_SEQ_LEN = 12
PROMPT_LENS = [4, 6, 3]     # → generate 8 / 6 / 9 tokens under a budget of 12


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
    """306's seeded tiny weights (seed 42) — byte-identical to 401's model."""
    rng = np.random.default_rng(42)
    _ = rng.integers(0, V, size=(1, 5))  # consume 306's prompt draw (keeps weights aligned)

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
    return W


def _greedy_to_budget(T: dict, prompt: list[int], budget: int) -> list[int]:
    """Greedy-decode ``prompt`` until the total sequence length reaches ``budget``."""
    ids = list(prompt)
    while len(ids) < budget:
        logits = _composed_oracle(T, np.array(ids)[None, :])
        ids.append(int(np.argmax(logits[0, -1])))
    return ids


def _make_prompts() -> list[list[int]]:
    rng = np.random.default_rng(123)
    return [rng.integers(0, V, size=L).tolist() for L in PROMPT_LENS]


def _eos_probe(seq: list[int], prompt_len: int) -> tuple[int, int]:
    """A token request-0 emits, whose FIRST occurrence in its generation is at
    step k >= 2. Used to test the EOS retirement branch: an engine given this eos
    must retire req 0 after exactly k generated tokens (and be running before)."""
    gen = seq[prompt_len:]
    for k in range(1, len(gen) + 1):
        tok = gen[k - 1]
        if tok not in gen[: k - 1] and k >= 2:
            return int(tok), int(k)
    raise RuntimeError("no clean eos probe found in request 0")


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

    W = _build_weights()
    T = {k: torch.from_numpy(v.astype(np.float64)) for k, v in W.items()}

    prompts = _make_prompts()
    seqs = [_greedy_to_budget(T, pr, MAX_SEQ_LEN) for pr in prompts]
    n_news = [len(s) - len(p) for s, p in zip(seqs, prompts)]
    for i, (pr, seq, n_new) in enumerate(zip(prompts, seqs, n_news)):
        print(f"  req {i}: prompt_len={len(pr)} n_new={n_new} seq={seq}")

    assert n_news[0] != n_news[1], "first two admitted requests must finish at different steps"
    eos_probe_id, eos_probe_nnew = _eos_probe(seqs[0], len(prompts[0]))
    print(f"  eos_probe: id={eos_probe_id} first appears at req0 gen-step {eos_probe_nnew}")

    # HF-anchor on the longest sequence.
    longest = np.array(max(seqs, key=len), dtype=np.int64)
    _hf_anchor(T, longest[None, :], _composed_oracle(T, longest[None, :]))

    out: dict[str, np.ndarray] = dict(
        n_req=np.array(len(prompts)),
        slots=np.array(SLOTS),
        eos_probe_id=np.array(eos_probe_id),
        eos_probe_nnew=np.array(eos_probe_nnew),
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
    for i, (pr, seq, n_new) in enumerate(zip(prompts, seqs, n_news)):
        out[f"prompt_len_{i}"] = np.array(len(pr))
        out[f"n_new_{i}"] = np.array(n_new)
        out[f"seq_{i}"] = np.array(seq, dtype=np.int64)

    np.savez(FIX / "continuous_batching.npz", **out)
    print(f"  wrote continuous_batching.npz  n_req={len(prompts)} slots={SLOTS} budget={MAX_SEQ_LEN}")


if __name__ == "__main__":
    main()
