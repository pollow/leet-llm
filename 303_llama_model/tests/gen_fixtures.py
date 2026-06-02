"""303 — frozen goldens from a tiny COMPOSED interleaved-RoPE Llama (float64 torch).

AUTHORING ONLY (gen group):
    uv run --group gen python 303_llama_model/tests/gen_fixtures.py

NOT HuggingFace LlamaForCausalLM: stories15M / llama3.np use the INTERLEAVED (Meta) RoPE
convention, whereas HF Llama uses rotate-half. We compose the genuine torch primitives
(F.linear, F.rms_norm, F.silu, F.scaled_dot_product_attention) with interleaved RoPE,
exactly as task 216's oracle, so the whole-model goldens match the block the learner built.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
import torch.nn.functional as F

FIX = pathlib.Path(__file__).parent / "fixtures"
EPS = 1e-6
BASE = 10000.0


def _split(t, n):
    *lead, L, d = t.shape
    return t.reshape(*lead, L, n, d // n).transpose(-3, -2)


def _merge(t):
    *lead, h, L, dk = t.shape
    return t.transpose(-3, -2).reshape(*lead, L, h * dk)


def _rope_i(x, positions):
    d = x.shape[-1]
    inv = 1.0 / (BASE ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    ang = torch.outer(positions.to(torch.float64), inv)
    cis = torch.polar(torch.ones_like(ang), ang)
    xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * cis).reshape(x.shape)


def main() -> None:
    FIX.mkdir(exist_ok=True)
    for old in FIX.glob("*.npz"):
        old.unlink()
    rng = np.random.default_rng(0)
    V, d, NL, H, KV, Fd = 64, 16, 2, 4, 2, 32
    dk = d // H
    L = 5
    ids = rng.integers(0, V, size=(1, L))
    W = {"model.embed_tokens.weight": rng.standard_normal((V, d)),
         "model.norm.weight": rng.standard_normal(d),
         "lm_head.weight": rng.standard_normal((V, d))}
    for i in range(NL):
        p = f"model.layers.{i}"
        W[f"{p}.self_attn.q_proj.weight"] = rng.standard_normal((d, d))
        W[f"{p}.self_attn.k_proj.weight"] = rng.standard_normal((KV * dk, d))
        W[f"{p}.self_attn.v_proj.weight"] = rng.standard_normal((KV * dk, d))
        W[f"{p}.self_attn.o_proj.weight"] = rng.standard_normal((d, d))
        W[f"{p}.mlp.gate_proj.weight"] = rng.standard_normal((Fd, d))
        W[f"{p}.mlp.up_proj.weight"] = rng.standard_normal((Fd, d))
        W[f"{p}.mlp.down_proj.weight"] = rng.standard_normal((d, Fd))
        W[f"{p}.input_layernorm.weight"] = rng.standard_normal(d)
        W[f"{p}.post_attention_layernorm.weight"] = rng.standard_normal(d)

    T = {k: torch.from_numpy(v) for k, v in W.items()}
    pos = torch.arange(L)
    am = torch.from_numpy(np.where(np.triu(np.ones((L, L), bool), 1), -np.inf, 0.0))
    h = T["model.embed_tokens.weight"][torch.from_numpy(ids)]
    for i in range(NL):
        p = f"model.layers.{i}"
        a = F.rms_norm(h, (d,), weight=T[f"{p}.input_layernorm.weight"], eps=EPS)
        q = _rope_i(_split(F.linear(a, T[f"{p}.self_attn.q_proj.weight"]), H), pos)
        k = _rope_i(_split(F.linear(a, T[f"{p}.self_attn.k_proj.weight"]), KV), pos)
        v = _split(F.linear(a, T[f"{p}.self_attn.v_proj.weight"]), KV)
        k = k.repeat_interleave(H // KV, dim=-3)
        v = v.repeat_interleave(H // KV, dim=-3)
        o = F.linear(_merge(F.scaled_dot_product_attention(q, k, v, attn_mask=am)),
                     T[f"{p}.self_attn.o_proj.weight"])
        z = h + o
        f = F.rms_norm(z, (d,), weight=T[f"{p}.post_attention_layernorm.weight"], eps=EPS)
        swi = F.linear(F.silu(F.linear(f, T[f"{p}.mlp.gate_proj.weight"]))
                       * F.linear(f, T[f"{p}.mlp.up_proj.weight"]), T[f"{p}.mlp.down_proj.weight"])
        h = z + swi
    h = F.rms_norm(h, (d,), weight=T["model.norm.weight"], eps=EPS)
    logits = (h @ T["lm_head.weight"].T).numpy()

    np.savez(FIX / "tiny_llama.npz", input_ids=ids, logits=logits,
             dim=np.array(d), n_layers=np.array(NL), n_heads=np.array(H),
             n_kv_heads=np.array(KV), vocab_size=np.array(V), max_seq_len=np.array(32),
             norm_eps=np.array(EPS), rope_base=np.array(BASE), **W)
    print(f"  wrote tiny_llama.npz  logits{logits.shape}")


if __name__ == "__main__":
    main()
