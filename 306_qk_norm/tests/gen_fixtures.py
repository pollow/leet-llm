"""306 — generate frozen golden fixtures for per-head Q/K RMSNorm.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 306_qk_norm/tests/gen_fixtures.py

Oracle: genuine ``Qwen3ForCausalLM`` from transformers 5.9.0, float64.

The fixture captures q and k **before** ``q_norm``/``k_norm`` is applied, the
learned weight vectors ``q_norm.weight`` and ``k_norm.weight``, and the
expected post-norm q and k computed in pure float64 numpy.

Why not capture HF's post-norm directly?  Qwen3RMSNorm internally casts its
input to float32 for the variance computation, then casts the result back.
This float32 intermediate breaks float64 parity at ``rtol=1e-9``.  We
therefore compute the expected output ourselves in numpy float64 from the
captured float64 pre-norm tensors — the inputs are genuine Qwen3 values; the
expected outputs are the float64 reference.

Fixture ``qknorm.npz`` stores:
  - ``q_pre``     shape ``(n_q_heads, L, head_dim)``   — pre-norm Q
  - ``k_pre``     shape ``(n_kv_heads, L, head_dim)``  — pre-norm K
  - ``q_post``    shape ``(n_q_heads, L, head_dim)``   — expected post-norm Q
  - ``k_post``    shape ``(n_kv_heads, L, head_dim)``  — expected post-norm K
  - ``q_weight``  shape ``(head_dim,)``                — q_norm.weight
  - ``k_weight``  shape ``(head_dim,)``                — k_norm.weight
  - ``eps``       scalar                               — variance_epsilon (1e-6)
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

FIX = pathlib.Path(__file__).parent / "fixtures"


def _rms_norm_f64(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """Pure float64 RMSNorm over the last axis — the reference computation."""
    rms = np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def _extract_qk_fixtures(seq_len: int = 5) -> dict[str, np.ndarray]:
    """Run a forward pass through a tiny Qwen3ForCausalLM and capture pre-norm
    Q and K tensors, plus the norm weights.

    Returns a dict with numpy float64 arrays.
    """
    cfg = Qwen3Config(
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

    # Randomise q_norm and k_norm weights so the fixture exercises the scaling
    # path (default init is all-ones, which makes the weight trivial to ignore).
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

    # Patch the attention forward to intercept q/k before q_norm is applied.
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

        # These are float64 tensors: (batch=1, L, n_heads, head_dim)
        q_pre = attn.q_proj(hidden_states).view(hidden_shape)
        k_pre = attn.k_proj(hidden_states).view(hidden_shape)

        # (batch=1, L, n_heads, head_dim) → (n_heads, L, head_dim) after squeeze+permute
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

    attn.forward = orig_forward  # restore

    q_pre = captured["q_pre"].numpy()  # (n_q_heads, L, head_dim)
    k_pre = captured["k_pre"].numpy()  # (n_kv_heads, L, head_dim)
    q_weight = captured["q_weight"].numpy()  # (head_dim,)
    k_weight = captured["k_weight"].numpy()  # (head_dim,)
    eps = float(captured["eps"])

    # Expected output: pure float64 reference (avoids HF's internal f32 cast)
    q_post = _rms_norm_f64(q_pre, q_weight, eps)
    k_post = _rms_norm_f64(k_pre, k_weight, eps)

    # HF-anchor: apply the real q_norm/k_norm modules to the captured pre-norm
    # tensors in float64.  This verifies the numpy oracle matches Qwen3's actual
    # semantics (axis, eps placement, weight application) and catches wrong-axis
    # or wrong-formula oracles.  Note: HF's Qwen3RMSNorm internally casts to
    # float32 for variance, so we allow a small tolerance (~1e-7 gap).
    with torch.no_grad():
        # q_pre shape: (n_q_heads, L, head_dim) — HF norm expects (batch, seq, dim)
        # so we reshape temporarily.
        q_pre_t = torch.from_numpy(q_pre).to(torch.float64)  # (n_q_heads, L, head_dim)
        k_pre_t = torch.from_numpy(k_pre).to(torch.float64)  # (n_kv_heads, L, head_dim)
        q_post_hf = attn.q_norm(q_pre_t.reshape(-1, q_pre.shape[-1])).reshape(q_pre.shape)
        k_post_hf = attn.k_norm(k_pre_t.reshape(-1, k_pre.shape[-1])).reshape(k_pre.shape)
    q_post_hf = q_post_hf.numpy()
    k_post_hf = k_post_hf.numpy()

    return {
        "q_pre": q_pre,
        "k_pre": k_pre,
        "q_post": q_post,
        "k_post": k_post,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "eps": np.array(eps),
        # HF-anchor values — used in main() for oracle verification only;
        # NOT saved to the fixture (float32-tainted, not suitable as graded oracle).
        "q_post_hf": q_post_hf,
        "k_post_hf": k_post_hf,
    }


def main() -> None:
    FIX.mkdir(exist_ok=True)

    data = _extract_qk_fixtures(seq_len=5)
    # Save only the graded fixture arrays (exclude HF-anchor values used for
    # oracle verification in this script only).
    fixture_keys = {"q_pre", "k_pre", "q_post", "k_post", "q_weight", "k_weight", "eps"}
    np.savez(FIX / "qknorm.npz", **{k: data[k] for k in fixture_keys})

    q_pre = data["q_pre"]
    q_post = data["q_post"]
    print(f"  wrote qknorm.npz")
    print(f"  q_pre  shape={q_pre.shape}  dtype={q_pre.dtype}")
    print(f"  q_post shape={q_post.shape}")
    print(f"  k_pre  shape={data['k_pre'].shape}")
    print(f"  k_post shape={data['k_post'].shape}")
    print(f"  q_weight={data['q_weight']}")
    print(f"  k_weight={data['k_weight']}")
    print(f"  eps={float(data['eps'])}")

    # Sanity check 1: float64 self-consistency (oracle = _rms_norm_f64)
    eps = float(data["eps"])
    rms_q = np.sqrt((q_pre**2).mean(axis=-1, keepdims=True) + eps)
    q_check = (q_pre / rms_q) * data["q_weight"]
    assert np.allclose(q_post, q_check, rtol=1e-12, atol=0), "Q self-consistency check failed"
    rms_k = np.sqrt((data["k_pre"] ** 2).mean(axis=-1, keepdims=True) + eps)
    k_check = (data["k_pre"] / rms_k) * data["k_weight"]
    assert np.allclose(data["k_post"], k_check, rtol=1e-12, atol=0), "K self-consistency check failed"
    print("  sanity check 1 passed (float64 self-consistency for Q and K)")

    # Sanity check 2: HF-anchor — numpy oracle matches genuine Qwen3 q_norm/k_norm
    # (rtol=1e-4/atol=1e-5 tolerates HF's internal float32 variance cast)
    q_post_hf = data["q_post_hf"]
    k_post_hf = data["k_post_hf"]
    np.testing.assert_allclose(
        q_post, q_post_hf, rtol=1e-4, atol=1e-5,
        err_msg="Q oracle diverges from genuine Qwen3 q_norm output",
    )
    np.testing.assert_allclose(
        data["k_post"], k_post_hf, rtol=1e-4, atol=1e-5,
        err_msg="K oracle diverges from genuine Qwen3 k_norm output",
    )
    print("  sanity check 2 passed (HF-anchor: numpy oracle matches Qwen3 q_norm/k_norm)")


if __name__ == "__main__":
    main()
