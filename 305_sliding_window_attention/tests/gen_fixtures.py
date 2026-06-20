"""305 — generate frozen golden fixtures for the sliding-window causal mask.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 305_sliding_window_attention/tests/gen_fixtures.py

Oracle: genuine ``MistralForCausalLM`` from transformers 5.9.0, float64.
The fixture captures the ``(L, L)`` additive mask (0.0 where attended,
-inf where masked) that Mistral's sliding-window attention applies.
We use a forward hook on the first attention layer to intercept the boolean
``attention_mask`` (shape ``(1, 1, L, L)``) that HF passes internally, then
convert it to additive float64 form.

Fixture ``band.npz`` stores: ``mask`` (L,L), ``seq_len``, ``window``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from transformers import MistralConfig, MistralForCausalLM

FIX = pathlib.Path(__file__).parent / "fixtures"


def _extract_hf_band(seq_len: int, window: int) -> np.ndarray:
    """Run a forward pass through a tiny MistralForCausalLM and capture the
    sliding-window attention mask produced internally by HF.

    Returns the ``(seq_len, seq_len)`` additive float64 mask:
      0.0  where attended  (i - window < j <= i)
      -inf where masked    (j > i or j <= i - window)
    """
    cfg = MistralConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        vocab_size=64,
        sliding_window=window,
        torch_dtype=torch.float64,
    )
    torch.manual_seed(42)
    model = MistralForCausalLM(cfg)
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def hook_fn(module, args, kwargs, output):  # noqa: ARG001
        bool_mask = kwargs.get("attention_mask")
        if bool_mask is not None and "mask" not in captured:
            # Shape is (batch=1, 1, L, L); True = attended, False = masked
            captured["mask"] = bool_mask[0, 0].clone()
        return output

    hook = model.model.layers[0].self_attn.register_forward_hook(
        hook_fn, with_kwargs=True
    )
    input_ids = torch.ones(1, seq_len, dtype=torch.long)
    with torch.no_grad():
        model(input_ids)
    hook.remove()

    bool_mask = captured["mask"].numpy()  # (L, L), bool, True=attended
    # Convert to additive float64: 0.0 where attended, -inf where masked
    additive = np.where(bool_mask, 0.0, -np.inf).astype(np.float64)
    return additive


def main() -> None:
    FIX.mkdir(exist_ok=True)

    seq_len = 6
    window = 3

    mask = _extract_hf_band(seq_len, window)

    np.savez(
        FIX / "band.npz",
        mask=mask,
        seq_len=np.array(seq_len),
        window=np.array(window),
    )
    print(f"  wrote band.npz  seq_len={seq_len} window={window}")
    print("  mask:")
    # Print finite entries as 0, -inf as -∞ for readability
    rows = []
    for row in mask:
        rows.append("  " + " ".join("  0" if v == 0.0 else "-inf" for v in row))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
