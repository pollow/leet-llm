"""302 — frozen greedy-decode goldens from the tiny genuine MarianMTModel.

AUTHORING ONLY (gen group):
    uv run --group gen python 302_translate/tests/gen_fixtures.py

Same tiny config + seed as task 301 so the committed weights match. We capture HF's
greedy ``generate`` (num_beams=1, do_sample=False) output ids as the token-sequence oracle.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from transformers import MarianConfig, MarianMTModel

FIX = pathlib.Path(__file__).parent / "fixtures"


def main() -> None:
    FIX.mkdir(exist_ok=True)
    for old in FIX.glob("*.npz"):
        old.unlink()
    torch.manual_seed(0)
    cfg = MarianConfig(
        vocab_size=64, decoder_vocab_size=64, d_model=16,
        encoder_layers=2, decoder_layers=2,
        encoder_attention_heads=4, decoder_attention_heads=4,
        encoder_ffn_dim=32, decoder_ffn_dim=32,
        max_position_embeddings=32, activation_function="gelu",
        scale_embedding=True, share_encoder_decoder_embeddings=True,
        pad_token_id=63, eos_token_id=0, bos_token_id=63,
        decoder_start_token_id=63, forced_eos_token_id=None,  # None ⇒ HF greedy == pure argmax
    )
    model = MarianMTModel(cfg).double().eval()
    src = np.array([[5, 6, 7, 8, 0]])
    with torch.no_grad():
        gen = model.generate(torch.tensor(src), max_length=12, num_beams=1, do_sample=False)
    arrays = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    arrays.update(src_ids=src, expected_ids=gen.numpy(),
                  d_model=np.array(16), n_heads=np.array(4),
                  n_enc_layers=np.array(2), n_dec_layers=np.array(2),
                  d_ff=np.array(32), vocab_size=np.array(64), max_pos=np.array(32),
                  scale_embedding=np.array(True), pad_id=np.array(63), eos_id=np.array(0),
                  decoder_start_id=np.array(63))
    np.savez(FIX / "tiny_greedy.npz", **arrays)
    print("  wrote tiny_greedy.npz  expected_ids", gen.numpy().tolist())


if __name__ == "__main__":
    main()
