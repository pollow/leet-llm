"""301 — frozen goldens from a tiny genuine HuggingFace MarianMTModel.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 301_transformer_model/tests/gen_fixtures.py

A tiny MarianMTModel (random init, float64) is the oracle. We dump its full state_dict
(HF names), the input ids, and three goldens: encoder output, decoder output, final logits.
``scale_embedding=True`` and ``decoder_attention_heads=4`` exercise the embed-scale and
multi-head wrinkles; ``encoder_layers=2`` localizes block-stacking bugs.
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
        decoder_start_token_id=63, forced_eos_token_id=0,
    )
    model = MarianMTModel(cfg).double().eval()

    src = np.array([[5, 6, 7, 8, 0]])                 # ends with eos
    tgt = np.array([[63, 9, 10, 11]])                 # starts with decoder_start
    with torch.no_grad():
        out = model(input_ids=torch.tensor(src), decoder_input_ids=torch.tensor(tgt),
                    output_hidden_states=True)

    arrays = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    arrays.update(
        src_ids=src, tgt_ids=tgt,
        enc_out=out.encoder_hidden_states[-1].numpy(),
        dec_out=out.decoder_hidden_states[-1].numpy(),
        logits=out.logits.numpy(),
        # config scalars for the test to rebuild TransformerConfig
        d_model=np.array(16), n_heads=np.array(4),
        n_enc_layers=np.array(2), n_dec_layers=np.array(2),
        d_ff=np.array(32), vocab_size=np.array(64), max_pos=np.array(32),
        scale_embedding=np.array(True), pad_id=np.array(63), eos_id=np.array(0),
        decoder_start_id=np.array(63),
    )
    np.savez(FIX / "tiny_marian.npz", **arrays)
    print(f"  wrote tiny_marian.npz  ({len(arrays)} arrays)  logits{out.logits.shape}")


if __name__ == "__main__":
    main()
