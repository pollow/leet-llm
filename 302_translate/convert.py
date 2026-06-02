"""302 — convert the real opus-mt-en-zh checkpoint to our .npz (AUTHORING/DEMO, gen group).

    uv run --group gen python 302_translate/convert.py

Writes opus_mt_en_zh.npz (full HF state_dict, float64) next to this file, plus a small
committed reference fixture tests/fixtures/real_ref.npz holding the HF greedy ids for a
fixed English prompt. The big .npz is git-ignored; only real_ref.npz is committed.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer

HERE = pathlib.Path(__file__).parent
NAME = "Helsinki-NLP/opus-mt-en-zh"
PROMPT = "I have a dream that one day this nation will rise up."


def main() -> None:
    tok = MarianTokenizer.from_pretrained(NAME)
    model = MarianMTModel.from_pretrained(NAME).double().eval()
    arrays = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    np.savez(HERE / "opus_mt_en_zh.npz", **arrays)

    enc = tok([PROMPT], return_tensors="pt")
    with torch.no_grad():
        gen = model.generate(**enc, num_beams=1, do_sample=False, max_length=64)
    cfg = model.config
    (HERE / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    np.savez(HERE / "tests" / "fixtures" / "real_ref.npz",
             src_ids=enc["input_ids"].numpy(), expected_ids=gen.numpy(),
             d_model=np.array(cfg.d_model), n_heads=np.array(cfg.decoder_attention_heads),
             n_enc_layers=np.array(cfg.encoder_layers), n_dec_layers=np.array(cfg.decoder_layers),
             d_ff=np.array(cfg.decoder_ffn_dim), vocab_size=np.array(cfg.vocab_size),
             max_pos=np.array(cfg.max_position_embeddings),
             scale_embedding=np.array(bool(cfg.scale_embedding)),
             pad_id=np.array(cfg.pad_token_id), eos_id=np.array(cfg.eos_token_id),
             decoder_start_id=np.array(cfg.decoder_start_token_id),
             activation=np.array(cfg.activation_function))
    print("translation:", tok.decode(gen[0], skip_special_tokens=True))
    print("wrote opus_mt_en_zh.npz (gitignored) + tests/fixtures/real_ref.npz")


if __name__ == "__main__":
    main()
