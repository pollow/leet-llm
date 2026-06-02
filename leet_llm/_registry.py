"""The single map from a friendly public name to the task that owns it.

This is what lets later tasks reuse the learner's earlier work:

    from leet_llm import group_last_axis, softmax

resolves ``softmax`` to whatever the learner implemented in ``005_softmax/``
(or the reference, when ``LEET_LLM_TARGET=solution``). Add an entry here the
moment a task introduces a reusable building block.
"""

from __future__ import annotations

# public name -> (task folder, attribute name within that task's module)
REGISTRY: dict[str, tuple[str, str]] = {
    # L0 — NumPy Foundations
    "group_last_axis": ("001_numpy_array_basics", "group_last_axis"),
    "ungroup_last_axis": ("001_numpy_array_basics", "ungroup_last_axis"),
    "add_bias": ("002_broadcasting", "add_bias"),
    "standardize": ("002_broadcasting", "standardize"),
    "affine": ("003_affine", "affine"),
    "batched_matmul": ("004_batched_matmul", "batched_matmul"),
    "outer_product": ("004_batched_matmul", "outer_product"),
    "batched_trace": ("004_batched_matmul", "batched_trace"),
    "softmax": ("005_softmax", "softmax"),
    "logsumexp": ("006_logsumexp", "logsumexp"),
    "log_softmax": ("006_logsumexp", "log_softmax"),
    "top_k": ("007_topk", "top_k"),
    "argmax": ("007_topk", "argmax"),
    "gather_rows": ("008_gather_onehot", "gather_rows"),
    "one_hot": ("008_gather_onehot", "one_hot"),
    "masked_fill": ("009_masking", "masked_fill"),
    "triangular_mask": ("009_masking", "triangular_mask"),
    "sample_categorical": ("010_rng_sampling", "sample_categorical"),
    "interleave": ("011_interleave_halves", "interleave"),
    "deinterleave": ("011_interleave_halves", "deinterleave"),
    "split_halves": ("011_interleave_halves", "split_halves"),
    "join_halves": ("011_interleave_halves", "join_halves"),
    # L1 — Tokenization & Batching
    "build_char_vocab": ("101_char_vocab", "build_char_vocab"),
    "char_encode": ("101_char_vocab", "char_encode"),
    "char_decode": ("101_char_vocab", "char_decode"),
    "text_to_byte_ids": ("102_byte_tokenizer", "text_to_byte_ids"),
    "byte_ids_to_text": ("102_byte_tokenizer", "byte_ids_to_text"),
    "count_pairs": ("103_bpe_step", "count_pairs"),
    "apply_merge": ("103_bpe_step", "apply_merge"),
    "bpe_train": ("104_bpe_train", "bpe_train"),
    "save_tokenizer": ("105_tokenizer_io", "save_tokenizer"),
    "load_tokenizer": ("105_tokenizer_io", "load_tokenizer"),
    "bpe_encode": ("106_bpe_encode", "bpe_encode"),
    "bpe_decode": ("107_bpe_decode", "bpe_decode"),
    "add_special_tokens": ("108_special_tokens", "add_special_tokens"),
    "strip_special_tokens": ("108_special_tokens", "strip_special_tokens"),
    "regex_split": ("109_regex_pretokenize", "regex_split"),
    "tiktoken_encode": ("110_tiktoken_load_encode", "tiktoken_encode"),
    "tiktoken_decode": ("110_tiktoken_load_encode", "tiktoken_decode"),
    "pad_batch": ("111_padding_and_mask", "pad_batch"),
    "padding_mask": ("111_padding_and_mask", "padding_mask"),
    "position_ids": ("112_position_indices", "position_ids"),
    "build_batch": ("113_build_batch", "build_batch"),
    # L2 — Operators & Layers
    "embedding": ("201_embedding", "embedding"),
    "gelu": ("202_activations", "gelu"),
    "silu": ("202_activations", "silu"),
    "layer_norm": ("203_layer_norm", "layer_norm"),
    "sinusoidal_pe": ("204_positional_encoding", "sinusoidal_pe"),
    "sdpa": ("205_scaled_dot_product_attention", "sdpa"),
    "mha": ("206_multi_head_attention", "mha"),
    "AttnParams": ("206_multi_head_attention", "AttnParams"),
    "ffn": ("207_feed_forward", "ffn"),
    "FFNParams": ("207_feed_forward", "FFNParams"),
    "add_residual": ("208_residual_norm", "add_residual"),
    "encoder_block": ("209_encoder_block", "encoder_block"),
    "EncoderBlockParams": ("209_encoder_block", "EncoderBlockParams"),
    "decoder_block": ("210_decoder_block", "decoder_block"),
    "DecoderBlockParams": ("210_decoder_block", "DecoderBlockParams"),
    "gpt_block": ("211_gpt_block", "gpt_block"),
    "GPTBlockParams": ("211_gpt_block", "GPTBlockParams"),
    "rms_norm": ("212_rms_norm", "rms_norm"),
    "rope_interleaved": ("213_rope", "rope_interleaved"),
    "rope_half": ("213_rope", "rope_half"),
    "rope_qk_dot": ("213_rope", "rope_qk_dot"),
    "swiglu_ffn": ("214_swiglu", "swiglu_ffn"),
    "SwiGLUParams": ("214_swiglu", "SwiGLUParams"),
    "gqa": ("215_gqa", "gqa"),
    "llama_decoder_block": ("216_llama_decoder_block", "llama_decoder_block"),
    "LlamaBlockParams": ("216_llama_decoder_block", "LlamaBlockParams"),
    # L3 — Whole-Model & Inference
    "TransformerConfig": ("301_transformer_model", "TransformerConfig"),
    "MarianParams": ("301_transformer_model", "MarianParams"),
    "load_marian": ("301_transformer_model", "load_marian"),
    "encoder": ("301_transformer_model", "encoder"),
    "decoder": ("301_transformer_model", "decoder"),
    "transformer_logits": ("301_transformer_model", "transformer_logits"),
    "translate": ("302_translate", "translate"),
}
