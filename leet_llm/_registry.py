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
}
