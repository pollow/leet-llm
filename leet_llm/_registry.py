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
}
