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
}
