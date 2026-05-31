"""leet-llm — the facade package.

Import the building blocks you've implemented as if they were one library::

    from leet_llm import group_last_axis, softmax

Each name is resolved lazily through the registry (``_registry.py``) to the task
that owns it, loading the learner's stub by default or the reference solution
when ``LEET_LLM_TARGET=solution``. Names whose task isn't implemented yet raise
a clear, actionable error.
"""

from __future__ import annotations

from ._loader import load_task
from ._registry import REGISTRY

__all__ = sorted(REGISTRY)


def __getattr__(name: str):
    try:
        folder, attr = REGISTRY[name]
    except KeyError:
        raise AttributeError(f"module 'leet_llm' has no attribute {name!r}") from None

    try:
        module = load_task(folder)
    except FileNotFoundError as exc:
        raise ImportError(
            f"'{name}' is provided by task {folder}, which doesn't exist yet."
        ) from exc

    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(
            f"'{name}' is provided by task {folder}, but it isn't implemented yet — "
            f"implement {folder}/ first."
        ) from exc


def __dir__() -> list[str]:
    return __all__
