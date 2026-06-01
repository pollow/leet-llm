"""108 — Special Tokens.

Implement the two functions below. See README.md for the full explanation.
Run `uv run grade 108` to check your work.
"""

from __future__ import annotations

from collections.abc import Iterable


def add_special_tokens(
    ids: list[int], bos_id: int | None = None, eos_id: int | None = None
) -> list[int]:
    """Return ``ids`` with ``bos_id`` prepended and ``eos_id`` appended (each if not None)."""
    raise NotImplementedError(
        "Implement add_special_tokens — see 108_special_tokens/README.md"
    )


def strip_special_tokens(ids: list[int], special_ids: Iterable[int]) -> list[int]:
    """Return ``ids`` with every id in ``special_ids`` removed."""
    raise NotImplementedError(
        "Implement strip_special_tokens — see 108_special_tokens/README.md"
    )
