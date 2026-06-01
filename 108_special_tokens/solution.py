"""108 — Special Tokens — reference solution."""

from __future__ import annotations

from collections.abc import Iterable


def add_special_tokens(
    ids: list[int], bos_id: int | None = None, eos_id: int | None = None
) -> list[int]:
    out = list(ids)
    if bos_id is not None:
        out.insert(0, bos_id)
    if eos_id is not None:
        out.append(eos_id)
    return out


def strip_special_tokens(ids: list[int], special_ids: Iterable[int]) -> list[int]:
    specials = set(special_ids)
    return [i for i in ids if i not in specials]
