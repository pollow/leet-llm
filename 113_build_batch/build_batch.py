"""113 — Build Batch.

Implement ``build_batch``. See README.md for the full explanation.
Run `uv run grade 113` to check your work.

Hint: compose your earlier work —
``from leet_llm import pad_batch, padding_mask, position_ids``.
"""

from __future__ import annotations

import numpy as np


def build_batch(
    seqs: list[list[int]], pad_id: int = 0, max_len: int | None = None
) -> dict[str, np.ndarray]:
    """Assemble ``{"input_ids", "pad_mask", "position_ids"}``, each a ``(B, L)`` array."""
    raise NotImplementedError("Implement build_batch — see 113_build_batch/README.md")
