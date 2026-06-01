"""201 — Embedding.

Implement ``embedding``. See README.md for the full explanation.
Run `uv run grade 201` to check your work.

Hint: you may reuse ``from leet_llm import gather_rows`` (008).
"""

from __future__ import annotations

import numpy as np


def embedding(ids: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Look up rows of ``table`` (V, d) by integer ``ids`` (...), returning (..., d)."""
    raise NotImplementedError("Implement embedding — see 201_embedding/README.md")
