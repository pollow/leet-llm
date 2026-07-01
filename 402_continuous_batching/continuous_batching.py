"""402 — Continuous (iteration-level) batching engine (Qwen3).

Wraps 401's single-sequence ``KVCache`` + ``prefill`` / ``decode_step`` into a
multi-request serving engine that mirrors **Orca iteration-level scheduling**
(= vLLM continuous batching): a *ragged* running set advanced **one token per
``step()``**, with finished requests **retired mid-batch** and waiting requests
**admitted onto the freed slots** — no head-of-line blocking.

Registered surface (see README.md):

- ``Engine``  — holds one 401 ``KVCache`` per live request; a running set (up to
                ``MAX_CONCURRENT_REQUESTS`` slots) plus a waiting queue.
    - ``add_request(prompt_ids) -> req_id`` — enqueue a request; returns its id.
    - ``step() -> list[(req_id, token_id)]`` — one scheduler iteration: exactly one
      token per currently-running request; admits waiting requests onto free slots
      (``prefill`` on admission); retires on eos / length budget.
    - ``is_finished(req_id) -> bool`` — whether that request has retired.

Run ``uv run grade 402`` to check your work.

Reuse (do NOT re-inline): ``KVCache`` / ``prefill`` / ``decode_step`` from 401 — the
engine *orchestrates* them, it does not re-author any forward math. Use ``np.argmax``
for the greedy next token so your stream matches 401's ``kv_generate``.
"""

from __future__ import annotations

import numpy as np

from leet_llm import KVCache, decode_step, prefill  # noqa: F401  (the primitives to drive)

__all__ = ["Engine", "MAX_CONCURRENT_REQUESTS"]

# GIVEN systems fact: the number of KV-cache slots (max requests running at once).
# Small so a 3-request scenario exercises the waiting queue + slot reuse. Production
# engines size this from available KV memory.
MAX_CONCURRENT_REQUESTS = 2


class Engine:
    """Iteration-level continuous-batching scheduler over 401's KV cache.

    - ``Engine(params, cfg)`` — build a scheduler for one model. ``eos_token_id`` is
      an optional retirement trigger (defaults to the length budget only).
    - ``add_request(prompt_ids) -> req_id`` — enqueue a request; returns its id.
    - ``step() -> list[(req_id, token_id)]`` — one scheduler iteration: exactly one
      token per currently-running request; admits waiting requests onto free slots.
    - ``is_finished(req_id) -> bool`` — whether that request has retired.
    """

    def __init__(self, params, cfg, eos_token_id: int | None = None) -> None:
        raise NotImplementedError

    def add_request(self, prompt_ids) -> int:
        raise NotImplementedError

    def is_finished(self, req_id: int) -> bool:
        raise NotImplementedError

    def step(self) -> list[tuple[int, int]]:
        raise NotImplementedError
