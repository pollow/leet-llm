"""402 — Continuous (iteration-level) batching engine (Qwen3) — REFERENCE SOLUTION.

Wraps 401's single-sequence ``KVCache`` + ``prefill`` / ``decode_step`` into a
multi-request serving engine that mirrors **Orca iteration-level scheduling**
(= vLLM continuous batching): a *ragged* running set advanced **one token per
``step()``**, with finished requests **retired mid-batch** and waiting requests
**admitted onto the freed slots** — no head-of-line blocking.

Design (the learner's job; stated here for the reference):

- Each live request owns exactly one 401 ``KVCache`` (independent K/V timeline).
- ``add_request`` enqueues a request; it does not run until a slot is free.
- ``step`` (one scheduler iteration): admit waiting requests onto slots freed by
  the *previous* step (``prefill`` on admission → first token), then advance every
  already-running request by one ``decode_step``. Each running request emits exactly
  one token this step. A request that emits ``eos`` or reaches the length budget
  (``cfg.max_seq_len``) is retired at the end of the step; its slot is filled on the
  next step.

Reuse (do NOT re-inline): ``KVCache`` / ``prefill`` / ``decode_step`` from 401.
Greedy argmax uses ``np.argmax`` so the token stream is byte-identical to 401's
``kv_generate`` (same tie-breaking).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("LEET_LLM_TARGET", "solution")

from leet_llm import KVCache, decode_step, prefill

__all__ = ["Engine", "MAX_CONCURRENT_REQUESTS"]

# GIVEN systems fact: the number of KV-cache slots (max requests running at once).
# Small so a 3-request fixture exercises the waiting queue + slot reuse. Production
# engines size this from available KV memory.
MAX_CONCURRENT_REQUESTS = 2


@dataclass
class _Req:
    """Per-request scheduler state (one 401 cache, GQA-shaped)."""

    prompt_len: int
    cache: KVCache
    last: int          # the token most recently emitted (fed to the next decode)
    gen: int           # number of tokens generated so far


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
        self.params = params
        self.cfg = cfg
        self.eos_token_id = eos_token_id
        self.slots = MAX_CONCURRENT_REQUESTS
        self.max_len = cfg.max_seq_len            # length budget (context cap)
        self._next_id = 0
        self._waiting: list[tuple[int, np.ndarray]] = []   # (req_id, prompt_ids)
        self._running: dict[int, _Req] = {}
        self._finished: dict[int, bool] = {}

    # -- public API ---------------------------------------------------------

    def add_request(self, prompt_ids) -> int:
        rid = self._next_id
        self._next_id += 1
        self._waiting.append((rid, np.asarray(prompt_ids).reshape(-1)))
        self._finished[rid] = False
        return rid

    def is_finished(self, req_id: int) -> bool:
        return self._finished.get(req_id, False)

    def step(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []

        # (1) admit waiting requests onto slots freed by earlier steps — a newly
        #     admitted request is prefilled and emits its first token this step.
        while self._waiting and len(self._running) < self.slots:
            rid, prompt = self._waiting.pop(0)
            cache = KVCache(self.cfg)
            logits = prefill(prompt[None, :], self.params, self.cfg, cache)
            tok = int(np.argmax(logits[0]))
            self._running[rid] = _Req(
                prompt_len=int(prompt.shape[0]), cache=cache, last=tok, gen=1
            )
            out.append((rid, tok))
            self._retire_if_done(rid, tok)

        # (2) advance every request that was already running *before* this step
        #     (i.e. not one just admitted above) by exactly one decode token.
        admitted = {rid for rid, _ in out}
        for rid in list(self._running):
            if rid in admitted:
                continue
            req = self._running[rid]
            logits = decode_step(req.last, self.params, self.cfg, req.cache)
            tok = int(np.argmax(logits[0]))
            req.last = tok
            req.gen += 1
            out.append((rid, tok))
            self._retire_if_done(rid, tok)

        return out

    # -- internals ----------------------------------------------------------

    def _retire_if_done(self, rid: int, tok: int) -> None:
        """Retire ``rid`` if it emitted eos or hit the length budget."""
        req = self._running[rid]
        hit_eos = self.eos_token_id is not None and tok == self.eos_token_id
        hit_budget = req.prompt_len + req.gen >= self.max_len
        if hit_eos or hit_budget:
            self._finished[rid] = True
            del self._running[rid]
