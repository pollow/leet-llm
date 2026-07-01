"""402 — tests for the continuous (iteration-level) batching engine.

Categories:
  A. Correctness — each request's emitted ids (concatenated across ``step``s, keyed
     by req_id) equal its standalone 401 ``kv_generate`` exactly, regardless of how
     it is interleaved with the other concurrently-running requests.
  B. Mechanism (§10.7, fails-on-naive) — observed through the registered API:
       (a) slot reuse   — a queued request begins the step right after a short
           request retires, *while a longer one is still running* (a batch-until-
           all-done scheduler would not admit it until the whole batch drains).
       (b) no wasted compute — a retired request never reappears in a later step.
       (c) iteration-level — every running request advances exactly one token per
           step; each request's emissions are contiguous (it is advanced every step
           while alive), and no step advances more than ``slots`` requests.
  C. Invariants — admitting more requests than free slots queues the overflow;
     ``is_finished`` flips exactly when a request hits the length budget (and, with
     an eos set, exactly when it emits eos).

Every check inspects only ``step()`` outputs / ``is_finished`` — so a NotImplemented
stub cannot pass any of them vacuously.
"""

from __future__ import annotations

import pathlib

import numpy as np

from leet_llm import Qwen3Config, kv_generate, load_qwen3
from leet_llm.grader import load

_m = load(__file__)
Engine = _m.Engine

FIX = pathlib.Path(__file__).parent / "fixtures"
_F = np.load(FIX / "continuous_batching.npz")

N_REQ = int(_F["n_req"])
SLOTS = int(_F["slots"])
MAX_STEPS = 100


def _cfg():
    return Qwen3Config(
        dim=int(_F["dim"]),
        n_layers=int(_F["n_layers"]),
        n_heads=int(_F["n_heads"]),
        n_kv_heads=int(_F["n_kv_heads"]),
        head_dim=int(_F["head_dim"]),
        vocab_size=int(_F["vocab_size"]),
        max_seq_len=int(_F["max_seq_len"]),
        norm_eps=float(_F["norm_eps"]),
        qk_norm_eps=float(_F["qk_norm_eps"]),
        rope_base=float(_F["rope_base"]),
    )


def _params(cfg):
    return load_qwen3({k: _F[k] for k in _F.files}, cfg)


def _prompt(i):
    return _F[f"seq_{i}"][: int(_F[f"prompt_len_{i}"])]


def _run(engine, rids):
    """Drive ``engine`` to completion. Returns:
    emitted[rid] -> list of tokens; steps -> list of per-step [(rid, tok), ...];
    finished_at[rid] -> index of the step after which is_finished(rid) became True.
    """
    emitted: dict[int, list[int]] = {rid: [] for rid in rids}
    steps: list[list[tuple[int, int]]] = []
    finished_at: dict[int, int] = {}
    for s in range(MAX_STEPS):
        out = [(int(r), int(t)) for r, t in engine.step()]
        if not out:
            break
        steps.append(out)
        for rid, tok in out:
            emitted[rid].append(tok)
        for rid in rids:
            if rid not in finished_at and engine.is_finished(rid):
                finished_at[rid] = s
        if all(engine.is_finished(rid) for rid in rids):
            break
    return emitted, steps, finished_at


# ---------------------------------------------------------------------------
# A. Correctness
# ---------------------------------------------------------------------------


def test_each_request_matches_standalone_kv_generate():
    """Concurrent interleaving must not change any request's token stream: each
    request's emitted ids equal its standalone 401 ``kv_generate`` output."""
    cfg = _cfg()
    params = _params(cfg)
    engine = Engine(params, cfg)
    rids = [engine.add_request(_prompt(i)) for i in range(N_REQ)]

    emitted, _, _ = _run(engine, rids)

    for i, rid in enumerate(rids):
        n_new = int(_F[f"n_new_{i}"])
        ref = kv_generate(_prompt(i)[None, :], params, cfg, n_new=n_new)
        expected_gen = list(ref[int(_F[f"prompt_len_{i}"]):])
        assert emitted[rid] == expected_gen, (
            f"request {i} diverged from its standalone kv_generate under batching"
        )
        # And equal to the frozen oracle sequence.
        assert emitted[rid] == _F[f"seq_{i}"][int(_F[f"prompt_len_{i}"]):].tolist()


# ---------------------------------------------------------------------------
# B. Mechanism (fails-on-naive)
# ---------------------------------------------------------------------------


def test_slot_reuse_admits_before_longer_request_finishes():
    """With more requests than slots, the queued request must begin the step right
    after a *short* request retires — while a *longer* concurrent request is still
    running. A batch-until-all-done scheduler admits it only after the whole initial
    batch drains (by which point the longer request has already finished), so it
    fails this assertion.
    """
    cfg = _cfg()
    params = _params(cfg)
    engine = Engine(params, cfg)
    rids = [engine.add_request(_prompt(i)) for i in range(N_REQ)]
    assert N_REQ > SLOTS, "fixture must have more requests than slots"

    emitted, steps, _ = _run(engine, rids)

    queued = rids[SLOTS]        # first request that had to wait (req index 2)
    long_running = rids[0]      # an initially-admitted, longer request (index 0)

    def first_step(rid):
        return next(s for s, out in enumerate(steps) if any(r == rid for r, _ in out))

    def last_step(rid):
        return max(s for s, out in enumerate(steps) if any(r == rid for r, _ in out))

    queued_start = first_step(queued)

    # The queued request only starts once a slot frees — never in the very first
    # step (all slots were taken by the initial batch).
    assert queued_start > 0, "queued request must wait for a slot before starting"

    # The discriminator vs a batch-until-all-done scheduler: the longer initial
    # request is STILL RUNNING when the queued one is admitted (it emits at or after
    # queued_start). A static scheduler would only admit the queued request after
    # the whole initial batch drained, i.e. after the longer request had finished.
    assert last_step(long_running) >= queued_start, (
        "queued request must be admitted while a longer request is still running "
        "(a batch-until-all-done scheduler waits for the whole batch to drain)"
    )


def test_retired_request_never_reappears():
    """No wasted compute: once a request retires it is never advanced again — its
    req_id must not appear in any later step's output."""
    cfg = _cfg()
    params = _params(cfg)
    engine = Engine(params, cfg)
    rids = [engine.add_request(_prompt(i)) for i in range(N_REQ)]

    _, steps, _ = _run(engine, rids)

    for rid in rids:
        appearances = [s for s, out in enumerate(steps) if any(r == rid for r, _ in out)]
        if appearances:
            last = max(appearances)
            for s in range(last + 1, len(steps)):
                assert all(r != rid for r, _ in steps[s]), (
                    f"retired request {rid} reappeared at step {s} (wasted compute)"
                )


def test_iteration_level_one_token_per_running_request():
    """Every step advances each running request by exactly one token: no req_id
    appears twice in a step, at most ``slots`` requests advance per step, and each
    request's emissions are contiguous (it is advanced every step while alive)."""
    cfg = _cfg()
    params = _params(cfg)
    engine = Engine(params, cfg)
    rids = [engine.add_request(_prompt(i)) for i in range(N_REQ)]

    _, steps, _ = _run(engine, rids)

    for s, out in enumerate(steps):
        ids = [r for r, _ in out]
        assert len(ids) == len(set(ids)), f"a request advanced twice in step {s}"
        assert len(ids) <= SLOTS, f"more than {SLOTS} requests advanced in step {s}"

    # Contiguity: each request appears in an unbroken run of steps (advanced every
    # step while alive — no stalling mid-flight).
    for rid in rids:
        present = [s for s, out in enumerate(steps) if any(r == rid for r, _ in out)]
        assert present, f"request {rid} never ran"
        assert present == list(range(present[0], present[-1] + 1)), (
            f"request {rid} was skipped mid-flight (not advanced every step while alive)"
        )


# ---------------------------------------------------------------------------
# C. Invariants
# ---------------------------------------------------------------------------


def test_overflow_is_queued_not_run():
    """Admitting more requests than free slots must queue the overflow: the first
    step advances exactly ``slots`` requests and the queued one has not started."""
    cfg = _cfg()
    params = _params(cfg)
    engine = Engine(params, cfg)
    rids = [engine.add_request(_prompt(i)) for i in range(N_REQ)]

    first = [(int(r), int(t)) for r, t in engine.step()]
    started = {r for r, _ in first}
    assert len(started) == SLOTS, f"first step should run exactly {SLOTS} requests"
    assert rids[SLOTS] not in started, "the overflow request must be queued, not run"
    assert engine.is_finished(rids[SLOTS]) is False


def test_is_finished_flips_exactly_at_length_budget():
    """Run a single request alone; is_finished is False after every step until the
    step whose emission reaches the length budget, then True — never before."""
    cfg = _cfg()
    params = _params(cfg)
    engine = Engine(params, cfg)
    rid = engine.add_request(_prompt(0))
    n_new = int(_F["n_new_0"])

    for gen in range(1, n_new + 1):
        engine.step()
        if gen < n_new:
            assert engine.is_finished(rid) is False, (
                f"request retired early (after {gen} of {n_new} tokens)"
            )
        else:
            assert engine.is_finished(rid) is True, "request did not retire at budget"


def test_is_finished_flips_exactly_at_eos():
    """With an eos set, a request retires exactly when it emits that token — not
    before, and it emits it as its final token."""
    cfg = _cfg()
    params = _params(cfg)
    eos = int(_F["eos_probe_id"])
    n_expected = int(_F["eos_probe_nnew"])
    engine = Engine(params, cfg, eos_token_id=eos)
    rid = engine.add_request(_prompt(0))

    emitted = []
    for _ in range(MAX_STEPS):
        out = engine.step()
        if not out:
            break
        for r, t in out:
            if r == rid:
                emitted.append(int(t))
        if engine.is_finished(rid):
            break

    assert len(emitted) == n_expected, "eos request retired at the wrong step"
    assert emitted[-1] == eos, "the final emitted token must be the eos token"
    assert engine.is_finished(rid) is True
