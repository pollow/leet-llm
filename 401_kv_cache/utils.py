"""Interactive KV-cache generation demo for 401 (skippable Tier-C payoff).

Reuses 306's weight/tokenizer loaders (shared serving infra) but drives the
*cache-based* ``kv_generate`` — so the learner watches their own ``KVCache``
generate real Qwen3-0.6B text, token-equal to the stateless 306/304 run but
produced incrementally. NOT a grading path: needs downloaded real weights
(see ``download.sh``); the graded tests are hermetic on the frozen fixture.

This file is named ``utils.py`` on purpose — the task loader excludes
``utils.py`` from student-stub detection (same convention as 306), so it never
collides with the single-stub rule.
"""

from __future__ import annotations

import importlib.util
import pathlib
from typing import Any, Callable

import numpy as np

# 306's utils.py already implements the weight/tokenizer/arg plumbing. Import it
# by file path under a distinct module name — a plain ``import utils`` would
# resolve to THIS module (401's), not 306's, whenever 401 is the script dir.
_QWEN3_UTILS = pathlib.Path(__file__).resolve().parent.parent / "306_qk_norm" / "utils.py"
_spec = importlib.util.spec_from_file_location("_qwen3_cli_utils_306", _QWEN3_UTILS)
_u = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_u)  # type: ignore[union-attr]


def run_kv_generate_cli(
    *,
    module_name: str,
    load_fn: Callable[[dict[str, np.ndarray], Any], Any],
    kv_generate_fn: Callable[..., list[int]],
    config_cls: Callable[..., Any],
) -> None:
    """Prompt → cache-based greedy generation → decoded text.

    ``kv_generate_fn(input_ids, params, cfg, n_new=...) -> list[int]`` runs the
    learner's prefill + incremental decode loop over a ``KVCache``.
    """
    parser = _u._build_parser(f"KV-cache generation via {module_name}")
    args = parser.parse_args()

    prompt = args.prompt if args.prompt is not None else input("Prompt> ").strip()
    if not prompt:
        raise ValueError("Prompt is empty. Pass --prompt or provide interactive input.")

    weights = _u.load_local_weights(args.weights)
    cfg = _u.build_qwen3_config(weights, config_cls, model_name=args.model_name)
    params = load_fn(weights, cfg)

    tokenizer = _u.load_tokenizer(args.model_name)
    input_ids = tokenizer(prompt, return_tensors="np").input_ids.astype(np.int32)
    if input_ids.shape[1] == 0:
        raise ValueError("Tokenizer produced an empty prompt.")

    out_ids = kv_generate_fn(input_ids, params, cfg, n_new=args.max_new_tokens)
    print(tokenizer.decode(out_ids, skip_special_tokens=True))
