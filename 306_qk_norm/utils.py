from __future__ import annotations

import argparse
import pathlib
import re
from typing import Any, Callable

import numpy as np
from leet_llm import sample


def _default_weights_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "qwen3_0_6b.npz"


def _build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt text.")
    parser.add_argument(
        "--weights",
        type=pathlib.Path,
        default=_default_weights_path(),
        help="Path to local qwen3_0_6b.npz weights.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HF model id for tokenizer/config lookup.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eos-id", type=int, default=None)
    return parser


def load_local_weights(weights_path: pathlib.Path) -> dict[str, np.ndarray]:
    if not weights_path.exists():
        raise FileNotFoundError(
            f"weights not found: {weights_path}. Run: bash 306_qk_norm/download.sh"
        )
    with np.load(weights_path) as data:
        return {key: data[key] for key in data.files}


def _infer_layer_count(weights: dict[str, np.ndarray]) -> int:
    patt = re.compile(r"^model\.layers\.(\d+)\.")
    layer_ids: set[int] = set()
    for key in weights:
        m = patt.match(key)
        if m:
            layer_ids.add(int(m.group(1)))
    if not layer_ids:
        raise ValueError("Could not infer n_layers from weight keys.")
    return max(layer_ids) + 1


def build_qwen3_config(
    weights: dict[str, np.ndarray],
    config_cls: Callable[..., Any],
    model_name: str = "Qwen/Qwen3-0.6B",
) -> Any:
    tok_embed = weights["model.embed_tokens.weight"]
    dim = int(tok_embed.shape[1])
    vocab_size = int(tok_embed.shape[0])
    n_layers = _infer_layer_count(weights)
    head_dim = int(weights["model.layers.0.self_attn.q_norm.weight"].shape[0])
    n_heads = int(weights["model.layers.0.self_attn.q_proj.weight"].shape[0] // head_dim)
    n_kv_heads = int(weights["model.layers.0.self_attn.k_proj.weight"].shape[0] // head_dim)

    max_seq_len = 4096
    norm_eps = 1e-6
    qk_norm_eps = 1e-6
    rope_base = 10000.0

    try:
        from transformers import AutoConfig

        try:
            hf_cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
        except Exception:
            hf_cfg = AutoConfig.from_pretrained(model_name)
        max_seq_len = int(getattr(hf_cfg, "max_position_embeddings", max_seq_len))
        norm_eps = float(getattr(hf_cfg, "rms_norm_eps", norm_eps))
        qk_norm_eps = norm_eps
        rope_base = float(getattr(hf_cfg, "rope_theta", rope_base))
    except Exception:
        pass

    return config_cls(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        norm_eps=norm_eps,
        qk_norm_eps=qk_norm_eps,
        rope_base=rope_base,
    )


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def generate_ids(
    input_ids: np.ndarray,
    forward_fn: Callable[[np.ndarray, Any, Any], np.ndarray],
    params: Any,
    cfg: Any,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_id: int | None,
    seed: int,
    on_token: Callable[[int], None] | None = None,
) -> list[int]:
    rng = np.random.default_rng(seed)
    ids = input_ids[0].tolist()

    for _ in range(max_new_tokens):
        logits = forward_fn(np.array([ids], dtype=np.int32), params, cfg)
        next_id = int(
            sample(
                logits[0, -1],
                rng,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        )
        ids.append(next_id)
        if on_token is not None:
            on_token(next_id)
        if eos_id is not None and next_id == eos_id:
            break

    return ids


def run_qwen3_cli(
    *,
    module_name: str,
    load_fn: Callable[[dict[str, np.ndarray], Any], Any],
    forward_fn: Callable[[np.ndarray, Any, Any], np.ndarray],
    config_cls: Callable[..., Any],
) -> None:
    parser = _build_parser(f"Interactive generation via {module_name}")
    args = parser.parse_args()

    prompt = args.prompt
    if prompt is None:
        prompt = input("Prompt> ").strip()
    if not prompt:
        raise ValueError("Prompt is empty. Pass --prompt or provide interactive input.")

    weights = load_local_weights(args.weights)
    cfg = build_qwen3_config(weights, config_cls, model_name=args.model_name)
    params = load_fn(weights, cfg)

    tokenizer = load_tokenizer(args.model_name)
    input_ids = tokenizer(prompt, return_tensors="np").input_ids.astype(np.int32)
    if input_ids.shape[1] == 0:
        raise ValueError(
            "Tokenizer produced an empty prompt. Ensure tokenizer assets are available "
            f"for {args.model_name} and retry."
        )
    eos_id = args.eos_id if args.eos_id is not None else tokenizer.eos_token_id

    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(prompt_text, end="", flush=True)

    def _on_token(token_id: int) -> None:
        piece = tokenizer.decode([token_id], skip_special_tokens=True)
        if piece:
            print(piece, end="", flush=True)

    out_ids = generate_ids(
        input_ids,
        forward_fn,
        params,
        cfg,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_id=eos_id,
        seed=args.seed,
        on_token=_on_token,
    )
    _ = out_ids
    print()
