"""304_generate benchmark — greedy story-generation baseline for future optimization.

Usage after fetching weights:
    bash 304_generate/download.sh                       # writes stories15M.model.npz + tokenizer.model
    LEET_LLM_TARGET=solution uv run --group gen python 304_generate/tools/benchmark.py --limit 20

This drives generate() in stateless-recompute mode (no KV-cache — that is L4), so the cost
is quadratic in sequence length. The KV-cache speedup measured against this baseline is the
whole point of L4. Captures wall time, per-story latency percentiles, and tokens/sec, and
prints the first few decoded stories so you can watch your own NumPy model write.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).parent.parent
DEFAULT_WEIGHTS = HERE / "stories15M.model.npz"
DEFAULT_REF = HERE / "tests" / "fixtures" / "real_ref.npz"
DEFAULT_TOKENIZER = HERE / "tokenizer.model"
DEFAULT_DATASET = HERE / "benchmark_data" / "prompts.txt"
MODEL_NAME = "stories15M (Xenova/llama2.c-stories15M)"


def load_cfg_and_params():
    from leet_llm import LlamaConfig, load_llama

    if not DEFAULT_WEIGHTS.exists():
        raise SystemExit(f"Missing {DEFAULT_WEIGHTS}. Run: bash 304_generate/download.sh")
    if not DEFAULT_REF.exists():
        raise SystemExit(f"Missing {DEFAULT_REF}. Run: uv run --group gen python 304_generate/convert.py")

    R = np.load(DEFAULT_REF)
    cfg = LlamaConfig(
        dim=int(R["dim"]), n_layers=int(R["n_layers"]), n_heads=int(R["n_heads"]),
        n_kv_heads=int(R["n_kv_heads"]), vocab_size=int(R["vocab_size"]),
        max_seq_len=int(R["max_seq_len"]), norm_eps=float(R["norm_eps"]),
        rope_base=float(R["rope_base"]),
    )
    W = np.load(DEFAULT_WEIGHTS)
    params = load_llama({k: W[k] for k in W.files}, cfg)
    return cfg, params


def load_tokenizer():
    import sentencepiece as spm

    if not DEFAULT_TOKENIZER.exists():
        raise SystemExit(f"Missing {DEFAULT_TOKENIZER}. Run: bash 304_generate/download.sh")
    return spm.SentencePieceProcessor(model_file=str(DEFAULT_TOKENIZER))


def load_dataset(path: pathlib.Path, limit: int) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Prompt file not found at {path}.")
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines[:limit]


def main():
    ap = argparse.ArgumentParser(description="304_generate greedy story-generation benchmark")
    ap.add_argument("--limit", type=int, default=20, help="number of prompts to generate from")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy/deterministic baseline (llama3.np uses 0.8)")
    ap.add_argument("--dataset", type=pathlib.Path, default=DEFAULT_DATASET)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--output", type=pathlib.Path, default=HERE / "benchmark_baseline.json")
    ap.add_argument("--print-samples", type=int, default=5,
                    help="number of first prompts to decode and print as full stories")
    ap.add_argument("--samples-output", type=pathlib.Path, default=HERE / "benchmark_samples.jsonl")
    args = ap.parse_args()

    # lazy import to respect LEET_LLM_TARGET env
    from leet_llm import generate

    print(f"[benchmark] loading tokenizer {DEFAULT_TOKENIZER.name} ...")
    sp = load_tokenizer()
    print("[benchmark] loading cfg + params ...")
    cfg, params = load_cfg_and_params()
    prompts = load_dataset(args.dataset, args.limit)
    eos_id = sp.eos_id()
    print(f"[benchmark] {len(prompts)} prompts from {args.dataset}, temperature={args.temperature}")

    def encode(text: str) -> np.ndarray:
        return np.array([[sp.bos_id()] + sp.encode(text)])

    if args.warmup > 0:
        _ = generate(encode(prompts[0]), params, cfg,
                     max_new_tokens=args.max_new_tokens, temperature=args.temperature, eos_id=eos_id)

    total_prompt_tokens = 0
    total_gen_tokens = 0
    latencies = []
    samples = []

    start = time.perf_counter()
    for i, text in enumerate(prompts, 1):
        ids = encode(text)
        total_prompt_tokens += int(ids.shape[1])

        t0 = time.perf_counter()
        out = generate(ids, params, cfg, max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature, eos_id=eos_id)
        t1 = time.perf_counter()

        latencies.append(t1 - t0)
        total_gen_tokens += len(out) - int(ids.shape[1])

        if len(samples) < args.print_samples:
            try:
                story = sp.decode(out)
            except Exception:
                story = "<decode error>"
            samples.append({
                "prompt": text, "story": story,
                "prompt_ids_len": int(ids.shape[1]), "total_ids_len": len(out),
                "latency_ms": round((t1 - t0) * 1000, 2),
            })

        if i % 5 == 0 or i == len(prompts):
            print(f"  {i}/{len(prompts)}  avg_latency={(sum(latencies) / len(latencies)) * 1000:.1f} ms")

    wall = time.perf_counter() - start
    lat = np.array(latencies)
    metrics = {
        "num_samples": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "wall_time_s": round(wall, 3),
        "avg_latency_ms": round(float(lat.mean() * 1000), 2),
        "p50_latency_ms": round(float(np.percentile(lat, 50) * 1000), 2),
        "p90_latency_ms": round(float(np.percentile(lat, 90) * 1000), 2),
        "p99_latency_ms": round(float(np.percentile(lat, 99) * 1000), 2),
        "stories_per_sec": round(len(prompts) / wall, 3),
        "prompt_tokens": total_prompt_tokens,
        "gen_tokens": total_gen_tokens,
        "gen_tokens_per_sec": round(total_gen_tokens / wall, 2),
        "dataset": str(args.dataset),
        "model": MODEL_NAME,
    }

    print("\n=== 304_generate baseline ===")
    print(json.dumps(metrics, indent=2))
    args.output.write_text(json.dumps(metrics, indent=2))
    print(f"saved -> {args.output}")

    if samples:
        print(f"\n=== Sample stories (first {len(samples)}) ===")
        for idx, s in enumerate(samples, 1):
            print(f"[{idx}] ({s['total_ids_len']} tok, {s['latency_ms']} ms) {s['story']}")
        try:
            with open(args.samples_output, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print(f"saved samples -> {args.samples_output}")
        except Exception as e:
            print(f"warn: could not save samples: {e}")


if __name__ == "__main__":
    main()
