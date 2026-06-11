"""302_translate benchmark — greedy decode baseline for future optimization.

Usage after downloading weights:
    uv sync --group gen
    uv run --group gen python 302_translate/convert.py   # writes opus_mt_en_zh.npz
    uv run --group gen python 302_translate/tools/download_dataset.py   # writes 1k sentences
    LEET_LLM_TARGET=solution uv run --group gen python 302_translate/tools/benchmark.py --batch-size 1 --limit 1000

Takes --batch-size as param (current translate() is single-sample stateless recompute,
so batch-size is reserved for future batched API; loop still processes one by one).
Captures wall time, latency, throughput, tokens/sec as baseline.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).parent.parent
DEFAULT_WEIGHTS = HERE / "opus_mt_en_zh.npz"
DEFAULT_REF = HERE / "tests" / "fixtures" / "real_ref.npz"
DEFAULT_DATASET = HERE / "benchmark_data" / "en_1k.txt"
MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"


def load_cfg_and_params():
    from leet_llm import TransformerConfig, load_marian

    if not DEFAULT_WEIGHTS.exists():
        raise SystemExit(
            f"Missing {DEFAULT_WEIGHTS}. Run: uv run --group gen python 302_translate/convert.py"
        )
    if not DEFAULT_REF.exists():
        raise SystemExit(f"Missing {DEFAULT_REF}. Run convert.py first.")

    R = np.load(DEFAULT_REF)
    act = str(R["activation"].item()) if "activation" in R else "gelu"
    cfg = TransformerConfig(
        d_model=int(R["d_model"]),
        n_heads=int(R["n_heads"]),
        n_enc_layers=int(R["n_enc_layers"]),
        n_dec_layers=int(R["n_dec_layers"]),
        d_ff=int(R["d_ff"]),
        vocab_size=int(R["vocab_size"]),
        max_pos=int(R["max_pos"]),
        scale_embedding=bool(R["scale_embedding"]),
        pad_id=int(R["pad_id"]),
        eos_id=int(R["eos_id"]),
        decoder_start_id=int(R["decoder_start_id"]),
        activation=act,
    )
    W = np.load(DEFAULT_WEIGHTS)
    params = load_marian({k: W[k] for k in W.files}, cfg)
    return cfg, params


def load_tokenizer():
    from transformers import MarianTokenizer

    return MarianTokenizer.from_pretrained(MODEL_NAME)


def load_dataset(path: pathlib.Path, limit: int) -> list[str]:
    if not path.exists():
        raise SystemExit(
            f"Dataset not found at {path}.\n"
            f"Generate it with: uv run --group gen python 302_translate/download_dataset.py\n"
            f"Or download manually — see 302_translate/README.md benchmark section."
        )
    lines = [
        l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    return lines[:limit]


def main():
    ap = argparse.ArgumentParser(description="302_translate greedy decode benchmark")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="batch size param for future batched API (currently looped)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="number of source sentences to translate",
    )
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--dataset", type=pathlib.Path, default=DEFAULT_DATASET)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument(
        "--output", type=pathlib.Path, default=HERE / "benchmark_baseline.json"
    )
    ap.add_argument(
        "--print-samples",
        type=int,
        default=5,
        help="number of first samples to decode and print source -> translation",
    )
    ap.add_argument(
        "--samples-output",
        type=pathlib.Path,
        default=HERE / "benchmark_samples.jsonl",
        help="where to write source/target pairs as jsonl",
    )
    args = ap.parse_args()

    # lazy import to respect LEET_LLM_TARGET env
    from leet_llm import translate

    print(f"[benchmark] loading tokenizer {MODEL_NAME} ...")
    tok = load_tokenizer()
    print("[benchmark] loading cfg + params ...")
    cfg, params = load_cfg_and_params()
    sents = load_dataset(args.dataset, args.limit)
    print(f"[benchmark] dataset: {len(sents)} sentences from {args.dataset}")
    print(
        f"[benchmark] batch-size param: {args.batch_size} (reserved, current translate is single-sample)"
    )

    # warmup
    if args.warmup > 0:
        enc = tok([sents[0]], return_tensors="np", padding=True)
        _ = translate(enc["input_ids"], params, cfg, max_new_tokens=args.max_new_tokens)

    total_src_tokens = 0
    total_tgt_tokens = 0
    latencies = []
    samples = []

    start = time.perf_counter()
    for i, sent in enumerate(sents, 1):
        enc = tok([sent], return_tensors="np", padding=True)
        src_ids = enc["input_ids"]
        total_src_tokens += int(src_ids.shape[1])

        t0 = time.perf_counter()
        out_ids = translate(src_ids, params, cfg, max_new_tokens=args.max_new_tokens)
        t1 = time.perf_counter()

        latencies.append(t1 - t0)
        total_tgt_tokens += len(out_ids)  # includes decoder_start + eos

        # collect first N samples with decoded translation for inspection
        if len(samples) < args.print_samples:
            try:
                translation = tok.decode(out_ids, skip_special_tokens=True)
            except Exception:
                translation = "<decode error>"
            samples.append(
                {
                    "src": sent,
                    "tgt": translation,
                    "src_ids_len": int(src_ids.shape[1]),
                    "tgt_ids_len": len(out_ids),
                    "latency_ms": round((t1 - t0) * 1000, 2),
                }
            )

        if i % 50 == 0 or i == len(sents):
            print(
                f"  {i}/{len(sents)}  avg_latency={(sum(latencies) / len(latencies)) * 1000:.1f} ms"
            )

    wall = time.perf_counter() - start
    latencies = np.array(latencies)
    metrics = {
        "num_samples": len(sents),
        "batch_size_param": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "wall_time_s": round(wall, 3),
        "avg_latency_ms": round(float(latencies.mean() * 1000), 2),
        "p50_latency_ms": round(float(np.percentile(latencies, 50) * 1000), 2),
        "p90_latency_ms": round(float(np.percentile(latencies, 90) * 1000), 2),
        "p99_latency_ms": round(float(np.percentile(latencies, 99) * 1000), 2),
        "sentences_per_sec": round(len(sents) / wall, 3),
        "src_tokens": total_src_tokens,
        "tgt_tokens": total_tgt_tokens,
        "src_tokens_per_sec": round(total_src_tokens / wall, 2),
        "tgt_tokens_per_sec": round(total_tgt_tokens / wall, 2),
        "dataset": str(args.dataset),
        "model": MODEL_NAME,
    }

    print("\n=== 302_translate baseline ===")
    print(json.dumps(metrics, indent=2))
    args.output.write_text(json.dumps(metrics, indent=2))
    print(f"saved -> {args.output}")

    # print source -> translation samples
    if samples:
        print(f"\n=== Sample translations (first {len(samples)}) ===")
        for idx, s in enumerate(samples, 1):
            print(f"[{idx}] SRC ({s['src_ids_len']} tok): {s['src']}")
            print(f"    TGT ({s['tgt_ids_len']} tok, {s['latency_ms']} ms): {s['tgt']}")
        # save jsonl
        try:
            with open(args.samples_output, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print(f"saved samples -> {args.samples_output}")
        except Exception as e:
            print(f"warn: could not save samples: {e}")


if __name__ == "__main__":
    main()
