"""Download 1k English source sentences for 302_translate benchmark.

Default source: Helsinki-NLP/opus-100 en-zh test split (~2000 sentences, CC-BY-4.0).
We take first 1000 English sides and write to benchmark_data/en_1k.txt — diff-friendly text,
one sentence per line, easy to review in Phabricator/Git.

Lightweight default: downloads only en-zh/test-00000-of-00001.parquet (~355KB) via
huggingface_hub hf_hub_download, not the full train/validation splits. Use --mode datasets
only if you want the HF datasets library path.

Alternative sources:
 * facebook/flores devtest eng_Latn 1012 sentences CC-BY-SA-4.0  (--source flores)
 * Helsinki-NLP/tatoeba eng-cmn  ~31k CC-BY-2.0                (--source tatoeba)

Usage:
    uv run --group gen python 302_translate/tools/download_dataset.py --source opus100 --limit 1000
    # default mode=parquet downloads single 355KB parquet
Requires: huggingface_hub (already in gen), pandas/pyarrow (transitive via datasets or torch ecosystem).
Add datasets only if using --mode datasets or other sources: uv add --group gen datasets
"""

from __future__ import annotations

import argparse
import pathlib

HERE = pathlib.Path(__file__).parent.parent
OUT_DIR = HERE / "benchmark_data"
OUT_DIR.mkdir(exist_ok=True)


def download_opus100(limit=1000, mode="parquet"):
    """
    Default mode='parquet' downloads only en-zh/test-00000-of-00001.parquet (~355KB)
    via huggingface_hub, avoiding full dataset cache.
    mode='datasets' falls back to datasets library (downloads more metadata).
    """
    out = OUT_DIR / "en_1k.txt"
    if mode == "parquet":
        from huggingface_hub import hf_hub_download
        import pandas as pd

        parquet_path = hf_hub_download(
            repo_id="Helsinki-NLP/opus-100",
            repo_type="dataset",
            filename="en-zh/test-00000-of-00001.parquet",
        )
        df = pd.read_parquet(parquet_path)
        # df has column 'translation' as struct/dict with 'en','zh' or similar
        # pandas reads struct as dict-like objects
        sents = []
        for tr in df["translation"].tolist()[:limit]:
            # tr can be dict or pyarrow struct converted to dict
            if isinstance(tr, dict):
                sents.append(tr.get("en") or tr.get("src") or "")
            else:
                # fallback: assume object with keys
                try:
                    sents.append(tr["en"])
                except Exception:
                    sents.append(str(tr))
        sents = [s for s in sents if s][:limit]
        out.write_text("\n".join(sents) + "\n", encoding="utf-8")
        print(
            f"wrote {len(sents)} sentences -> {out}  source=Helsinki-NLP/opus-100 en-zh/test.parquet CC-BY-4.0 (~355KB download)"
        )
        return out
    else:  # datasets fallback
        from datasets import load_dataset

        ds = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="test")
        sents = [
            ex["translation"]["en"] for ex in ds.select(range(min(limit, len(ds))))
        ]
        out.write_text("\n".join(sents) + "\n", encoding="utf-8")
        print(
            f"wrote {len(sents)} sentences -> {out}  source=Helsinki-NLP/opus-100 en-zh test CC-BY-4.0"
        )
        return out


def download_flores(limit=1012):
    from datasets import load_dataset

    # flores-200 devtest English side
    ds = load_dataset("facebook/flores", "eng_Latn", split="devtest")
    sents = [ex["sentence"] for ex in ds.select(range(min(limit, len(ds))))]
    out = OUT_DIR / "en_1k_flores.txt"
    out.write_text("\n".join(sents) + "\n", encoding="utf-8")
    print(
        f"wrote {len(sents)} sentences -> {out}  source=facebook/flores eng_Latn devtest CC-BY-SA-4.0"
    )
    return out


def download_tatoeba(limit=1000):
    from datasets import load_dataset

    ds = load_dataset("Helsinki-NLP/tatoeba", lang1="en", lang2="cmn", split="test")
    # tatoeba_mt format varies; try generic fallback
    sents = []
    for ex in ds.select(range(min(limit, len(ds)))):
        # dataset has 'sourceString' or 'translation'
        if "sourceString" in ex:
            sents.append(ex["sourceString"])
        elif "translation" in ex:
            sents.append(ex["translation"]["en"])
        else:
            sents.append(str(ex))
    out = OUT_DIR / "en_1k_tatoeba.txt"
    out.write_text("\n".join(sents) + "\n", encoding="utf-8")
    print(f"wrote {len(sents)} -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source", choices=["opus100", "flores", "tatoeba"], default="opus100"
    )
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument(
        "--mode",
        choices=["parquet", "datasets"],
        default="parquet",
        help="parquet = hf_hub_download single test parquet only (~355KB); datasets = full HF datasets library",
    )
    args = ap.parse_args()

    if args.source == "opus100":
        download_opus100(args.limit, mode=args.mode)
    elif args.source == "flores":
        download_flores(args.limit)
    else:
        download_tatoeba(args.limit)

    print(
        "\nNext: uv run --group gen python 302_translate/tools/benchmark.py --limit",
        args.limit,
    )


if __name__ == "__main__":
    main()
