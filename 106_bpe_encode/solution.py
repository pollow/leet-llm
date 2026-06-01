"""106 — BPE Encode (score-greedy) — reference solution.

Mirrors llama3.np/tokenizer.py: repeatedly merge the highest-scoring adjacent pair.
"""

from __future__ import annotations


def bpe_encode(text: str, tokens: list[str], scores: list[float]) -> list[int]:
    stoi = {t: i for i, t in enumerate(tokens)}
    ids = [stoi[ch] for ch in text if ch in stoi]

    while True:
        best_score = float("-inf")
        best_id = -1
        best_idx = -1
        for i in range(len(ids) - 1):
            merged = tokens[ids[i]] + tokens[ids[i + 1]]
            j = stoi.get(merged, -1)
            if j != -1 and scores[j] > best_score:
                best_score = scores[j]
                best_id = j
                best_idx = i
        if best_idx == -1:
            break
        ids = ids[:best_idx] + [best_id] + ids[best_idx + 2 :]

    return ids
