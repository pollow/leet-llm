"""104 — BPE Training Loop — reference solution."""

from __future__ import annotations


def bpe_train(text: str, vocab_size: int) -> tuple[list[str], list[float]]:
    tokens: list[str] = sorted(set(text))
    scores: list[float] = [0.0] * len(tokens)
    stoi = {t: i for i, t in enumerate(tokens)}
    seq = [stoi[ch] for ch in text]

    merge_index = 0
    while len(tokens) < vocab_size:
        counts: dict[tuple[int, int], int] = {}
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
        if not counts:
            break

        best = max(counts.values())
        pair = min(
            (p for p, c in counts.items() if c == best),
            key=lambda p: (tokens[p[0]], tokens[p[1]]),
        )

        new_id = len(tokens)
        tokens.append(tokens[pair[0]] + tokens[pair[1]])
        merge_index += 1
        scores.append(float(-merge_index))

        merged: list[int] = []
        i, n = 0, len(seq)
        while i < n:
            if i < n - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                merged.append(new_id)
                i += 2
            else:
                merged.append(seq[i])
                i += 1
        seq = merged

    return tokens, scores
