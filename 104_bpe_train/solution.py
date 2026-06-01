"""104 — BPE Training Loop — reference solution."""

from __future__ import annotations

from leet_llm import count_pairs, apply_merge, build_char_vocab, char_encode


def bpe_train(text: str, vocab_size: int) -> tuple[list[str], list[float]]:
    stoi, tokens = build_char_vocab(text)
    scores = [0.0] * len(tokens)
    seq = char_encode(text, stoi)
    rep = -1.0

    while len(tokens) < vocab_size:
        counts = count_pairs(seq)
        if not counts:
            break

        best = max(counts.values())
        cands = [pair for pair, cnt in counts.items() if cnt == best]
        pair = min(cands, key=lambda p: (tokens[p[0]], tokens[p[1]]))

        new_id = len(tokens)
        tokens.append(tokens[pair[0]] + tokens[pair[1]])
        scores.append(rep)

        seq = apply_merge(seq, pair, new_id)
        rep -= 1

    return tokens, scores
