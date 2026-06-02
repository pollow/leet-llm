import pathlib
import numpy as np

from leet_llm.grader import load

_m = load(__file__)
sample = _m.sample
generate = _m.generate

FIX = pathlib.Path(__file__).parent / "fixtures"
_W = np.load(FIX / "warpers.npz")


def test_greedy_is_argmax():
    logits = _W["logits"][0]
    assert sample(logits, temperature=0.0) == int(np.argmax(logits))


def test_top_k_keeps_hf_support():
    # tokens HF's TopKLogitsWarper keeps (finite) are exactly those top-k should keep.
    logits = _W["logits"][0]
    kept = set(np.where(np.isfinite(_W["topk_5"][0]))[0].tolist())
    # sampling many times with top_k=5 must only ever return ids in `kept`.
    rng = np.random.default_rng(0)
    got = {sample(logits, rng, temperature=1.0, top_k=5) for _ in range(200)}
    assert got <= kept and len(kept) == 5


def test_top_p_keeps_hf_support():
    logits = _W["logits"][0]
    kept = set(np.where(np.isfinite(_W["topp_0p9"][0]))[0].tolist())
    rng = np.random.default_rng(1)
    got = {sample(logits, rng, temperature=1.0, top_p=0.9) for _ in range(300)}
    assert got <= kept


def test_seeded_reproducible():
    logits = _W["logits"][0]
    a = sample(logits, np.random.default_rng(7), temperature=1.0)
    b = sample(logits, np.random.default_rng(7), temperature=1.0)
    assert a == b


# --- generation loop over a tiny model (reuse 303's fixture weights) ---
from leet_llm import LlamaConfig, load_llama  # noqa: E402

_L = np.load(pathlib.Path(__file__).parents[2] / "303_llama_model/tests/fixtures/tiny_llama.npz")


def _cfg():
    return LlamaConfig(dim=int(_L["dim"]), n_layers=int(_L["n_layers"]),
                       n_heads=int(_L["n_heads"]), n_kv_heads=int(_L["n_kv_heads"]),
                       vocab_size=int(_L["vocab_size"]), max_seq_len=int(_L["max_seq_len"]),
                       norm_eps=float(_L["norm_eps"]), rope_base=float(_L["rope_base"]))


def _params():
    return load_llama({k: _L[k] for k in _L.files}, _cfg())


def test_generate_greedy_deterministic_and_grows():
    cfg = _cfg()
    prompt = _L["input_ids"]
    out = generate(prompt, _params(), cfg, max_new_tokens=4, temperature=0.0)
    assert out[: prompt.shape[1]] == prompt[0].tolist()
    assert len(out) == prompt.shape[1] + 4               # no eos given ⇒ runs full budget
    again = generate(prompt, _params(), cfg, max_new_tokens=4, temperature=0.0)
    assert out == again                                   # greedy is deterministic


def test_generate_stops_at_eos():
    cfg = _cfg()
    prompt = _L["input_ids"]
    full = generate(prompt, _params(), cfg, max_new_tokens=4, temperature=0.0)
    eos = full[prompt.shape[1]]                           # force eos = first generated token
    out = generate(prompt, _params(), cfg, max_new_tokens=4, temperature=0.0, eos_id=eos)
    assert out == full[: prompt.shape[1] + 1] and out[-1] == eos
