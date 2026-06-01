import pathlib

import numpy as np
import pytest

from leet_llm import AttnParams  # owned by task 206
from leet_llm.grader import load

_m = load(__file__)
gqa = _m.gqa

FIX = pathlib.Path(__file__).parent / "fixtures"
_FIXTURES = sorted(FIX.glob("*.npz"))


@pytest.mark.parametrize("path", _FIXTURES, ids=[p.stem for p in _FIXTURES])
def test_matches_torch_fixture(path):
    """Frozen goldens from a float64 torch GQA reference (incl. MQA and MHA-equivalent)."""
    d = np.load(path)
    p = AttnParams(Wq=d["Wq"], Wk=d["Wk"], Wv=d["Wv"], Wo=d["Wo"])
    mask = d["mask"] if "mask" in d.files else None
    out = gqa(d["x"], p, int(d["n_heads"]), int(d["n_kv_heads"]), mask=mask)
    np.testing.assert_allclose(out, d["out"], rtol=1e-9, atol=1e-9)


def test_shape_preserved():
    rng = np.random.default_rng(1)
    d, n_heads, n_kv = 12, 4, 2
    dk = d // n_heads
    p = AttnParams(
        Wq=rng.standard_normal((d, d)),
        Wk=rng.standard_normal((n_kv * dk, d)),
        Wv=rng.standard_normal((n_kv * dk, d)),
        Wo=rng.standard_normal((d, d)),
    )
    x = rng.standard_normal((3, 6, d))
    assert gqa(x, p, n_heads, n_kv).shape == (3, 6, d)
