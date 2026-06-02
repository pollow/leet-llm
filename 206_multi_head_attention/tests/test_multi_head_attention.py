import pathlib

import numpy as np
import pytest

from leet_llm.grader import load

_m = load(__file__)
mha = _m.mha
AttnParams = _m.AttnParams

FIX = pathlib.Path(__file__).parent / "fixtures"
_FIXTURES = sorted(FIX.glob("*.npz"))


def _params(d):
    opt = {k: d[k] for k in ("bq", "bk", "bv", "bo") if k in d.files}
    return AttnParams(Wq=d["Wq"], Wk=d["Wk"], Wv=d["Wv"], Wo=d["Wo"], **opt)


@pytest.mark.parametrize("path", _FIXTURES, ids=[p.stem for p in _FIXTURES])
def test_matches_torch_fixture(path):
    """Frozen goldens from a float64 torch multi-head attention reference."""
    d = np.load(path)
    n_heads = int(d["n_heads"])
    x_kv = d["x_kv"] if "x_kv" in d.files else None
    mask = d["mask"] if "mask" in d.files else None
    out = mha(d["x_q"], _params(d), n_heads, x_kv=x_kv, mask=mask)
    np.testing.assert_allclose(out, d["out"], rtol=1e-9, atol=1e-9)


def test_x_kv_defaults_to_x_q():
    # self-attention: passing x_kv=x_q explicitly equals leaving it as the default
    rng = np.random.default_rng(1)
    d_model = 8
    x = rng.standard_normal((2, 5, d_model))
    p = AttnParams(
        Wq=rng.standard_normal((d_model, d_model)),
        Wk=rng.standard_normal((d_model, d_model)),
        Wv=rng.standard_normal((d_model, d_model)),
        Wo=rng.standard_normal((d_model, d_model)),
    )
    np.testing.assert_allclose(mha(x, p, 2), mha(x, p, 2, x_kv=x), atol=1e-12)


def test_shape_preserved():
    rng = np.random.default_rng(2)
    d_model = 12
    x = rng.standard_normal((3, 7, d_model))
    p = AttnParams(
        Wq=rng.standard_normal((d_model, d_model)),
        Wk=rng.standard_normal((d_model, d_model)),
        Wv=rng.standard_normal((d_model, d_model)),
        Wo=rng.standard_normal((d_model, d_model)),
    )
    assert mha(x, p, 3).shape == (3, 7, d_model)
