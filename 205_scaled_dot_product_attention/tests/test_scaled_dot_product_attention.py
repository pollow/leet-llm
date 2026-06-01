import pathlib

import numpy as np
import pytest

from leet_llm.grader import load

_m = load(__file__)
sdpa = _m.sdpa

FIX = pathlib.Path(__file__).parent / "fixtures"
_FIXTURES = sorted(FIX.glob("*.npz"))


@pytest.mark.parametrize("path", _FIXTURES, ids=[p.stem for p in _FIXTURES])
def test_matches_torch_fixture(path):
    """Frozen goldens from a float64 torch oracle (see gen_fixtures.py)."""
    d = np.load(path)
    mask = d["mask"] if "mask" in d.files else None
    out = sdpa(d["q"], d["k"], d["v"], mask)
    np.testing.assert_allclose(out, d["out"], rtol=1e-9, atol=1e-9)


def test_weights_average_identical_values():
    # Attention weights sum to 1, so if every value row is identical the output
    # equals that value — regardless of the (finite) scores.
    rng = np.random.default_rng(1)
    q = rng.standard_normal((2, 4, 8))
    k = rng.standard_normal((2, 4, 8))
    v = np.broadcast_to(rng.standard_normal((2, 1, 8)), (2, 4, 8)).copy()
    np.testing.assert_allclose(sdpa(q, k, v), v, atol=1e-10)


def test_causal_mask_hides_future():
    # With a causal mask, perturbing future value rows must not change the outputs
    # of earlier queries (which cannot attend to them).
    rng = np.random.default_rng(2)
    q = rng.standard_normal((1, 5, 8))
    k = rng.standard_normal((1, 5, 8))
    v = rng.standard_normal((1, 5, 8))
    mask = np.triu(np.ones((5, 5), dtype=bool), k=1)
    out1 = sdpa(q, k, v, mask)
    v2 = v.copy()
    v2[0, 3:] += 10.0  # perturb positions 3 and 4
    out2 = sdpa(q, k, v2, mask)
    np.testing.assert_allclose(out1[0, :3], out2[0, :3], atol=1e-10)
