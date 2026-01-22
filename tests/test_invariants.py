from functools import partial

import numpy as np
import numpy.testing as npt
import pytest

import ahe


@pytest.mark.parametrize(
    "adaptive_strategy, rtol",
    [
        pytest.param(None, 0.0, id="non-adaptive"),
        pytest.param({"kind": "sliding-tile", "tile-size": 5}, 0.0, id="sliding-tile"),
        pytest.param(
            {"kind": "tile-interpolation", "tile-size": 2},
            3e-16,
            id="tile-interpolation",
        ),
    ],
)
def test_directional_invariance(adaptive_strategy, rtol, subtests):
    IMAGE_SHAPE = (4, 4)
    prng = np.random.default_rng(0)

    image = np.clip(
        prng.normal(loc=0.5, scale=0.25, size=np.prod(IMAGE_SHAPE)).reshape(
            IMAGE_SHAPE
        ),
        a_min=0.0,
        a_max=1.0,
        dtype="float64",
    )

    heq = partial(ahe.equalize_histogram, nbins=8, adaptive_strategy=adaptive_strategy)
    t_invt = {
        "LR": (np.fliplr, np.fliplr),
        "UD": (np.flipud, np.flipud),
        "LR+UD": (lambda a: np.flipud(np.fliplr(a)), lambda a: np.fliplr(np.flipud(a))),
        "UD+LR": (lambda a: np.fliplr(np.flipud(a)), lambda a: np.flipud(np.fliplr(a))),
        "transpose": (np.transpose, np.transpose),
        "rot90": (np.rot90, lambda a: np.rot90(np.rot90(np.rot90(a)))),
    }
    res0 = heq(image)
    for name, (t, invt) in t_invt.items():
        with subtests.test(name):
            res = invt(heq(t(image)))
            npt.assert_allclose(res, res0, rtol=rtol)
