from itertools import product

import numpy as np
import numpy.testing as npt
import pytest

import ahe


class PyHistEq:
    # quick and dirty Python/numpy base implementations of basic histeq and full AHE
    @staticmethod
    def _get_tile(image, center_pixel: tuple[int, int], max_size: int):
        wing_size = max_size // 2
        isize, jsize = image.shape
        i0, j0 = center_pixel

        pimage = np.pad(image, pad_width=wing_size, mode="reflect")
        iL = i0
        iR = i0 + 2 * wing_size
        jL = j0
        jR = j0 + 2 * wing_size
        return pimage[iL : iR + 1, jL : jR + 1]

    @staticmethod
    def _get_normalized_cdf(tile, nbins: int):
        hist, bin_edges = np.histogram(tile.ravel(), bins=nbins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # cumulative distribution function
        cdf = hist.cumsum(dtype=tile.dtype)
        return (bin_centers, cdf / cdf[-1])

    @staticmethod
    def _he_numpy(image, nbins: int):
        # ref implementation, adapted from scikit-image (exposure.equalize_hist)
        bin_centers, normalized_cdf = PyHistEq._get_normalized_cdf(image, nbins)

        # As of version 2.4, np.interp always promotes to float64, so we
        # have to cast back to single precision when float32 output is desired
        return (
            np.interp(image.flat, bin_centers, normalized_cdf)
            .reshape(image.shape)
            .astype(image.dtype, copy=False)
        )

    @staticmethod
    def _ahe_numpy(image, nbins: int, tile_size: int):
        out = np.zeros_like(image)
        for i, j in product(*(range(size) for size in image.shape)):
            tile = PyHistEq._get_tile(image, center_pixel=(i, j), max_size=tile_size)
            bin_centers, normalized_cdf = PyHistEq._get_normalized_cdf(tile, nbins)
            out[i, j] = np.interp(image[i, j], bin_centers, normalized_cdf)
        return out


@pytest.mark.parametrize(
    "nbins, min_rms_reduction",
    [
        (12, 2.0),
        (64, 10.0),
        (256, 50.0),
    ],
)
@pytest.mark.parametrize("dtype, rtol", [("float32", 2e-6), ("float64", 2.5e-15)])
def test_historgram_equalization(nbins, min_rms_reduction, dtype, rtol, subtests):
    # histogram equalization produces a new image whose cumulative
    # distribution function (cdf) should be close(r) to a straight line
    # (i.e., approaching a flat intensity distribution)
    # This test check this property from an initial image made of gaussian
    # noise.
    # Expected rms reduction factors (min_rms_reduction) are empirical, i.e.,
    # slightly looser than what the original implementation was able to achieve

    IMAGE_SHAPE = (256, 128)
    prng = np.random.default_rng(0)
    image = np.clip(
        prng.normal(loc=5.0, scale=1.0, size=np.prod(IMAGE_SHAPE)).reshape(IMAGE_SHAPE),
        a_min=0.0,
        a_max=None,
        dtype=dtype,
    )

    def normalized_cdf(a):
        hist, bin_edges = np.histogram(a.ravel(), bins=nbins)
        cdf = hist.cumsum(dtype=dtype)
        return cdf / cdf[-1]

    def rms(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    image_eq = ahe.equalize_histogram(image, nbins=nbins)
    cdf_in = normalized_cdf(image)
    cdf_eq = normalized_cdf(image_eq)

    id_func = np.linspace(0, 1, nbins)
    rms_in = rms(cdf_in, id_func)
    rms_eq = rms(cdf_eq, id_func)
    with subtests.test("basic properties"):
        assert rms_eq < rms_in
        assert (rms_in / rms_eq) > min_rms_reduction

    image_ref = PyHistEq._he_numpy(image, nbins=nbins)

    with subtests.test("compare to ref impl"):
        npt.assert_allclose(image_eq, image_ref, rtol=rtol)


@pytest.mark.parametrize("dtype, rtol", [("float32", 6e-7), ("float64", 5e-16)])
def test_historgram_equalization_sliding_tile_full_ahe(dtype, rtol):
    IMAGE_SHAPE = (4, 6)
    TILE_SIZE = 3
    NBINS = 3
    prng = np.random.default_rng(0)
    image = np.clip(
        prng.normal(loc=5.0, scale=1.0, size=np.prod(IMAGE_SHAPE)).reshape(IMAGE_SHAPE),
        a_min=0.0,
        a_max=None,
        dtype=dtype,
    )

    res_ahe = PyHistEq._ahe_numpy(image, nbins=NBINS, tile_size=TILE_SIZE)

    res_st = ahe.equalize_histogram(
        image,
        nbins=NBINS,
        adaptive_strategy={
            "kind": "sliding-tile",
            "tile-size": TILE_SIZE,
        },
    )
    npt.assert_allclose(res_st, res_ahe, rtol=rtol)


# https://github.com/neutrinoceros/ahe/issues/8
@pytest.mark.parametrize("dtype, rtol", [("float32", 1.25), ("float64", 1.25)])
def test_historgram_equalization_tile_interpolation_full_ahe(dtype, rtol):
    IMAGE_SHAPE = (64, 64)
    TILE_SIZE = 32
    NBINS = 3
    prng = np.random.default_rng(0)
    image = np.clip(
        prng.normal(loc=5.0, scale=1.0, size=np.prod(IMAGE_SHAPE)).reshape(IMAGE_SHAPE),
        a_min=0.0,
        a_max=None,
        dtype=dtype,
    )

    res_ahe = PyHistEq._ahe_numpy(image, nbins=NBINS, tile_size=TILE_SIZE)

    res_ti = ahe.equalize_histogram(
        image,
        nbins=NBINS,
        adaptive_strategy={
            "kind": "tile-interpolation",
            "tile-size": TILE_SIZE,
        },
    )
    npt.assert_allclose(res_ti, res_ahe, rtol=rtol)
