from __future__ import annotations

__all__ = [
    "equalize_histogram",
]

from typing import TYPE_CHECKING, Literal, assert_never

import numpy as np

from ahe._api import (
    SUPPORTED_AHE_KINDS,
    SlidingTile,
    TileInterpolation,
)
from ahe._core import (
    equalize_histogram_f32,
    equalize_histogram_f64,
    equalize_histogram_sliding_tile_f32,
    equalize_histogram_sliding_tile_f64,
    equalize_histogram_tile_interpolation_f32,
    equalize_histogram_tile_interpolation_f64,
)
from ahe._typing import UNSET, UnsetType

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeVar

    from numpy import dtype, ndarray
    from numpy import float32 as f32
    from numpy import float64 as f64

    from ahe._api import Strategy, StrategySpec
    from ahe._typing import Pair

    F = TypeVar("F", f32, f64)

_SUPPORTED_DTYPES: list[np.dtype[np.floating]] = [
    np.dtype("float32"),
    np.dtype("float64"),
]


def _resolve_nbins(nbins: int | Literal["auto"], shape: Pair[int]) -> int:
    if nbins == "auto":
        npix = shape[0] * shape[1]
        return min(npix, 256)
    else:
        return nbins


def _resolve_max_bincount(max_normalized_bincount: float, *, tile_size: int) -> int:
    return max(1, int(max_normalized_bincount * tile_size))


def equalize_histogram(
    image: ndarray[tuple[int, int], dtype[F]],
    /,
    *,
    nbins: int | Literal["auto"] = "auto",
    boundaries: Literal["reflect", "periodic"] = "reflect",
    adaptive_strategy: StrategySpec | None = None,
    max_normalized_bincount: float = 1.0,
) -> ndarray[tuple[int, int], dtype[F]]:
    """(Contrast Limited) (Adaptive) Histogram Equalization for gray-scale images.

    Parameters
    ----------
    image : 2D array, positional only
      The input gray-scale image.

    nbins: int or 'auto', keyword-only
      number of bins to use in histograms
      By default ('auto'), this is set to 256 or the number maximum of pixels in a tile,
      whichever is smallest.
      Reduce this number for faster computations.
      Increase it to improve the overall contrast of the result.

    boundaries: 'reflect' (default), 'periodic', keyword-only
        If an adapative_strategy is defined, the image domain is extended to ensure that
        all pixels, including the vicinity of image edges, are part of tiles with equal
        shapes.
        This argument controls how this domain extension is performed in both directions.
        It has no effect if combined with adaptive_strategy=None.

    adaptive_strategy: None (default) a sliding-tile, or tile-interpolation specification, keyword-only
        The default behavior is simple Histogram Equalization[^1] (HE).
        If this argument is not None, Adaptive Histogram Equalization[^2] (AHE) is used
        instead. Both flavors of efficient AHE, sliding-tile and tile-interpolation, are
        available, and specified as a dictionary. See Examples for details.

        Note that the tile-size must be odd in the case of a sliding-tile, and even in
        the case of a tile-interpolation strategy. These restrictions are necessary to
        prevent left/right biases in the output. Furthermore, tile-interpolation is only
        supported for images with even sizes in both directions.

    max_normalized_bincount: float, default is 1.0
        should satisfy 0 < max_normalized_bincount <= 1.0
        Set a value <1.0 to enable contrast limitation ((CL)(A)HE).
        This value represents the maximum bincount, normalized by the number of pixels
        in a tile. Values in excess of this threshold are re-distributed evenly over the
        whole tile. For instance, max_normalized_bincount=0.01 means that any bin
        totalling more that 1% of the pixels in a tile will be clipped.
        Effectively, this factor limits the maximum enhancement of intensity
        for any given pixel. Higher values result in higher contrast.
        See Sec. 4.2 from reference [1] for details.

    Returns
    -------
    2D array
        The processed image with values normalized to the [0, 1] interval.

    Examples
    --------
    Here's a basic application of basic (non-adaptive) histogram equalization

    >>> import numpy as np
    >>> image_shape = (128, 128)
    >>> prng = np.random.default_rng(0)
    >>> image = np.clip(prng.normal(loc=0.5, scale=0.25, size=np.prod(image_shape)).reshape(image_shape), a_min=0.0, a_max=1.0)
    >>> equalize_histogram(image) # doctest:+ELLIPSIS
    array([[...
    ...]], shape=(128, 128))

    To instead compute exact AHE, one can specify a sliding-tile as follow
    >>> equalize_histogram(
    ...    image,
    ...    adaptive_strategy={
    ...        'kind': 'sliding-tile',
    ...        'tile-size': 15,
    ...    },
    ... ) # doctest:+ELLIPSIS
    array([[...
    ...]], shape=(128, 128))

    It is also possible to reasonably approximate AHE at a fraction of the computational
    cost using tile-interpolation instead
    >>> equalize_histogram(
    ...    image,
    ...    adaptive_strategy={
    ...        'kind': 'tile-interpolation',
    ...        'tile-size': 16,
    ...    },
    ... ) # doctest:+ELLIPSIS
    array([[...
    ...]], shape=(128, 128))

    This strategy may also be specified with a number of contextual regions used to map
    the entire image, by providing tile-into instead of tile-size
    >>> equalize_histogram(
    ...    image,
    ...    adaptive_strategy={
    ...        'kind': 'tile-interpolation',
    ...        'tile-into': 8,
    ...    },
    ... ) # doctest:+ELLIPSIS
    array([[...
    ...]], shape=(128, 128))

    Both tile-size and tile-into will accept either a pair of integers (n, m) to
    distinguish sizes (or divisors, respectively) or a single integer n, as illustrated
    above, which is a shorthand for (n, n).

    References
    ----------
    [1] Pizer, S. M. et al. (1987)
        Adaptive Histogram Equalization and Its Variations.
        Compute Vizion, Graphics, and Image Processing, 39, 355-368
        DOI: 10.1016/S0734-189X(87)80186-X
    """
    if image.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"Found unsupported data type: {image.dtype}. "
            f"Expected of of {_SUPPORTED_DTYPES}."
        )
    if np.any(~np.isfinite(image)):
        raise ValueError(
            "Image contains infinite values or NaNs, none of which are supported."
        )

    pad_mode: Literal["reflect", "wrap"]
    match boundaries:
        case "reflect":
            pad_mode = "reflect"
        case "periodic":
            pad_mode = "wrap"
        case _:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError(  # pyright: ignore[reportUnreachable]
                f"Received unknown value {boundaries=}. "
                "Expected 'reflect' or 'periodic'."
            )

    if not (0 < max_normalized_bincount <= 1.0):
        raise ValueError(
            f"{max_normalized_bincount=} is invalid: expected a value within ]0, 1]"
        )

    input_dtype = image.dtype
    if adaptive_strategy is None:
        if not image.flags.c_contiguous:
            # - low-level interpolation routines require the array be contiguous
            # - hot loops are optimized for C order arrays
            # the image is never to be mutated so this won't compromise
            # correctness or consistency
            # this type:ignore comment can be removed when Python 3.10 is dropped
            image = np.copy(image, order="C")
        histeq: Callable[
            [ndarray[tuple[int, int], dtype[F]], int, int],
            ndarray[tuple[int, int], dtype[F]],
        ]
        if input_dtype == np.dtype("float32"):
            histeq = equalize_histogram_f32  # type: ignore[assignment]
        elif input_dtype == np.dtype("float64"):
            histeq = equalize_histogram_f64  # type: ignore[assignment]
        else:
            raise AssertionError
        nbins = _resolve_nbins(nbins, image.shape)
        max_bincount = _resolve_max_bincount(
            max_normalized_bincount, tile_size=image.size
        )
        return histeq(image, nbins, max_bincount)

    ahe_type: type[Strategy]
    match adaptive_strategy.get("kind", UNSET):
        case "sliding-tile":
            ahe_type = SlidingTile
        case "tile-interpolation":
            ahe_type = TileInterpolation
        case str() as unknown:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError(  # pyright: ignore[reportUnreachable]
                f"Unknown strategy kind {unknown!r}. "
                f"Expected one of {sorted(SUPPORTED_AHE_KINDS)}"
            )
        case UnsetType():  # pyright: ignore[reportUnnecessaryComparison]
            raise TypeError("adaptive_strategy is missing a 'kind' key.")  # pyright: ignore[reportUnreachable]  s
        case _ as invalid:  # pyright: ignore[reportUnnecessaryComparison]
            raise TypeError(  # pyright: ignore[reportUnreachable]
                f"Invalid strategy kind {invalid!r} with type {type(invalid)}. "
                f"Expected one of {sorted(SUPPORTED_AHE_KINDS)}"
            )

    strat = ahe_type.from_spec(adaptive_strategy)
    ts = strat.resolve_tile_shape(image.shape)
    strat = type(strat)(tile_shape=ts)
    assert strat.tile_shape == ts
    nbins = _resolve_nbins(nbins, ts)

    pad_width = strat.resolve_pad_width(image.shape)
    pimage = np.pad(image, pad_width=pad_width, mode=pad_mode)
    max_bincount = _resolve_max_bincount(
        max_normalized_bincount, tile_size=ts[0] * ts[1]
    )

    match strat:
        case SlidingTile():
            histeq_st: Callable[
                [ndarray[tuple[int, int], dtype[F]], int, Pair[int], int],
                ndarray[tuple[int, int], dtype[F]],
            ]
            if input_dtype == np.dtype("float32"):
                histeq_st = equalize_histogram_sliding_tile_f32  # type: ignore[assignment]
            elif input_dtype == np.dtype("float64"):
                histeq_st = equalize_histogram_sliding_tile_f64  # type: ignore[assignment]
            else:
                raise AssertionError
            res = histeq_st(pimage, nbins, ts, max_bincount)
        case TileInterpolation():
            histeq_ti: Callable[
                [ndarray[tuple[int, int], dtype[F]], int, Pair[int], int],
                ndarray[tuple[int, int], dtype[F]],
            ]
            if input_dtype == np.dtype("float32"):
                histeq_ti = equalize_histogram_tile_interpolation_f32  # type: ignore[assignment]
            elif input_dtype == np.dtype("float64"):
                histeq_ti = equalize_histogram_tile_interpolation_f64  # type: ignore[assignment]
            else:
                raise AssertionError
            res = histeq_ti(pimage, nbins, ts, max_bincount)
        case _ as unreachable:  # pyright: ignore[reportUnnecessaryComparison]
            assert_never(unreachable)

    # unpad result
    return res[pad_width[0][0] : -pad_width[0][1], pad_width[1][0] : -pad_width[1][1]]
