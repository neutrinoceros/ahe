from __future__ import annotations

__all__ = [
    "equalize_histogram",
]

import sys
from typing import TYPE_CHECKING, Literal

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

if sys.version_info >= (3, 11):
    from typing import assert_never  # pyright: ignore[reportUnreachable]
else:
    from typing_extensions import assert_never  # pyright: ignore[reportUnreachable]

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


def equalize_histogram(
    image: ndarray[tuple[int, int], dtype[F]],
    /,
    *,
    nbins: int | Literal["auto"] = "auto",
    boundaries: Literal["reflect", "periodic"] = "reflect",
    adaptive_strategy: StrategySpec | None = None,
    contrast_limitation: None = None,
) -> ndarray[tuple[int, int], dtype[F]]:
    """Equalize histogram of a gray-scale image.

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

    contrast_limitation: None, keyword-only
      not implemented
      https://github.com/neutrinoceros/rlic/issues/304

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
    """
    if contrast_limitation is not None:
        raise NotImplementedError  # type: ignore

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

    input_dtype = image.dtype
    if adaptive_strategy is None:
        if not image.flags.c_contiguous:
            # - low-level interpolation routines require the array be contiguous
            # - hot loops are optimized for C order arrays
            # the image is never to be mutated so this won't compromise
            # correctness or consistency
            # this type:ignore comment can be removed when Python 3.10 is dropped
            image = np.copy(image, order="C")  # type: ignore[assignment]
        histeq: Callable[
            [ndarray[tuple[int, int], dtype[F]], int],
            ndarray[tuple[int, int], dtype[F]],
        ]
        if input_dtype == np.dtype("float32"):
            histeq = equalize_histogram_f32  # type: ignore[assignment] # pyright: ignore[reportAssignmentType]
        elif input_dtype == np.dtype("float64"):
            histeq = equalize_histogram_f64  # type: ignore[assignment] # pyright: ignore[reportAssignmentType]
        else:
            raise AssertionError
        nbins = _resolve_nbins(nbins, image.shape)
        return histeq(image, nbins)

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

    match strat:
        case SlidingTile():
            histeq_st: Callable[
                [ndarray[tuple[int, int], dtype[F]], int, Pair[int]],
                ndarray[tuple[int, int], dtype[F]],
            ]
            if input_dtype == np.dtype("float32"):
                histeq_st = equalize_histogram_sliding_tile_f32  # type: ignore[assignment] # pyright: ignore[reportAssignmentType]
            elif input_dtype == np.dtype("float64"):
                histeq_st = equalize_histogram_sliding_tile_f64  # type: ignore[assignment] # pyright: ignore[reportAssignmentType]
            else:
                raise AssertionError
            res = histeq_st(pimage, nbins, ts)  # type: ignore[arg-type]
        case TileInterpolation():
            histeq_ti: Callable[
                [ndarray[tuple[int, int], dtype[F]], int, Pair[int]],
                ndarray[tuple[int, int], dtype[F]],
            ]
            if input_dtype == np.dtype("float32"):
                histeq_ti = equalize_histogram_tile_interpolation_f32  # type: ignore[assignment] # pyright: ignore[reportAssignmentType]
            elif input_dtype == np.dtype("float64"):
                histeq_ti = equalize_histogram_tile_interpolation_f64  # type: ignore[assignment] # pyright: ignore[reportAssignmentType]
            else:
                raise AssertionError
            res = histeq_ti(pimage, nbins, ts)  # type: ignore[arg-type]
        case _ as unreachable:  # pyright: ignore[reportUnnecessaryComparison]
            assert_never(unreachable)

    # unpad result
    return res[pad_width[0][0] : -pad_width[0][1], pad_width[1][0] : -pad_width[1][1]]  # type: ignore[return-value]
