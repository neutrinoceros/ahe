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
    # boundaries: BoundarySpec = "closed",
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

    boundaries: 'closed' (default), 'periodic', or a dict with keys 'x' and 'y',
                and values are either of these strings, or 2-tuples (left, right)
                thereof. Keyword-only

      Only 'closed' boundaries are accepted at the moment.
      https://github.com/neutrinoceros/rlic/issues/303

    adaptive_strategy: None (default) or a sliding-tile specification, keyword-only
      not implemented
      https://github.com/neutrinoceros/rlic/issues/301
      https://github.com/neutrinoceros/rlic/issues/302

    contrast_limitation: None, keyword-only
      not implemented
      https://github.com/neutrinoceros/rlic/issues/304

    Returns
    -------
    2D array
        The processed image with values normalized to the [0, 1] interval.
    """
    if contrast_limitation is not None:
        raise NotImplementedError  # type: ignore

    if image.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"Found unsupported data type: {image.dtype}. "
            f"Expected of of {_SUPPORTED_DTYPES}."
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
    nbins = _resolve_nbins(nbins, ts)

    pad_width = strat.resolve_pad_width(image.shape)
    pimage = np.pad(image, pad_width=pad_width, mode="reflect")

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
