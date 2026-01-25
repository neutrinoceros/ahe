from numpy import dtype, ndarray
from numpy import float32 as f32
from numpy import float64 as f64

from ahe._typing import Pair

def equalize_histogram_f32(
    image: ndarray[tuple[int, int], dtype[f32]],
    nbins: int,
    max_bincount: int,
) -> ndarray[tuple[int, int], dtype[f32]]: ...
def equalize_histogram_f64(
    image: ndarray[tuple[int, int], dtype[f64]],
    nbins: int,
    max_bincount: int,
) -> ndarray[tuple[int, int], dtype[f64]]: ...
def equalize_histogram_sliding_tile_f32(
    image: ndarray[tuple[int, int], dtype[f32]],
    nbins: int,
    tile_shape: Pair[int],
    max_bincount: int,
) -> ndarray[tuple[int, int], dtype[f32]]: ...
def equalize_histogram_sliding_tile_f64(
    image: ndarray[tuple[int, int], dtype[f64]],
    nbins: int,
    tile_shape: Pair[int],
    max_bincount: int,
) -> ndarray[tuple[int, int], dtype[f64]]: ...
def equalize_histogram_tile_interpolation_f32(
    image: ndarray[tuple[int, int], dtype[f32]],
    nbins: int,
    tile_shape: Pair[int],
    max_bincount: int,
) -> ndarray[tuple[int, int], dtype[f32]]: ...
def equalize_histogram_tile_interpolation_f64(
    image: ndarray[tuple[int, int], dtype[f64]],
    nbins: int,
    tile_shape: Pair[int],
    max_bincount: int,
) -> ndarray[tuple[int, int], dtype[f64]]: ...
