import re

import pytest

import ahe
from ahe._api import SlidingTile, TileInterpolation, minimal_divisor_size


@pytest.mark.parametrize(
    "nbins, shape, expected",
    [
        (32, (8, 8), 32),
        (64, (8, 8), 64),
        (256, (8, 8), 256),
        ("auto", (8, 8), 64),
        ("auto", (4, 4), 16),
        ("auto", (1024, 2048), 256),
    ],
)
def test_auto_nbins(nbins, shape, expected):
    res = ahe._lib._resolve_nbins(nbins, shape)
    assert res == expected


@pytest.mark.parametrize(
    "size, into, expected",
    [
        (3, 1, 3),
        (3, 2, 2),
        (3, 3, 1),
        (4, 1, 4),
        (4, 2, 2),
        (4, 3, 2),
        (4, 4, 1),
        (5, 1, 5),
        (5, 2, 3),
        (5, 3, 2),
        (5, 4, 2),
        (5, 5, 1),
    ],
)
def test_minimal_size_divisor(size, into, expected):
    assert minimal_divisor_size(size, into) == expected


@pytest.mark.parametrize("image_shape", [(5, 6), (6, 5), (5, 5)])
def test_resolve_pad_width_tile_interpolation_odd_image(image_shape):
    strat = TileInterpolation(tile_shape=(4, 4))
    with pytest.raises(
        ValueError,
        match="^"
        + re.escape(
            "Tile interpolation is only supported for images with even sizes "
            f"in all directions. Received image with shape {image_shape}"
        )
        + "$",
    ):
        strat.resolve_pad_width(image_shape)


@pytest.mark.parametrize(
    "strat, image_shape, expected",
    [
        (SlidingTile(tile_shape=(3, 3)), (4, 4), ((1, 1), (1, 1))),
        (SlidingTile(tile_shape=(5, 5)), (5, 6), ((2, 2), (2, 2))),
        (TileInterpolation(tile_shape=(2, 2)), (4, 4), ((2, 2), (2, 2))),
        (TileInterpolation(tile_shape=(4, 4)), (4, 4), ((4, 4), (4, 4))),
        (TileInterpolation(tile_shape=(2, 4)), (8, 8), ((2, 2), (4, 4))),
    ],
)
def test_resolve_pad_width(strat, image_shape, expected):
    assert strat.resolve_pad_width(image_shape) == expected


@pytest.mark.parametrize(
    "tile_shape, image_shape, expected_shape",
    [
        ((-1, -1), (5, 7), (5, 7)),
        ((-1, 3), (5, 7), (5, 3)),
        ((3, -1), (5, 7), (3, 7)),
        # even values are allowed in containing shapes, but padded in the output
        ((-1, -1), (6, 8), (7, 9)),
        ((-1, 3), (6, 8), (7, 3)),
        ((3, -1), (6, 8), (3, 9)),
    ],
)
def test_sliding_tile_resolve(tile_shape, image_shape, expected_shape):
    st = SlidingTile(tile_shape=tile_shape)
    assert st.resolve_tile_shape(image_shape) == expected_shape


@pytest.mark.parametrize(
    "strat, image_shape, expected_shape",
    [
        (TileInterpolation(tile_into=(2, 2)), (256, 256), (128, 128)),
        (TileInterpolation(tile_into=(2, 3)), (256, 256), (128, 86)),
        (TileInterpolation(tile_into=(5, 1)), (256, 256), (52, 256)),
        (TileInterpolation(tile_shape=(64, 64)), (256, 256), (64, 64)),
    ],
)
def test_tile_interpolation_resolve(strat, image_shape, expected_shape):
    assert strat.resolve_tile_shape(image_shape) == expected_shape
