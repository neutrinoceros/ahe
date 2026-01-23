# high level interface tests
import re

import numpy as np
import numpy.testing as npt
import pytest
from pytest import RaisesExc, RaisesGroup

import ahe
from ahe._api import (
    INTO_NEG,
    TS_EVEN,
    TS_INVALID_EXPECTED_EVEN,
    TS_INVALID_EXPECTED_ODD,
    TS_ODD,
    SlidingTile,
    TileInterpolation,
    as_pair,
)


class TestInvalidStrategy:
    @pytest.mark.parametrize("key", ["tile-into", "tile-size"])
    def test_missing_strategy_kind(self, key):
        with pytest.raises(
            TypeError, match=r"^adaptive_strategy is missing a 'kind' key\.$"
        ):
            ahe.equalize_histogram(
                np.eye(5, dtype="float64"),
                adaptive_strategy={},
            )

    def test_unknown_strategy_kind(self):
        with pytest.raises(
            ValueError,
            match=(
                r"^Unknown strategy kind 'not-a-kind'\. "
                r"Expected one of \['sliding-tile', 'tile-interpolation'\]$"
            ),
        ):
            ahe.equalize_histogram(
                np.eye(5, dtype="float64"),
                adaptive_strategy={"kind": "not-a-kind"},
            )

    def test_invalid_strategy_kind_type(self):
        with pytest.raises(
            TypeError,
            match=(
                r"^Invalid strategy kind None with type <class 'NoneType'>\. "
                r"Expected one of \['sliding-tile', 'tile-interpolation'\]$"
            ),
        ):
            ahe.equalize_histogram(
                np.eye(5, dtype="float64"),
                adaptive_strategy={"kind": None},
            )

    def test_unsupported_dtype(self):
        with pytest.raises(
            TypeError,
            match=(
                r"^Found unsupported data type: int64\. "
                r"Expected of of \[dtype\('float32'\), dtype\('float64'\)\]\.$"
            ),
        ):
            ahe.equalize_histogram(np.eye(3, dtype="int64"), nbins=3)


class TestSlidingTileFromSpec:
    def test_missing_tile_size(self):
        with pytest.raises(
            TypeError,
            match=(
                r"Sliding tile specification is missing a 'tile-size' key\. "
                r"Expected a single int, or a pair thereof\."
            ),
        ):
            SlidingTile.from_spec({"kind": "sliding-tile"})

    def test_invalid_type_tile_size(self):
        with pytest.raises(
            TypeError,
            match=(
                r"Incorrect type associated with key 'tile-size'\. "
                r"Received 1\.5 with type <class 'float'>\. "
                r"Expected a single int, or a pair thereof\."
            ),
        ):
            SlidingTile.from_spec({"kind": "sliding-tile", "tile-size": 1.5})

    @pytest.mark.parametrize(
        "size, axis, msg",
        [
            ((-3, 3), "y", TS_INVALID_EXPECTED_ODD),
            ((3, -3), "x", TS_INVALID_EXPECTED_ODD),
            ((1, 3), "y", TS_INVALID_EXPECTED_ODD),
            ((3, 1), "x", TS_INVALID_EXPECTED_ODD),
            ((4, 3), "y", TS_EVEN),
            ((3, 4), "x", TS_EVEN),
        ],
    )
    def test_single_invalid_tile_size_value(self, size, axis, msg):
        if axis == "y":
            s = size[0]
        else:
            s = size[1]
        with pytest.raises(
            ValueError,
            match=re.escape(msg.format(axis=axis, size=s)),
        ):
            SlidingTile.from_spec({"kind": "sliding-tile", "tile-size": size})

    @pytest.mark.parametrize(
        "size, msg",
        [
            (1, TS_INVALID_EXPECTED_ODD),
            ((1, 1), TS_INVALID_EXPECTED_ODD),
            (4, TS_EVEN),
        ],
    )
    def test_invalid_tile_size_value(self, size, msg):
        ps = as_pair(size)
        with RaisesGroup(
            RaisesExc(
                ValueError,
                match=re.escape(msg.format(axis="x", size=ps[1])),
            ),
            RaisesExc(
                ValueError,
                match=re.escape(msg.format(axis="y", size=ps[0])),
            ),
        ):
            SlidingTile.from_spec({"kind": "sliding-tile", "tile-size": size})

    @pytest.mark.parametrize(
        "spec, expected",
        [
            pytest.param(
                {"kind": "sliding-tile", "tile-size": 13},
                SlidingTile(tile_shape=(13, 13)),
                id="int-tile-size",
            ),
            pytest.param(
                {"kind": "sliding-tile", "tile-size": (13, 15)},
                SlidingTile(tile_shape=(13, 15)),
                id="tuple-tile-size",
            ),
        ],
    )
    def test_valid_inputs(self, spec, expected):
        strategy = SlidingTile.from_spec(spec)
        assert strategy == expected

    @pytest.mark.parametrize("spec", [-1, (-1, -1), (5, -1), (-1, 5)])
    def test_negative_tile_size(self, spec):
        strat = SlidingTile.from_spec({"kind": "sliding-tile", "tile-size": spec})
        assert strat.tile_shape == as_pair(spec)


class TestTileInterpolationFromSpec:
    @pytest.mark.parametrize(
        "size, axis, msg",
        [
            ((-4, 4), "y", TS_INVALID_EXPECTED_EVEN),
            ((4, -4), "x", TS_INVALID_EXPECTED_EVEN),
            ((3, 4), "y", TS_ODD),
            ((4, 3), "x", TS_ODD),
        ],
    )
    def test_single_invalid_tile_size_value(self, size, axis, msg):
        if axis == "y":
            s = size[0]
        else:
            s = size[1]
        with pytest.raises(
            ValueError,
            match=re.escape(msg.format(axis=axis, size=s)),
        ):
            TileInterpolation.from_spec(
                {"kind": "tile-interpolation", "tile-size": size}
            )

    @pytest.mark.parametrize(
        "size, msg",
        [
            (0, TS_INVALID_EXPECTED_EVEN),
            ((0, 0), TS_INVALID_EXPECTED_EVEN),
            (3, TS_ODD),
        ],
    )
    def test_invalid_tile_size_value(self, size, msg):
        ps = as_pair(size)
        with RaisesGroup(
            RaisesExc(
                ValueError,
                match=re.escape(msg.format(axis="x", size=ps[0])),
            ),
            RaisesExc(
                ValueError,
                match=re.escape(msg.format(axis="y", size=ps[1])),
            ),
        ):
            TileInterpolation.from_spec(
                {"kind": "tile-interpolation", "tile-size": size}
            )

    @pytest.mark.parametrize("spec", [-2, -1, 0])
    @pytest.mark.parametrize("axis", ["x", "y"])
    def test_single_invalid_tile_into_value(self, spec, axis):
        if axis == "x":
            into = (spec, 1)
        else:
            into = (1, spec)
        with pytest.raises(
            ValueError,
            match=re.escape(INTO_NEG.format(axis=axis, into=spec)),
        ):
            TileInterpolation.from_spec(
                {"kind": "tile-interpolation", "tile-into": into}
            )

    def test_no_tile_spec(self):
        with pytest.raises(
            TypeError,
            match=(
                r"Neither 'tile-into' nor 'tile-size' keys were found\. "
                r"Either are allowed, but exactly one is expected\."
            ),
        ):
            TileInterpolation.from_spec({"kind": "tile-interpolation"})

    def test_both_tile_spec_keys(self):
        with pytest.raises(
            TypeError,
            match=(
                r"Both 'tile-into' and 'tile-size' keys were provided\. "
                r"Only one of them can be specified at a time\."
            ),
        ):
            TileInterpolation.from_spec(
                {"kind": "tile-interpolation", "tile-into": 11, "tile-size": 13}
            )

    @pytest.mark.parametrize("key", ["tile-into", "tile-size"])
    def test_invalid_type(self, key):
        with pytest.raises(
            TypeError,
            match=(
                f"Incorrect type associated with key {key!r}. "
                f"Received None with type <class 'NoneType'>. "
                "Expected a single int, or a pair thereof."
            ),
        ):
            TileInterpolation.from_spec({"kind": "tile-interpolation", key: None})

    @pytest.mark.parametrize(
        "spec, expected",
        [
            pytest.param(
                {"kind": "tile-interpolation", "tile-size": 14},
                TileInterpolation(tile_shape=(14, 14)),
                id="int-tile-size",
            ),
            pytest.param(
                {"kind": "tile-interpolation", "tile-size": (12, 14)},
                TileInterpolation(tile_shape=(12, 14)),
                id="tuple-tile-size",
            ),
            pytest.param(
                {"kind": "tile-interpolation", "tile-into": 4},
                TileInterpolation(tile_into=(4, 4)),
                id="int-tile-into",
            ),
            pytest.param(
                {"kind": "tile-interpolation", "tile-into": (2, 3)},
                TileInterpolation(tile_into=(2, 3)),
                id="tuple-tile-into",
            ),
        ],
    )
    def test_valid_inputs(self, spec, expected):
        strategy = TileInterpolation.from_spec(spec)
        assert strategy == expected


class TestBoundaries:
    IMAGE_SHAPE = (128, 256)

    def test_invalid_boundaries(self):
        with pytest.raises(
            ValueError,
            match=(
                r"^Received unknown value boundaries=None\. "
                r"Expected 'reflect' or 'periodic'\.$"
            ),
        ):
            ahe.equalize_histogram(np.eye(5, dtype="float64"), boundaries=None)

    @pytest.mark.parametrize(
        "adaptive_strategy",
        [
            pytest.param({"kind": "sliding-tile", "tile-size": 5}, id="sliding-tile"),
            pytest.param(
                {"kind": "tile-interpolation", "tile-size": 64}, id="tile-interpolation"
            ),
        ],
    )
    def test_boundaries(self, adaptive_strategy, subtests):
        prng = np.random.default_rng(0)
        image = prng.normal(size=np.prod(self.IMAGE_SHAPE)).reshape(self.IMAGE_SHAPE)
        res0 = ahe.equalize_histogram(
            image, nbins=8, adaptive_strategy=adaptive_strategy
        )
        res_reflect = ahe.equalize_histogram(
            image,
            nbins=8,
            adaptive_strategy=adaptive_strategy,
            boundaries="reflect",
        )
        with subtests.test("default='reflect'"):
            npt.assert_array_equal(res_reflect, res0)

        res_periodic = ahe.equalize_histogram(
            image,
            nbins=8,
            adaptive_strategy=adaptive_strategy,
            boundaries="periodic",
        )
        with subtests.test("'periodic'!='reflect'"):
            assert np.ptp(np.abs(res_periodic - res_reflect)) > 0.0


@pytest.mark.parametrize(
    "adaptive_strategy",
    [
        pytest.param(None, id="non-adaptive"),
        pytest.param({"kind": "sliding-tile", "tile-size": 5}, id="sliding-tile"),
        pytest.param(
            {"kind": "tile-interpolation", "tile-size": 64}, id="tile-interpolation"
        ),
    ],
)
class TestRegressions:
    IMAGE_SHAPE = (128, 256)

    def test_uniform_image(self, adaptive_strategy):
        image = np.ones(self.IMAGE_SHAPE, dtype="float64")
        res = ahe.equalize_histogram(
            image, nbins=8, adaptive_strategy=adaptive_strategy
        )
        assert res is not image
        npt.assert_array_equal(res, image)

    def test_non_contiguous_image(self, adaptive_strategy):
        image = np.ones(self.IMAGE_SHAPE, dtype="float64")[::2, ::2]
        res = ahe.equalize_histogram(
            image, nbins=8, adaptive_strategy=adaptive_strategy
        )
        assert res is not image
        npt.assert_array_equal(res, image)


def test_non_finite_values(subtests):
    image = np.ones((128, 256), dtype="float64")
    msg = r"^Image contains infinite values or NaNs, none of which are supported\.$"

    for outlier, val in {"NaN": np.nan}.items():
        test_double = image.copy()
        test_double[5, 200] = val
        with subtests.test(outlier=outlier), pytest.raises(ValueError, match=msg):
            ahe.equalize_histogram(test_double)
