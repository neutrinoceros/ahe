# pyright: reportUnreachable=false, reportUnnecessaryComparison=false
__all__ = [
    "SlidingTile",
    "Strategy",
    "StrategySpec",
    "SUPPORTED_AHE_KINDS",
    "TileInterpolation",
]

import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from math import ceil
from typing import Literal, Protocol, TypeAlias, TypedDict, TypeVar, final

from ahe._typing import UNSET, Pair, PairSpec, UnsetType

if sys.version_info >= (3, 11):
    from typing import Self, assert_never
else:
    from exceptiongroup import ExceptionGroup
    from typing_extensions import Self, assert_never

SUPPORTED_AHE_KINDS = frozenset({"sliding-tile", "tile-interpolation"})
StrategyKind: TypeAlias = Literal["sliding-tile", "tile-interpolation"]


SlidingTileSpec = TypedDict(
    "SlidingTileSpec", {"kind": Literal["sliding-tile"], "tile-size": PairSpec[int]}
)
TileInterpolationIntoSpec = TypedDict(
    "TileInterpolationIntoSpec",
    {
        "kind": Literal["tile-interpolation"],
        "tile-into": PairSpec[int],
    },
)
TileInterpolationSizeSpec = TypedDict(
    "TileInterpolationSizeSpec",
    {
        "kind": Literal["tile-interpolation"],
        "tile-size": PairSpec[int],
    },
)

StrategySpec: TypeAlias = (
    SlidingTileSpec | TileInterpolationIntoSpec | TileInterpolationSizeSpec
)


def as_pair(s: PairSpec[int], /) -> Pair[int]:
    match s:
        case int():
            return (s, s)
        case (int(s1), int(s2)):
            return (s1, s2)
        case _ as unreachable:
            assert_never(unreachable)  # type: ignore[arg-type]


def report(exceptions: list[Exception]) -> None:
    if len(exceptions) == 1:
        raise exceptions[0]
    elif exceptions:
        raise ExceptionGroup(
            "Found multiple issues with adaptive strategy specifications",
            exceptions,
        )


TS_EVEN = "{axis} tile size {size} is even. Only odd values are allowed."
TS_ODD = "{axis} tile size {size} is odd. Only even values are allowed."
TS_INVALID_BASE = "{axis} tile size {size} is invalid. "
TS_INVALID_EXPECTED = "Expected an {parity} value >={min_value} (or exactly -1)."
TS_INVALID_EXPECTED_ODD = TS_INVALID_BASE + TS_INVALID_EXPECTED.format(
    parity="odd", min_value=3
)
TS_INVALID_EXPECTED_EVEN = TS_INVALID_BASE + TS_INVALID_EXPECTED.format(
    parity="even", min_value=2
)


class Parity(Enum):
    ODD = auto()
    EVEN = auto()


def collect_exceptions_tile_size(
    tile_size: Pair[int], *, require: Parity
) -> list[Exception]:
    exceptions: list[Exception] = []

    match require:
        case Parity.ODD:
            ts_invalid = TS_INVALID_EXPECTED_ODD
            min_value = 3
        case Parity.EVEN:
            ts_invalid = TS_INVALID_EXPECTED_EVEN
            min_value = 2
        case _ as unreachable:
            assert_never(unreachable)

    for axis, size in zip(("y", "x"), tile_size, strict=True):
        if size == -1:
            continue
        if size < min_value:
            exceptions.append(ValueError(ts_invalid.format(axis=axis, size=size)))
        if require is Parity.ODD and not size % 2:
            exceptions.append(ValueError(TS_EVEN.format(axis=axis, size=size)))
        if require is Parity.EVEN and size > 0 and size % 2:
            exceptions.append(ValueError(TS_ODD.format(axis=axis, size=size)))
    return exceptions


INTO_NEG = (
    "Cannot produce {into} tiles on axis {axis}. The minimum meaningful value is 1"
)


def collect_exceptions_tile_into(tile_into: Pair[int]) -> list[Exception]:
    exceptions: list[Exception] = []
    for axis, into in zip(("x", "y"), tile_into, strict=True):
        if into < 1:
            exceptions.append(ValueError(INTO_NEG.format(axis=axis, into=into)))

    return exceptions


def resolve_neg_tile_shapes(tile_shape: Pair[int], image_shape: Pair[int]) -> Pair[int]:
    assert all(s > 0 for s in image_shape)
    base_shape = tile_shape
    ret_shape_mut = list(base_shape)
    for i in range(2):
        s = image_shape[i]
        if base_shape[i] == -1:
            ret_shape_mut[i] = s
    ret_shape = (ret_shape_mut[0], ret_shape_mut[1])
    assert all(s > 0 for s in ret_shape)
    return ret_shape


def minimal_divisor_size(size: int, into: int) -> int:
    d, r = divmod(size, into)
    # 1 if there's a non-zero remainder, 0 otherwise
    round_up = int(bool(r))
    return max(1, d + round_up)


V = TypeVar("V")


class Strategy(Protocol):
    @classmethod
    def from_spec(cls, spec: Mapping[str, V], /) -> Self: ...
    def resolve_tile_shape(self, image_shape: Pair[int], /) -> Pair[int]: ...
    def resolve_pad_width(self, image_shape: Pair[int], /) -> Pair[Pair[int]]: ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class SlidingTile:
    tile_shape: Pair[int]

    @classmethod
    def from_spec(cls, spec: Mapping[str, V], /) -> Self:
        if spec.get("kind") != "sliding-tile":
            raise AssertionError

        tsp: Pair[int] | None = None
        match spec.get("tile-size", UNSET):
            case (int() | (int(), int())) as ts:
                tsp = as_pair(ts)  # type: ignore[arg-type]
            case UnsetType():
                raise TypeError(
                    "Sliding tile specification is missing a 'tile-size' key. "
                    "Expected a single int, or a pair thereof."
                )
            case _ as invalid:
                raise TypeError(
                    "Incorrect type associated with key 'tile-size'. "
                    f"Received {invalid} with type {type(invalid)}. "
                    "Expected a single int, or a pair thereof."
                )

        exceptions = collect_exceptions_tile_size(tsp, require=Parity.ODD)
        report(exceptions)
        return cls(tile_shape=tsp)

    def resolve_tile_shape(self, image_shape: Pair[int], /) -> Pair[int]:
        # only return odd values, rounding up to the next odd number if needed
        ret_mut = [s | 1 for s in resolve_neg_tile_shapes(self.tile_shape, image_shape)]
        return ret_mut[0], ret_mut[1]

    def resolve_pad_width(self, _image_shape: Pair[int], /) -> Pair[Pair[int]]:
        # it is assumed that a tile has an odd size >=3 in both directions.
        # wing size is then trivially half of the preceding even number.

        ts = self.tile_shape
        assert all(s >= 3 for s in ts)
        assert all(s % 2 for s in ts)
        pw = (ts[0] // 2, ts[1] // 2)
        return ((pw[0], pw[0]), (pw[1], pw[1]))


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TileInterpolation:
    tile_into: Pair[int] | None = None
    tile_shape: Pair[int] | None = None

    @classmethod
    def from_spec(cls, spec: Mapping[str, V], /) -> Self:
        if spec.get("kind") != "tile-interpolation":
            raise AssertionError

        exceptions: list[Exception] = []

        tsp: Pair[int] | None = None
        tip: Pair[int] | None = None
        match (spec.get("tile-into", UNSET), spec.get("tile-size", UNSET)):
            case (UnsetType(), UnsetType()):
                exceptions.append(
                    TypeError(
                        "Neither 'tile-into' nor 'tile-size' keys were found. "
                        "Either are allowed, but exactly one is expected."
                    )
                )
            case (((int() | (int(), int())) as ti), UnsetType()):
                tip = as_pair(ti)  # type: ignore[arg-type]
            case (UnsetType(), ((int() | (int(), int())) as ts)):
                tsp = as_pair(ts)  # type: ignore[arg-type]
            case (ts, UnsetType()):
                exceptions.append(
                    TypeError(
                        "Incorrect type associated with key 'tile-into'. "
                        f"Received {ts} with type {type(ts)}. "
                        "Expected a single int, or a pair thereof."
                    )
                )
            case (UnsetType(), ts):
                exceptions.append(
                    TypeError(
                        "Incorrect type associated with key 'tile-size'. "
                        f"Received {ts} with type {type(ts)}. "
                        "Expected a single int, or a pair thereof."
                    )
                )
            case _:
                exceptions.append(
                    TypeError(
                        "Both 'tile-into' and 'tile-size' keys were provided. "
                        "Only one of them can be specified at a time."
                    )
                )

        if tip is not None:
            exceptions.extend(collect_exceptions_tile_into(tip))

        if tsp is not None:
            exceptions.extend(collect_exceptions_tile_size(tsp, require=Parity.EVEN))

        report(exceptions)
        return cls(tile_into=tip, tile_shape=tsp)

    def resolve_tile_shape(self, image_shape: Pair[int], /) -> Pair[int]:
        assert all(s > 0 for s in image_shape)
        assert Counter([self.tile_into, self.tile_shape])[None] == 1

        if self.tile_into is not None:
            ts_mut = [
                minimal_divisor_size(image_shape[0], self.tile_into[0]),
                minimal_divisor_size(image_shape[1], self.tile_into[1]),
            ]
        elif self.tile_shape is not None:
            ts_mut = list(resolve_neg_tile_shapes(self.tile_shape, image_shape))
        else:
            raise AssertionError

        # only return even values, rounding down to the previous even number if needed
        ts_mut = [(s | 1) ^ 1 for s in ts_mut]
        return ts_mut[0], ts_mut[1]

    def resolve_pad_width(self, image_shape: Pair[int], /) -> Pair[Pair[int]]:
        # the padded image should be (in every direction):
        # - an integral multiple of the tile size
        # - as small as possible
        # further more, the pad_width must be at least half the size of a single tile
        if self.tile_shape is None:
            raise AssertionError
        if any(s % 2 for s in image_shape):
            # a direct consequence of using a staggered tiling scheme in the hot loop,
            # and not wanting to allow left/right biases, is that we can allow allow
            # even-sized tile shapes. Combined with the requirements on pad width above,
            # this implies that solutions only exists for even-sized images.
            raise ValueError(
                "Tile interpolation is only supported for images with even sizes in all directions. "
                f"Received image with shape {image_shape}"
            )

        ts = self.tile_shape
        sh = image_shape

        pw = (
            (ceil(sh[0] / ts[0] + 1) * ts[0] - sh[0]) // 2,
            (ceil(sh[1] / ts[1] + 1) * ts[1] - sh[1]) // 2,
        )
        assert not (sh[0] + 2 * pw[0]) % ts[0]
        assert not (sh[1] + 2 * pw[1]) % ts[1]
        assert pw[0] >= ts[0] // 2
        assert pw[1] >= ts[1] // 2
        return ((pw[0], pw[0]), (pw[1], pw[1]))
