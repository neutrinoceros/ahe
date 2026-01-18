__all__ = [
    "Pair",
    "PairSpec",
    "UnsetType",
    "UNSET",
]
from enum import Enum, auto
from typing import TypeAlias, TypeVar

T = TypeVar("T")
Pair: TypeAlias = tuple[T, T]
PairSpec: TypeAlias = T | Pair[T]


class UnsetType(Enum):
    UNSET = auto()


UNSET = UnsetType.UNSET
