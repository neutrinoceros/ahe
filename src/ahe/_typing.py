__all__ = [
    "Pair",
    "PairSpec",
    "UNSET",
]
import sys
from typing import TypeAlias, TypeVar

if sys.version_info < (3, 15):
    from typing_extensions import sentinel

T = TypeVar("T")
Pair: TypeAlias = tuple[T, T]
PairSpec: TypeAlias = T | Pair[T]


UNSET = sentinel("UNSET")
