from __future__ import annotations
from typing import Sequence, TypeVar
import numpy as np

# TODO move somewhere else
#  (for now i want to keep changes in refactor_npc branch contained to tenpy.linalg as much as possible

# TODO make sure everything is still needed


def force_str_len(obj, length: int, rjust: bool = True, placeholder: str = '[...]') -> str:
    """Convert an object to a string, then force the string to a given length.
    If `str(obj)` is too short, right (rjust=True) or left (rjust=False) justify it, filling with spaces.
    If it is too long, replace a central portion with the placeholder.
    """
    assert length >= 0
    obj = str(obj)
    if len(obj) <= length:
        return obj.rjust(length) if rjust else obj.ljust(length)
    else:
        num_chars = length - len(placeholder)
        assert num_chars >= 0, f'Placeholder {placeholder} is longer than length={length}!'
        left_chars = num_chars // 2
        right_chars = num_chars - left_chars
        return obj[:left_chars] + placeholder + obj[-right_chars:]


UNSPECIFIED = object()


def inverse_permutation(permutation: list[int]):
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv


T = TypeVar('T')


def duplicate_entries(seq: Sequence[T], ignore: Sequence[T] = []) -> set[T]:
    return set(ele for idx, ele in enumerate(seq) if ele in seq[idx + 1:] and ele not in ignore)
