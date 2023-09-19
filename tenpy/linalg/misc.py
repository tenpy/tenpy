# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

from __future__ import annotations
from typing import Sequence, TypeVar
import numpy as np
from functools import wraps

__all__ = ['force_str_len', 'UNSPECIFIED', 'inverse_permutation', 'duplicate_entries',
           'join_as_many_as_possible', 'make_stride', 'find_row_differences']

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
        res = obj[:left_chars] + placeholder
        if right_chars > 0:
            res = res + obj[-right_chars:]
        return res


UNSPECIFIED = object()


# TODO is actually implemented in tools.misc ...
def inverse_permutation(permutation: list[int]):
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv


_T = TypeVar('_T')


def duplicate_entries(seq: Sequence[_T], ignore: Sequence[_T] = []) -> set[_T]:
    return set(ele for idx, ele in enumerate(seq) if ele in seq[idx + 1:] and ele not in ignore)


def join_as_many_as_possible(msgs: Sequence[str], separator: str, priorities: Sequence[int] = None,
                             max_len: int = None, fill: str = '...') -> str:
    """Like ``separator.join(msgs)`` but if the result is too long, only some of the `msgs`
    will be included. Higher values in `priorities` are prioritized
    (default: prioritize what appears earlier in `msgs`).
    """
    if len(msgs) == 0:
        return ''
    if sum(len(m) for m in msgs) + len(separator) * (len(msgs) - 1) <= max_len:
        return separator.join(msgs)

    if priorities is None:
        order = range(len(msgs))
    else:
        order = np.argsort(-np.array(priorities))

    # build arrays whose n-th elemnt represent the resulting length if n+1 msgs are kept
    cum_lengths = np.cumsum([len(msgs[n]) for n in order])
    candidate_lengths = cum_lengths + np.arange(1, len(msgs) + 1) * len(separator) + len(fill)
    num_msgs = np.where(candidate_lengths > max_len)[0][0]

    return separator.join([msgs[n] for n in order[:num_msgs]] + [fill])


_MAX_INT = np.iinfo(int).max


def make_stride(shape, cstyle=True):
    """Create the strides for C- (or F-style) arrays with a given shape.

    Equivalent to ``x = np.zeros(shape); return np.array(x.strides, np.intp) // x.itemsize``.

    Note that ``np.sum(inds * _make_stride(np.max(inds, axis=0), cstyle=False), axis=1)`` is
    sorted for (positive) 2D `inds` if ``np.lexsort(inds.T)`` is sorted.
    """
    L = len(shape)
    stride = 1
    res = np.empty([L], np.intp)
    if cstyle:
        res[L - 1] = 1
        for a in range(L - 1, 0, -1):
            stride *= shape[a]
            res[a - 1] = stride
        assert stride * shape[0] < _MAX_INT
    else:
        res[0] = 1
        for a in range(0, L - 1):
            stride *= shape[a]
            res[a + 1] = stride
        assert stride * shape[0] < _MAX_INT
    return res


def find_row_differences(sectors, include_len: bool=False):
    """Return indices where the rows of the 2D array `sectors` change.

    Parameters
    ----------
    sectors : 2D array
        The rows of this array are compared.
    include_len : bool
        If ``len(sectors)`` should be included or not.
    
    Returns
    -------
    diffs: 1D array
        The indices where rows change, including the first and last. Equivalent to:
        ``[0] + [i for i in range(1, len(sectors)) if np.any(qflat[i-1] != qflat[i])]``
    """
    # note: by default remove last entry [len(sectors)] compared to old.charges
    len_sectors = len(sectors)
    diff = np.ones(len_sectors + int(include_len), dtype=np.bool_)
    diff[1:len_sectors] = np.any(sectors[1:] != sectors[:-1], axis=1)
    return np.nonzero(diff)[0]  # get the indices of True-values
