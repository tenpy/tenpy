from __future__ import annotations

from tenpy.linalg.matrix_operations.leg_bipartition import leg_bipartition
from tenpy.linalg.tensors import Tensor


def exp(t: Tensor, legs1: list[int | str], legs2: list[int | str]) -> Tensor:
    """
    The exponential of t, viewed as a linear map from legs1 to legs2.
    Requires the two groups of legs to be mutually dual.
    Contrary to numpy, this is *not* the element-wise exponential function.
    """
    idcs1, labels1, spaces1, idcs2, labels2, spaces2 = leg_bipartition(t, legs1, legs2)
    assert len(spaces1) == len(spaces2)
    assert all(s1.can_contract_with(s2) for s1, s2 in zip(spaces1, spaces2))
    res_data = t.backend.exp(t.data, idcs1, idcs2)
    return Tensor(res_data, t.backend, legs=t.legs, leg_labels=t.leg_labels, dtype=t.dtype)


def log(t: Tensor, legs1: list[int | str], legs2: list[int | str]) -> Tensor:
    """
    The logarithm of t, viewed as a linear map from legs1 to legs2.
    Requires the two groups of legs to be mutually dual.
    Contrary to numpy, this is *not* the element-wise logarithm.
    """
    idcs1, labels1, spaces1, idcs2, labels2, spaces2 = leg_bipartition(t, legs1, legs2)
    assert len(spaces1) == len(spaces2)
    assert all(s1.can_contract_with(s2) for s1, s2 in zip(spaces1, spaces2))
    res_data = t.backend.log(t.data, idcs1, idcs2)
    return Tensor(res_data, t.backend, legs=t.legs, leg_labels=t.leg_labels, dtype=t.dtype)
