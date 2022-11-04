from __future__ import annotations

import numpy as np

from tenpy.linalg.dummy_config import config
from tenpy.linalg.misc import UNSPECIFIED
from tenpy.linalg.symmetries import ProductSpace
from tenpy.linalg.tensors import Tensor, LegPipe


def tdot(t1: Tensor, t2: Tensor, legs1: int | str | list[int | str], legs2: int | str | list[int | str],
         relabel1: dict[str, str] = None, relabel2: dict[str, str] = None) -> Tensor:
    """
    TODO: decide name, eg from tensordot, tdot, contract

    Contraction of two tensors

    Parameters
    ----------
    t1 : Tensor
    t2 : Tensor
    legs1 : int or str or list of int or list of str
        the leg(s) on t1 to be contracted, referenced either by index or by label
    legs2 : int or str of list of int or list of str
        the leg(s) on t2 to be contracted, referenced either by index or by label
    relabel1 : dict
        labels of the result are determined as if t1 had been relabelled by this mapping before contraction
    relabel2
        labels of the result are determined as if t2 had been relabelled by this mapping before contraction

    Returns
    -------

    """

    ax1 = t1.get_leg_idcs(legs1)
    ax2 = t2.get_leg_idcs(legs2)

    assert len(ax1) == len(ax2)
    assert all(t1.legs[n1].can_contract_with(t2.legs[n2]) for n1, n2 in zip(ax1, ax2))

    open_legs1 = [l for n, l in enumerate(t1.legs) if n not in ax1]
    open_legs2 = [l for n, l in enumerate(t2.legs) if n not in ax2]
    open_labels1 = [l for n, l in enumerate(t1._leg_labels) if n not in ax1]
    open_labels2 = [l for n, l in enumerate(t2._leg_labels) if n not in ax2]
    new_labels = result_leg_labels(open_labels1, open_labels2, relabel1, relabel2)

    res_data = t1.backend.tdot(t1.data, t2.data, ax1, ax2)
    res_dtype = t1.backend.infer_dtype(res_data)
    return Tensor(res_data, backend=t1.backend, legs=open_legs1 + open_legs2, leg_labels=new_labels, dtype=res_dtype)


def outer(t1: Tensor, t2: Tensor, relabel1: dict[str, str] = None, relabel2: dict[str, str] = None) -> Tensor:
    """outer product, aka tensor product, aka direct product of two tensors"""
    res_data = t1.backend.outer(t1.data, t2.data)
    res_labels = result_leg_labels(t1.leg_labels, t2.leg_labels, relabel1, relabel2)
    return Tensor(res_data, backend=t1.backend, legs=t1.legs + t2.legs, leg_labels=res_labels,
                  dtype=t1.backend.infer_dtype(res_data))


# TODO return tensor scalar?
def inner(t1: Tensor, t2: Tensor, do_conj: bool = True) -> complex:
    """
    Inner product of two tensors with the same legs.
    t1 and t2 live in the same space, the inner product is the contraction of the dual ("conjugate") of t1 with t2.
    # DOC do_conj arg
    """
    if not do_conj:
        raise NotImplementedError  # TODO
    if config.strict_labels and t1.leg_labels != t2.leg_labels:
        # TODO transpose st labels match. if not possible, raise.
        raise NotImplementedError
    assert t1.legs == t2.legs
    return t1.backend.inner(t1.data, t2.data)


def transpose(t: Tensor, permutation: list[int]) -> Tensor:
    # TODO call this permute_legs or sth?
    if config.strict_labels:
        # TODO: strict labels means position of legs should be irrelevant, there is no need to transpose.
        print('dummy warning!')
    assert len(permutation) == t.num_legs
    assert set(permutation) == set(range(t.num_legs))
    res_data = t.backend.transpose(t.data, permutation)
    return Tensor(res_data, t.backend, [t.legs[n] for n in permutation],
                  leg_labels=[t._leg_labels[n] for n in permutation], dtype=t.dtype)


def trace(t: Tensor, legs1: int | str | list[int | str] = None, legs2: int | str | list[int | str] = None
          ) -> Tensor | float | complex:
    """default leg args: assume there are only two legs and trace over them"""
    if legs1 is None or legs2 is None:
        if legs1 is not None or legs2 is not None:
            raise ValueError
        if t.num_legs != 2:
            raise ValueError
        legs1 = [0]
        legs2 = [1]

    idcs1 = t.get_leg_idcs(legs1)
    idcs2 = t.get_leg_idcs(legs2)
    remaining_idcs = [n for n in range(t.num_legs) if n not in idcs1 and n not in idcs2]
    assert len(idcs1) == len(idcs2)
    assert all(idx not in idcs2 for idx in idcs1)
    assert all(t.legs[idx1].can_contract_with(t.legs[idx2]) for idx1, idx2 in zip(idcs1, idcs2))
    data = t.backend.trace(t, idcs1, idcs2)  # FIXME use t.data
    if len(remaining_idcs) == 0:
        # result is a scalar
        return t.backend.item(data)
    else:
        return Tensor(data, t.backend, legs=[t.legs[n] for n in remaining_idcs],
                      leg_labels=[t._leg_labels[n] for n in remaining_idcs], dtype=t.dtype)


def conj(t: Tensor) -> Tensor:
    """the conjugate of t, living in the dual space"""
    # TODO (Jakob) think about this in the context of pivotal category with duals
    return Tensor(t.backend.conj(t), backend=t.backend, legs=[s.dual for s in t.legs], leg_labels=t.leg_labels,
                  dtype=t.dtype)


# TODO there should be an operation that converts only one or some of the legs to dual
#  i.e. vectorization of density matrices
#  formally, this is contraction with the (co-)evaluation map, aka cup or cap


def combine_legs(t: Tensor, legs: list[int | str], label: str = UNSPECIFIED) -> Tensor:
    """
    Combine a group of legs of a tensor. Result is at the previous position of legs[0].
    # TODO support multiple combines in one function call? what would the signature be?
    """
    idcs = t.get_leg_idcs(legs)
    assert len(idcs) > 1
    pipe = LegPipe([t.legs[n] for n in idcs], old_labels=[t._leg_labels[n] for n in idcs])
    new_legs = [pipe if n == idcs[0] else s for n, s in enumerate(t.legs) if n not in idcs[1:]]
    if label is UNSPECIFIED:
        label = t._leg_labels[idcs[0]]
    new_labels = [label if n == idcs[0] else l for n, l in enumerate(t._leg_labels) if n not in idcs[1:]]
    data = t.backend.combine_legs(t.data, legs)
    return Tensor(data, t.backend, legs=new_legs, leg_labels=new_labels, dtype=t.dtype)


def split_leg(t: Tensor, leg: int | str) -> Tensor:
    """if the legs were contiguous in t.legs before combining, this is the inverse operation of combine_legs,
    otherwise it is only inverse up to a permute_legs"""
    # TODO support multiple splits in one function call? make consistent with combine
    idx = t.get_leg_idx(leg)
    leg = t.legs[idx]
    assert isinstance(leg, LegPipe)
    data = t.backend.split_leg(t.data, idx, orig_spaces=leg.spaces)
    legs = t.legs[:idx] + leg.spaces + t.legs[idx + 1:]
    labels = t._leg_labels[:idx] + leg.old_labels + t.leg_labels[idx + 1:]
    return Tensor(data, t.backend, legs=legs, leg_labels=labels, dtype=t.dtype)


def fuse_transpose(t: Tensor, legs: list[int, list[int]]):
    """
    Permutes and combines legs of a tensor.
    If legs[n] is an int, it describes a leg that is permuted from t.
    If it is a list, it describes several legs of t and the result has a LegPipe which combines them
    """
    # TODO support backend-specific more efficient implementations
    legs = list(map(t.get_leg_idcs, legs))
    flat_legs = sum(legs, [])
    res = transpose(t, flat_legs)
    for m, leg_grp in legs:
        if len(leg_grp) > 1:
            res = combine_legs(t, list(range(m, m + len(leg_grp))))
    return res


def is_scalar(obj) -> bool:
    """If obj is a scalar, meaning either a python scalar like float or complex, or a Tensor
    which has only one-dimensional legs with trivial grading"""
    if isinstance(obj, (int, float, complex)):
        return True
    if isinstance(obj, Tensor):
        return all(l.is_trivial for l in obj.legs)
    else:
        raise TypeError(f'Type not supported for is_scalar: {type(obj)}')


def allclose(a: Tensor, b: Tensor, rtol=1e-05, atol=1e-08) -> bool:
    """
    TODO: name? dont necessarily need to follow numpy convention
    If a and b are equal up to numerical tolerance.
    If the following is True "element-wise" for all numerical parameters (what exactly this means is backend-dependent)
    `abs(a - b) <= atol + rtol * abs(b)`.
    TODO think about edge-cases (nan, inf, ...)
    """
    # TODO what to do for different backends
    backend = None
    if isinstance(a, Tensor):
        backend = a.backend
        a_data = a.data
    else:
        assert is_scalar(a)
        a_data = a

    if isinstance(b, Tensor):
        backend = b.backend
        b_data = b.data
    else:
        assert is_scalar(b)
        b_data = b

    if backend is None:
        return np.allclose(a_data, b_data, rtol=rtol, atol=atol)
    else:
        return backend.allclose(a_data, b_data, rtol=rtol, atol=atol)


def squeeze_leg(t: Tensor, leg: int | str | list[int | str]) -> Tensor:
    """Remove trivial leg from tensor"""
    idcs = t.get_leg_idcs(leg)
    assert all(t.legs[n].is_trivial for n in idcs)
    new_legs = [l for n, l in enumerate(t.legs) if n not in idcs]
    new_labels = [l for n, l in enumerate(t._leg_labels) if n not in idcs]
    return Tensor(t.backend.squeeze_legs(t.data, idcs), t.backend, legs=new_legs, leg_labels=new_labels, dtype=t.dtype)


def norm(t: Tensor) -> float:
    """2-norm of a tensor, i.e. sqrt(inner(t, t))"""
    return t.backend.norm(t.data)


# TODO remaining:
#  Tensor > DiagonalTensor > Scalar
#  define elementwise ops like min, sqrt, real.. for Diagonal
#  scalar edge case
#  do we allow min, max, abs, real, imag...?  maybe "to_real" input is approx real, maybe only Scalar
#  scale_axis ...? not trivial what that even means for non-abelian...
#  pinv?
#  scale_axis (with a DiagonalTensor), or special case of tensordot
#  look at OLD svd options eg degen, need them...?

# TODO in other modules:
#  think about Tensor.__del__, standard probably ok
#  QR
#  eigen
#  elementary functions, such as sin, cos, sqrt, exp, ... which only work on scalars or DiagonalTensors?
#  indexing...; dont even allow?
#  random generation


def result_leg_labels(labels1: list[str | None], labels2: list[str | None],
                      relabel1: dict[str, str] | None, relabel2: dict[str, str] | None
                      ) -> list[str | None]:
    """basically just list concatenation, i.e. labels1 + labels2,
    but with duplicate checks and (optional) relabelling."""

    if relabel1 is not None:
        relabel1 = relabel1.copy()  # this may be inefficient, but should be rewritten in C anyway
        labels1 = [relabel1.pop(l, l) for l in labels1]
        if len(relabel1) > 0:
            raise ValueError(f'relabel1 has superfluous entries: {list(relabel1.keys())}')

    if relabel2 is not None:
        relabel2 = relabel2.copy()
        labels2 = [relabel2.pop(l, l) for l in labels1]
        if len(relabel2) > 0:
            raise ValueError(f'relabel2 has superfluous entries: {list(relabel1.keys())}')

    res_labels = labels1 + labels2
    duplicates = []
    for n, l in enumerate(res_labels):
        if l is None:
            continue

        if l in duplicates:
            res_labels[n] = None
        elif l in res_labels[n + 1:]:
            duplicates.append(l)
            res_labels[n] = None
    # TODO warn if there were duplicates?

    return res_labels
