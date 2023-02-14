from __future__ import annotations

from .backends.abstract_backend import AbstractBackend
from .dummy_config import config
from .symmetries import ProductSpace
from .tensors import Tensor, combine_leg_labels, split_leg_label, match_label_order


def tdot(t1: Tensor, t2: Tensor,
         legs1: int | str | list[int | str] = -1, legs2: int | str | list[int | str] = 0,
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
    leg_idcs1 = t1.get_leg_idcs(legs1)
    leg_idcs2 = t2.get_leg_idcs(legs2)
    if len(leg_idcs1) != len(leg_idcs2):
        # checking this for leg_idcs* instead of legs* allows us to assume that they are both lists
        raise ValueError('Must specify the same number of legs for both tensors')
    if not all(t1.legs[idx1].can_contract_with(t2.legs[idx2]) for idx1, idx2 in zip(leg_idcs1, leg_idcs2)):
        raise ValueError('Incompatible legs.')  # TODO show which
    backend = get_same_backend(t1, t2)
    open_legs1 = [leg for idx, leg in enumerate(t1.legs) if idx not in leg_idcs1]
    open_legs2 = [leg for idx, leg in enumerate(t2.legs) if idx not in leg_idcs2]
    open_labels1 = [leg for idx, leg in enumerate(t1.labels) if idx not in leg_idcs1]
    open_labels2 = [leg for idx, leg in enumerate(t2.labels) if idx not in leg_idcs2]
    res_labels = get_result_labels(open_labels1, open_labels2, relabel1, relabel2)
    res_data = backend.tdot(t1.data, t2.data, leg_idcs1, leg_idcs2)
    return Tensor(res_data, backend=backend, legs=open_legs1 + open_legs2, labels=res_labels)


def outer(t1: Tensor, t2: Tensor, relabel1: dict[str, str] = None, relabel2: dict[str, str] = None) -> Tensor:
    """outer product, aka tensor product, aka direct product of two tensors"""
    backend = get_same_backend(t1, t2)
    res_labels = get_result_labels(t1.labels, t2.labels, relabel1, relabel2)
    res_data = backend.outer(t1.data, t2.data)
    return Tensor(res_data, backend=backend, legs=t1.legs + t2.legs, labels=res_labels)


def inner(t1: Tensor, t2: Tensor) -> complex:
    """
    Inner product of two tensors with the same legs.
    t1 and t2 live in the same space, the inner product is the contraction of the dual ("conjugate") of t1 with t2

    If config.strict_labels, legs with matching labels are contracted.
    Otherwise the n-th leg of t1 is contracted with the n-th leg of t2
    """
    if t1.num_legs != t2.num_legs:
        raise ValueError('Tensors need to have the same number of legs')
    leg_order_2 = match_label_order(t1, t2)
    if not all(t1.legs[n1].space == t2.legs[n2].space for n1, n2 in enumerate(leg_order_2)):
        raise ValueError('Incompatible legs')
    backend = get_same_backend(t1, t2)
    res = backend.inner(t1.data, t2.data, axs2=leg_order_2)
    # TODO: Scalar(Tensor) class...?
    return res


def transpose(t: Tensor, permutation: list[int]) -> Tensor:
    """Change the order of legs of a Tensor.
    TODO: also have an inplace version?
    TODO: name it permute_legs or sth instead?
    """
    if config.strict_labels:
        # TODO: proper warning:
        # strict labels means position of legs should be irrelevant, there is no need to transpose.
        print('dummy warning!')
    assert len(permutation) == t.num_legs
    assert set(permutation) == set(range(t.num_legs))
    res_data = t.backend.transpose(t.data)
    return Tensor(res_data, backend=t.backend, legs=[t.legs[n] for n in permutation],
                  labels=[t.labels[n] for n in permutation])


def trace(t: Tensor, legs1: int | str | list[int | str] = -2, legs2: int | str | list[int | str] = -1
          ) -> Tensor | float | complex:
    """
    Trace over one or more pairs of legs, that is contract these pairs.
    """
    leg_idcs1 = t.get_leg_idcs(legs1)
    leg_idcs2 = t.get_leg_idcs(legs2)
    if len(leg_idcs1) != len(leg_idcs2):
        raise ValueError('Must specify same number of legs')
    remaining_leg_idcs = [n for n in range(t.num_legs) if n not in leg_idcs1 and n not in leg_idcs2]
    res_data = t.backend.trace(t, leg_idcs1, leg_idcs2)
    if len(remaining_leg_idcs) == 0:
        # result is a scalar
        return t.backend.item(res_data)
    else:
        return Tensor(res_data, backend=t.backend, legs=[t.legs[n] for n in remaining_leg_idcs],
                      labels=[t.labels[n] for n in remaining_leg_idcs])


def conj(t: Tensor) -> Tensor:
    """
    The conjugate of t, living in the dual space.
    Labels are adjuste as `'p'` -> `'p*'` and `'p*'` -> `'p'`
    """
    # TODO (Jakob) think about this in the context of pivotal category with duals
    return Tensor(t.backend.conj(t), backend=t.backend, legs=[l.dual for l in t.legs])


# TODO there should be an operation that converts only one or some of the legs to dual
#  i.e. vectorization of density matrices
#  formally, this is contraction with the (co-)evaluation map, aka cup or cap


def combine_legs(t: Tensor, legs: list[int | str], new_leg: ProductSpace = None) -> Tensor:
    """
    Combine a group of legs of a tensor. Resulting leg (of type ProductSpace) is at the
    previous position of legs[0].
    # TODO support multiple combines in one function call? what would the signature be
    # TODO inplace version
    """
    if len(legs) < 2:
        raise ValueError('expected at least two legs')

    leg_idcs = t.get_leg_idcs(legs)
    if new_leg is None:
        new_leg = ProductSpace([t.legs[idx] for idx in leg_idcs])
    old_legs = [t.legs[idx] for idx in leg_idcs]
    res_legs = [new_leg if idx == leg_idcs[0] else leg for idx, leg in enumerate(t.legs)
            if idx not in leg_idcs[1:]]
    new_label = combine_leg_labels(t.leg_labels)
    res_labels = [new_label if idx == leg_idcs[0] else label for idx, label in enumerate(t.leg_labels)
              if idx not in leg_idcs[1:]]
    res_data = t.backend.combine_legs(t.data, leg_idcs=leg_idcs, old_legs=old_legs, new_leg=new_leg)
    return Tensor(res_data, backend=t.backend, legs=res_legs, labels=res_labels)


def split_leg(t: Tensor, leg: int | str) -> Tensor:
    """
    Split a leg that was previously combined.
    If the legs were contiguous in t.legs before combining, this is the inverse operation of combine_legs,
    otherwise it is the inverse up to a permute_legs
    # TODO support multiple splits? -> make consistent with combine
    # TODO inplace version
    """
    leg_idx = t.get_leg_idx(leg)
    if not isinstance(t.legs[leg_idx]):
        raise ValueError(f'Leg {leg} is not a ProductSpace.')
    legs = t.legs[:leg_idx] + t.legs[leg_idx].spaces + t.legs[leg_idx + 1:]
    labels = t.labels[:leg_idx] + split_leg_label(t.labels[leg_idx]) + t.labels[leg_idx + 1:]
    res_data = t.backend.split_leg(t.data, leg_idx=leg_idx, leg=t.legs[leg_idx])
    return Tensor(res_data, backend=t.backend, legs=legs, labels=labels)


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
    If a and b are equal up to numerical tolerance, that is if `norm(a - b) <= atol + rtol * norm(a)`.
    Note that the definition is not symmetric under exchanging `a` and `b`.
    """
    assert rtol >= 0
    assert atol >= 0
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        diff = norm(a - b)
        a_norm = norm(a)
    else:
        if isinstance(a, Tensor):
            try:
                a = a.item()
            except ValueError:
                raise ValueError('Can not compare non-scalar Tensor and scalar') from None
        if isinstance(b, Tensor):
            try:
                b = b.item()
            except ValueError:
                raise ValueError('Can not compare scalar and non-scalar Tensor') from None
        diff = abs(a - b)
        a_norm = abs(a)
    return diff <= atol + rtol * a_norm


ALL_TRIVIAL_LEGS = object()


def squeeze_legs(t: Tensor, legs: int | str | list[int | str] = ALL_TRIVIAL_LEGS) -> Tensor:
    """
    Remove trivial leg from tensor.
    If legs are specified, they are squeezed if they are trivial and a ValueError is raised if not.
    If no legs are specified, all trivial legs are squeezed
    """
    if legs is ALL_TRIVIAL_LEGS:
        leg_idcs = [n for n, l in enumerate(t.legs) if l.is_trivial]
    else:
        leg_idcs = t.get_leg_idcs(legs)
        if not all(t.legs[idx].is_trivial for idx in leg_idcs):
            raise ValueError('Tried to squeeze non-trivial legs.')
    res_legs = [l for idx, l in enumerate(t.legs) if idx not in leg_idcs]
    res_labels = [label for idx, label in enumerate(t.labels) if idx not in leg_idcs]
    res_data = t.backend.squeeze_legs(t.data, leg_idcs)
    return Tensor(res_data, backend=t.backend, legs=res_legs, labels=res_labels)


def norm(t: Tensor) -> float:
    """2-norm of a tensor, i.e. sqrt(inner(t, t))"""
    return t.backend.norm(t.data)


def get_result_labels(legs1: list[str | None], legs2: list[str | None],
                      relabel1: dict[str, str] | None, relabel2: dict[str, str] | None) -> list[str]:
    """
    Utility function to combine two lists of leg labels, such that they can appear on the same tensor.
    Labels are changed by the mappings relabel1 and relabel2.
    Any conflicting labels (after relabelling) are dropped
    """
    relabel1 = relabel1 or {}
    relabel2 = relabel2 or {}

    labels1 = [relabel1.get(leg.label, leg.label) for leg in legs1]
    labels2 = [relabel2.get(leg.label, leg.label) for leg in legs2]
    conflicting = [label for label in labels1 if label in labels2]
    labels = labels1 + labels2
    if conflicting:
        # TODO issue warning
        labels = [None if label in conflicting else label for label in labels]
    return labels


# TODO move this somewhere else?
def get_same_backend(*tensors: Tensor, error_msg: str = 'Incompatible backends.') -> AbstractBackend:
    """If all tensors have the same backend, return it. Otherwise raise a ValueError"""
    try:
        backend = tensors[0].backend
    except IndexError:
        raise ValueError('expected at least one tensor') from None
    if not all(tens.backend == backend for tens in tensors):
        raise ValueError(error_msg)
    return backend
