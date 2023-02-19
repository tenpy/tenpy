# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

from __future__ import annotations
import math
from .tensors import DiagonalTensor, Tensor, get_same_backend


def svd(a: Tensor, u_legs: list[int | str] = None, vh_legs: list[int | str] = None,
        new_labels: tuple[str, ...] = None, options=None) -> tuple[Tensor, DiagonalTensor, Tensor]:
    """SVD of a tensor, viewed as a linear map (i.e. matrix) from one set of its legs to the rest.

    TODO: document input format for u_legs / vh_legs, can probably to it centrally for all matrix ops

    Parameters
    ----------
    a : Tensor
    u_legs : list[int  |  str], optional
        Which of the legs belong to "matrix rows" and end up as legs of U.
    vh_legs : list[int  |  str], optional
        Which of the legs belong to "matrix rows" and end up as legs of Vh.
    new_labels : tuple[str, ...], optional
        Specification for the labels of the newly generated leg, i.e. those legs that would need to 
        be contracted to recover `a`.
        If two `str`s, the first is used for "right" legs (that of U and the second of S)
                      the second is used for "left" legs (that of Vh and the first of S)
        If four `str`s, all the above are specified, in the order U, first of S, second of S, Vh
        If None (default), the new legs will not be labelled
    options : dict | Config | None
        TODO: things like driver / stable / ... ?

    Returns
    -------
    tuple[Tensor, DiagonalTensor, Tensor]
        The tensors U, S, Vh that form the SVD, such that `tdot(U, tdot(S, Vh, 1, 0), -1, 0)` is equal
        to `a` up to numerical precsision
    """
    u_idcs, vh_idcs = leg_bipartition(a, u_legs, vh_legs)
    l_u, l_su, l_sv, l_vh = _svd_new_labels(new_labels)

    # TODO read algorithm etc from config
    u_data, s_data, vh_data, new_leg = a.backend.svd(a, u_idcs, vh_idcs)

    U = Tensor(u_data, backend=a.backend, legs=[a.legs[n] for n in u_idcs] + [new_leg], 
               labels=[a.labels[n] for n in u_idcs] + [l_u])
    # TODO revisit this once DiagonalTensor is defined
    S = DiagonalTensor(s_data, backend=a.backend, legs=[new_leg.dual, new_leg], labels=[l_su, l_sv])
    Vh = Tensor(vh_data, backend=a.backend, legs=[new_leg.dual] + [a.legs[n] for n in vh_idcs], 
                labels=[l_vh] + [a.labels[n] for n in vh_idcs])
    return U, S, Vh


def truncate_svd(U: Tensor, S: DiagonalTensor, Vh: Tensor, options=None
                 ) -> tuple[Tensor, DiagonalTensor, Tensor, float]:
    """Truncate an SVD decomposition
    
    Returns
    -------
    U, S, Vh, trunc_err
    """
    backend = get_same_backend(U, S, Vh)
    # TODO implement backend.truncate_svd
    u_data, s_data, vh_data, new_leg, trunc_err = backend.truncate_svd(U, S, Vh, options)
    U = Tensor(u_data, backend=backend, legs=U.legs[:-1] + [new_leg], labels=U.labels)
    # TODO revisit this once DiagonalTensor is defined
    S = DiagonalTensor(s_data, backend=backend, legs=[new_leg.dual, new_leg], labels=S.labels)
    Vh = Tensor(vh_data, backend=backend, legs=[new_leg.dual] + Vh.legs[1:], labels=Vh.labels)
    return U, S, Vh, trunc_err


def svd_split(a: Tensor, legs1: list[int | str] = None, legs2: list[int | str] = None, 
              new_labels: tuple[str, str] = None, options=None, s_exponent: float = .5):
    """Split a tensor via (truncated) svd, 
    i.e. compute (U @ S ** s_exponent) and (S ** (1 - s_exponent) @ Vh)"""
    raise NotImplementedError  # TODO


def _svd_new_labels(new_labels: tuple[str, ...] | None) -> tuple[str, str, str, str]:
    if new_labels is None:
        l_u = l_su = l_sv = l_vh = None
    else:
        try:
            num_labels = len(new_labels)
        except AttributeError:
            raise ValueError('new_labels must be a sequence')
        if num_labels == 2:
            l_u = l_sv = new_labels[0]
            l_su = l_vh = new_labels[1]
        elif num_labels == 4:
            l_u, l_su, l_sv, l_vh = new_labels
        else:
            raise ValueError('expected 2 or 4 new_labels')
    return l_u, l_su, l_sv, l_vh


def leg_bipartition(a: Tensor, legs1: list[int | str] | None, legs2: list[int | str] | None
                    ) -> tuple[list[int] | list[int]]:
    """Utility function for partitioning the legs of a Tensor into two groups.
    The tensor can then unambiguously be understood as a linear map, and linear algebra concepts
    such as SVD, eigensystems, ... make sense.

    The lists `legs1` and `legs2` can bother either be a list, describing via indices (`int`) 
    or via labels (`str`) a subset of the legs of `a`, or they can be `None`.
    - if both are `None`, use the default `legs1=[0]` and `legs2=[1]`, which requires `a` to be a 
      two-leg tensor, which we understand as a matrix
    - if exactly one is `None`, it implies "all remaining legs", 
      i.e. those legs not contained in the other list
    - if both are lists, a `ValueError` is raised, if they are not a bipartition of all legs of `a`

    Returns a list of indices for both groups
    """

    if legs1 is None and legs2 is None:
        if a.num_legs != 2:
            raise ValueError('Expected a two-leg tensor')
        idcs1 = [0]
        idcs2 = [1]
    elif legs1 is None:
        idcs2 = a.get_leg_idcs(legs2)
        idcs1 = [n for n in range(a.num_legs) if n not in idcs2]
    elif legs2 is None:
        idcs1 = a.get_leg_idcs(legs1)
        idcs2 = [n for n in range(a.num_legs) if n not in idcs1]
    else:
        idcs1 = a.get_leg_idcs(legs1)
        idcs2 = a.get_leg_idcs(legs2)
        in_both_lists = [l for l in idcs1 if l in idcs2]
        if in_both_lists:
            raise ValueError('leg groups must be disjoint')
        missing = [l for l in range(a.num_legs) if a not in idcs1 and a not in idcs2]
        if missing:
            raise ValueError('leg groups together must contain all legs')
    return idcs1, idcs2


def exp(t: Tensor | complex | float, legs1: list[int | str] = None, legs2: list[int | str] = None) -> Tensor:
    """
    The exponential of t, viewed as a linear map from legs1 to legs2.
    Requires the two groups of legs to be mutually dual.
    Contrary to numpy, this is *not* the element-wise exponential function.
    """
    if not isinstance(t, Tensor):
        return math.exp(t)
    idcs1, idcs2 = leg_bipartition(legs1, legs2)
    assert len(idcs1) == len(idcs2)
    assert all(t.legs[i1].is_dual_of(t.legs[i2]) for i1, i2 in zip(idcs1, idcs2))
    res_data = t.backend.exp(t, idcs1, idcs2)
    return Tensor(res_data, backend=t.backend, legs=t.legs, labels=t.labels)


def log(t: Tensor | complex | float, legs1: list[int | str] = None, legs2: list[int | str] = None) -> Tensor:
    """
    The (natural) logarithm of t, viewed as a linear map from legs1 to legs2.
    Requires the two groups of legs to be mutually dual.
    Contrary to numpy, this is *not* the element-wise exponential function.
    """
    if not isinstance(t, Tensor):
        return math.log(t)
    idcs1, idcs2 = leg_bipartition(legs1, legs2)
    assert len(idcs1) == len(idcs2)
    assert all(t.legs[i1].is_dual_of(t.legs[i2]) for i1, i2 in zip(idcs1, idcs2))
    res_data = t.backend.log(t, idcs1, idcs2)
    return Tensor(res_data, backend=t.backend, legs=t.legs, labels=t.labels)
    