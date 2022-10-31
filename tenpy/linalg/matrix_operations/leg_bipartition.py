from __future__ import annotations

from tenpy.linalg.symmetries import VectorSpace
from tenpy.linalg.tensors import Tensor


def leg_bipartition(a: Tensor, legs1: list[int | str] | None, legs2: list[int | str] | None
                    ) -> tuple[list[int], list[str | None], list[VectorSpace],
                               list[int], list[str | None], list[VectorSpace]]:
    """
    Utility function for partitioning the legs of a Tensor into two groups.
    The tensor can then unambiguously be understood as a linear map, and linear algebra concepts
    such as SVD, eigensystems, ... make sense.

    The lists `legs1` and `legs2` can bother either be a list, describing via indices (int) or via labels (str)
    a subset of the legs of `a`, or they can be None.
    - if both are None, use the default legs1=[0] and legs2=[1], which requires `a` to be a two-leg tensor,
      which we understand as a matrix
    - if exactly one is None, it implies "all remaining legs", i.e. those legs not contained in the other list
    - if both are list, a ValueError is raised, if they are not a bipartition of all legs of `a`

    Returns
    -------
    idcs1, labels1, spaces1, idcs2, labels2, vh_spaces2
        For both subsets of legs, their indices (i.e. positions in a.legs), labels and spaces (i.e. entries of a.legs)
    """

    if legs1 is None and legs2 is None:
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
            raise ValueError
        missing = [l for l in range(a.num_legs) if a not in idcs1 and a not in idcs2]
        if missing:
            raise ValueError

    labels1 = [a._leg_labels[n] for n in idcs1]
    labels2 = [a._leg_labels[n] for n in idcs2]
    spaces1 = [a.legs[n] for n in idcs1]
    spaces2 = [a.legs[n] for n in idcs2]

    return idcs1, labels1, spaces1, idcs2, labels2, spaces2
