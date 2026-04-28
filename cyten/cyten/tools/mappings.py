"""Tools for the sparse mappings that occur when manipulating fusion trees."""
# Copyright (C) TeNPy Developers, Apache license

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Generic, TypeAlias, TypeVar

__all__ = ['SparseMapping', 'IdentityMapping']


_KT = TypeVar('_KT')  # type for keys, labelling basis elements
_Scalar: TypeAlias = float | complex  # type for "entries" of a SparseMapping


class SparseMapping(Generic[_KT], dict[_KT, dict[_KT, _Scalar]]):
    r"""A sparse matrix, where the labels of basis states are a structured type, not just int.

    Used in :class:`cyten.backends.fusion_tree_backend.TreePairMapping` and related objects.

    To represent the mapping ``e_j -> \sum_i A_{ij} e_i``, we store ``self[j][i] = A_{ij}``.
    I.e. a single entry ``self[j][i] = a`` represents the contribution ``e_j -> a e_i``.
    """

    @classmethod
    def from_identity(cls, keys: Iterable[_KT]):
        """The identity mapping ``e_j -> e_j`` on the given keys"""
        res = cls()
        for i in keys:
            res[i] = {i: 1}
        return res

    def pre_compose(self, other: SparseMapping[_KT] | dict[_KT, dict[_KT, _Scalar]]) -> SparseMapping[_KT]:
        r"""The composite ``res_{ik} = \sum_j other_{ij} self{jk}``, such that self acts first.

        I.e. we pre-compose self with other, i.e. compose other with self, i.e.::

            pre_compose(self, other) : x ↦ other(self(x)) = (other ∘ self)(x)
        """
        # e_k -> \sum_j self_{jk} e_j -> \sum_j self_{jk} \sum_i other_{ij} e_i
        # res_{ik} = \sum_j other_{ij} self_{jk}
        # res[k][i] = \sum_j other[j][i] * self[k][j]
        res = SparseMapping()
        for k, self_k in self.items():
            res[k] = res_k = {}
            for j, self_jk in self_k.items():
                if j not in other:
                    continue
                for i, other_ij in other[j].items():
                    res_k[i] = res_k.get(i, 0) + other_ij * self_jk
        return res

    def nonzero_rows(self) -> set[_KT]:
        """The idcs ``i`` for which there are entries ``self_{ij} = self[j][i]`` set."""
        return set(i for self_j in self.values() for i in self_j.keys())

    def nonzero_cols(self) -> set[_KT]:
        """The idcs ``j`` for which there are entries ``self_{ij} = self[j][i]`` set."""
        return set(self.keys())

    def prune(self, tol: float) -> SparseMapping[_KT]:
        """Remove small contributions with ``abs(coefficient) <= tol`` in-place."""
        for j in self.keys():
            self[j] = {i: a for i, a in self[j].items() if abs(a) > tol}
        return self


class IdentityMapping(Generic[_KT]):
    """An identity mapping with same call structure as :class:`SparseMapping`"""

    def __init__(self, keys: Sequence[_KT]):
        self.keys = set(keys)

    def pre_compose(self, other: SparseMapping[_KT] | dict[_KT, dict[_KT, _Scalar]]) -> SparseMapping[_KT]:
        r"""The composite ``res_{ik} = \sum_j other_{ij} self{jk}``, such that self acts first.

        I.e. we pre-compose self with other, i.e. compose other with self, i.e.::

            pre_compose(self, other) : x ↦ other(self(x)) = (other ∘ self)(x)
        """
        # res_{ik} = \sum_j other_{ij} self_{jk} = delta_{k in self} other_{ik}
        res = SparseMapping()
        for k in self.keys:
            if k not in other:
                continue
            res[k] = other[k].copy()
        return res

    def nonzero_rows(self) -> set[_KT]:
        """The idcs ``i`` for which there are entries ``self_{ij} = self[j][i]`` set."""
        return self.keys

    def nonzero_cols(self) -> set[_KT]:
        """The idcs ``j`` for which there are entries ``self_{ij} = self[j][i]`` set."""
        return self.keys

    def prune(self, tol: float):
        """Remove small entries, in-place."""
        pass
