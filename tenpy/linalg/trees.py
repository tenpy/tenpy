"""TODO module docstring"""
# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations
from typing import Iterator
import numpy as np

from .symmetries import Symmetry, Sector, SectorArray, FusionStyle, SymmetryError
from .dtypes import Dtype
from .backends.abstract_backend import BlockBackend, Block

__all__ = ['FusionTree', 'fusion_trees']


class FusionTree:
    r"""A fusion tree, which represents the map from uncoupled to coupled sectors.

    .. warning ::
        Should think of FusionTrees as immutable.
        Do not act on their attributes with inplace operations, unless you know *exactly* what you
        are doing.

    TODO expand docstring. maybe move drawing to module level docstring.

    Example fusion tree with
        uncoupled = [a, b, c, d]
        are_dual = [False, True, True, False]
        inner_sectors = [x, y]
        multiplicities = [m0, m1, m2]

    |    |
    |    coupled
    |    |
    |    m2
    |    |  \
    |    y   \
    |    |    \
    |    m1    \
    |    |  \   \
    |    x   \   \
    |    |    \   \
    |    m0    \   \
    |    |  \   \   \
    |    a   b   c   d
    |    |   |   |   |
    |    |   Z   Z   |
    |    |   |   |   |


    Notes
    -----
    Consider the ``n``-th vertex (counting 0-based from bottom to top).
    It fuses :math:`a \otimes b \to c` with multiplicity label ``multiplicities[n]``.

        - ``a = uncoupled[0] if n == 0 else inner_sectors[n - 1]``
        - ``b = uncoupled[n + 1]``
        - ``c = coupled if (n == num_vertices - 1) else inner_sectors[n]``
    

    """

    def __init__(self, symmetry: Symmetry,
                 uncoupled: SectorArray | list[Sector],  # N uncoupled sectors
                 coupled: Sector,
                 are_dual: np.ndarray | list[bool],  # N flags: is there a Z isomorphism below the uncoupled sector
                 inner_sectors: SectorArray | list[Sector],  # N - 2 internal sectors
                 multiplicities: np.ndarray | list[int] = None,  # N - 1 multiplicity labels; all 0 per default
                 ):
        # OPTIMIZE demand SectorArray / ndarray (not list) and skip conversions?
        self.symmetry = symmetry
        self.uncoupled = np.asarray(uncoupled)
        self.num_uncoupled = len(uncoupled)
        self.num_vertices = num_vertices = max(len(uncoupled) - 1, 0)
        self.num_inner_edges = max(len(uncoupled) - 2, 0)
        self.coupled = coupled
        self.are_dual = np.asarray(are_dual)
        if len(inner_sectors) == 0:
            inner_sectors = symmetry.empty_sector_array
        self.inner_sectors = np.asarray(inner_sectors)
        if multiplicities is None:
            multiplicities = np.zeros((num_vertices,), dtype=int)
        self.multiplicities = np.asarray(multiplicities)
        self.fusion_style = symmetry.fusion_style
        self.is_abelian = symmetry.is_abelian
        self.braiding_style = symmetry.braiding_style

    def test_sanity(self):
        assert self.symmetry.are_valid_sectors(self.uncoupled)
        assert self.symmetry.is_valid_sector(self.coupled)
        assert len(self.are_dual) == self.num_uncoupled
        assert len(self.inner_sectors) == self.num_inner_edges
        assert self.symmetry.are_valid_sectors(self.inner_sectors)
        assert len(self.multiplicities) == self.num_vertices

        # special cases: no vertices
        if self.num_uncoupled == 0:
            assert np.all(self.coupled == self.symmetry.trivial_sector)
        if self.num_uncoupled == 1:
            assert np.all(self.uncoupled[0] == self.coupled)
        # otherwise, check fusion rules at every vertex
        for vertex in range(self.num_vertices):
            # the two sectors below this vertex
            a = self.uncoupled[0] if vertex == 0 else self.inner_sectors[vertex - 1]
            b = self.uncoupled[vertex + 1]
            # the sector above this vertex
            c = self.inner_sectors[vertex] if vertex < self.num_inner_edges else self.coupled
            N = self.symmetry.n_symbol(a, b, c)
            assert N > 0, 'inconsistent fusion'
            assert 0 <= self.multiplicities[vertex] < N, 'invalid multiplicity label'

    @property
    def pre_Z_uncoupled(self):
        res = self.uncoupled.copy()
        for i, dual in enumerate(self.are_dual):
            if dual:
                res[i, :] = self.symmetry.dual_sector(res[i, :])
        return res

    def __hash__(self) -> int:
        if self.fusion_style == FusionStyle.single:
            # inner sectors are completely determined by uncoupled, all multiplicities are 0
            unique_identifier = (self.are_dual, self.coupled, self.uncoupled)
        elif self.fusion_style == FusionStyle.multiple_unique:
            # all multiplicities are 0
            unique_identifier = (self.are_dual, self.coupled, self.uncoupled, self.inner_sectors)
        else:
            unique_identifier = (self.are_dual, self.coupled, self.uncoupled, self.inner_sectors, self.multiplicities)

        return hash(unique_identifier)

    def __eq__(self, other) -> bool:
        if not isinstance(other, FusionTree):
            return False
        return all([
            np.all(self.coupled == other.coupled),
            np.all(self.uncoupled == other.uncoupled),
            np.all(self.inner_sectors == other.inner_sectors),
            np.all(self.multiplicities == other.multiplicities),
        ])

    @staticmethod
    def _str_uncoupled_coupled(symmetry, uncoupled, coupled, are_dual) -> str:
        """Helper function for string representation.

        Generates a string that represents the uncoupled sectors before the Z isos,
        the uncoupled sectors after and the coupled sector.

        Is also used by ``fusion_trees.__str__``.
        """
        uncoupled_1 = []  # before Zs
        uncoupled_2 = []  # after Zs
        for a, is_dual in zip(uncoupled, are_dual):
            a_str = symmetry.sector_str(a)
            uncoupled_2.append(a_str)
            if is_dual:
                uncoupled_1.append(f'dual({symmetry.sector_str(symmetry.dual_sector(a))})')
            else:
                uncoupled_1.append(a_str)

        before_Z = f'({", ".join(uncoupled_1)})'
        after_Z = f'({", ".join(uncoupled_2)})'
        final = symmetry.sector_str(coupled)
        return f'{before_Z} -> {after_Z} -> {final}'

    def __str__(self) -> str:
        signature = self._str_uncoupled_coupled(
            self.symmetry, self.uncoupled, self.coupled, self.are_dual
        )
        entries = [signature]
        if self.fusion_style in [FusionStyle.multiple_unique, FusionStyle.general]:
            inner_sectors_str = ', '.join(self.symmetry.sector_str(x) for x in self.inner_sectors)
            entries.append(f'({inner_sectors_str})')
        if self.fusion_style == FusionStyle.general:
            entries.append(str(self.multiplicities))
        return f'FusionTree[{str(self.symmetry)}]({", ".join(entries)})'

    def __repr__(self) -> str:
        inner = str(self.inner_sectors).replace('\n', ',')
        uncoupled = str(self.uncoupled).replace('\n', ',')
        return (f'FusionTree({self.symmetry}, {uncoupled}, {self.coupled}, {self.are_dual}, '
                f'{inner}, {self.multiplicities})')

    def as_block(self, backend: BlockBackend = None, dtype: Dtype = None) -> Block:
        """Get the matrix elements of the map as a backend Block.

        If no backend is given, we return it as a numpy array.

        Returns
        -------
        The matrix elements with axes ``[m_a1, m_a2, ..., m_aJ, m_c]``.
        """
        if not self.symmetry.can_be_dropped:
            raise SymmetryError(f'Can not convert to block for symmetry {self.symmetry}')
        if backend is None:
            from .backends.numpy import NumpyBlockBackend
            block_backend = NumpyBlockBackend()
        else:
            block_backend = backend.block_backend
        if dtype is None:
            dtype = self.symmetry.fusion_tensor_dtype
        # handle special cases of small trees
        if dtype is None:
            dtype = self.symmetry.fusion_tensor_dtype
        if self.num_uncoupled == 0:
            # must be identity on the trivial sector. But since there is no uncoupled sector,
            # do not even give it an axis.
            return block_backend.ones_block([1], dtype=dtype)
        if self.num_uncoupled == 1:
            if self.are_dual[0]:
                return self.symmetry.Z_iso(self.coupled)
            else:
                dim_c = self.symmetry.sector_dim(self.coupled)
                return block_backend.eye_block([dim_c], dtype)
        if self.num_uncoupled == 2:
            mu = self.multiplicities[0]
            # OPTIMIZE should we offer a symmetry function to compute only the mu slice?
            X = self.symmetry.fusion_tensor(*self.uncoupled, self.coupled, *self.are_dual)[mu]
            return block_backend.block_from_numpy(X, dtype)  # [a0, a1, c]
        # larger trees: iterate over vertices
        mu0 = self.multiplicities[0]
        X0 = self.symmetry.fusion_tensor(
            self.uncoupled[0], self.uncoupled[1], self.inner_sectors[0],
            Z_a=self.are_dual[0], Z_b=self.are_dual[1]
        )[mu0]
        res = block_backend.block_from_numpy(X0, dtype)  # [a0, a1, i0]
        for vertex in range(1, self.num_vertices):
            mu = self.multiplicities[vertex]
            a = self.inner_sectors[vertex - 1]
            b = self.uncoupled[vertex + 1]
            c = self.inner_sectors[vertex] if vertex < self.num_inner_edges else self.coupled
            X = self.symmetry.fusion_tensor(a, b, c, Z_b=self.are_dual[vertex + 1])[mu]
            X_block =block_backend.block_from_numpy(X, dtype)
            #  [a0, a1, ..., an, i{n-1}] & [i{n-1}, a{n+1}, in] -> [a0, a1, ..., a{n+1}, in]
            res = block_backend.block_tdot(res, X_block, [-1], [0])
        return res

    def copy(self, deep=False) -> FusionTree:
        """Return a shallow (or deep) copy."""
        if deep:
            return FusionTree(self.symmetry, self.uncoupled.copy(), self.coupled.copy(),
                              self.are_dual.copy(), self.inner_sectors.copy())
        return FusionTree(self.symmetry, self.uncoupled, self.coupled, self.are_dual,
                          self.inner_sectors)

    def insert(self, t2: FusionTree) -> FusionTree:
        """Insert a tree `t2` below the first uncoupled sector.

        See Also
        --------
        insert_at
            Inserting at general position
        split
        """
        return FusionTree(
            symmetry=self.symmetry,
            uncoupled=np.concatenate([t2.uncoupled, self.uncoupled[1:]]),
            coupled=self.coupled,
            are_dual=np.concatenate([t2.are_dual, self.are_dual[1:]]),
            inner_sectors=np.concatenate([t2.inner_sectors, self.uncoupled[:1], self.inner_sectors]),
            multiplicities=np.concatenate([t2.multiplicities, self.multiplicities])
        )
        
    def insert_at(self, n: int, t2: FusionTree) -> dict[FusionTree, complex]:
        r"""Insert a tree `t2` below the `n`-th uncoupled sector.

        The result is (in general) not a canonical tree::

            TODO draw
        
        We transform it to canonical form via a series of F moves.
        This yields the result as a linear combination of canonical trees.
        We return a dictionary, with those trees as keys and the prefactors as values.

        Parameters
        ----------
        n : int
            The position to insert at. `t2` is inserted below ``t1.uncoupled[n]``.
            We must have have ``self.are_dual[n] is False``, as we can not have a Z between trees.
        t2 : :class:`FusionTree`
            The fusion tree to insert

        Returns
        -------
        coefficients : dict
            Trees and coefficients that form the above map as a linear combination.
            Abusing notation (``FusionTree`` instances can not actually be scaled or added),
            this means ``map = sum(c * t for t, c in coefficient.items())``.

        See Also
        --------
        insert
            The same insertion, but restricted to ``n=0``, and returns that tree directly, no dict.
        split
        """
        assert self.symmetry == t2.symmetry
        assert np.all(self.uncoupled[n] == t2.coupled)
        assert not self.are_dual[n]

        if t2.num_vertices == 0:
            # t2 has no actual fusion, it is either identity or a Z iso
            if t2.are_dual[0]:
                res = self.copy()
                res.are_dual = self.are_dual.copy()
                res.are_dual[n] = True
                return {res: 1}
            return {self: 1}

        if self.num_vertices == 0:
            return {t2: 1}

        if n == 0:
            # result is already a canonical tree -> no need to do F moves
            return {self.insert(t2): 1}
        
        if t2.num_vertices == 1:
            # inserting a single X tensor
            raise NotImplementedError # TODO
            # - can assume n > 0
            # - do F moves right to left

        # remaining case: t1 has at least 1 vertex and t2 has at least 2.
        # recursively insert: split t2 into a 1-vertex tree and a rest.
        raise NotImplementedError # TODO

    def split(self, n: int) -> tuple[FusionTree, FusionTree]:
        """Split into two separate fusion trees.

        TODO cartoon?

        Parameters
        ----------
        n : int
            Where to split. Must fulfill ``2 <= n < self.num_uncoupled``.

        Returns
        -------
        t1 : :class:`FusionTree`
            The part that fuses the ``uncoupled_sectors[:n]`` to ``inner_sectors[n - 2]``
        t2 : :class:`FusionTree`
            The part that fuses ``inner_sectors[n - 2]`` and ``uncoupled_sectors[n:]``
            to ``coupled``.

        See Also
        --------
        insert
        """
        if n < 2:
            raise ValueError('Left tree has no vertices (n < 2)')
        if n >= self.num_uncoupled:
            raise ValueError('Right tree has no vertices (n >= num_uncoupled)')
        cut_sector = self.inner_sectors[n - 2]
        t1 = FusionTree(
            self.symmetry,
            uncoupled=self.uncoupled[:n],
            coupled=cut_sector,
            are_dual=self.are_dual[:n],
            inner_sectors=self.inner_sectors[:n - 2],
            multiplicities=self.multiplicities[:n - 1],
        )
        t2 = FusionTree(
            self.symmetry,
            uncoupled=np.concatenate([cut_sector[None, :], self.uncoupled[n:]]),
            coupled=self.coupled,
            are_dual=np.insert(self.are_dual[n:], 0, False),
            inner_sectors=self.inner_sectors[n - 1:],
            multiplicities=self.multiplicities[n - 1:],
        )
        return t1, t2


class fusion_trees:
    """Iterator over all :class:`FusionTree`s with given uncoupled and coupled sectors.

    This custom iterator has efficient implementations of ``len`` and :meth:`index`, which
    avoid generating all intermediate trees.

    TODO elaborate on canonical order of trees -> reference in module level docstring.
    """
    def __init__(self, symmetry: Symmetry, uncoupled: SectorArray | list[Sector], coupled: Sector,
                 are_dual=None):
        # DOC: coupled = None means trivial sector
        self.symmetry = symmetry
        if len(uncoupled) == 0:
            uncoupled = symmetry.empty_sector_array
        self.uncoupled = np.asarray(uncoupled)  # OPTIMIZE demand SectorArray (not list) and skip?
        self.num_uncoupled = num_uncoupled = len(uncoupled)
        self.coupled = coupled
        if are_dual is None:
            are_dual = np.zeros((num_uncoupled,), bool)
        else:
            are_dual = np.asarray(are_dual)
        self.are_dual = are_dual

    def __iter__(self) -> Iterator[FusionTree]:
        if len(self.uncoupled) == 0:
            if np.all(self.coupled == self.symmetry.trivial_sector):
                yield FusionTree(self.symmetry, self.uncoupled, self.coupled, [], [], [])
            return
        
        if len(self.uncoupled) == 1:
            if np.all(self.uncoupled[0] == self.coupled):
                yield FusionTree(self.symmetry, self.uncoupled, self.coupled, self.are_dual, [], [])
            return
        
        if len(self.uncoupled) == 2:
            # OPTIMIZE does handling of multiplicities introduce significant overhead?
            #          could do a specialized version for multiplicity-free fusion
            for mu in range(self.symmetry.n_symbol(*self.uncoupled, self.coupled)):
                yield FusionTree(self.symmetry, self.uncoupled, self.coupled, self.are_dual, [], [mu])
            return
            
        a1 = self.uncoupled[0]
        a2 = self.uncoupled[1]
        for b in self.symmetry.fusion_outcomes(a1, a2):
            uncoupled = np.concatenate([b[None, :], self.uncoupled[2:]])
            are_dual = np.concatenate([[False], self.are_dual[2:]])
            # set multiplicity index to 0 for now. will adjust it later.
            left_tree = FusionTree(self.symmetry, self.uncoupled[:2], b, self.are_dual[:2],
                                    [], [0])
            for rest_tree in fusion_trees(self.symmetry, uncoupled, self.coupled, are_dual):
                tree = rest_tree.insert(left_tree)
                for mu in range(self.symmetry._n_symbol(a1, a2, b)):
                    res = tree.copy()
                    res.multiplicities = res.multiplicities.copy()
                    res.multiplicities[0] = mu
                    yield res

    def __len__(self) -> int:
        # OPTIMIZE caching ?

        if len(self.uncoupled) == 0:
            if np.all(self.coupled == self.symmetry.trivial_sector):
                return 1
            return 0

        if len(self.uncoupled) == 1:
            if np.all(self.uncoupled[0] == self.coupled):
                return 1
            return 0

        if len(self.uncoupled) == 2:
            return self.symmetry.n_symbol(*self.uncoupled, self.coupled)

        a1 = self.uncoupled[0]
        a2 = self.uncoupled[1]
        count = 0
        for b in self.symmetry.fusion_outcomes(a1, a2):
            uncoupled = np.concatenate([b[None, :], self.uncoupled[2:]])
            num_subtrees = len(fusion_trees(self.symmetry, uncoupled, self.coupled))
            count += self.symmetry.n_symbol(a1, a2, b) * num_subtrees
        return count

    def __str__(self):
        signature = FusionTree._str_uncoupled_coupled(
            self.symmetry, self.uncoupled, self.coupled, self.are_dual
        )
        return f'fusion_trees[{str(self.symmetry)}]({signature})'

    def __repr__(self):
        uncoupled = str(self.uncoupled).replace('\n', ',')
        return f'fusion_trees({self.symmetry}, {uncoupled}, {self.coupled}, {self.are_dual})'

    def index(self, tree: FusionTree) -> int:
        # TODO check compatibility first (same symmetry, same uncoupled, same coupled)
        # TODO inefficient dummy implementation, can exploit __len__ of iterator over subtrees
        # to know how many we need to skip.
        for n, t in enumerate(self):
            if t == tree:
                return n
        raise ValueError(f'Tree not found.')
