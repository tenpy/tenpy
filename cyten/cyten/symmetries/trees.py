"""TODO module docstring"""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import prod
from typing import TYPE_CHECKING

import numpy as np

from ..block_backends import Block, NumpyBlockBackend
from ..block_backends.dtypes import Dtype
from ..tools import to_valid_idx
from ._symmetries import FusionStyle, Sector, SectorArray, Symmetry, SymmetryError

if TYPE_CHECKING:
    from ..backends import TensorBackend


class FusionTree:
    r"""A fusion tree, which represents the map from uncoupled to coupled sectors.

    Consider the following example tree::

        FusionTree(
            symmetry=symmetry,
            coupled=coupled,
            uncoupled=[a, b, c, d],
            are_dual=[False, True, True, False],
            inner_sectors=[x, y],
            multiplicities=[i, j, k],
        )

    Graphically::

        |    a     b     c     d     <- isomorphic to pre_Z_uncoupled
        |    v     ^     ^     v        e.g. dual(b) iso to pre_Z_uncoupled[1]
        |    │     Z     Z     │
        |    v     v     v     v
        |    a     b     c     d     <- uncoupled
        |    ╰──i──╯     │     │
        |      x│        │     │
        |       ╰───j────╯     │
        |          y│          │
        |           ╰────k─────╯
        |                │
        |                coupled

    Attributes
    ----------
    symmetry : Symmetry
        The symmetry.
    uncoupled : 2D array of int
        N uncoupled sectors. These are the sectors *below* any Z isos.
        I.e. the generalized tree, including the Zs, maps from the :attr:`pre_Z_sectors` instead.
    coupled : 1D array of int
        The coupled sector at the bottom of the tree.
    are_dual : 1D array of bool
        N flags: is there a Z isomorphism above the uncoupled sector.
    inner_sectors : 2D array of int
        N - 2 internal sectors, at the internal edges of the tree.
    multiplicities : 1D array of int
        N - 1 multiplicity labels, at the fusion vertices of the tree.

    Notes
    -----
    Consider the ``n``-th vertex (counting 0-based from top to bottom).
    It fuses :math:`e \otimes f \to g` with multiplicity label ``multiplicities[n]``.

        - ``e = uncoupled[0] if n == 0 else inner_sectors[n - 1]``
        - ``f = uncoupled[n + 1]``
        - ``g = coupled if (n == num_vertices - 1) else inner_sectors[n]``

    """

    def __init__(
        self,
        symmetry: Symmetry,
        uncoupled: SectorArray | list[Sector],  # N uncoupled sectors
        coupled: Sector,
        are_dual: np.ndarray | list[bool],  # N flags: is there a Z isomorphism above the uncoupled sector
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
        self.are_dual = np.asarray(are_dual, dtype=bool)
        if len(inner_sectors) == 0:
            inner_sectors = symmetry.empty_sector_array
        # empty lists are by default converted to arrays with dtype=float, which leads to issues in __hash__
        self.inner_sectors = np.asarray(inner_sectors, dtype=int)
        if multiplicities is None:
            multiplicities = np.zeros((num_vertices,), dtype=int)
        self.multiplicities = np.asarray(multiplicities, dtype=int)
        self.fusion_style = symmetry.fusion_style
        self.is_abelian = symmetry.is_abelian
        self.braiding_style = symmetry.braiding_style

    def test_sanity(self):
        """Perform sanity checks."""
        assert self.symmetry.are_valid_sectors(self.uncoupled), 'invalid uncoupled'
        assert self.symmetry.is_valid_sector(self.coupled), 'invalid coupled'
        assert len(self.are_dual) == self.num_uncoupled, 'wrong length of are_dual'
        assert len(self.inner_sectors) == self.num_inner_edges, 'wrong length of inner_sectors'
        assert self.symmetry.are_valid_sectors(self.inner_sectors), 'invalid inner sectors'
        assert len(self.multiplicities) == self.num_vertices, 'invalid length of multiplicities'

        # special cases: no vertices
        if self.num_uncoupled == 0:
            assert np.all(self.coupled == self.symmetry.trivial_sector)
        if self.num_uncoupled == 1:
            assert np.all(self.uncoupled[0] == self.coupled)
        # otherwise, check fusion rules at every vertex
        for n in range(self.num_vertices):
            a, b, mu, c = self.vertex_labels(n)
            N = self.symmetry.n_symbol(a, b, c)
            assert N > 0, 'inconsistent fusion'
            assert 0 <= mu < N, 'invalid multiplicity label'

    @classmethod
    def from_abelian_symmetry(
        cls, symmetry: Symmetry, uncoupled: Sequence[Sector], are_dual: Sequence[bool]
    ) -> FusionTree:
        """Assume an abelian symmetry and build the unique tree with the given `uncoupled`.

        For an abelian symmetry, two sectors fuse to a single other sector, such that the entire
        tree is determined by the uncoupled sectors alone.
        """
        assert symmetry.is_abelian
        if len(uncoupled) == 0:
            return cls.from_empty(symmetry=symmetry)
        if len(uncoupled) == 1:
            return cls.from_sector(symmetry=symmetry, sector=uncoupled[0], is_dual=are_dual[0])
        fusion_outcomes = []
        last_sector = uncoupled[0]
        for a in uncoupled[1:]:
            f = symmetry.fusion_outcomes(last_sector, a)[0]
            fusion_outcomes.append(f)
            last_sector = f
        return FusionTree(
            symmetry=symmetry,
            uncoupled=uncoupled,
            coupled=fusion_outcomes[-1],
            are_dual=are_dual,
            inner_sectors=fusion_outcomes[:-1],
            multiplicities=None,
        )

    @classmethod
    def from_empty(cls, symmetry: Symmetry):
        """The empty tree with no uncoupled sectors."""
        return FusionTree(
            symmetry,
            uncoupled=symmetry.empty_sector_array,
            coupled=symmetry.trivial_sector,
            are_dual=[],
            inner_sectors=symmetry.empty_sector_array,
            multiplicities=[],
        )

    @classmethod
    def from_sector(cls, symmetry: Symmetry, sector: Sector, is_dual: bool):
        """A tree with a single uncoupled sector and no nodes."""
        return FusionTree(
            symmetry,
            uncoupled=[sector],
            coupled=sector,
            are_dual=[is_dual],
            inner_sectors=symmetry.empty_sector_array,
            multiplicities=[],
        )

    @property
    def pre_Z_uncoupled(self):
        """The uncoupled sectors *above* any Z isomorphisms."""
        res = self.uncoupled.copy()
        res[self.are_dual, :] = self.symmetry.dual_sectors(res[self.are_dual, :])
        return res

    def __hash__(self) -> int:
        if self.fusion_style == FusionStyle.single:
            # inner sectors are completely determined by uncoupled, all multiplicities are 0
            unique_identifier = [self.are_dual, self.coupled, self.uncoupled]
        elif self.fusion_style == FusionStyle.multiple_unique:
            # all multiplicities are 0
            unique_identifier = [self.are_dual, self.coupled, self.uncoupled, self.inner_sectors]
        else:
            unique_identifier = [self.are_dual, self.coupled, self.uncoupled, self.inner_sectors, self.multiplicities]

        return hash(tuple(hash(tuple(arr.flatten().tolist())) for arr in unique_identifier))

    def __eq__(self, other) -> bool:
        if not isinstance(other, FusionTree):
            return False
        return (
            np.all(self.are_dual == other.are_dual)
            and np.all(self.coupled == other.coupled)
            and np.all(self.uncoupled == other.uncoupled)
            and np.all(self.inner_sectors == other.inner_sectors)
            and np.all(self.multiplicities == other.multiplicities)
        )

    def _ascii_diagram(self, dagger: bool, uncoupled_padding=2, inner_sector_padding=0) -> np.ndarray:
        """The :meth:`ascii_diagram` as a 2D array of single characters."""
        # We build a splitting tree (dagger=True), and if dagger=False, we mirror it at the end

        assert uncoupled_padding > 0
        assert inner_sector_padding >= 0

        uncoupled_strs = [self.symmetry.sector_str(a) for a in self.uncoupled]
        pre_Z_uncoupled_strs = [self.symmetry.sector_str(a) for a in self.pre_Z_uncoupled]

        # single-letter sectors dont work with the design choice of attaching wires to the
        # second character of a sector -> make them at least 2 characters
        uncoupled_strs = [s.rjust(2) for s in uncoupled_strs]
        pre_Z_uncoupled_strs = [s.rjust(2) for s in pre_Z_uncoupled_strs]

        # pad the uncoupled sectors in a single column to a consistent width
        uncoupled_widths = [max(len(s), len(s2)) for s, s2 in zip(uncoupled_strs, pre_Z_uncoupled_strs)]
        uncoupled_strs = [s.ljust(w) for w, s in zip(uncoupled_widths, uncoupled_strs)]
        pre_Z_uncoupled_strs = [s.ljust(w) for w, s in zip(uncoupled_widths, pre_Z_uncoupled_strs)]

        # special cases with no fusion vertices
        if self.num_uncoupled == 0:
            return np.array(list('empty FusionTree'), dtype=str)[:, None]
        if self.num_uncoupled == 1:
            ascii = np.full((uncoupled_widths[0], 5), ' ', dtype=str)
            ascii[:, 0] = list(uncoupled_strs[0])
            # note: arrows are the same no matter if dagger or not
            ascii[1, [1, 2, 3]] = ['v', 'Z', '^'] if self.are_dual[0] else ['v', '│', 'v']
            ascii[:, 4] = list(pre_Z_uncoupled_strs[0])
            if not dagger:
                ascii = ascii[:, ::-1]
            return ascii

        coupled_str = self.symmetry.sector_str(self.coupled)
        inner_sector_strs = [self.symmetry.sector_str(s) for s in self.inner_sectors]

        # step 1: build just the bare tree without inner sectors or multiplicity labels
        #         only remember where they would go
        vertex_positions = []  # positions of the ┴
        num_rows_uncoupled = 5
        num_rows_coupled = 1
        num_rows = num_rows_uncoupled + 2 * self.num_vertices + num_rows_coupled
        uncoupled_pos = [sum(uncoupled_widths[:i]) + i * uncoupled_padding for i in range(self.num_uncoupled)]
        num_cols = sum(uncoupled_widths) + self.num_uncoupled * uncoupled_padding
        ascii = np.full((num_cols, num_rows), ' ', dtype=str)
        # last line: pre_Z_uncoupled
        for pos, s in zip(uncoupled_pos, uncoupled_strs):
            ascii[pos : pos + len(s), -1] = list(s)
        # line -2: Z or vertical wires
        for pos, has_Z in zip(uncoupled_pos, self.are_dual):
            # note: arrows are the same no matter if dagger or not
            ascii[pos + 1, [-4, -3, -2]] = ['v', 'Z', '^'] if has_Z else ['v', '│', 'v']
        # line -3: uncoupled
        for pos, s in zip(uncoupled_pos, pre_Z_uncoupled_strs):
            ascii[pos : pos + len(s), -5] = list(s)
        # fusion vertices
        row = num_rows - 1 - num_rows_uncoupled
        left_wire = uncoupled_pos[0] + 1
        for n in range(self.num_vertices):
            right_wire = uncoupled_pos[n + 1] + 1
            ascii[right_wire, row + 1 : -num_rows_uncoupled] = '│'
            vertex = (left_wire + right_wire) // 2
            ascii[left_wire, row] = '╭' if dagger else '╰'
            ascii[left_wire + 1 : vertex, row] = '─'
            ascii[vertex, row] = '┴' if dagger else '┬'
            vertex_positions.append((vertex, row))
            ascii[vertex + 1 : right_wire, row] = '─'
            ascii[right_wire, row] = '╮' if dagger else '╯'
            ascii[vertex, row - 1] = '│'
            # for next iteration:
            left_wire = vertex
            row = row - 2
        assert row == 0
        coupled_pos = left_wire - 1
        ascii[coupled_pos : coupled_pos + len(coupled_str), 0] = list(coupled_str)

        left_overhangs = {}  # {row: extra_str}
        for (x, y), s in zip(vertex_positions[:-1], inner_sector_strs):
            row = y - 1  # one above the vertex
            start = x - len(s)
            if start < 0:
                left_overhangs[row] = s[: abs(start)]
                ascii[:x, row] = list(s[abs(start) :])
            else:
                ascii[start:x, row] = list(s)
        if len(left_overhangs) > 0:
            extra_left = np.full((max(len(s) for s in left_overhangs.values()), num_rows), ' ', str)
            for row, extra_s in left_overhangs.items():
                extra_left[-len(extra_s) :, row] = list(extra_s)
        else:
            extra_left = np.zeros((0, num_rows), str)

        if self.symmetry.fusion_style > FusionStyle.multiple_unique:
            # need to print multiplicities
            for (x, y), mult in zip(vertex_positions, self.multiplicities):
                mult = str(mult)
                if len(mult) == 1:
                    ascii[x, y] = mult
                elif len(mult) == 2:
                    ascii[x : x + 2, y] = list(mult)
                elif len(mult) == 3:
                    ascii[x - 1 : x + 2, y] = list(mult)
                else:
                    raise NotImplementedError('Multiplicity with >3 digits not supported.')

        # finalize
        ascii = np.concatenate([extra_left, ascii], axis=0)
        if not dagger:
            ascii = ascii[:, ::-1]

        return ascii

    def ascii_diagram(self, dagger=False) -> str:
        """Visual representation of the tree as ASCII art."""
        ascii = self._ascii_diagram(dagger=dagger)
        return '\n'.join(''.join(row) for row in ascii.T)

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

    @staticmethod
    def bend_leg(
        X: FusionTree,
        Y: FusionTree,
        bend_downward: bool,
        do_conj: bool = False,
    ) -> dict[tuple[FusionTree, FusionTree], float | complex]:
        r"""Bend a leg on a tree-pair, return the resulting linear combination of tree-pairs.

        Graphically::

            |    bend_downward=True                    bend_downward=False
            |
            |   │   │   │   ╭────╮                    │   │   │   │    │
            |   ┢━━━┷━━━┷━━━┷━┓  │                    ┢━━━┷━━━┷━━━┷━┓  │
            |   ┡━━━━━━━━━━━━━┛  │                    ┡━━━━━━━━━━━━━┛  │
            |   │                │                    │                │
            |   ┢━━━━━━━━━━━━━┓  │                    ┢━━━━━━━━━━━━━┓  │
            |   ┡━━━┯━━━┯━━━┯━┛  │                    ┡━━━┯━━━┯━━━┯━┛  │
            |   │   │   │   │    │                    │   │   │   ╰────╯

        Parameters
        ----------
        X, Y : FusionTree
            The original tree pair, such that we modify ``hconj(X) @ Y``.
            Note that `X` is a fusion tree that represents the splitting tree ``hconj(X)``.
        bend_downward : bool
            Whether the rightmost leg of `Y` is bent down (``bend_downward == True``) or the rightmost
            leg of ``hconj(X)`` is bent up (``bend_downward == False``).
        do_conj : bool
            If ``True``, return the conjugate of the coefficients instead.

        Returns
        -------
        linear_combination : dict {FusionTree: complex}
            The bent tree pair is a linear combination ``bent = sum_i a_i hconj(Y_i) @ X_i`` of tree
            pairs (where ``Y_i`` is a fusion tree and thus ``hconj(Y_i)`` a splitting tree).
            The returned dictionary has entries ``linear_combination[Y_i, X_i] = a_i`` for the
            contributions to this linear combination (i.e. tree pairs for which the coefficient
            vanishes are omitted).

        """
        if not bend_downward:
            # OPTIMIZE: do it explicitly instead?
            # bend_up(dagger(Y) @ X)
            # == dagger(dagger(bend_up(dagger(Y) @ X))
            # == dagger(bend_down(dagger(dagger(Y) @ X))))
            # == dagger(bend_down(dagger(X) @ Y))
            # == dagger(sum_i b_i (dagger(X_i) @ Y_i))
            # == sum_i conj(b_i) dagger(Y_i) @ X_i
            # i.e. we need to swap the order of inputs and invert bend_downward,
            # then for the result, swap the trees back and conj the coefficients (invert do_conj)
            other = FusionTree.bend_leg(Y, X, bend_downward=True, do_conj=not do_conj)
            return {(Y_i, X_i): b_i for (X_i, Y_i), b_i in other.items()}

        # OPTIMIZE remove input checks?
        assert Y.symmetry == X.symmetry
        symmetry = Y.symmetry
        assert np.all(Y.coupled == X.coupled)
        c = Y.coupled

        if Y.num_uncoupled == 0:
            raise ValueError('No leg to be bent.')

        is_dual = Y.are_dual[-1]

        if Y.num_uncoupled == 1:
            X_i = FusionTree.from_empty(symmetry)
            Y_i = X.extended(
                new_uncoupled=symmetry.dual_sector(c), mu=0, new_coupled=symmetry.trivial_sector, is_dual=not is_dual
            )
            b_i = symmetry.sqrt_qdim(c)
            if is_dual:
                b_i = b_i * symmetry.frobenius_schur(c)
            return {(Y_i, X_i): b_i}

        X_i, c, mu, z = Y.split_bottom_vertex()

        if X.num_uncoupled == 0:
            e = X_i.coupled
            Y_i = FusionTree.from_sector(symmetry, e, is_dual=not is_dual)
            b_i = symmetry.inv_sqrt_qdim(e)
            if not is_dual:
                b_i = b_i * symmetry.frobenius_schur(e)
            return {(Y_i, X_i): b_i}

        B = symmetry.b_symbol(X_i.coupled, z, c)
        chi_z = symmetry.frobenius_schur(z)
        zbar = symmetry.dual_sector(z)
        res = {}
        for nu in range(B.shape[1]):
            b_i = B[mu, nu]
            Y_i = X.extended(zbar, nu, X_i.coupled, not is_dual)
            if is_dual:
                b_i = b_i * chi_z
            if do_conj:
                b_i = np.conj(b_i)
            res[Y_i, X_i] = b_i
        return res

    def braid(
        self,
        j: int,
        overbraid: bool,
        cutoff: float = 1e-16,
        do_conj: bool = False,
    ) -> dict[FusionTree, float | complex]:
        r"""Braid a leg on a fusion tree, return the resulting linear combination of trees.

        Graphically::

            |   overbraid:                  underbraid
            |
            |   │   │   │   │               │   │   │   │
            |   │    ╲ ╱    │               │    ╲ ╱    │
            |   │     ╱     │               │     ╲     │
            |   │    ╱ ╲    │               │    ╱ ╲    │
            |   │   j  j+1  │               │   j  j+1  │
            |   ┢━━━┷━━━┷━━━┷━┓             ┢━━━┷━━━┷━━━┷━┓
            |   ┡━━━━━━━━━━━━━┛             ┡━━━━━━━━━━━━━┛
            |   │                           │

        .. warning ::
            When braiding splitting trees (daggers of fusion trees), consider the notes below.

        Parameters
        ----------
        j : int
            The index for the braid. We braid ``uncoupled[j]`` with ``uncoupled[j + 1]``.
        overbraid : bool
            If we apply an overbraid or an underbraid (see graphic above).
        cutoff : float
            We skip contributions with a prefactor below this.
        do_conj : bool
            If ``True``, return the conjugate of the coefficients instead.

        Returns
        -------
        linear_combination : dict {FusionTree: complex}
            The braided fusion tree is a linear combination ``braided_self = sum_i a_i X_i``.
            The returned dictionary has entries ``linear_combination[X_i] = a_i`` for the
            contributions to this linear combination (i.e. trees for which the coefficient vanishes
            may be omitted).

        """
        assert 0 <= j < self.num_uncoupled - 1
        if j == 0:  # R-move
            a, b, mu, c = self.vertex_labels(0)
            if overbraid:
                a_i = self.symmetry.r_symbol(a, b, c)[mu]
            else:
                a_i = np.conj(self.symmetry.r_symbol(b, a, c)[mu])
            if do_conj:
                a_i = np.conj(a_i)
            X_i = self.copy(deep=True)
            X_i.uncoupled[0] = b
            X_i.uncoupled[1] = a
            X_i.are_dual[:2] = X_i.are_dual[1::-1]
            return {X_i: a_i}

        # C-move
        res = {}
        a, b, mu, e = self.vertex_labels(j - 1)
        _e, c, nu, d = self.vertex_labels(j)
        X_new = self.copy(deep=True)
        X_new.uncoupled[j] = c
        X_new.uncoupled[j + 1] = b
        X_new.are_dual[j] = self.are_dual[j + 1]
        X_new.are_dual[j + 1] = self.are_dual[j]
        for f in self.symmetry.fusion_outcomes(a, c):
            if not self.symmetry.can_fuse_to(f, b, d):
                continue
            if overbraid:
                C_sym = self.symmetry.c_symbol(a, b, c, d, e, f)[mu, nu]
            else:
                # underbraid compared to overbraid:
                #  - conj
                #  - b <-> c  [in args of c_symbol(...)]
                #  - e <-> f  [in args of c_symbol(...)]
                #  - (mu,nu) <-> (kappa,lambda)  [by indexing c_symbol(...) differently]
                C_sym = np.conj(self.symmetry.c_symbol(a, c, b, d, f, e)[:, :, mu, nu])
            if do_conj:
                C_sym = np.conj(C_sym)
            for kappa, C_kappa in enumerate(C_sym):
                for lambda_, a_i in enumerate(C_kappa):
                    if abs(a_i) < cutoff:
                        continue
                    X_i = X_new.copy(deep=True)
                    X_i.inner_sectors[j - 1] = f
                    X_i.multiplicities[j - 1] = kappa
                    X_i.multiplicities[j] = lambda_
                    assert X_i not in res  # OPTIMIZE rm check
                    res[X_i] = a_i
        return res

    def vertex_labels(self, n: int) -> tuple[Sector, Sector, int, Sector]:
        r"""For the ``n``-th fusion vertex, get the respective sectors.

        Returns
        -------
        a, b, mu, c
            The sectors and multiplicity label around the ``n``-th vertex of the tree::

                |   (n-1 higher vertices)      │
                |                      │       │
                |                      a       b
                |                      ╰───µ───╯
                |                          c
                |                          │
                |                          (possibly lower vertices)

        """
        if n == 0:
            a, b = self.uncoupled[:2]
        else:
            a = self.inner_sectors[n - 1]
            b = self.uncoupled[n + 1]
        if n == self.num_vertices - 1:
            c = self.coupled
        else:
            c = self.inner_sectors[n]
        return a, b, self.multiplicities[n], c

    def modify_vertex_labels(self, n: int, a: Sector, b: Sector, mu: int, c: Sector, copy: bool = True) -> FusionTree:
        """Update the multiplicity and the three sectors around the ``n``-th vertex.

        Parameters
        ----------
        n : int
            The vertex.
        a, b, mu, c
            Three sectors and a multiplicity, like the returns of :meth:`vertex_labels`.
            ``None`` place-holders indicate to not update that value.
        copy : bool
            If ``True``, we return a modified copy. If ``False``, we modify in place and return
            the modified instance.

        """
        if copy:
            return self.copy(deep=True).modify_vertex_labels(n, a=a, b=b, mu=mu, c=c, copy=False)
        if n == 0:
            self.uncoupled[0] = a
        else:
            self.inner_sectors[n - 1] = a
        self.uncoupled[n + 1] = b
        if n == self.num_vertices - 1:
            self.coupled = c
        else:
            self.inner_sectors[n] = c
        self.multiplicities[n] = mu
        return self

    def __str__(self) -> str:
        ascii = self._ascii_diagram(dagger=False)
        res = f'<FusionTree   symmetry: {self.symmetry!s}>'
        for row in ascii.T:
            res = res + '\n    |   ' + ''.join(row)
        return res

    def __repr__(self) -> str:
        inner = str(self.inner_sectors).replace('\n', ',')
        uncoupled = str(self.uncoupled).replace('\n', ',')
        return (
            f'FusionTree({self.symmetry}, {uncoupled}, {self.are_dual}, coupled={self.coupled}, '
            f'inner_sectors={inner}, multiplicities={self.multiplicities})'
        )

    def as_block(self, backend: TensorBackend = None, dtype: Dtype = None) -> Block:
        """Get the matrix elements of the map as a backend Block.

        If no backend is given, we return it as a numpy array.

        Returns
        -------
        The matrix elements with axes ``[m_a1, m_a2, ..., m_aJ, m_c]``.

        """
        if not self.symmetry.can_be_dropped:
            raise SymmetryError(f'Can not convert to block for symmetry {self.symmetry}')
        if backend is None:
            block_backend = NumpyBlockBackend()
        else:
            block_backend = backend.block_backend
        if dtype is None:
            dtype = self.symmetry.fusion_tensor_dtype
        # handle special cases of small trees
        if self.num_uncoupled == 0:
            # must be identity on the trivial sector. But since there is no uncoupled sector,
            # do not even give it an axis.
            return block_backend.ones_block([1], dtype=dtype)
        if self.num_uncoupled == 1:
            if self.are_dual[0]:
                Z = self.symmetry.Z_iso(self.symmetry.dual_sector(self.uncoupled[0]))
                # [m_c, m_a1] -> need to transpose!
                return block_backend.block_from_numpy(Z.T, dtype=dtype)
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
            self.uncoupled[0], self.uncoupled[1], self.inner_sectors[0], Z_a=self.are_dual[0], Z_b=self.are_dual[1]
        )[mu0]
        res = block_backend.block_from_numpy(X0, dtype)  # [a0, a1, i0]
        for vertex in range(1, self.num_vertices):
            mu = self.multiplicities[vertex]
            a = self.inner_sectors[vertex - 1]
            b = self.uncoupled[vertex + 1]
            c = self.inner_sectors[vertex] if vertex < self.num_inner_edges else self.coupled
            X = self.symmetry.fusion_tensor(a, b, c, Z_b=self.are_dual[vertex + 1])[mu]
            X_block = block_backend.block_from_numpy(X, dtype)
            #  [a0, a1, ..., an, i{n-1}] & [i{n-1}, a{n+1}, in] -> [a0, a1, ..., a{n+1}, in]
            res = block_backend.tdot(res, X_block, [-1], [0])
        return res

    def copy(self, deep=True) -> FusionTree:
        """Return a shallow (or deep) copy."""
        if deep:
            return FusionTree(
                self.symmetry,
                self.uncoupled.copy(),
                self.coupled.copy(),
                self.are_dual.copy(),
                self.inner_sectors.copy(),
                self.multiplicities.copy(),
            )
        return FusionTree(
            self.symmetry, self.uncoupled, self.coupled, self.are_dual, self.inner_sectors, self.multiplicities
        )

    def extended(self, new_uncoupled: Sector, mu: int, new_coupled: Sector, is_dual: bool):
        r"""A new tree, from adding a new fusion node at the bottom, below the coupled sector.

        Graphically::

            |               │
            |              (Z)
            |               v
            |   (self)     new_uncoupled
            |       │       │
            |       ╰───µ───╯
            |           │
            |          new_coupled

        See Also
        --------
        insert
            Can insert nodes "above"
        split_topmost
            Split off the topmost node.

        """
        if self.num_uncoupled == 0:
            assert mu == 0
            multiplicities = []
        else:
            multiplicities = np.append(self.multiplicities, mu)
        if self.num_uncoupled < 2:
            # result has one vertex, and thus no inner sectors
            inner_sectors = self.inner_sectors
        else:
            inner_sectors = np.append(self.inner_sectors, self.coupled[None, :], axis=0)
        return FusionTree(
            self.symmetry,
            uncoupled=np.append(self.uncoupled, new_uncoupled[None, :], axis=0),
            coupled=new_coupled,
            are_dual=np.append(self.are_dual, is_dual),
            inner_sectors=inner_sectors,
            multiplicities=multiplicities,
        )

    def insert(self, t2: FusionTree) -> FusionTree:
        """Insert a tree `t2` above the first uncoupled sector.

        See Also
        --------
        insert_at
            Inserting at general position
        split
            Split into two separate fusion trees.

        """
        return FusionTree(
            symmetry=self.symmetry,
            uncoupled=np.concatenate([t2.uncoupled, self.uncoupled[1:]]),
            coupled=self.coupled,
            are_dual=np.concatenate([t2.are_dual, self.are_dual[1:]]),
            inner_sectors=np.concatenate([t2.inner_sectors, self.uncoupled[:1], self.inner_sectors]),
            multiplicities=np.concatenate([t2.multiplicities, self.multiplicities]),
        )

    def insert_at(self, n: int, t2: FusionTree, eps: float = 1.0e-14) -> dict[FusionTree, complex]:
        r"""Insert a tree `t2` above the `n`-th uncoupled sector.

        The result is (in general) not a canonical tree.
        We transform it to canonical form via a series of F moves.
        This yields the result as a linear combination of canonical trees.
        We return a dictionary, with those trees as keys and the prefactors as values.

        Parameters
        ----------
        n : int
            The position to insert at. `t2` is inserted above ``t1.uncoupled[n]``.
            We must have have ``self.are_dual[n] is False``, as we can not have a Z between trees.
        t2 : :class:`FusionTree`
            The fusion tree to insert
        eps : float
            F symbols whose absolute values are smaller than this number are treated as zero.

        Returns
        -------
        coefficients : dict
            Trees and coefficients that form the composite map as a linear combination.
            Abusing notation (``FusionTree`` instances can not actually be scaled or added),
            this means ``map = sum(c * t for t, c in coefficient.items())``.

        See Also
        --------
        insert
            The same insertion, but restricted to ``n=0``, and returns that tree directly, no dict.
        split
            Split into two separate fusion trees.

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

        # should be more efficient than using recursion
        sym = self.symmetry
        coefficients = {}
        new_unc = np.vstack((self.uncoupled[:n], t2.uncoupled, self.uncoupled[n + 1 :]))
        new_dual = np.concatenate([self.are_dual[:n], t2.are_dual, self.are_dual[n + 1 :]])
        new_inners_left = self.inner_sectors[: n - 1]
        new_inners_right = self.inner_sectors[n - 1 :]
        new_multis_left = self.multiplicities[: n - 1]
        new_multis_right = self.multiplicities[n:]

        # build the remaining parts (inner and multiplicities) from the right
        a = self.uncoupled[0] if len(new_inners_left) == 0 else new_inners_left[-1]
        d_initial = self.coupled if n == self.num_uncoupled - 1 else new_inners_right[0]
        tree_parts = {(tuple(), (self.multiplicities[n - 1],)): 1}
        for i in range(t2.num_uncoupled - 1, 0, -1):
            new_tree_parts = {}  # contains new inner_sectors and multiplicities
            for (inners, multis), amplitude in tree_parts.items():
                b = t2.inner_sectors[i - 2] if i > 1 else t2.uncoupled[0]
                c = t2.uncoupled[i]
                d = np.asarray(inners[0], dtype=int) if len(inners) > 0 else d_initial
                e = t2.coupled if len(inners) == 0 else t2.inner_sectors[i - 1]
                multi = t2.multiplicities[i - 1]
                for f in sym.fusion_outcomes(a, b):
                    if not sym.can_fuse_to(f, c, d):
                        continue
                    fs = sym._f_symbol(a, b, c, d, e, f)[multi, multis[0], :, :]
                    for (kap, lam), factor in np.ndenumerate(fs):
                        if abs(factor) < eps:
                            continue
                        new_parts = ((tuple(f), *inners), (kap, lam, *multis[1:]))
                        if new_parts in new_tree_parts:
                            new_tree_parts[new_parts] += amplitude * factor
                        else:
                            new_tree_parts[new_parts] = amplitude * factor
            tree_parts = new_tree_parts

        for (inners, multis), amplitude in tree_parts.items():
            inners = np.asarray(inners, dtype=int)
            new_inners = np.vstack((new_inners_left, inners, new_inners_right))
            new_multis = np.concatenate([new_multis_left, multis, new_multis_right])
            new_tree = FusionTree(sym, new_unc, self.coupled, new_dual, new_inners, new_multis)
            coefficients[new_tree] = amplitude
        return coefficients

    def outer(self, right_tree: FusionTree, eps: float = 1.0e-14) -> dict[FusionTree, complex]:
        r"""Outer product with another tree.

        Fuse with `right_tree` at the coupled sector (-> new coupled sectors are all sectors that
        are allowed fusion channels of the coupled sectors).

        Parameters
        ----------
        right_tree : FusionTree
            Tree to be combined with at the coupled sector from the right.
        eps : float
            F symbols whose absolute values are smaller than this number are treated as zero.

        Returns
        -------
        linear_combination : dict {FusionTree: complex}
            Result expressed as linear combination of fusion trees in the canonical basis with the
            corresponding coefficients.

        See Also
        --------
        insert_at
            Similar insertion, but the tree is inserted above of an uncoupled sector rather than
            fused with the coupled sector.

        """
        # trivial cases
        if self.num_uncoupled == 0:
            return {right_tree: 1}
        if right_tree.num_uncoupled == 0:
            return {self: 1}

        # use self.insert_at(right_tree) -> construct new tree with
        # right_tree.coupled as uncoupled sector at the end
        sym = self.symmetry
        res = {}
        unc = np.vstack((self.uncoupled, right_tree.coupled))
        dual = np.concatenate([self.are_dual, [False]])
        if self.num_uncoupled <= 1:
            inner = np.zeros((0, unc.shape[1]), dtype=int)
        else:
            inner = np.vstack((self.inner_sectors, self.coupled))
        for new_coupled in sym.fusion_outcomes(self.coupled, right_tree.coupled):
            for m in range(sym._n_symbol(self.coupled, right_tree.coupled, new_coupled)):
                multi = np.concatenate([self.multiplicities, [m]])
                tree = FusionTree(
                    symmetry=sym,
                    uncoupled=unc,
                    coupled=new_coupled,
                    are_dual=dual,
                    inner_sectors=inner,
                    multiplicities=multi,
                )
                res.update(tree.insert_at(self.num_uncoupled, right_tree, eps=eps))
        return res

    def split(self, n: int) -> tuple[FusionTree, FusionTree]:
        """Split into two separate fusion trees.

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
            inner_sectors=self.inner_sectors[: n - 2],
            multiplicities=self.multiplicities[: n - 1],
        )
        t2 = FusionTree(
            self.symmetry,
            uncoupled=np.concatenate([cut_sector[None, :], self.uncoupled[n:]]),
            coupled=self.coupled,
            are_dual=np.insert(self.are_dual[n:], 0, False),
            inner_sectors=self.inner_sectors[n - 1 :],
            multiplicities=self.multiplicities[n - 1 :],
        )
        return t1, t2

    def split_bottom_vertex(self) -> tuple[FusionTree, Sector, int, Sector]:
        """Split off the bottom vertex.

        Graphically::

            |   a b x y z           a  b  x  y     z
            |   │ │ │ │ │           │  │  │  │     │
            |   (self_tree)    =    (rest_tree)    │
            |       │                    │         │
            |       c                    ╰────µ────╯
            |                                 │
            |                                 c

        where `rest_tree` might be empty if ``self.num_uncoupled == 1`` or consist of
        only a single sector with no fusion vertex if ``self.num_uncoupled == 2``.

        Returns
        -------
        rest_tree : FusionTree
            The remaining tree, with one fewer vertex.
        c : Sector
            The old coupled sector.
        mu : int
            The old bottom multiplicity label.
        z : Sector
            The old last uncoupled sector.

        See Also
        --------
        extended

        """
        if self.num_uncoupled == 0:
            raise ValueError('Cant split empty tree')
        if self.num_uncoupled == 1:
            return FusionTree.from_empty(self.symmetry), self.coupled, 0, self.coupled
        if self.num_uncoupled == 2:
            rest_tree = FusionTree.from_sector(self.symmetry, self.uncoupled[0], is_dual=self.are_dual[0])
            return rest_tree, self.coupled, self.multiplicities[0], self.uncoupled[-1]
        rest_tree = FusionTree(
            self.symmetry,
            uncoupled=self.uncoupled[:-1],
            coupled=self.inner_sectors[-1],
            are_dual=self.are_dual[:-1],
            inner_sectors=self.inner_sectors[:-1],
            multiplicities=self.multiplicities[:-1],
        )
        return rest_tree, self.coupled, self.multiplicities[-1], self.uncoupled[-1]

    def twist(self, idcs: Sequence[int], overtwist: bool) -> dict[FusionTree, float | complex]:
        """Twist some legs above a tree, return the resulting linear combination of trees.

        Parameters
        ----------
        idcs : list of int
            Which uncoupled legs to twist
        overtwist : bool
            The chirality of the twist. If the loop is to the right of the wires, an overtwist is
            such that the free end is on top. See notes below.

        Returns
        -------
        linear_combination : dict {FusionTree: complex}
            The composite object of tree and twist is a linear combination
            ``twisted_self = sum_i a_i X_i``. The returned dictionary has entries
            ``linear_combination[X_i] = a_i`` for the contributions to this linear combination
            (i.e. trees for which the coefficient vanishes may be omitted).

        Notes
        -----
        See the following graphical examples for braid chiralities::

            |   idcs = [-1]                    idcs = [-1]
            |   overtwist = True               overtwist = False
            |
            |   │   │   │   │                  │   │   │   │
            |   │   │   │   │   ╭─╮            │   │   │   │   ╭─╮
            |   │   │   │    ╲ ╱  │            │   │   │    ╲ ╱  │
            |   │   │   │     ╱   │            │   │   │     ╲   │
            |   │   │   │    ╱ ╲  │            │   │   │    ╱ ╲  │
            |   ┢━━━┷━━━┷━━━┷━┓ ╰─╯            ┢━━━┷━━━┷━━━┷━┓ ╰─╯
            |   ┡━━━━━━━━━━━━━┛                ┡━━━━━━━━━━━━━┛
            |   │                              │

        For multiple legs (``len(idcs) > 1``), we twist the together, e.g. here for
        ``idcs=[-2, -1]`` and ``overtwist=True``::

            |   │   │   │   │   ╭──────╮
            |   │   │    ╲   ╲ ╱       │
            |   │   │     ╲   ╱   ╭─╮  │
            |   │   │      ╲ ╱ ╲ ╱  │  │
            |   │   │       ╱   ╱   │  │
            |   │   │      ╱ ╲ ╱ ╲  │  │
            |   │   │     ╱   ╱   ╰─╯  │
            |   │   │    ╱   ╱ ╲       │
            |   ┢━━━┷━━━┷━━━┷━┓ ╰──────╯
            |   ┡━━━━━━━━━━━━━┛
            |   │

        """
        if self.symmetry.has_symmetric_braid:
            # twists are trivial
            return {self: 1}

        if len(idcs) == 0:
            return {self: 1}

        if len(idcs) == 1:
            # single wire twist
            i = to_valid_idx(idcs[0], self.num_uncoupled)
            theta = self.symmetry.topological_twist(self.uncoupled[i])
            if not overtwist:
                theta = np.conj(theta)
            return {self: theta}

        idcs = sorted([to_valid_idx(i, self.num_uncoupled) for i in idcs])
        assert all(i2 > i1 for i2, i1 in zip(idcs[1:], idcs[:-1])), 'duplicate idcs'

        if len(idcs) == self.num_uncoupled:
            # we can just slide the whole tree through the twist and end up with a twist of the
            # coupled sector
            theta = self.symmetry.topological_twist(self.coupled)
            if not overtwist:
                theta = np.conj(theta)
            return {self: theta}

        if idcs == [*range(len(idcs))]:
            # we can slide a subtree through the twist and get a twist on an inner sector
            a = self.inner_sectors[idcs[-1] - 1]
            # note: have already excluded the special cases where this index would be out of bounds
            theta = self.symmetry.topological_twist(a)
            if not overtwist:
                theta = np.conj(theta)
            return {self: theta}

        # Not sure what the best strategy is in the general case.
        # Option A: we could do the twist on range(i, j) as:
        #           - twist on range(j)
        #           - inverse twist on range(i)
        #           - some extra braiding
        # Option B: break it down recursively
        #           - twist range(i, mid)
        #           - twist range(mid, j)
        #           - braid twice
        raise NotImplementedError


class fusion_trees(Iterable[FusionTree]):
    r"""Iterable over all :class:`FusionTree`\ s with given uncoupled and coupled sectors.

    This custom iterator has efficient implementations of ``len`` and :meth:`index`, which
    avoid generating all intermediate trees.

    TODO elaborate on canonical order of trees -> reference in module level docstring.
    """

    def __init__(self, symmetry: Symmetry, uncoupled: SectorArray | list[Sector], coupled: Sector, are_dual=None):
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

    def __iter__(self):
        if self.num_uncoupled == 0:
            if np.all(self.coupled == self.symmetry.trivial_sector):
                yield FusionTree(self.symmetry, self.uncoupled, self.coupled, [], [], [])
            return

        if self.num_uncoupled == 1:
            if np.all(self.uncoupled[0] == self.coupled):
                yield FusionTree(self.symmetry, self.uncoupled, self.coupled, self.are_dual, [], [])
            return

        if self.num_uncoupled == 2:
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
            left_tree = FusionTree(self.symmetry, self.uncoupled[:2], b, self.are_dual[:2], [], [0])
            for rest_tree in fusion_trees(self.symmetry, uncoupled, self.coupled, are_dual):
                tree = rest_tree.insert(left_tree)
                for mu in range(self.symmetry._n_symbol(a1, a2, b)):
                    res = tree.copy()
                    res.multiplicities = res.multiplicities.copy()
                    res.multiplicities[0] = mu
                    yield res

    def __len__(self) -> int:
        # OPTIMIZE caching ?

        if self.num_uncoupled == 0:
            if np.all(self.coupled == self.symmetry.trivial_sector):
                return 1
            return 0

        if self.num_uncoupled == 1:
            if np.all(self.uncoupled[0] == self.coupled):
                return 1
            return 0

        if self.num_uncoupled == 2:
            return self.symmetry.n_symbol(*self.uncoupled, self.coupled)

        a1 = self.uncoupled[0]
        a2 = self.uncoupled[1]
        count = 0
        for b in self.symmetry.fusion_outcomes(a1, a2):
            uncoupled = np.concatenate([b[None, :], self.uncoupled[2:]])
            num_subtrees = len(fusion_trees(self.symmetry, uncoupled, self.coupled))
            # no need to check if the fusion is allowed in n_symbol -> use _n_symbol
            count += self.symmetry._n_symbol(a1, a2, b) * num_subtrees
        return count

    def __str__(self):
        signature = FusionTree._str_uncoupled_coupled(self.symmetry, self.uncoupled, self.coupled, self.are_dual)
        return f'fusion_trees[{str(self.symmetry)}]({signature})'

    def __repr__(self):
        uncoupled = str(self.uncoupled).replace('\n', ',')
        return f'fusion_trees({self.symmetry}, {uncoupled}, {self.coupled}, {self.are_dual})'

    def index(self, tree: FusionTree) -> int:
        """The index of a given tree in the iterator."""
        # check compatibility first (same symmetry, same uncoupled, same coupled, same are_dual)
        if not self.symmetry.is_same_symmetry(tree.symmetry):
            raise ValueError(f'Inconsistent symmetries, {self.symmetry} != {tree.symmetry}')
        if not np.all(self.uncoupled == tree.uncoupled):
            raise ValueError(f'Inconsistent uncoupled sectors, {self.uncoupled} != {tree.uncoupled}')
        if not np.all(self.coupled == tree.coupled):
            raise ValueError(f'Inconsistent coupled sector, {self.coupled} != {tree.coupled}')
        if not np.all(self.are_dual == tree.are_dual):
            raise ValueError(f'Inconsistent dualities, {self.are_dual} != {tree.are_dual}')
        return self._compute_index(tree)

    def _compute_index(self, tree: FusionTree) -> int:
        if self.num_uncoupled < 2:
            if self.num_uncoupled == 0 and np.all(self.coupled == self.symmetry.trivial_sector):
                return 0
            elif self.num_uncoupled == 1 and np.all(self.uncoupled[0] == self.coupled):
                return 0
            raise ValueError(f'Inconsistent coupled sector.')

        idx = 0
        # product of all multiplicities to the left of left_sec in for loop below
        left_multi = 1
        # upper limit for the values multiplicities take at each vertex (of the tree)
        max_multis = []
        for i in range(self.num_uncoupled - 2):
            # coupled sector is unique, no need to shift idx for target_sec == self.coupled
            target_sec = tree.inner_sectors[i]
            left_sec = self.uncoupled[i] if i == 0 else tree.inner_sectors[i - 1]
            sector_found = False
            for fusion_sec in self.symmetry.fusion_outcomes(left_sec, self.uncoupled[i + 1]):
                multi = self.symmetry._n_symbol(left_sec, self.uncoupled[i + 1], fusion_sec)
                if np.all(fusion_sec == target_sec):
                    sector_found = True
                    left_multi *= multi
                    max_multis.append(multi)
                    break
                uncoupled = np.concatenate([fusion_sec[None, :], self.uncoupled[i + 2 :]])
                are_dual = np.concatenate([[False], self.are_dual[i + 2 :]])
                idx += left_multi * multi * len(fusion_trees(self.symmetry, uncoupled, self.coupled, are_dual))
            if not sector_found:
                raise ValueError(f'Inconsistent inner sector.')

        left_sec = self.uncoupled[0] if self.num_uncoupled == 2 else tree.inner_sectors[-1]
        if not self.symmetry.can_fuse_to(left_sec, self.uncoupled[-1], self.coupled):
            raise ValueError(f'Inconsistent inner sector.')

        max_multis.append(self.symmetry._n_symbol(left_sec, self.uncoupled[-1], self.coupled))
        if not np.all(tree.multiplicities < max_multis):
            raise ValueError(f'Inconsistent multiplicity.')

        # idx shift from multiplicities
        if not self.symmetry.is_abelian:
            idx += sum([multi * prod(max_multis[:i]) for i, multi in enumerate(tree.multiplicities)])
        return idx
