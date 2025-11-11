"""Matrix product operator (MPO).

An MPO is the generalization of an :class:`~tenpy.networks.mps.MPS` to operators. Graphically::

    |      ^        ^        ^
    |      |        |        |
    |  ->- W[0] ->- W[1] ->- W[2] ->- ...
    |      |        |        |
    |      ^        ^        ^

So each 'matrix' has two physical legs ``p, p*`` instead of just one,
i.e. the entries of the 'matrices' are local operators.
Valid boundary conditions of an MPO are the same as for an MPS
(i.e. ``'finite' | 'segment' | 'infinite'``).
(In general, you can view the MPO as an MPS with larger physical space and bring it into
canonical form. However, unlike for an MPS, this doesn't simplify calculations.
Thus, an MPO has no `form`.)

We use the following label convention for the `W` (where arrows indicate `qconj`)::

    |            p*
    |            ^
    |            |
    |     wL ->- W ->- wR
    |            |
    |            ^
    |            p


If an MPO describes a sum of local terms (e.g. most Hamiltonians),
some bond indices correspond to 'only identities to the left/right'.
We store these indices in `IdL` and `IdR` (if there are such indices).

Similar as for the MPS, a bond index ``i`` is *left* of site `i`,
i.e. between sites ``i-1`` and ``i``.
"""
# Copyright (C) TeNPy Developers, Apache license

import copy
import logging
import warnings

import numpy as np
from scipy.linalg import expm
from scipy.special import comb

from ..linalg import np_conserved as npc
from ..linalg.krylov_based import GMRES
from ..linalg.sparse import FlatLinearOperator, NpcLinearOperator, ShiftNpcLinearOperator
from ..linalg.truncation import TruncationError, svd_theta
from ..tools.math import lcm
from ..tools.misc import add_with_None_0, inverse_permutation, to_iterable
from ..tools.params import asConfig
from ..tools.string import vert_join
from .mps import BaseEnvironment, MPSGeometry, TransferMatrix
from .site import group_sites
from .terms import TermList

logger = logging.getLogger(__name__)

__all__ = [
    'MPO',
    'make_W_II',
    'MPOGraph',
    'MPOEnvironment',
    'MPOEnvironmentBuilder',
    'MPOTransferMatrix',
    'grid_insert_ops',
]


class MPO(MPSGeometry):
    """Matrix product operator, finite (MPO) or infinite (iMPO).

    Parameters
    ----------
    Ws : list of :class:`~tenpy.linalg.np_conserved.Array`
        The matrices of the MPO. Should have labels ``wL, wR, p, p*``.
        Finite boundary conditions require ``Ws[0].get_leg('wL').ind_len == 1``, and similarly
        ``Ws[-1].get_leg('wR').ind_len == 1``
    IdL : (iterable of) {int | None}
        Indices on the bonds, which correspond to 'only identities to the left'.
        A single entry holds for all bonds.
    IdR : (iterable of) {int | None}
        Indices on the bonds, which correspond to 'only identities to the right'.
    max_range : int | np.inf | None
        Maximum range of hopping/interactions (in unit of sites) of the MPO. ``None`` for unknown.
    explicit_plus_hc : bool
        If True, this flag indicates that the hermitian conjugate of the MPO should be
        computed and added at runtime, i.e., `self` is not (necessarily) hermitian.
    unit_cell_width : int
        See :attr:`~tenpy.models.lattice.Lattice.mps_unit_cell_width`.

    Attributes
    ----------
    dtype : type
        The data type of the `_W`.
    bc : {'finite' | 'segment' | 'infinite'}
        Boundary conditions as described in :mod:`~tenpy.networks.mps`.
        ``'finite'`` requires ``Ws[0].get_leg('wL').ind_len = 1``.
    IdL : list of {int | None}
        Indices on the bonds (length `L`+1), which correspond to 'only identities to the left'.
        ``None`` for bonds where it is not set.
        In standard form, this is `0` (except for unset bonds in finite case)
    IdR : list of {int | None}
        Indices on the bonds (length `L`+1), which correspond to 'only identities to the right'.
        ``None`` for bonds where it is not set.
        In standard form, this is the last index on the bond (except for unset bonds in finite case).
    max_range : int | np.inf | None
        Maximum range of hopping/interactions (in unit of sites) of the MPO. ``None`` for unknown.
    grouped : int
        Number of sites grouped together, see :meth:`group_sites`.
    explicit_plus_hc : bool
        If True, this flag indicates that the hermitian conjugate of the MPO should be
        computed and added at runtime, i.e., `self` is not (necessarily) hermitian.
    _W : list of :class:`~tenpy.linalg.np_conserved.Array`
        The matrices of the MPO. Labels are ``'wL', 'wR', 'p', 'p*'``.
    _valid_bc : tuple of str
        Class attribute. Valid boundary conditions; the same as for an MPS.
    _graph : None | list of {dict of {(int,int): :class:`~tenpy.linalg.np_conserved.Array`}}
        Represents the graph structure of `self`.
        `_graph[j_site][(i,j)] = op` where `op=self._W[j_site][i,j]` iff `npc.norm(op)>0`
        Defaults to `None` if the graph is invalid or has not yet been built
    _outer_permutation : None | False | list of int
        Relevant only for iMPOs and segment MPOs with contractible outer virtual legs:
        `self._W[0].get_leg("wL")`, `self._W[-1].get_leg("wR")`
        Ordering of the outer virtual legs such that `self` is upper triangular w.r.t. the whole unit cell.
        Follows the constraint `_outer_permutation[0] = self.IdL[0]`, `_outer_permutation[-1] = self.IdR[-1]`
        Defaults to `None` if the ordering has not yet been checked of False if ordering the MPO is not possible
        **Note:** This ordering is valid only with respect to the **whole unit cell**, not for
        internal `'wL'/'wR'` legs.
    _cycles : None | dict {int: list of int}
        Contains one entry `_cycles[i0] = [i0, i1, ..., iL = i0]` for each index `i0` of the outer
        virtual leg that connects to itself.
        The cycle is `_W[0][i0, i1] * _W[1][i1, i2] * ... * _W[L-1][iL-1, iL]`
        Defaults to None if :attr:`_outer_permutation` does not exist.

    """

    def __init__(
        self,
        sites,
        Ws,
        bc='finite',
        IdL=None,
        IdR=None,
        max_range=None,
        explicit_plus_hc=False,
        mps_unit_cell_width=None,
    ):
        super().__init__(sites, bc, unit_cell_width=mps_unit_cell_width)
        self.dtype = dtype = np.result_type(*[W.dtype for W in Ws])
        self._W = [W.astype(dtype, copy=True) for W in Ws]
        self.IdL = self._get_Id(IdL, len(sites))
        self.IdR = self._get_Id(IdR, len(sites))
        self.grouped = 1
        self.max_range = max_range
        self.explicit_plus_hc = explicit_plus_hc
        self._graph = None
        # for iterative environment initialization
        self._outer_permutation = None
        self._cycles = None
        self.test_sanity()

    def _reset_graph(self):
        # set proper defaults and avoid carrying graph if not valid
        self._graph = None
        self._outer_permutation = None
        self._cycles = None

    def _make_graph(self, norm_tol=1e-12):
        """Construct :attr:`_graph`

        This function builds the MPOGraph represented by `self` in a sligthly different format.

        Parameters
        ----------
        norm_tol : float
            Entries in :attr:`_W` are considered zero if their norm is smaller than `norm_tol`.

        """
        if self._graph is not None:
            return
        # build graph, no checks for loops etc.
        self._graph = [{} for _ in range(len(self.sites))]
        for i in range(self.L):
            W = self.get_W(i)
            W.itranspose(['wL', 'wR', 'p', 'p*'])
            chiL, chiR, _, _ = W.shape
            for jL in range(chiL):
                for jR in range(chiR):
                    op = W[jL, jR]
                    if npc.norm(op) > norm_tol:
                        self._graph[i][(jL, jR)] = op

    def _order_graph(self):
        """Find an ordering for :attr:`_graph` if possible

        Checks whether `self` can be brought into upper triangular form
        and updates :attr:`_outer_permutation` and :attr:`_cycles` accordingly.

        .. note ::

            - Attempting to do this makes only sense if the MPO is periodic.
              The upper triangular form applies to **the whole unit cell**.
              In particular, the MPO matrices :attr:`_W` are **not** ordered.
            - Ordering the graph can fail for multiple reasons. This is generally not
              critical and :attr:`_outer_permutation` should be set to `False`.
              If an error occurs, we raise a `ValueError`, catch it at the end and
              prompt a distinct warning. Functions that require the upper triangular form
              should check the status via :attr:`_outer_permutation`.

        """
        # check whether ordering the graph makes sense
        if self._outer_permutation is not None and not self._outer_permutation:
            warnings.warn(
                'Ordering the MPO was already tried and failed. '
                'If intentional, make sure that the MPO satisfies the requirements.'
            )
            return
        if self.bc == 'finite':
            warnings.warn("_order_graph() called for 'finite' MPO. This does not make sense.")
            self._outer_permutation = False
            return
        if self._graph is None:
            self._make_graph()
        if self.finite:  # segment
            try:
                self._W[0].get_leg('wL').test_contractible(self._W[-1].get_leg('wR'))
            except ValueError:
                warnings.warn(
                    "_order_graph() called for 'segment' MPO with different "
                    'left and right outer virtual leg. If intentional, '
                    'ensure that outer virtual legs are contractible.'
                )
                self._outer_permutation = False
                return
        # attempting to order the graph makes sense
        try:
            cycle_params = self._graph_connections()
            j_IdL, j_IdR, j_other_cycles, j_upper, j_lower = self._sort_connections(cycle_params)
            outer_connections, _, cycles = cycle_params
            perm = [-1 for _ in range(self.chi[0])]
            offsets = [len(j_upper), -1]
            # IdL, IdR
            if j_IdL is not None:
                perm[0] = j_IdL
                offsets[0] += 1
            if j_IdR is not None:
                perm[-1] = j_IdR
                offsets[1] = -2
            # cycles
            for j, j_cycle in enumerate(j_other_cycles):
                perm[offsets[0] + j] = j_cycle

            # order j_upper / j_lower by iteratively removing all indices that do not couple to the block
            def sort_block(block):
                ordering = []
                while block:  # still indices left
                    js_step = [j for j in block if not outer_connections[j] & block]
                    if len(js_step) == 0:  # Illegal cycle
                        raise ValueError(
                            'Index ordering failed: '
                            'Illegal cycle A1 -> A2 -> ... -> A1 over multiple '
                            'unit cells found. '
                            'Increasing the unit cell might fix this problem.'
                        )
                    # reverse: order indices with same depth in ascending order
                    for j in reversed(js_step):
                        ordering.append(j)
                        block.remove(j)
                return ordering

            # upper indices
            j_upper_ordered = sort_block(j_upper)
            for index, j in enumerate(j_upper_ordered):
                perm[offsets[0] - 1 - index] = j
            # lower indices
            j_lower_ordered = sort_block(j_lower)
            for index, j in enumerate(j_lower_ordered):
                perm[offsets[1] - index] = j
            # ordering was successful
            self._cycles = {cycle[0]: cycle for cycle in cycles}
            self._outer_permutation = perm
        except ValueError as e:
            # graph cannot be ordered
            warnings.warn('Ordering the MPO failed: ' + str(e))
            self._outer_permutation = False

    def _graph_connections(self):
        """Helper function for :meth:`_order_graph`.

        Determine all connections of the outer virtual leg
            outer_leg[i] -> {outer_leg[j] | i,j connected by MPO graph}

        Returns
        -------
        outer_connections : list of {set of int}
            One entry for each index of the outer leg,
            containing a set of all indices it connects to.
        j_cycles : set of int
            Indices of the outer virtual leg that connect to themselves
        cycles : list of {list of int}
            The corresponding cycles as in :attr:`_cycles`.

        """
        j_cycles = []
        cycles = []
        outer_connections = []
        for j_outer in range(self.chi[0]):
            grid = [[-1 for _ in range(_chi)] for _chi in self.chi]
            grid[0][j_outer] = j_outer
            # forward
            for j_site, layer in enumerate(self._graph):
                js_connected = set(j for i, j in layer if i in grid[j_site])
                for j_edge in js_connected:
                    grid[j_site + 1][j_edge] = j_edge
            # connected indices
            outer_connections.append([j for j in grid[-1] if j != -1])
            # loop present
            if grid[-1][j_outer] == j_outer:
                j_cycles.append(j_outer)
                loop = [j_outer]
                j_current = j_outer
                # backward propagation
                for j_right in range(self.L - 1, -1, -1):
                    js_backward = [i for i, j in self._graph[j_right] if j == j_current and i in grid[j_right]]
                    if len(js_backward) != 1:
                        # only simple paths supported
                        # NOTE: Paths with multiple branches that reconnect can work in principle,
                        #       but are not supported by MPOEnvironmentBuilder
                        raise ValueError(f'Loop missing or multiple loops found for outer index {j_outer}')
                    j_current = js_backward[0]
                    loop.append(j_current)
                cycles.append(list(reversed(loop)))
        return [set(x) for x in outer_connections], set(j_cycles), cycles

    def _sort_connections(self, graph_connections):
        """Sort the outer virtual leg into blocks

        Helper function for `self._order_graph()`

        Categorize the indices of the outer virtual leg into:
            1) `IdL`
            2) `IdR`
            3) other cycles -> indices with cycles distinct from `IdL` and `IdR`
            4) upper indices -> indices without connections FROM other cycles
            5) lower indices -> indices that other cycles connects to

        Returns
        -------
        j_IdL : int | None
            `IdL`
        j_IdR : int | None
            `IdR`
        other_cycles : set of int
            other cycles
        j_upper : set of int
            upper indices
        j_lower : set of int
            lower indices

        """
        outer_connections, j_cycles, _ = graph_connections
        # check IdL, IdR valid
        j_IdL, j_IdR = self.IdL[0], self.IdR[-1]
        if (j_IdL is not None) and (j_IdL < 0):
            j_IdL = self.chi[0] + j_IdL
        if (j_IdR is not None) and (j_IdR < 0):
            j_IdR = self.chi[-1] + j_IdR
        if (j_IdL is not None) and (j_IdL not in j_cycles):
            raise ValueError('Connection IdL -> IdL missing')
        if (j_IdR is not None) and (j_IdR not in j_cycles):
            raise ValueError('Connection IdR -> IdR missing')
        for j, connection in enumerate(outer_connections):
            if j != j_IdL and j_IdL in connection:
                raise ValueError(f'Outer index {j} -> IdL connection found ?!')
        if (j_IdR is not None) and (len(outer_connections[j_IdR]) != 1):
            raise ValueError('IdR connects to different index ?!')
        # check loops and indices
        other_cycles = set(j for j in j_cycles if j != j_IdL and j != j_IdR)
        if not other_cycles:  # only loops are IdL, IdR
            return j_IdL, j_IdR, set(), set(j for j in range(self.chi[0]) if j not in j_cycles), set()
        j_upper = set()
        j_lower = set()
        other_cycle_connections = set().union(*(outer_connections[j_loop] for j_loop in other_cycles))
        for j, connection in enumerate(outer_connections):
            if j not in j_cycles:
                if j in other_cycle_connections:  # existing connection some_loop -> label_j
                    j_lower.add(j)
                    if connection & other_cycles:  # existing connection label_j -> some loop
                        raise ValueError(f'Connection I -> loop1 and loop2 -> I found for Index I={j}')

                else:  # Default: add to j_upper if not a lower index
                    j_upper.add(j)
        return j_IdL, j_IdR, other_cycles, j_upper, j_lower

    def copy(self):
        """Make a shallow copy of `self`."""
        return copy.copy(self)

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        """Export `self` into a HDF5 file.

        This method saves all the data it needs to reconstruct `self` with :meth:`from_hdf5`.

        Specifically, it saves
        :attr:`sites`,
        :attr:`chinfo`,
        :attr:`max_range`,
        :attr:`unit_cell_width` (under these names),
        :attr:`_W` as ``"tensors"``,
        :attr:`IdL` as ``"index_identity_left"``,
        :attr:`IdR` as ``"index_identity_right"``, and
        :attr:`bc` as ``"boundary_condition"``.
        Moreover, it saves :attr:`L`, :attr:`explicit_plus_hc` and :attr:`grouped` as HDF5 attributes,
        as well as the maximum of :attr:`chi` under the name :attr:`max_bond_dimension`.

        Parameters
        ----------
        hdf5_saver : :class:`~tenpy.tools.hdf5_io.Hdf5Saver`
            Instance of the saving engine.
        h5gr : :class`Group`
            HDF5 group which is supposed to represent `self`.
        subpath : str
            The `name` of `h5gr` with a ``'/'`` in the end.

        """
        hdf5_saver.save(self.sites, subpath + 'sites')
        hdf5_saver.save(self.chinfo, subpath + 'chinfo')
        hdf5_saver.save(self._W, subpath + 'tensors')
        hdf5_saver.save(self.IdL, subpath + 'index_identity_left')
        hdf5_saver.save(self.IdR, subpath + 'index_identity_right')
        h5gr.attrs['grouped'] = self.grouped
        hdf5_saver.save(self.bc, subpath + 'boundary_condition')
        hdf5_saver.save(self.max_range, subpath + 'max_range')
        hdf5_saver.save(self.unit_cell_width, subpath + 'unit_cell_width')
        h5gr.attrs['explicit_plus_hc'] = self.explicit_plus_hc
        h5gr.attrs['L'] = self.L  # not needed for loading, but still useful metadata
        h5gr.attrs['max_bond_dimension'] = np.max(self.chi)  # same
        # building the graph / ordering it takes <1s for reasonable MPOs, so not worth saving it

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        """Load instance from a HDF5 file.

        This method reconstructs a class instance from the data saved with :meth:`save_hdf5`.

        Parameters
        ----------
        hdf5_loader : :class:`~tenpy.tools.hdf5_io.Hdf5Loader`
            Instance of the loading engine.
        h5gr : :class:`Group`
            HDF5 group which is represent the object to be constructed.
        subpath : str
            The `name` of `h5gr` with a ``'/'`` in the end.

        Returns
        -------
        obj : cls
            Newly generated class instance containing the required data.

        """
        obj = cls.__new__(cls)  # create class instance, no __init__() call
        hdf5_loader.memorize_load(h5gr, obj)

        obj.sites = hdf5_loader.load(subpath + 'sites')
        obj.chinfo = hdf5_loader.load(subpath + 'chinfo')
        obj._W = hdf5_loader.load(subpath + 'tensors')
        obj.dtype = np.result_type(*[W.dtype for W in obj._W])
        obj.IdL = hdf5_loader.load(subpath + 'index_identity_left')
        obj.IdR = hdf5_loader.load(subpath + 'index_identity_right')
        obj.grouped = hdf5_loader.get_attr(h5gr, 'grouped')
        obj.bc = hdf5_loader.load(subpath + 'boundary_condition')
        obj.max_range = hdf5_loader.load(subpath + 'max_range')
        obj.unit_cell_width = hdf5_loader.load(subpath + 'unit_cell_width')
        obj.explicit_plus_hc = h5gr.attrs.get('explicit_plus_hc', False)
        obj._graph = None
        obj._outer_permutation = None
        obj._cycles = None
        obj.test_sanity()
        return obj

    @classmethod
    def from_grids(
        cls,
        sites,
        grids,
        bc='finite',
        IdL=None,
        IdR=None,
        Ws_qtotal=None,
        legs=None,
        max_range=None,
        explicit_plus_hc=False,
        mps_unit_cell_width=None,
    ):
        """Initialize an MPO from `grids`.

        Parameters
        ----------
        sites : list of :class:`~tenpy.models.lattice.Site`
            Defines the local Hilbert space for each site.
        grids : list of list of list of entries
            For each site (outer-most list) a matrix-grid (corresponding to ``wL, wR``)
            with entries being or representing (see :func:`grid_insert_ops`) onsite-operators.
        bc : {'finite' | 'segment' | 'infinite'}
            Boundary conditions as described in :mod:`~tenpy.networks.mps`.
        IdL : (iterable of) {int | None}
            Indices on the bonds, which correspond to 'only identities to the left'.
            A single entry holds for all bonds.
        IdR : (iterable of) {int | None}
            Indices on the bonds, which correspond to 'only identities to the right'.
        Ws_qtotal : (list of) total charge
            The `qtotal` to be used for each grid. Defaults to zero charges.
        legs : list of :class:`~tenpy.linalg.charge.LegCharge`
            List of charges for 'wL' legs left of each `W`, L + 1 entries.
            The last entry should be the conjugate of the 'wR' leg,
            i.e. identical to ``legs[0]`` for 'infinite' `bc`.
            By default, determine the charges automatically. This is limited to cases where
            there are no "dangling open ends" in the MPO graph. (The :class:`MPOGraph` can handle
            those cases, though.)
        max_range : int | np.inf | None
            Maximum range of hopping/interactions (in unit of sites) of the MPO.
            ``None`` for unknown.
        explicit_plus_hc : bool
            If True, the Hermitian conjugate of the MPO is computed at runtime,
            rather than saved in the MPO.
        unit_cell_width : int
            See :attr:`~tenpy.models.lattice.Lattice.mps_unit_cell_width`.

        See Also
        --------
        grid_insert_ops : used to plug in `entries` of the grid.
        tenpy.linalg.np_conserved.grid_outer : used for final conversion.

        """
        chinfo = sites[0].leg.chinfo
        L = len(sites)
        assert len(grids) == L  # wrong arguments?
        grids = [grid_insert_ops(site, grid) for site, grid in zip(sites, grids)]
        if Ws_qtotal is None:
            Ws_qtotal = [chinfo.make_valid()] * L
        else:
            Ws_qtotal = chinfo.make_valid(Ws_qtotal)
            if Ws_qtotal.ndim == 1:
                Ws_qtotal = [Ws_qtotal] * L
        IdL = cls._get_Id(IdL, L)
        IdR = cls._get_Id(IdR, L)
        if legs is None:
            if bc != 'infinite':
                # ensure that we have only a single entry in the first and last leg
                # i.e. project grids[0][:, :] -> grids[0][IdL[0], :]
                # and         grids[-1][:, :] -> grids[-1][:,IdR[-1], :]
                first_grid = grids[0]
                last_grid = grids[-1]
                if len(first_grid) > 1:
                    grids[0] = [first_grid[IdL[0]]]
                    IdL[0] = 0
                    IdR[0] = None
                if len(last_grid[0]) > 1:
                    grids[-1] = [[row[IdR[-1]]] for row in last_grid]
                    IdR[-1] = 0
                    IdL[-1] = None
                legs = _calc_grid_legs_finite(chinfo, grids, Ws_qtotal, None)
            else:
                legs = _calc_grid_legs_infinite(chinfo, grids, Ws_qtotal, None, IdL[0])
        # now build the `W` from the grid
        assert len(legs) == L + 1
        Ws = []
        for i in range(L):
            W = npc.grid_outer(grids[i], [legs[i], legs[i + 1].conj()], Ws_qtotal[i], ['wL', 'wR'])
            Ws.append(W)
        # no graph
        return cls(sites, Ws, bc, IdL, IdR, max_range, explicit_plus_hc, mps_unit_cell_width)

    @classmethod
    def from_wavepacket(cls, sites, coeff, op, eps=1.0e-15, unit_cell_width=None):
        r"""Create a (finite) MPO wave packet representing ``sum_i coeff[i] op_i``.

        Note that we define it only for finite systems; a generalization to infinite systems
        is not straight forward due to normalization issues: the individual terms vanish in
        the thermodynamic limit!

        Parameters
        ----------
        sites : list of :class:`~tenpy.models.lattice.Site`
            Defines the local Hilbert space for each site.
        coeff : list of float/complex
            Wave packet coefficients.
        op : str
            Name of the operator to be applied.
        eps : float
            Discard terms where ``abs(coeff[i]) < eps``.

        Examples
        --------
        Say you have fermions, so ``op='Cd'``, and want to create
        a gaussian wave packet :math:`\sum_x \alpha_x c^\dagger_x` with
        :math:`\alpha_x \propto e^{-0.5(x-x_0)^2/\sigma^2} e^{i k_0 x}`.
        Then you would use

        .. testsetup :: from_wavepacket

            from tenpy.networks.site import FermionSite
            from tenpy.networks.mpo import MPO
            from tenpy.networks.mps import MPS
            import numpy as np

        .. doctest :: from_wavepacket

            >>> (
            ...     L,
            ...     k0,
            ...     x0,
            ...     sigma,
            ... ) = 50, np.pi / 8.0, 10.0, 5.0
            >>> x = np.arange(L)
            >>> coeff = np.exp(-1.0j * k0 * x) * np.exp(-0.5 * (x - x0) ** 2 / sigma**2)
            >>> coeff /= np.linalg.norm(coeff)
            >>> site = FermionSite(conserve='N')
            >>> wp = MPO.from_wavepacket([site] * L, coeff, 'Cd')
            >>> wp.chi == [1] + [2] * (L - 1) + [1]
            True

        Indeed, we can apply this to a (vacuum) MPS and get the correct state:

        .. doctest :: from_wavepacket

            >>> psi = MPS.from_product_state([site] * L, ['empty'] * L, unit_cell_width=L)
            >>> wp.apply(psi, dict(compression_method='SVD'))
            TruncationError()
            >>> C = psi.correlation_function('Cd', 'C')
            >>> C_expected = np.conj(coeff)[:, np.newaxis] * coeff[np.newaxis, :]
            >>> bool(np.max(np.abs(C - C_expected)) < 1.0e-10)
            True

        """
        coeff = np.asarray(coeff)
        assert coeff.shape == (len(sites),)
        L = len(sites)
        assert L >= 2
        first_nonzero = np.nonzero(coeff)[0][0]
        needs_JW = sites[first_nonzero].op_needs_JW(op)
        upper_left = 'JW' if needs_JW else 'Id'

        grids = []
        for i in range(L):
            local = None if abs(coeff[i]) < eps else [(op, coeff[i])]
            grid = [[upper_left, local], [None, 'Id']]
            if i == 0:
                grid = grid[:1]  # first row only
            if i == L - 1:  # last column only
                grid = [grid[0][1:], grid[1][1:]]
            grids.append(grid)
        IdL = [0] + [None] * L
        # note: for finite bc, the JW string ends at site 0, so we don't need to worry about
        # extending it to the left; but for infinite MPS, the first environment for applying the
        # MPO to an MPS would need a non-trivial modification that is not captured when setting
        # IdL=0!
        IdR = [None] * L + [0]
        # no graph
        return cls.from_grids(sites, grids, 'finite', IdL, IdR, mps_unit_cell_width=unit_cell_width)

    def test_sanity(self):
        """Sanity check, raises ValueErrors, if something is wrong."""
        super().test_sanity()
        for i in range(self.L):
            S = self.sites[i]
            W = self.get_W(i)
            S.leg.test_equal(W.get_leg('p'))
            S.leg.test_contractible(W.get_leg('p*'))
            if self.bc == 'infinite' or i + 1 < self.L:
                W2 = self.get_W(i + 1)
                W.get_leg('wR').test_contractible(W2.get_leg('wL'))
        if not (len(self.IdL) == len(self.IdR) == self.L + 1):
            raise ValueError('wrong len of `IdL`/`IdR`')

    @property
    def chi(self):
        """Dimensions of the virtual bonds."""
        return [W.get_leg('wL').ind_len for W in self._W] + [self._W[-1].get_leg('wR').ind_len]

    def get_W(self, i, copy=False):
        """Return `W` at site `i`."""
        i_in_unit_cell, num_unit_cells = self._to_valid_site_index(i, return_num_unit_cells=True)
        W = self._W[i_in_unit_cell]
        if copy:
            W = W.copy()
        return self.shift_Array_unit_cells(W, num_unit_cells=num_unit_cells, inplace=copy)

    def set_W(self, i, W):
        """Set `W` at site `i`. Note that ``W`` may be modified in-place."""
        i_in_unit_cell, num_unit_cells = self._to_valid_site_index(i, return_num_unit_cells=True)
        self._W[i_in_unit_cell] = self.shift_Array_unit_cells(W, -num_unit_cells)
        self._reset_graph()

    def get_IdL(self, i):
        """Return index of `IdL` at bond to the *left* of site `i`.

        May be ``None``.
        """
        return self.IdL[self._to_valid_site_index(i)]

    def get_IdR(self, i):
        """Return index of `IdR` at bond to the *right* of site `i`.

        May be ``None``.
        """
        # The convention for the order of IdR is incompatible with something like
        #  self.IdR[self._to_valid_bond_index(i, is_left=False)]
        return self.IdR[self._to_valid_site_index(i) + 1]

    def enlarge_mps_unit_cell(self, factor=2):
        """Repeat the unit cell for infinite MPS boundary conditions; in place.

        Parameters
        ----------
        factor : int
            The new number of sites in the unit cell will be increased from `L` to ``factor*L``.

        """
        if int(factor) != factor:
            raise ValueError('`factor` should be integer!')
        if factor <= 1:
            raise ValueError("can't shrink!")
        if self.finite:
            raise ValueError("can't enlarge finite MPO")
        factor = int(factor)
        L = self.L
        self._W = [self.get_W(j) for j in range(0, factor * L)]
        self.sites = [self.get_site(j) for j in range(0, factor * self.L)]
        self.IdL = factor * self.IdL[:-1] + [self.IdL[-1]]
        self.IdR = factor * self.IdR[:-1] + [self.IdR[-1]]
        self.unit_cell_width *= factor
        if self._graph is not None:
            self._graph = factor * self._graph
        # can keep self._ordering_checked, outer_permutations
        if self._outer_permutation:
            self._cycles = {i0: factor * cycle[:-1] + [cycle[-1]] for i0, cycle in self._cycles.items()}
        self.test_sanity()

    def group_sites(self, n=2, grouped_sites=None):
        """Modify `self` inplace to group sites.

        Group each `n` sites together using the :class:`~tenpy.networks.site.GroupedSite`.
        This might allow to do TEBD with a Trotter decomposition,
        or help the convergence of DMRG (in case of too long range interactions).

        Parameters
        ----------
        n : int
            Number of sites to be grouped together.
        grouped_sites : None | list of :class:`~tenpy.networks.site.GroupedSite`
            The sites grouped together.

        """
        if grouped_sites is None:
            grouped_sites = group_sites(self.sites, n, charges='same')
        else:
            assert grouped_sites[0].n_sites == n
        if self.max_range is not None and self.max_range != np.inf:
            min_n = max(min([gs.n_sites for gs in grouped_sites]), 1)
            self.max_range = int(np.ceil(self.max_range / min_n))
        Ws = []
        IdL = []
        IdR = [self.IdR[0]]
        i = 0
        for gs in grouped_sites:
            new_W = self.get_W(i).itranspose(['wL', 'p', 'p*', 'wR'])
            for j in range(1, gs.n_sites):
                W = self.get_W(i + j).itranspose(['wL', 'p', 'p*', 'wR'])
                new_W = npc.tensordot(new_W, W, axes=[-1, 0])
            comb = [list(range(1, 1 + 2 * gs.n_sites, 2)), list(range(2, 2 + 2 * gs.n_sites, 2))]
            new_W = new_W.combine_legs(comb, pipes=[gs.leg, gs.leg.conj()])
            Ws.append(new_W.iset_leg_labels(['wL', 'p', 'p*', 'wR']))
            IdL.append(self.get_IdL(i))
            i += gs.n_sites
            IdR.append(self.get_IdR(i - 1))
        IdL.append(self.IdL[-1])
        self.IdL = IdL
        self.IdR = IdR
        self._W = Ws
        self.sites = grouped_sites
        self.grouped = self.grouped * n
        self._reset_graph()

    def extract_segment(self, first, last):
        """Extract a segment from the MPO.

        Parameters
        ----------
        first, last : int
            The first and last site to *include* into the segment.

        Returns
        -------
        cp : :class:`MPO`
            A `copy` of self with "segment" boundary conditions.

        See Also
        --------
        tenpy.networks.mps.MPS.extract_segment : similar method for MPS.

        """
        sites_per_ring = self.L // self.unit_cell_width
        unit_cell_width, remainder = divmod(last + 1 - first, sites_per_ring)
        if remainder != 0:
            msg = f'Number of sites must be an integer multiple of unit_cell_width={unit_cell_width}.'
            raise ValueError(msg)
        L = self.L
        sites = [self.sites[i % L] for i in range(first, last + 1)]
        W = [self.get_W(i) for i in range(first, last + 1)]
        IdL = [self.IdL[i % L] for i in range(first, last + 1)]
        IdL.append(self.IdL[last % L + 1])
        IdR = [self.IdR[i % L] for i in range(first, last + 1)]
        IdR.append(self.IdR[last % L + 1])
        cp = self.__class__(
            sites, W, 'segment', IdL, IdR, self.max_range, self.explicit_plus_hc, unit_cell_width
        )  # no graph
        cp.grouped = self.grouped
        return cp

    def sort_legcharges(self):
        """Sort virtual legs by charges. In place.

        The MPO seen as matrix of the ``wL, wR`` legs is usually very sparse. This sparsity is
        captured by the LegCharges for these bonds not being sorted and bunched. This requires a
        tensordot to do more block-multiplications with smaller blocks. This is in general faster
        for large blocks, but might lead to a larger overhead for small blocks. Therefore, this
        function allows to sort the virtual legs by charges.
        """
        new_W = [None] * self.L
        perms = [None] * (self.L + 1)
        for i, w in enumerate(self._W):
            w = w.transpose(['wL', 'wR', 'p', 'p*'])
            p, w = w.sort_legcharge([True, True, False, False], [True, True, False, False])
            if perms[i] is not None:
                assert np.all(p[0] == perms[i])
            perms[i] = p[0]
            perms[i + 1] = p[1]
            new_W[i] = w
        self._W = new_W
        chi = self.chi
        for b, p in enumerate(perms):
            IdL = self.IdL[b]
            if IdL is not None:
                self.IdL[b] = np.nonzero(p == IdL)[0][0]
            IdR = self.IdR[b]
            if IdR is not None:
                IdR = IdR % chi[b]
                self.IdR[b] = np.nonzero(p == IdR)[0][0]
        if self._graph is not None:  # makes sense to only permute the indices
            inv_perms = []
            for perm in perms:
                inv_perm = np.empty_like(perm)
                inv_perm[perm] = np.arange(len(inv_perm), dtype=int)
                inv_perms.append(inv_perm)
            new_graph = [{} for _ in range(self.L)]
            for j_site, layer in enumerate(self._graph):
                for i, j in layer:
                    new_graph[j_site][(inv_perms[j_site][i], inv_perms[j_site + 1][j])] = self._graph[j_site][(i, j)]
            self._graph = new_graph
            if self._outer_permutation:
                self._outer_permutation = [inv_perms[0][j] for j in self._outer_permutation]
                perm_cycles = []
                for j_outer in self._cycles:
                    perm_cycles.append(
                        [inv_perms[j_bond][j_cycle] for j_bond, j_cycle in enumerate(self._cycles[j_outer])]
                    )
                self._cycles = {cycle[0]: cycle for cycle in perm_cycles}
        # done

    def make_U(self, dt, approximation='II'):
        r"""Creates the U_I or U_II propagator.

        Approximations of MPO exponentials following :cite:`zaletel2015`.

        Parameters
        ----------
        dt : float|complex
            The time step per application of the propagator.
            Should be imaginary for real time evolution!
        approximation : ``'I' | 'II'``
            Selects the approximation, :meth:`make_U_I` (``'I'``) or :meth:`make_U_II` (``'II'``).

        Returns
        -------
        U : :class:`~tenpy.networks.mpo.MPO`
            The propagator, i.e. approximation :math:`U ~= exp(H*dt)`

        """
        if approximation == 'II':
            return self.make_U_II(dt)
        elif approximation == 'I':
            return self.make_U_I(dt)
        raise ValueError(repr(approximation) + ' not implemented')

    def make_U_I(self, dt):
        r"""Creates the :math:`U_I` propagator with `W_I` tensors.

        Parameters
        ----------
        dt : float|complex
            The time step per application of the propagator.
            Should be imaginary for real time evolution!

        Returns
        -------
        UI : :class:`~tenpy.networks.mpo.MPO`
            The propagator, i.e. approximation :math:`U_I ~= exp(H*dt)`

        """
        if self.explicit_plus_hc:
            raise NotImplementedError(
                "MPO.make_U_I() assumes hermitian H, you can't use "
                'the `explicit_plus_hc=True` flag!\n'
                'See also https://github.com/tenpy/tenpy/issues/265'
            )
        U = [
            self.get_W(i).astype(np.result_type(dt, self.dtype), copy=True).itranspose(['wL', 'wR', 'p', 'p*'])
            for i in range(self.L)
        ]

        IdLR = []
        for i in range(0, self.L):  # correct?
            U1 = U[i]
            U2 = U[(i + 1) % self.L]
            IdL = self.IdL[i + 1]
            IdR = self.IdR[i + 1]
            assert IdL is not None and IdR is not None
            U1[:, IdL, :, :] = U1[:, IdL, :, :] + dt * U1[:, IdR, :, :]
            keep = np.ones(U1.shape[1], dtype=bool)
            keep[IdR] = False
            U1.iproject(keep, 1)
            if self.finite and i + 1 == self.L:
                keep = np.ones(U2.shape[0], dtype=bool)
                assert self.IdR[0] is not None
                keep[self.IdR[0]] = False
            U2.iproject(keep, 0)

            if IdL > IdR:
                IdLR.append(IdL - 1)
            else:
                IdLR.append(IdL)

        IdL = self.IdL[0]
        IdR = self.IdR[0]
        assert IdL is not None and IdR is not None
        if IdL > IdR:
            IdLR_0 = IdL - 1
        else:
            IdLR_0 = IdL
        IdLR = [IdLR_0] + IdLR

        return MPO(self.sites, U, self.bc, IdLR, IdLR, np.inf, mps_unit_cell_width=self.unit_cell_width)  # no graph

    def make_U_II(self, dt):
        r"""Creates the :math:`U_{II}` propagator.

        Parameters
        ----------
        dt : float|complex
            The time step per application of the propagator. Should be imaginary for real time evolution!

        Returns
        -------
        U_II : :class:`~tenpy.networks.mpo.MPO`
            The propagator, i.e. approximation :math:`UII ~= exp(H*dt)`

        """
        if self.explicit_plus_hc:
            raise NotImplementedError(
                "MPO.make_U_II() assumes hermitian H, you can't use "
                'the `explicit_plus_hc=True` flag!\n'
                'See also https://github.com/tenpy/tenpy/issues/265'
            )
        dtype = np.result_type(dt, self.dtype)
        IdL = self.IdL
        IdR = self.IdR

        chinfo = self.chinfo
        trivial = chinfo.make_valid()
        U = []
        for i in range(0, self.L):
            labels = ['wL', 'wR', 'p', 'p*']
            W = self.get_W(i).itranspose(labels)
            assert np.all(W.qtotal == trivial)
            DL, DR, _, _ = W.shape
            Wflat = W.to_ndarray()
            proj_L = np.ones(DL, dtype=np.bool_)
            proj_L[IdL[i]] = False
            proj_L[IdR[i]] = False
            proj_R = np.ones(DR, dtype=np.bool_)
            proj_R[IdL[i + 1]] = False
            proj_R[IdR[i + 1]] = False

            # Extract (A, B, C, D)
            D = Wflat[IdL[i], IdR[i + 1], :, :]
            C = Wflat[IdL[i], proj_R, :, :]
            B = Wflat[proj_L, IdR[i + 1], :, :]
            A = Wflat[proj_L, :, :, :][:, proj_R, :, :]  # numpy indexing requires two steps

            W_II = make_W_II(dt, A, B, C, D)

            leg_L, leg_R, leg_p, leg_pconj = W.legs
            new_leg_L = npc.LegCharge.from_qflat(chinfo, [chinfo.make_valid()], leg_L.qconj)
            new_leg_L = new_leg_L.extend(leg_L.project(proj_L)[2])
            new_leg_R = npc.LegCharge.from_qflat(chinfo, [chinfo.make_valid()], leg_R.qconj)
            new_leg_R = new_leg_R.extend(leg_R.project(proj_R)[2])

            W_II = npc.Array.from_ndarray(
                W_II,
                [new_leg_L, new_leg_R, leg_p, leg_pconj],
                dtype=dtype,
                qtotal=trivial,
                labels=labels,
            )
            # TODO: could sort by charges.
            U.append(W_II)
        Id = [0] * (self.L + 1)
        return MPO(
            self.sites, U, self.bc, Id, Id, max_range=self.max_range, mps_unit_cell_width=self.unit_cell_width
        )  # no graph

    def expectation_value(self, psi, tol=1.0e-10, max_range=100, init_env_data={}):
        """Calculate ``<psi|self|psi>/<psi|psi>`` (or density for infinite).

        For infinite MPS, it **assumes** that `self` is extensive, e.g. a Hamiltonian
        but not a unitary, and returns the expectation value *density*.
        For finite MPS, it just returns the total value.

        This function is just a small wrapper around :meth:`expectation_value_finite`,
        :meth:`expectation_value_power` or :meth:`expectation_value_TM`.

        Parameters
        ----------
        psi : :class:`~tenpy.networks.mps.MPS`
            The state in which to calculate the expectation value.
        tol, max_range :
            See  :meth:`expectation_value_powermethod`.
        init_env_data : dict
            Optional environment data, if known.

        Returns
        -------
        exp_val : float/complex
            The expectation value of `self` with respect to the state `psi`.
            For an infinite MPS: the (energy) density per site.

        """
        if self.finite:
            return self.expectation_value_finite(psi, **init_env_data)
        elif self.max_range is None or self.max_range > 10 * self.L:
            return self.expectation_value_TM(psi, tol=tol, **init_env_data)
        else:
            return self.expectation_value_power(psi, tol=tol, max_range=max_range, **init_env_data)

    def expectation_value_finite(self, psi, init_env_data={}):
        """Calculate ``<psi|self|psi>/<psi|psi>`` for finite MPS.

        Parameters
        ----------
        psi : :class:`~tenpy.networks.mps.MPS`
            The state in which to calculate the expectation value.
        init_env_data : dict
            Optional environment data (for segment MPS).

        Returns
        -------
        exp_val : float/complex
            The expectation value of `self` with respect to the state `psi`
            (extensive, not the density).

        """
        if psi.bc == 'segment':
            if len(init_env_data) == 0:
                init_env_data['start_env_sites'] = 0
                warnings.warn(
                    'MPO.expectation_value(psi) with segment psi needs environments! '
                    'Can only estimate value completely ignoring contributions '
                    'across segment boundaries!'
                )
        env = MPOEnvironment(psi, self, psi, **init_env_data)
        val = env.full_contraction(0)  # handles explicit_plus_hc
        return np.real_if_close(val)

    def expectation_value_TM(self, psi, tol=1.0e-10, init_env_data={}):
        """Calculate ``<psi|self|psi>/<psi|psi> / L`` from the MPOTransferMatrix.

        Only for infinite MPS, and **assumes** that the Hamiltonian is an extensive sum of
        (quasi)local terms, and that the MPO has all :attr:`IdL` and :attr:`IdR` defined.

        Diagonalizing the :class:`MPOTransferMatrix` allows to find energy densities for infinite
        systems even for hamiltonians with infinite (exponentially decaying) range.


        Parameters
        ----------
        psi : :class:`~tenpy.networks.mps.MPS`
            The state in which to calculate the expectation value.
        tol : float
            Precision for finding the eigenvectors of the transfer matrix.
        init_env_data : dict
            Optional guess for the environment data.

        Returns
        -------
        exp_val : float/complex
            The expectation value density of `self` with respect to the state `psi`.

        """
        if psi.finite:
            raise ValueError('not infinite MPS')
        if np.linalg.norm(psi.norm_test()) > tol:
            psi = psi.copy()
            psi.canonical_form()
        guess = init_env_data.get('init_RP', None)
        TM = MPOTransferMatrix(self, psi, transpose=False, guess=guess)
        val, vec = TM.dominant_eigenvector(tol=tol)
        if abs(1.0 - val) > tol * 10.0:
            logger.warning('MPOTransferMatrix eigenvalue not 1: got 1. - %.3e', 1.0 - val)
        E = TM.energy(vec)  #  handles explicit_plus_hc
        return np.real_if_close(E)

    def expectation_value_power(self, psi, tol=1.0e-10, max_range=100):
        """Calculate ``<psi|self|psi>/<psi|psi>`` with a power-method.

        Only for infinite MPS, and **assumes** that the Hamiltonian is an extensive sum of
        (quasi)local terms, and that the MPO has all :attr:`IdL` and :attr:`IdR` defined.
        Only for infinite MPS.

        Instead of diagonalizing the MPOTransferMatrix like :meth:`expectation_value_TM`, this
        method uses just considers terms of the MPO starting in the first unit cell and then
        continues to contract tensors until convergence. For infinite-range MPOs, this converges
        like a power-method (i.e. slower than :meth:`expectation_value_TM`), but for finite-range
        MPOs it's likely faster, and conceptually cleaner.

        Parameters
        ----------
        psi : :class:`~tenpy.networks.mps.MPS`
            The state in which to calculate the expectation value.
        tol : float
            For infinite MPO containing exponentially decaying long-range terms, stop evaluating
            further terms if the terms in `LP` have norm < `tol`.
        max_range : int
            Ignored for finite `psi`.
            Contract at most ``self.L * max_range`` sites, even if `tol` is not reached.
            In that case, issue a warning.

        Returns
        -------
        exp_val : float/complex
            The expectation value of `self` with respect to the state `psi`.
            For an infinite MPS: the density per site.

        """
        if psi.finite:
            raise ValueError('not infinite MPS')
        env = MPOEnvironment(psi, self, psi, start_env_sites=0)
        L = lcm(self.L, psi.L)
        LP0 = env.init_LP(0)
        masks_L_no_IdL = []
        masks_R_no_IdRL = []
        for i, W in enumerate(self._W):
            mask_L = np.ones(W.get_leg('wL').ind_len, np.bool_)
            mask_L[self.get_IdL(i)] = False
            masks_L_no_IdL.append(mask_L)
            mask_R = np.ones(W.get_leg('wR').ind_len, np.bool_)
            mask_R[self.get_IdL(i + 1)] = False
            mask_R[self.get_IdR(i)] = False
            masks_R_no_IdRL.append(mask_R)
        # contract first site with theta
        theta = psi.get_theta(0, 1)
        LP = npc.tensordot(LP0, theta, axes=['vR', 'vL'])
        LP = npc.tensordot(LP, self._W[0], axes=[['wR', 'p0'], ['wL', 'p*']])
        LP = npc.tensordot(LP, theta.conj(), axes=[['vR*', 'p'], ['vL*', 'p0*']])

        for i in range(1, max(max_range, 1) * L):
            i0 = i % self.L
            W = self.get_W(i)
            if i >= L:
                # have one full unit cell: don't use further terms starting with IdL
                mask_L = masks_L_no_IdL[i0]
                LP.iproject(mask_L, 'wR')
                W = W.copy()
                W.iproject(mask_L, 'wL')
            B = psi.get_B(i, form='B')
            LP = npc.tensordot(LP, B, axes=['vR', 'vL'])
            LP = npc.tensordot(LP, W, axes=[['wR', 'p'], ['wL', 'p*']])
            LP = npc.tensordot(LP, B.conj(), axes=[['vR*', 'p'], ['vL*', 'p*']])

            if i >= L - 1:
                RP = env.init_RP(i)
                current_value = npc.inner(LP, RP, axes=[['vR*', 'wR', 'vR'], ['vL*', 'wL', 'vL']], do_conj=False)
                LP_converged = LP.copy()
                LP_converged.iproject(masks_R_no_IdRL[i0], 'wR')
                if npc.norm(LP_converged) < tol:
                    break  # no more terms left
        else:  # no break
            msg = f'Tolerance {tol:.2e} not reached within {max_range:d} sites'
            warnings.warn(msg, stacklevel=2)
        if self.explicit_plus_hc:
            current_value = current_value + np.conj(current_value)
        return np.real_if_close(current_value / L)

    def _expectation_value_environment(self, psi, *args):
        # TODO: Might be worth implementing?
        raise NotImplementedError('Could be implemented using MPOEnvironmentBuilder')

    def variance(self, psi, exp_val=None):
        """Calculate ``<psi|self^2|psi> - <psi|self|psi>^2``.

        Works only for finite systems. Ignores the :attr:`~tenpy.networks.mps.MPS.norm` of `psi`.

        .. todo ::
            This is a naive, expensive implementation contracting the full network.
            Try to follow :arXiv:`1711.01104` for a better estimate; would that even work in
            the infinite limit?

        Parameters
        ----------
        psi : :class:`~tenpy.networks.mps.MPS`
            State for which the variance should be taken.
        exp_val : float/complex | None
            The result of ``<psi|self|psi> = self.expectation_value(psi)`` if known;
            otherwise obtained from :meth:`expectation_value`.
            (Set this to 0 to obtain only the part ``<psi|self^2|psi>``.)

        """
        if self.bc != 'finite':
            raise ValueError('works only for finite systems')
        if self.L != psi.L:
            raise ValueError('expect same L')
        if psi._p_label != ['p']:
            raise NotImplementedError('not adjusted for non-standard MPS.')
        if self.explicit_plus_hc:
            raise NotImplementedError('not implemented for explicit_plus_hc flag')
        assert self.L >= 1
        if exp_val is None:
            exp_val = self.expectation_value(psi)

        th = psi.get_theta(0, n=1)
        W = self.get_W(0).take_slice(self.get_IdL(0), 'wL')
        contr = npc.tensordot(th, W.replace_label('wR', 'wR1'), axes=['p0', 'p*'])
        contr = npc.tensordot(contr, W.replace_label('wR', 'wR2'), axes=['p', 'p*'])
        contr = npc.tensordot(th.conj(), contr, axes=[['vL*', 'p0*'], ['vL', 'p']])
        for i in range(1, self.L):
            B = psi.get_B(i, form='B')
            W = self.get_W(i)
            contr = npc.tensordot(contr, B, axes=['vR', 'vL'])
            contr = npc.tensordot(contr, W.replace_label('wR', 'wR1'), axes=[['wR1', 'p'], ['wL', 'p*']])
            contr = npc.tensordot(contr, W.replace_label('wR', 'wR2'), axes=[['wR2', 'p'], ['wL', 'p*']])
            contr = npc.tensordot(contr, B.conj(), axes=[['vR*', 'p'], ['vL*', 'p*']])
        contr = contr.take_slice([self.get_IdR(self.L - 1)] * 2, ['wR1', 'wR2'])
        contr = npc.trace(contr, 'vR', 'vR*')
        return np.real_if_close(contr - exp_val**2)

    def prefactor(self, i, ops):
        """Get prefactor for a given string of operators in self.

        Parameters
        ----------
        i : int
            First site with non-identity operator.
        ops : list of str
            String of operators for which the prefactor is to be determined;
            the first entry is the name for the operator acting on site `i`,
            second entry on site `i` + 1, etc.

        Returns
        -------
        prefactor : float
            The prefactor obtained from ``trace(dagger(ops), H) / norm``,
            where ``norm = trace(dagger(ops), ops)``

        """
        ops = to_iterable(ops)
        IdL = self.get_IdL(i)
        IdR_final = self.get_IdR(i + len(ops) - 1)
        if IdL is None or IdR_final is None:
            return 0.0
        contr = None
        for k, opname in enumerate(ops):
            j = i + k
            W = self.get_W(j)
            if contr is None:
                contr = W.take_slice(IdL, 'wL')
            else:
                proj = np.ones(contr.shape[0])
                IdL = self.get_IdL(j)
                IdR = self.get_IdR(j - 1)
                if IdL is not None:
                    proj[IdL] = 0.0
                if IdR is not None:
                    proj[IdR] = 0.0
                contr.iscale_axis(proj, 0)
                contr = npc.tensordot(contr, W, axes=['wR', 'wL'])
            site = self.sites[j % len(self.sites)]
            op = site.get_op(opname)
            op_norm = npc.tensordot(op.conj(), op, axes=[['p', 'p*'], ['p*', 'p']])
            contr = npc.tensordot(op.conj(), contr, axes=[['p', 'p*'], ['p*', 'p']]) / op_norm
        contr = contr[IdR_final]
        return contr

    def to_TermList(self, op_basis, start=None, max_range=None, cutoff=1.0e-12, ignore=['Id', 'JW']):
        """Obtain a `TermList` represented by self.

        This function is meant for debugging MPOs to make sure they have the terms one expects.
        Be aware of pitfalls with operator orthonormality, e.g. for fermions
        ``N = 0.5 * (Id + JW)`` might not appear as you expect due to `ignore`.


        Parameters
        ----------
        op_basis : (list of) list of str
            Local basis of operators in which to represent all terms of `self`,
            e.g. ``['Id', 'Sx', 'Sy', 'Sz']`` for spin-1/2 or ``['Id', 'JW', 'C', 'Cd']`` for
            fermions. Should be orthogonal with respect to the operator product
            ``<A|B> = tr(A^dagger B)``.
        start : (list of) int
            Extract terms starting on that/these sites, going to the right, i.e. the left-most
            index within each term is in `start`.
            If ``None``, take all terms starting in ``range(L)``, i.e. one MPS unit cell for
            infinite systems.
        cutoff : float
            Drop terms with prefactors (roughly) smaller than that.
            Strictly speaking, it might also drop larger terms if the term has larger weight on
            the right (in the MPO) than on the left.
        ignore : list of str
            Filter terms to not contain these operator names when they're not the left/rightmost
            operators in a term.

        Returns
        -------
        term_list : :class:`~tenpy.networks.terms.TermList`
            The terms in `self` with left-most index in `start`.

        """
        if start is not None:
            start = to_iterable(start)
        else:
            start = range(self.L)
        L = self.L
        if max_range is None:
            max_range = 5 * L
            if self.max_range is not None:
                max_range = min(max_range, self.max_range)
        if isinstance(op_basis[0], str):
            op_basis = [op_basis]
        all_terms = []
        all_prefs = []
        for i in start:
            partial_L = [None] * self.get_W(i).get_leg('wL').ind_len
            if self.get_IdL(i) is None:
                continue
            partial_L[self.get_IdL(i)] = [([], 1.0)]
            if self.finite:
                max_range = min(max_range, L - i - 1)
            for k in range(max_range + 1):
                j = i + k
                IdL = self.get_IdL(j)
                IdR = self.get_IdR(j)
                if IdR is None:
                    IdR = -1  # not equal to positive index
                site_j = self.sites[j % L]
                W = self.get_W(j)
                W = W.transpose(['wL', 'wR', 'p', 'p*'])
                op_basis_j = op_basis[j % len(op_basis)]
                partial_R = [None] * W.get_leg('wR').ind_len
                if k > 0 and IdL is not None:
                    partial_L[IdL] = None  # drop terms not starting at `start`
                for opname in op_basis_j:
                    op = site_j.get_op(opname)
                    op_dagger = op.conj().transpose()
                    op_norm = npc.tensordot(op, op_dagger, axes=[['p', 'p*'], ['p*', 'p']])
                    op_W = npc.tensordot(W, op_dagger, axes=[['p', 'p*'], ['p*', 'p']])
                    op_W = op_W.to_ndarray() / op_norm
                    op_W[np.abs(op_W) < cutoff] = 0.0
                    for x, y in zip(*np.nonzero(op_W)):
                        if partial_L[x] is None:
                            continue
                        pref_j = op_W[x, y]
                        if y == IdR:
                            # finish terms
                            for term, pref in partial_L[x]:
                                if abs(pref * pref_j) < cutoff:
                                    continue
                                all_terms.append(term + [(opname, j)])
                                all_prefs.append(pref * pref_j)
                        else:
                            if partial_R[y] is None:
                                partial_R[y] = []
                            new_partial = partial_R[y]
                            if k > 0 and opname in ignore:
                                for term, pref in partial_L[x]:
                                    new_partial.append((term, pref * pref_j))
                            else:
                                for term, pref in partial_L[x]:
                                    new_partial.append((term + [(opname, j)], pref * pref_j))
                partial_L = partial_R
                if all(t is None for t in partial_L):
                    break
        return TermList(all_terms, all_prefs)

    def dagger(self):
        """Return hermitian conjugate copy of self."""
        if self.explicit_plus_hc:
            return self.copy()
        # complex conjugate and transpose everything
        Ws = [w.conj().itranspose(['wL*', 'wR*', 'p', 'p*']) for w in self._W]
        # and now revert conjugation of the wL/wR legs
        # rename labels 'wL*' -> 'wL', 'wR*' -> 'wR'
        for w in Ws:
            w.ireplace_labels(['wL*', 'wR*'], ['wL', 'wR'])
        # flip charges and qconj back
        for i in range(self.L - 1):
            Ws[i].legs[1] = wR = Ws[i].legs[1].flip_charges_qconj()
            Ws[i + 1].legs[0] = wR.conj()
        Ws[-1].legs[1] = wR = Ws[-1].legs[1].flip_charges_qconj()
        if self.finite:
            Ws[0].legs[0] = Ws[0].legs[0].flip_charges_qconj()
        else:
            Ws[0].legs[0] = wR.conj()
        # could keep graph in principle and only conjugate the operators
        # but its probably not worth the effort since building it is very fast
        return MPO(
            self.sites, Ws, self.bc, self.IdL, self.IdR, self.max_range, mps_unit_cell_width=self.unit_cell_width
        )

    def is_hermitian(self, eps=1.0e-10, max_range=None):
        """Check if `self` is a hermitian MPO.

        Shorthand for ``self.is_equal(self.dagger(), eps, max_range)``.
        """
        if self.explicit_plus_hc:
            return True
        return self.is_equal(self.dagger(), eps, max_range)

    def is_equal(self, other, eps=1.0e-10, max_range=None):
        """Check if `self` and `other` represent the same MPO to precision `eps`.

        To compare them efficiently we view `self` and `other` as MPS and compare the overlaps
        ``abs(<self|self> + <other|other> - 2 Re(<self|other>)) < eps*(<self|self>+<other|other>)``

        Parameters
        ----------
        other : :class:`MPO`
            The MPO to compare to.
        eps : float
            Precision threshold what counts as zero.
        max_range : None | int
            Ignored for finite MPS; for finite MPS we consider only the terms contained in the
            sites with indices ``range(self.L + 2 * max_range)``.
            None defaults to :attr:`max_range` (or :attr:`L` in case this is infinite or None).

        Returns
        -------
        equal : bool
            Whether `self` equals `other` to the desired precision.

        """
        if self.finite:
            num_sites = self.L
        elif max_range is not None and max_range < np.inf:
            num_sites = self.L + 2 * max_range
        elif self.max_range is not None and self.max_range < np.inf:
            num_sites = self.L + 2 * self.max_range
        else:
            num_sites = self.L + 2 * self.L
        ov = self.overlap(other, understood_infinite=True, num_sites=num_sites)
        s_norm = self.overlap(self, understood_infinite=True, num_sites=num_sites)
        o_norm = other.overlap(other, understood_infinite=True, num_sites=num_sites)
        dist = abs(s_norm - 2 * np.real(ov) + o_norm)
        return dist < eps * abs(s_norm + o_norm)

    def apply(self, psi, options):
        """Apply `self` to an MPS `psi` and compress `psi` in place.

        For infinite MPS, the assumed form of `self` is a product (e.g. a time evolution operator
        :math:`U= e^{-iH dt}`, not an (extensive) sum as a Hamiltonian would have.
        See :ref:`iMPSWarning` for more details.

        Options
        -------
        .. cfg:config :: ApplyMPO
            :include: VariationalApplyMPO, ZipUpApplyMPO

            compression_method : ``'SVD' | 'variational' | 'zip_up'``
                Mandatory.
                Selects the method to be used for compression.
                For the `SVD` compression, `trunc_params` is the only other option used.
            trunc_params : dict
                Truncation parameters as described in :cfg:config:`truncation`.


        Parameters
        ----------
        psi : :class:`~tenpy.networks.mps.MPS`
            The state to which `self` should be applied, in place.
        options : dict
            See above.

        """
        options = asConfig(options, 'ApplyMPO')
        method = options['compression_method']
        trunc_params = options.subconfig('trunc_params')
        if method == 'SVD':
            self.apply_naively(psi)
            return psi.compress_svd(trunc_params)
        elif method == 'variational':
            from ..algorithms.mps_common import VariationalApplyMPO

            return VariationalApplyMPO(psi, self, options).run()
        elif method == 'zip_up':
            trunc_err = self.apply_zipup(psi, options)
            return trunc_err + psi.compress_svd(trunc_params)
        elif method == 'variationalQR':
            from ..algorithms.mps_common import QRBasedVariationalApplyMPO

            return QRBasedVariationalApplyMPO(psi, self, options).run()

        # TODO: zipup method infinite?
        raise ValueError('Unknown compression method: ' + repr(method))

    def apply_naively(self, psi):
        """Applies an MPO to an MPS (in place) naively, without compression.

        This function simply contracts the `W` tensors of the MPO to the `B` tensors of the
        MPS, resulting in an MPS with bond dimension `self.chi * psi.chi`.

        .. warning ::
            This function sets only a wild *guess* for the new singular values.
            You should either compress the MPS or at least call
            :meth:`~tenpy.networks.mps.MPS.canonical_form`.
            If you use :meth:`apply` instead, this will be done automatically.

        Parameters
        ----------
        psi : :class:`~tenpy.networks.mps.MPS`
            The MPS to which `self` should be applied. Modified in place!

        """
        bc = psi.bc
        if bc != self.bc:
            raise ValueError('Boundary conditions of MPS and MPO are not the same')
        if psi.L != self.L:
            raise ValueError('Length of MPS and MPO not the same')
        if self.explicit_plus_hc:
            raise NotImplementedError("Can't use explicit_plus_hc with apply_naively")
        for i in range(psi.L):
            B = npc.tensordot(psi.get_B(i, 'B'), self.get_W(i), axes=('p', 'p*'))
            if i == 0 and bc == 'finite':
                B = B.take_slice(self.get_IdL(i), 'wL')
                B = B.combine_legs(['wR', 'vR'], qconj=[-1])
                B.ireplace_labels(['(wR.vR)'], ['vR'])
                B.legs[B.get_leg_index('vR')] = B.get_leg('vR').to_LegCharge()
            elif i == psi.L - 1 and bc == 'finite':
                B = B.take_slice(self.get_IdR(i), 'wR')
                B = B.combine_legs(['wL', 'vL'], qconj=[1])
                B.ireplace_labels(['(wL.vL)'], ['vL'])
                B.legs[B.get_leg_index('vL')] = B.get_leg('vL').to_LegCharge()
            else:
                B = B.combine_legs([['wL', 'vL'], ['wR', 'vR']], qconj=[+1, -1])
                B.ireplace_labels(['(wL.vL)', '(wR.vR)'], ['vL', 'vR'])
                B.legs[B.get_leg_index('vL')] = B.get_leg('vL').to_LegCharge()
                B.legs[B.get_leg_index('vR')] = B.get_leg('vR').to_LegCharge()
            psi.set_B(i, B, 'B')

        # fix right leg of last tensor (reason: combine_legs sorts charges)
        if not psi.finite:
            left_leg = psi.get_B(psi.L).get_leg('vL')
            right_leg = psi.get_B(psi.L - 1).get_leg('vR')
            perm, _ = left_leg.sort()
            inv_perm = inverse_permutation(perm)
            inv_perm_flat = right_leg.perm_flat_from_perm_qind(inv_perm)
            # TODO: Do it without using `permute`. We do not need to touch the
            # blocks anyway, so we can just re-order B._qdata and the leg charges.
            B = psi.get_B(psi.L - 1).permute(inv_perm_flat, axis='vR')
            psi.set_B(psi.L - 1, B, 'B')

        if bc == 'infinite':
            # calculate (rather arbitrary) guess for S[0] (no we don't like it either)
            weight = np.ones(self.get_W(0).shape[self.get_W(0).get_leg_index('wL')]) * 0.05
            weight[self.get_IdL(0)] = 1
            weight = weight / np.linalg.norm(weight)
            S0 = np.kron(weight, psi.get_SL(0))  # order dictated by '(wL,vL)'
        else:
            S0 = np.ones(psi.get_B(0, None).get_leg('vL').ind_len)
        psi.set_SL(0, S0)
        for i in range(psi.L):
            psi.set_SR(i, np.ones(psi.get_B(i, None).get_leg('vR').ind_len))

    def apply_zipup(self, psi, options):
        """Applies an MPO to an MPS (in place) with the zip-up method.

        Described in Ref. :cite:`stoudenmire2010`.

        The 'W' tensors are contracted to the 'B' tensors with intermediate SVD
        compressions, truncated to bond dimensions `chi_max * m_temp`.

        .. warning ::
            The MPS afterwards is only approximately in canonical form
            (under the assumption that self is close to unity).
            You should either compress the MPS or at least call
            :meth:`~tenpy.networks.mps.MPS.canonical_form`.
            If you use :meth:`apply` instead, this will be done automatically.

        Parameters
        ----------
        psi : :class:`~tenpy.networks.mps.MPS`
            The MPS to which `self` should be applied. Modified in place!
        trunc_params : dict
            Truncation parameters as described in :cfg:config:`truncation`.


        Options
        -------
        .. cfg:config :: ZipUpApplyMPO

            trunc_params : dict
                Truncation parameters as described in :cfg:config:`truncation`.
            m_temp: int
                bond dimension will be truncated to `m_temp * chi_max`
            trunc_weight: float
                reduces cut for Schmidt values to `trunc_weight * svd_min`

        """
        options = asConfig(options, 'zip_up')
        m_temp = options.get('m_temp', 2, int)
        trunc_weight = options.get('trunc_weight', 1.0, 'real')
        trunc_params = options.subconfig('trunc_params')
        relax_trunc = trunc_params.copy()  # relaxed truncation criteria
        relax_trunc['chi_max'] *= m_temp
        if 'svd_min' in relax_trunc.keys():
            relax_trunc['svd_min'] *= trunc_weight
        trunc_err = TruncationError()
        bc = psi.bc
        if bc != self.bc:
            raise ValueError('Boundary conditions of MPS and MPO are not the same')
        if psi.L != self.L:
            raise ValueError('Length of MPS and MPO not the same')
        if bc != 'finite':
            raise ValueError('Only finite boundary conditions implemented')
        if self.explicit_plus_hc:
            raise NotImplementedError("Can't use explicit_plus_hc with apply_zipup")
        for i in range(psi.L):
            B = npc.tensordot(psi.get_B(i, 'B'), self.get_W(i), axes=('p', 'p*'))
            if i == 0 and bc == 'finite':
                B = B.take_slice(self.get_IdL(i), 'wL')
                B = B.combine_legs([['vL', 'p'], ['wR', 'vR']], qconj=[+1, -1])
                U, S, VH, err, norm_new = svd_theta(B, relax_trunc)
                trunc_err += err
                psi.norm *= norm_new
                U = U.split_legs()
                VH = VH.split_legs()
                VH.iscale_axis(S, 'vL')
                psi.set_SR(i, S)
                psi.set_B(i, U, 'A')
            elif i == psi.L - 1 and bc == 'finite':
                B = npc.tensordot(VH, B, axes=(['wR', 'vR'], ['wL', 'vL']))
                B = B.take_slice(self.get_IdR(i), 'wR')
                B = B.combine_legs(['vL', 'p'], qconj=[-1])
                U, S, VH, err, norm_new = svd_theta(B, relax_trunc, [B.qtotal, None])
                trunc_err += err
                psi.norm *= norm_new
                U = U.split_legs()
                psi.set_SR(i, S)
                psi.set_B(i, U, 'A')
            else:
                B = npc.tensordot(VH, B, axes=(['wR', 'vR'], ['wL', 'vL']))
                B = B.combine_legs([['vL', 'p'], ['wR', 'vR']], qconj=[1, -1])
                U, S, VH, err, norm_new = svd_theta(B, relax_trunc)
                trunc_err += err
                psi.norm *= norm_new
                U = U.split_legs()
                VH = VH.split_legs()
                VH.iscale_axis(S, 'vL')
                psi.set_SR(i, S)
                psi.set_B(i, U, 'A')

        return trunc_err

    def plus_identity(self, alpha, beta, sites=[0]):
        r"""Compute a new MPO :math:`alpha * 1 + beta * \mathtt{self}`.

        This can e.g. be used to make a simple (non-unitary) first-order approximation to
        the time evolution unitary :math:`e^{-i t H} \approx 1 - i t H`.

        This function only works for finite MPOs for now.

        Parameters
        ----------
        alpha : float|complex
            Coefficient for identity
        beta : float|complex
            Coefficient for existing MPO
        sites : list
            List of MPO indices of which tensors to modify

        Returns
        -------
        mpo : :class:`~tenpy.networks.mpo.MPO`
            MPO representing the operator :math:`\alpha * 1 + \beta O`

        Notes
        -----
        There is significant freedom in how we incorporate the linear
        combination into the MPO structure. One naive choice is to only modify the first tensor.

        The first tensor (ignoring the second wL entry since it's unnecessary)::

            [1 C D] -> [beta*1 beta*C alpha*1+beta*D]

        Another choice is to modify `N` tensors specified by the input argument `sites`.

        """
        if self.bc != 'finite':
            raise NotImplementedError('MPO.add_identity only works for finite MPO.')
        if self.explicit_plus_hc:
            raise NotImplementedError

        N = len(sites)
        assert N <= self.L
        if not set(sites).issubset(set(list(range(self.L)))):
            raise ValueError(f'The sites {sites} are not strictly contained in {{1, ..., {self.L - 1}}}.')
        if sorted(sites) != [*range(min(sites), max(sites) + 1)]:
            # test fails for non-contiguous sites. not sure why
            raise NotImplementedError

        t_beta = beta ** (1 / N)
        t_alpha = alpha / N

        IdL = self.IdL
        IdR = self.IdR

        chinfo = self.chinfo
        trivial = chinfo.make_valid()
        U = []
        counter = 0
        for k in range(0, self.L):
            labels = ['wL', 'wR', 'p', 'p*']
            W = self.get_W(k).itranspose(labels)
            assert np.all(W.qtotal == trivial)
            DL, DR, d, d = W.shape

            A_npc, B_npc, C_npc, D_npc = _partition_W(W, IdL[k], IdR[k], IdL[k + 1], IdR[k + 1])
            Id_npc = npc.eye_like(D_npc, labels=['p', 'p*'])
            dW = np.empty((DL, DR), dtype=object)

            # Get coefficients depending on if site k in sites and if so
            # what number site in sites it is.
            if k in sites:
                b = t_beta
                a = t_alpha
                g = 1 if counter != 0 else beta
                d = 1 if counter != N - 1 else beta
                counter = counter + 1
            else:
                b = g = d = 1.0
                a = 0.0

            # First Row - only this is modified
            dW[0, 0] = d * Id_npc
            for i in range(0, DR - 2):
                dW[0, i + 1] = b ** (counter) * C_npc[0, i]
            dW[0, -1] = (b**N) * D_npc + a * Id_npc
            # Middle Rows
            for i in range(0, DL - 2):
                for j in range(0, DR - 2):
                    dW[i + 1, j + 1] = b * A_npc[i, j]
                dW[i + 1, -1] = b ** (N - counter + 1) * B_npc[i, 0]
            # Bottom Rows
            dW[-1, -1] = g * Id_npc
            U.append(dW)

        assert counter == N
        # Sajant: We have enforced that the MPO look upper block triangular without any permutations
        IdL = [0] * (self.L + 1)
        IdR = [-1] * (self.L + 1)
        return MPO.from_grids(
            self.sites,
            U,
            self.bc,
            IdL,
            IdR,
            max_range=self.max_range,
            explicit_plus_hc=self.explicit_plus_hc,
            mps_unit_cell_width=self.unit_cell_width,
        )

    def overlap(self, other, understood_infinite: bool = False, num_sites: int = None):
        """Overlap between two MPOs.

        For finite MPOs, this is the Frobenius inner product::

            <self|other> = Tr[hconj(self) @ other]

        For infinite MPOs, the TD limit of that overlap is always either 0, 1 or infinite,
        i.e. it is not helpful. Instead we choose a finite section of the infinite overlap diagram
        and project onto ``IdL`` on the left and ``IdR`` on the right. This means we effectively
        compute the overlap between those contributions to the MPOs that act trivially outside
        the finite section. The main motivation is a distance measure in :meth:`is_equal`.

        Parameters
        ----------
        other : MPO
            The other operator. Must have the same :attr:`finite`, and if finite the same :attr:`L`.
        understood_infinite : bool
            For infinite MPOs, the overlap has an unusual definition, see above.
            Set this flag to confirm you understand this and suppress the warning.
        num_sites : int
            Ignored for finite MPOs. For infinite MPOs, the number of sites that we contract.
            We project onto IdL on site ``0``, contract tensors from ``range(num_sites)``, and
            then project onto IdR. By default, we use ``L + 2 * max_range`` of whichever MPO has the
            larger value, where we substitute ``L`` for an unknown or infinite ``max_range``.

        """
        if self.finite and other.finite:
            assert self.L == other.L
            num_sites = self.L
        elif not self.finite and not other.finite:
            if num_sites is None:
                self_max_range = self.max_range
                if self_max_range is None or self_max_range == np.inf:
                    self_max_range = self.L
                other_max_range = other.max_range
                if other_max_range is None or other_max_range == np.inf:
                    other_max_range = other.L
                num_sites = max(self.L + 2 * self_max_range, other.L + 2 * other.max_range)
            assert num_sites >= self.L
            if not understood_infinite:
                msg = (
                    'The overlap between infinite MPOs has an unusual definition. Make sure '
                    'you understand it, then set `understood_infinite=True` to suppress '
                    'this warning.'
                )
                warnings.warn(msg, stacklevel=2)
        else:
            raise ValueError('cant take overlap between finite and infinite MPO')

        if self.explicit_plus_hc and other.explicit_plus_hc:
            # <A|B> = Tr[hc(A) B] = conj(Tr[hc(B) A]) = conj(Tr[A hc(B)]) = conj(<hc(A)|hc(B)>)
            # <A + hc(A) | B + hc(B)> = <A|B> + <hc(A)|B> + <A|hc(B)> + <hc(A)|hc(B)>
            #                         = <A|B> + <hc(A)|B> + conj(<hc(A)|B>) + conj(<A|B>)
            #                         = 2 Re[ <A|B> + <hc(A)|B> ]
            A_B = self._overlap_no_hc(other, num_sites=num_sites)
            hcA_B = self._overlap_no_hc(other, num_sites=num_sites, hconj_self=True)
            ov = 2 * np.real(A_B + hcA_B)

        elif self.explicit_plus_hc:
            A_B = self._overlap_no_hc(other, num_sites=num_sites)
            hcA_B = self._overlap_no_hc(other, num_sites=num_sites, hconj_self=True)
            ov = A_B + hcA_B

        elif other.explicit_plus_hc:
            # <A|B + hc(B)> = <A|B> + <A|hc(B)> = <A|B> + conj(<hc(A)|B>)
            A_B = self._overlap_no_hc(other, num_sites=num_sites)
            hcA_B = self._overlap_no_hc(other, num_sites=num_sites, hconj_self=True)
            ov = A_B + np.conj(hcA_B)

        else:
            ov = self._overlap_no_hc(other, num_sites=num_sites)

        return ov

    def _overlap_no_hc(self, other, num_sites: int, hconj_self: bool = False):
        """Internal version of :meth:`overlap` that ignores :attr:`explicit_plus_hc`.

        This computes the overlap for the MPO given by the tensors, ignoring any explicit hc.
        If ``hconj_self``, we use the hc of self instead, i.e. compute
        ``<hc(self)|other> = Tr[self @ other]``.
        """
        wA = self.get_W(0).take_slice([self.get_IdL(0)], ['wL'])
        wB = other.get_W(0).take_slice([other.get_IdL(0)], ['wL'])

        if hconj_self:
            wA = wA.replace_label('wR', 'wR*')
        else:
            wA = wA.conj()
        res = npc.tensordot(wA, wB, axes=[['p*', 'p'], ['p', 'p*']])  # wR* wR

        for i in range(1, num_sites):
            if hconj_self:
                wA = self.get_W(i).replace_labels(['wL', 'wR'], ['wL*', 'wR*'])
            else:
                wA = self.get_W(i).conj()
            wB = other.get_W(i)
            res = npc.tensordot(res, wA, axes=['wR*', 'wL*'])
            res = npc.tensordot(res, wB, axes=[['wR', 'p*', 'p'], ['wL', 'p', 'p*']])

        IdR_idcs = (self.get_IdR(num_sites - 1), other.get_IdR(num_sites - 1))
        res = res.itranspose(['wR*', 'wR'])[IdR_idcs]
        return res

    def distance(self, other, understood_infinite: bool = False, num_sites: int = None):
        """The Frobenius distance induced by the inner product :meth:`overlap`."""
        ov = self.overlap(other, understood_infinite=understood_infinite, num_sites=num_sites)
        s_norm = self.overlap(self, understood_infinite=understood_infinite, num_sites=num_sites)
        o_norm = other.overlap(other, understood_infinite=understood_infinite, num_sites=num_sites)
        dist = s_norm - 2 * np.real(ov) + o_norm
        if dist < -1e-14 * (s_norm + o_norm):
            raise RuntimeError('Negative distance encountered.')
        return abs(dist)

    def _to_valid_index(self, i, bond=False):
        """Make sure `i` is a valid index of a site.

        .. deprecated :: 1.2.0
            Use :meth:`~tenpy.networks.mps.MPSGeometry._to_valid_site_index`
            or :meth:`~tenpy.networks.mps.MPSGeometry._to_valid_bond_index` instead.
            Note that they have an additional return value.

        For finite systems, we just check if ``i`` is within bounds.
        For infinite systems, we return the index *within* the MPS unit cell that is equivalent to
        ``i``, by adding a suitable multiple of ``self.L``.
        """
        msg = '_to_valid_index methods have been deprecated. Use _to_valid_site_index or _to_valid_bond_index instead.'
        warnings.warn(msg, category=FutureWarning, stacklevel=2)
        if not self.finite:
            return i % self.L
        if i < 0:
            msg = (
                'Negative site indices for open boundary conditions are deprecated and will '
                'raise a ValueError in the future'
            )
            warnings.warn(msg, category=FutureWarning, stacklevel=3)
            i += self.L
        if i >= self.L + int(bond) or i < 0:
            raise KeyError(f'i = {i:d} out of bounds for finite MPO')
        return i

    @staticmethod
    def _get_Id(Id, L):
        """Parse the IdL or IdR argument of __init__"""
        if Id is None:
            return [None] * (L + 1)
        try:
            Id = list(Id)
        except TypeError:
            return [Id] * (L + 1)
        if len(Id) != L + 1:
            raise ValueError(f'expected list with L+1={L + 1:d} entries')
        return Id

    def __add__(self, other):
        """Return an MPO representing `self + other`.

        Requires both `self` and `other` to be in standard sum form with `IdL` and `IdR` being set.

        This is a naive, block-wise addition without any compression!

        Parameters
        ----------
        other : :class:`MPO`
            MPO to be added to `self`.

        Returns
        -------
        sum_mpo : :class:`MPO`
            The sum `self + other`.

        """
        if self.explicit_plus_hc != other.explicit_plus_hc:
            raise ValueError('Can not add MPOs with different explicit_plus_hc flags')

        L = self.L
        assert self.bc == other.bc
        assert self.unit_cell_width == other.unit_cell_width
        assert other.L == L

        ps = [self._get_block_projections(i) for i in range(L + 1)]
        po = [other._get_block_projections(i) for i in range(L + 1)]

        def block(of, l, r):
            block_, pl, pr = of
            l = pl[l]
            r = pr[r]
            if l is None or r is None:
                return None
            # else
            return block_[l, r]

        # l/r = left/right,  s/o = self/other
        Ws = []
        IdL = [None] * (L + 1)
        IdL[0] = 0
        IdR = [None] * (L + 1)
        IdR[-1] = -1
        for i in range(L):
            ws = self._W[i].itranspose(['wL', 'wR', 'p', 'p*'])
            wo = other._W[i].itranspose(['wL', 'wR', 'p', 'p*'])
            s = (ws, ps[i], ps[i + 1])
            o = (wo, po[i], po[i + 1])
            onsite = add_with_None_0(block(s, 0, 2), block(o, 0, 2))

            w_grid = [
                [block(s, 0, 0), block(s, 0, 1), block(o, 0, 1), onsite        ],
                [None,           block(s, 1, 1), None,           block(s, 1, 2)],
                [None,           None,           block(o, 1, 1), block(o, 1, 2)],
                [None,           None,           None,           block(s, 2, 2)]
            ]  # fmt: skip
            w_grid = np.array(w_grid, dtype=object)
            if w_grid[0, 0] is None:
                w_grid[0, 0] = block(o, 0, 0)
            if w_grid[0, 0] is not None:
                IdL[i + 1] = 0
            if w_grid[-1, -1] is None:
                w_grid[-1, -1] = block(o, 2, 2)
            if w_grid[-1, -1] is not None:
                IdR[i] = -1
            # now drop rows and columns which are completely zero
            w_is_None = np.array([[(w is None) for w in w_row] for w_row in w_grid], dtype=bool)
            w_grid = w_grid[np.logical_not(np.all(w_is_None, 1)), :]
            w_grid = w_grid[:, np.logical_not(np.all(w_is_None, 0))]
            Ws.append(npc.grid_concat(w_grid, [0, 1]))
        if self.max_range is not None and other.max_range is not None:
            max_range = max(self.max_range, other.max_range)
        else:
            max_range = None
        return MPO(
            self.sites,
            Ws,
            self.bc,
            IdL,
            IdR,
            max_range,
            self.explicit_plus_hc,
            mps_unit_cell_width=self.unit_cell_width,
        )  # no graph

    def _get_block_projections(self, i):
        """Projections onto (IdL, other, IdR) on bond `i` in range(0, L+1)"""
        if self.finite:  # allows i = L for finite bc
            if i < self.L:
                length = self._W[i].get_leg('wL').ind_len
            else:
                assert i == self.L
                length = self._W[i - 1].get_leg('wR').ind_len
        else:
            i = i % self.L
            length = self._W[i].get_leg('wL').ind_len
        IdL = self.IdL[i]
        IdR = self.IdR[i]
        proj_other = np.ones(length, np.bool_)
        if IdL is None:
            proj_IdL = None
        else:
            proj_IdL = np.zeros(length, np.bool_)
            proj_IdL[IdL] = True
            proj_other[IdL] = False
        if IdR is None:
            proj_IdR = None
        else:
            proj_IdR = np.zeros(length, np.bool_)
            proj_IdR[IdR] = True
            proj_other[IdR] = False
            assert IdR != IdL
        if length == int(IdL is not None) + int(IdR is not None):
            proj_other = None
        return (proj_IdL, proj_other, proj_IdR)


def make_W_II(t, A, B, C, D):
    r"""W_II approx to exp(t H) from MPO parts (A, B, C, D).

    Get the W_II approximation of :cite:`zaletel2015`.

    In the paper, we have two formal parameter "phi_{r/c}" which satisfies
    :math:`\phi_r^2 = phi_c^2 = 0`.  To implement this, we temporarily extend the virtual Hilbert
    space with two hard-core bosons "br, bl". The components of Eqn (11) can be computed for each
    index of the virtual row/column independently
    The matrix exponential is done in the hard-core extended Hilbert space

    Parameters
    ----------
    t : float
        The time step per application of the propagator.
        Should be imaginary for real time evolution!
    A, B, C, D :  :class:`numpy.ndarray`
        Blocks of the MPO tensor to be exponentiated, as defined in :cite:`zaletel2015`.
        Legs ``'wL', 'wR', 'p', 'p*'``; legs projected to a single IdL/IdR can be dropped.

    """
    tC = np.sqrt(np.abs(t))  # spread time step across B, C
    tB = t / tC
    d = D.shape[0]

    # The virtual size of W is  (1+Nr, 1+Nc)
    Nr = A.shape[0]
    Nc = A.shape[1]
    W = np.zeros((1 + Nr, 1 + Nc, d, d), dtype=np.result_type(D, t))

    Id_ = np.array([[1, 0], [0, 1]])  # 2x2 operators in a hard-core boson space
    b = np.array([[0, 0], [1, 0]])

    Id = np.kron(Id_, Id_)  # 4x4 operators in the 2x hard core boson space
    Br = np.kron(b, Id_)
    Bc = np.kron(Id_, b)
    Brc = np.kron(b, b)
    for r in range(Nr):  # double loop over row / column of A
        for c in range(Nc):
            # Select relevant part of virtual space and extend by hardcore bosons
            h = (
                np.kron(Brc, A[r, c, :, :])
                + np.kron(Br, tB * B[r, :, :])
                + np.kron(Bc, tC * C[c, :, :])
                + t * np.kron(Id, D)
            )
            w = expm(h)  # Exponentiate in the extended Hilbert space
            w = w.reshape((2, 2, d, 2, 2, d))
            w = w[:, :, :, 0, 0, :]
            W[1 + r, 1 + c, :, :] = w[1, 1]  # extracts relevant parts according to Eqn 11
            if c == 0:
                W[1 + r, 0] = w[1, 0]
            if r == 0:
                W[0, 1 + c] = w[0, 1]
                if c == 0:
                    W[0, 0] = w[0, 0]
        if Nc == 0:  # technically only need one boson
            h = np.kron(Br, tB * B[r, :, :]) + t * np.kron(Id, D)
            w = expm(h)
            w = w.reshape((2, 2, d, 2, 2, d))
            w = w[:, :, :, 0, 0, :]
            W[1 + r, 0] = w[1, 0]
            if r == 0:
                W[0, 0] = w[0, 0]
    if Nr == 0:
        for c in range(Nc):
            h = np.kron(Bc, tC * C[c, :, :]) + t * np.kron(Id, D)
            w = expm(h)
            w = w.reshape((2, 2, d, 2, 2, d))
            w = w[:, :, :, 0, 0, :]
            W[0, 1 + c] = w[0, 1]
            if c == 0:
                W[0, 0] = w[0, 0]
        if Nc == 0:
            W = expm(t * D).reshape([1, 1, d, d])
    return W


class MPOGraph(MPSGeometry):
    """Representation of an MPO by a graph, based on a 'finite state machine'.

    This representation is used for building H_MPO from the interactions.
    The idea is to view the MPO as a kind of 'finite state machine'.
    The **states** or **keys** of this finite state machine life on the MPO bonds *between* the
    `Ws`. They label the indices of the virtual bonds of the MPOs, i.e., the indices on legs
    ``wL`` and ``wR``. They can be anything hash-able like a ``str``, ``int`` or a tuple of them.

    The **edges** of the graph are the entries ``W[keyL, keyR]``, which itself are onsite operators
    on the local Hilbert space. The indices `keyL` and `keyR` correspond to the legs ``'wL', 'wR'``
    of the MPO. The entry ``W[keyL, keyR]`` connects the state ``keyL`` on bond ``(i-1, i)``
    with the state ``keyR`` on bond ``(i, i+1)``.

    The keys ``'IdR'`` (for 'identity left') and ``'IdR'`` (for 'identity right') are reserved to
    represent only ``'Id'`` (=identity) operators to the left and right of the bond, respectively.

    .. todo ::
        might be useful to add a "cleanup" function which removes operators cancelling each other
        and/or unused states. Or better use a 'compress' of the MPO?

    Parameters
    ----------
    sites : list of :class:`~tenpy.models.lattice.Site`
        Local sites of the Hilbert space.
    bc : {'finite', 'infinite'}
        MPO boundary conditions.
    max_range : int | np.inf | None
        Maximum range of hopping/interactions (in unit of sites) of the MPO. ``None`` for unknown.
    unit_cell_width : int
        See :attr:`~tenpy.models.lattice.Lattice.num_unit_cell_width`.

    Attributes
    ----------
    max_range : int | np.inf | None
        Maximum range of hopping/interactions (in unit of sites) of the MPO. ``None`` for unknown.
    states : list of set of keys
        ``states[i]`` gives the possible keys at the virtual bond ``(i-1, i)`` of the MPO.
        `L+1` entries.
    graph : list of dict of dict of list of tuples
        For each site `i` a dictionary ``{keyL: {keyR: [(opname, strength)]}}`` with
        ``keyL in states[i]`` and ``keyR in states[i+1]``.
    _grid_legs : None | list of LegCharge
        The charges for the MPO

    """

    _valid_bc = ['finite', 'infinite']  # segment makes no sense for MPOGraph

    def __init__(self, sites, bc='finite', max_range=None, unit_cell_width=None):
        super().__init__(sites=sites, bc=bc, unit_cell_width=unit_cell_width)
        self.max_range = max_range
        # empty graph
        self.states = [set() for _ in range(self.L + 1)]
        self.graph = [{} for _ in range(self.L)]
        self._ordered_states = None
        self.test_sanity()

    @classmethod
    def from_terms(cls, terms, sites, bc, insert_all_id=True, unit_cell_width=None):
        """Initialize an :class:`MPOGraph` from OnsiteTerms and CouplingTerms.

        Parameters
        ----------
        terms : iterable of ``tenpy.networks.terms.*Terms`` classes
            Entries can be :class:`~tenpy.networks.terms.OnsiteTerms`,
            :class:`~tenpy.networks.terms.CouplingTerms`,
            :class:`~tenpy.networks.terms.MultiCouplingTerms` or
            :class:`~tenpy.networks.terms.ExponentialCouplingTerms`.
            All the entries get added to the new :class:`MPOGraph`.
        sites : list of :class:`~tenpy.networks.site.Site`
            Local sites of the Hilbert space.
        bc : ``'finite' | 'infinite'``
            MPO boundary conditions.
        insert_all_id : bool
            Whether to insert identities such that `IdL` and `IdR` are defined on each bond.
            See :meth:`add_missing_IdL_IdR`.
        unit_cell_width : int
            See :attr:`~tenpy.models.lattice.Lattice.mps_unit_cell_width`.

        Returns
        -------
        graph : :class:`MPOGraph`
            Initialized with the given terms.

        See Also
        --------
        from_term_list :
            equivalent for representation by :class:`~tenpy.networks.terms.TermList`.

        """
        graph = cls(sites, bc, 0, unit_cell_width=unit_cell_width)
        for term in terms:
            term.add_to_graph(graph)
            # add_to_graph increases `max_range` as necessary
        graph.add_missing_IdL_IdR(insert_all_id)
        return graph

    @classmethod
    def from_term_list(cls, term_list, sites, bc, insert_all_id=True, unit_cell_width=None):
        """Initialize from a list of operator terms and prefactors.

        Parameters
        ----------
        term_list : :class:`~tenpy.networks.mps.TermList`
            Terms to be added to the MPOGraph.
        sites : list of :class:`~tenpy.networks.site.Site`
            Local sites of the Hilbert space.
        bc : ``'finite' | 'infinite'``
            MPO boundary conditions.
        insert_all_id : bool
            Whether to insert identities such that `IdL` and `IdR` are defined on each bond.
            See :meth:`add_missing_IdL_IdR`.
        unit_cell_width : int
            See :attr:`~tenpy.models.lattice.Lattice.mps_unit_cell_width`.

        Returns
        -------
        graph : :class:`MPOGraph`
            Initialized with the given terms.

        See Also
        --------
        from_terms : equivalent for other representation of terms.

        """
        ot_ct = term_list.to_OnsiteTerms_CouplingTerms(sites)
        return cls.from_terms(ot_ct, sites, bc, insert_all_id, unit_cell_width=unit_cell_width)

    def test_sanity(self):
        """Sanity check, raises ValueErrors, if something is wrong."""
        super().test_sanity()
        assert len(self.graph) == self.L
        assert len(self.states) == self.L + 1
        for i, site in enumerate(self.sites):
            stL, stR = self.states[i : i + 2]
            # check graph
            gr = self.graph[i]
            for keyL in gr:
                assert keyL in stL
                for keyR in gr[keyL]:
                    assert keyR in stR
                    for opname, strength in gr[keyL][keyR]:
                        assert site.valid_opname(opname)
        # done

    def add(self, i, keyL, keyR, opname, strength, check_op=True, skip_existing=False):
        """Insert an edge into the graph.

        Parameters
        ----------
        i : int
            Site index at which the edge of the graph is to be inserted.
        keyL : hashable
            The state at bond (i-1, i) to connect from.
        keyR : hashable
            The state at bond (i, i+1) to connect to.
        opname : str
            Name of the operator.
        strength : str
            Prefactor of the operator to be inserted.
        check_op : bool
            Whether to check that 'opname' exists on the given `site`.
        skip_existing : bool
            If ``True``, skip adding the graph node if it exists (with same keys and `opname`).

        """
        i = i % self.L
        if check_op:
            if not self.sites[i].valid_opname(opname):
                raise ValueError(f'operator {opname!r} not existent on site {i:d}')
        G = self.graph[i]
        if keyL not in self.states[i]:
            self.states[i].add(keyL)
        if keyR not in self.states[i + 1]:
            self.states[i + 1].add(keyR)
        D = G.setdefault(keyL, {})
        if keyR not in D:
            D[keyR] = [(opname, strength)]
        else:
            entry = D[keyR]
            if not skip_existing or not any([op == opname for op, _ in entry]):
                entry.append((opname, strength))

    def add_string_left_to_right(self, i, j, key, opname='Id', check_op=True, skip_existing=True):
        r"""Insert a bunch of edges for an 'operator string' into the graph.

        Terms like :math:`S^z_i S^z_j` actually stand for
        :math:`S^z_i \otimes \prod_{i < k < j} \mathbb{1}_k \otimes S^z_j`.
        This function adds the :math:`\mathbb{1}` terms to the graph.

        Parameters
        ----------
        i, j: int
            An edge is inserted on all sites between `i` and `j`, `i < j`.
            `j` can be larger than :attr:`L`, in which case the operators are supposed to act on
            different MPS unit cells.
        key: tuple
            The key at bond (i+1, i) to connect from.
        opname : str
            Name of the operator to be used for the string.
            Useful for the Jordan-Wigner transformation to fermions.
        skip_existing : bool
            Whether existing graph nodes should be skipped.

        Returns
        -------
        key_i : tuple
            The `key` on the right of site i we connected to.

        """
        if j <= i:
            raise ValueError('j <= i not allowed')
        keyL = keyR = key
        for k in range(i + 1, j):
            if (k - i) % self.L == 0:
                # necessary to extend key because keyL is already in use at this bond
                keyR = keyL + (k, opname, opname)  # same structure as for other standard keys
                # (i, op_i, op_str_right_of_i) e.g. in MultiCouplingTerms.add_to_graph
            k = k % self.L
            if not self.has_edge(k, keyL, keyR):
                self.add(k, keyL, keyR, opname, 1.0, check_op=check_op, skip_existing=skip_existing)
            keyL = keyR
        return keyL

    def add_string_right_to_left(self, j, i, key, opname='Id', check_op=True, skip_existing=True):
        r"""Insert a bunch of edges for an 'operator string' into the graph.

        Similar as :meth:`add_string_left_to_right`, but in the other direction.

        Parameters
        ----------
        j, i: int
            An edge is inserted on all sites between `i` and `j`, `i < j`.
            Note the switched argument order compared to :meth:`add_string_left_to_right`.
        key: tuple
            The key at bond (j-1, j) to connect from.
        opname : str
            Name of the operator to be used for the string.
            Useful for the Jordan-Wigner transformation to fermions.
        skip_existing : bool
            Whether existing graph nodes should be skipped.

        Returns
        -------
        key_i : hashable
            The `key` on the right of site i we connected to.

        """
        if j <= i:
            raise ValueError('j <= i not allowed')
        keyL = keyR = key
        for k in range(j - 1, i, -1):
            if (j - k) % self.L == 0:
                # necessary to extend key because keyR is already in use at this bond
                keyL = keyR + (k, opname, opname)
            k = k % self.L
            if not self.has_edge(k, keyL, keyR):
                self.add(k, keyL, keyR, opname, 1.0, check_op=check_op, skip_existing=skip_existing)
            keyR = keyL
        return keyR

    def add_missing_IdL_IdR(self, insert_all_id=True):
        """Add missing identity ('Id') edges connecting ``'IdL'->'IdL' and ``'IdR'->'IdR'``.

        This function should be called *after* all other operators have been inserted.

        Parameters
        ----------
        insert_all_id : bool
            If ``True``, insert 'Id' edges on *all* bonds.
            If ``False`` and boundary conditions are finite, only insert
            ``'IdL'->'IdL'`` to the left of the rightmost existing 'IdL' and
            ``'IdR'->'IdR'`` to the right of the leftmost existing 'IdR'.
            The latter avoid "dead ends" in the MPO, but some functions (like `make_WI`) expect
            'IdL'/'IdR' to exist on all bonds.

        """
        if self.bc == 'infinite' or insert_all_id:
            max_IdL = self.L  # add identities for all sites
            min_IdR = 0
        else:
            max_IdL = max([0] + [i for i, s in enumerate(self.states[:-1]) if 'IdL' in s])
            min_IdR = min([self.L] + [i for i, s in enumerate(self.states[:-1]) if 'IdR' in s])
        for k in range(0, max_IdL):
            if not self.has_edge(k, 'IdL', 'IdL'):
                self.add(k, 'IdL', 'IdL', 'Id', 1.0)
        for k in range(min_IdR, self.L):
            if not self.has_edge(k, 'IdR', 'IdR'):
                self.add(k, 'IdR', 'IdR', 'Id', 1.0)
        # done

    def has_edge(self, i, keyL, keyR):
        """True if there is an edge from `keyL` on bond (i-1, i) to `keyR` on bond (i, i+1)."""
        return keyR in self.graph[i].get(keyL, [])

    def build_MPO(self, Ws_qtotal=None):
        """Build the MPO represented by the graph (`self`).

        Parameters
        ----------
        Ws_qtotal : None | (list of) charges
            The `qtotal` for each of the Ws to be generated, default (``None``) means 0 charge.
            A single qtotal holds for each site.
        unit_cell_width : int
            See :attr:`~tenpy.models.lattice.Lattice.mps_unit_cell_width`.

        Returns
        -------
        mpo : :class:`MPO`
            the MPO which self represents.

        """
        self.test_sanity()
        # pre-work: generate the grid
        self._set_ordered_states()
        grids = self._build_grids()
        IdL = [s.get('IdL', None) for s in self._ordered_states]
        IdR = [s.get('IdR', None) for s in self._ordered_states]
        legs, Ws_qtotal = self._calc_legcharges(Ws_qtotal)
        H = MPO.from_grids(
            self.sites,
            grids,
            self.bc,
            IdL,
            IdR,
            Ws_qtotal,
            legs,
            self.max_range,
            mps_unit_cell_width=self.unit_cell_width,
        )
        return H

    def __repr__(self):
        return f'<MPOGraph L={self.L:d}>'

    def __str__(self):
        """String showing the graph for debug output."""
        res = []
        for i in range(self.L):
            G = self.graph[i]
            strs = []
            for keyL in self.states[i]:
                s = [repr(keyL)]
                s.append('-' * len(s[-1]))
                D = G.get(keyL, [])
                for keyR in D:
                    s.append(repr(keyR) + ':')
                    for optuple in D[keyR]:
                        s.append('  ' + repr(optuple))
                strs.append('\n'.join(s))
            res.append(vert_join(strs, delim='|'))
            res.append('')
        # & states on last MPO bond
        res.append(vert_join([repr(keyR) for keyR in self.states[-1]], delim=' |'))
        return '\n'.join(res)

    def _set_ordered_states(self):
        """Define an ordering of the 'states' on each MPO bond.

        Set ``self._ordered_states`` to a list of dictionaries ``{state: index}``.
        """
        res = self._ordered_states = []
        for s in self.states:
            d = {}
            for i, key in enumerate(sorted(s, key=_mpo_graph_state_order)):
                d[key] = i
            res.append(d)

    def _build_grids(self):
        """Translate the graph dictionaries into grids for the `Ws`."""
        states = self._ordered_states
        assert states is not None  # make sure that _set_ordered_states was called
        grids = []
        for i in range(self.L):
            stL, stR = states[i : i + 2]
            graph = self.graph[i]  # ``{keyL: {keyR: [(opname, strength)]}}``
            grid = [None] * len(stL)
            for keyL, a in stL.items():
                row = [None] * len(stR)
                for keyR, lst in graph[keyL].items():
                    b = stR[keyR]
                    row[b] = lst
                grid[a] = row
            grids.append(grid)
        return grids

    def _calc_legcharges(self, Ws_qtotal):
        """Obtain charges for the virtual legs of the MPO.

        Should only be called after :meth:`_set_ordered_states`.

        Parameters
        ----------
        Ws_qtotal : None | (list of) charges
            The `qtotal` for each of the Ws to be generated, default (``None``) means 0 charge.
            A single qtotal holds for each site.

        Returns
        -------
        legs : list of :class:`~tenpy.linalg.charge.LegCharge`
            LegCharges to be used on each bond, `L+1` entries.
            Entry `i` contains the 'wL' leg of `W[i]`,
            entry `i+1` needs to be conjugated to be used as `wR` leg of `W[i]`.
        Ws_qtotal :  list of qtotal
            Same as argument, but parsed to a list of L charges defaulting to zeros.

        """
        L = self.L
        states = self._ordered_states
        sites = self.sites
        infinite = self.bc == 'infinite'
        chinfo = self.chinfo

        if Ws_qtotal is None:
            Ws_qtotal = [chinfo.make_valid()] * L
        else:
            Ws_qtotal = chinfo.make_valid(Ws_qtotal)
            if Ws_qtotal.ndim == 1:
                Ws_qtotal = [Ws_qtotal] * L

        charges = [[None] * len(st) for st in states]
        charges[0][states[0]['IdL']] = chinfo.make_valid(None)  # default charge = 0.

        def travel_q_LR(i, keyL):
            """Transport charges from left to right through the MPO graph.

            Inspect graph edges on site `i` starting on the left with `keyL` and add charges
            for all connections to the right.
            Originally we recursively transported charges from there, but now this is done
            iteratively to avoid the maximum recursion limit in python for large systems.
            """
            stack = []
            stack.append((i, keyL))
            while len(stack):
                i, keyL = stack.pop(-1)  # We are replacing system stack with one of our own
                l = states[i][keyL]
                site = sites[i]
                st_r = states[i + 1]
                ch_r = charges[i + 1]
                # charge rule: q_left - q_right + op_qtotal = Ws_qtotal
                qL_Wq = charges[i][l] - Ws_qtotal[i]  # q_left - Ws_qtotal
                edges = self.graph[i][keyL]
                edge_stack = []
                for keyR, ops in edges.items():
                    r = st_r[keyR]
                    qR = ch_r[r]
                    if qR is None:
                        op_qtotal = site.get_op(ops[0][0]).qtotal
                        ch_r[r] = qL_Wq + op_qtotal  # solve chargerule for q_right
                        if infinite or i + 1 < L:
                            edge_stack.append(((i + 1) % L, keyR))
                        if infinite and i + 1 == L:  # copy and shift to the left leg
                            charges[0][r] = self.shift_charges_unit_cells(ch_r[r], -1)
                stack = edge_stack + stack

        travel_q_LR(0, 'IdL')

        # now we can still have unknown edges in the case of "dead ends" in the MPO graph.

        def travel_q_RL(i, keyL):
            """Transport charges from the right to left through the MPO graph.

            Inspect graph edges on site `i` starting on the left with 'keyL', where
            the charge needs to be determined. If one of them has a charge defined on the right,
            use it to determine the charge for keyL and return True.
            If none of them has the charge defined, return False.
            """
            l = states[i][keyL]
            site = sites[i]
            st_r = states[i + 1]
            ch_r = charges[i + 1]
            # charge rule: q_left - q_right + op_qtotal = Ws_qtotal
            Wq = Ws_qtotal[i]  # q_left - Ws_qtotal
            edges = self.graph[i][keyL]
            for keyR, ops in edges.items():
                r = st_r[keyR]
                qR = ch_r[r]
                if qR is not None:
                    op_qtotal = site.get_op(ops[0][0]).qtotal
                    charges[i][l] = Wq + qR - op_qtotal  # solve chargerule for q_left
                    break
            else:  # no break
                return False
            return True

        if not infinite and any([ch is None for ch in charges[-1]]):
            raise ValueError("can't determine all charges on the very right leg of the MPO!")

        max_checks = 1000  # Hard-coded since for a properly set-up MPO graph, this loop will
        # terminate after one iteration
        for _ in range(max_checks):
            repeat = False
            for i in reversed(range(L)):
                ch = charges[i]
                for keyL, l in states[i].items():
                    if ch[l] is None:
                        if not travel_q_RL(i, keyL):
                            repeat = True  # couldn't find it out.
            if not repeat:
                break
        else:  # no break
            raise ValueError("MPOGraph with dead ends: can't determine charges")
        # have all charges determined
        # convert to LegCharges
        legs = []
        for ch in charges:
            ch = chinfo.make_valid(ch)
            leg = npc.LegCharge.from_qflat(chinfo, ch, qconj=+1)
            legs.append(leg)
        return legs, Ws_qtotal


class MPOEnvironment(BaseEnvironment):
    """Stores partial contractions of :math:`<bra|H|ket>` for an MPO `H`.

    The network for a contraction :math:`<bra|H|ket>` of an MPO `H` between two MPS looks like::

        |     .------>-M[0]-->-M[1]-->-M[2]-->- ...  ->--.
        |     |        |       |       |                 |
        |     |        ^       ^       ^                 |
        |     |        |       |       |                 |
        |     LP[0] ->-W[0]-->-W[1]-->-W[2]-->- ...  ->- RP[-1]
        |     |        |       |       |                 |
        |     |        ^       ^       ^                 |
        |     |        |       |       |                 |
        |     .------<-N[0]*-<-N[1]*-<-N[2]*-<- ...  -<--.

    We use the following label convention (where arrows indicate `qconj`)::

        |    .-->- vR           vL ->-.
        |    |                        |
        |    LP->- wR           wL ->-RP
        |    |                        |
        |    .--<- vR*         vL* -<-.

    See :class:`BaseEnvironment` for further details.

    Parameters
    ----------
    bra : :class:`~tenpy.networks.mps.MPS`
        The MPS to project on. Should be given in usual 'ket' form;
        we call `conj()` on the matrices directly.
    H : :class:`~tenpy.networks.mpo.MPO`
        The MPO sandwiched between `bra` and `ket`.
        Should have 'IdL' and 'IdR' set on the first and last bond.
    ket : :class:`~tenpy.networks.mpo.MPS`
        The MPS on which `H` acts. May be identical with `bra`.
    **init_env_data :
        Further keyword arguments with initialization data, as returned by
        :meth:`get_initialization_data`.
        See :meth:`init_first_LP_last_RP` for details on these parameters.

    Attributes
    ----------
    H : :class:`~tenpy.networks.mpo.MPO`
        The MPO sandwiched between `bra` and `ket`.

    """

    def __init__(self, bra, H, ket, cache=None, **init_env_data):
        self.H = H
        super().__init__(bra, ket, cache, **init_env_data)
        self.dtype = np.result_type(bra.dtype, ket.dtype, H.dtype)

    def init_first_LP_last_RP(
        self,
        init_LP=None,
        init_RP=None,
        age_LP=0,
        age_RP=0,
        start_env_sites=None,
        force_init_method='iter',
        gmres_options=None,
    ):
        """(Re)initialize first LP and last RP from the given data.

        If `init_LP` and `init_RP` are not given, we try to find sensible initial values.
        Dummy environments can by built with :meth:`init_LP` and :meth:`init_RP`, especially
        for **finite** MPS.

        For **infinite** MPS, we try to converge the environments with one of three methods:

        - If `start_env_sites` is given as an integer, contract that many sites into the
          environment from the given `init_LP` and `init_RP` or new trivial environments built
          with :meth:`init_LP` / :meth:`init_RP`.
        - If `start_env_sites` is None, and :attr:`bra` is :attr:`ket`,
          get `init_LP` and `init_RP` using one of two methods:

            'iter': :meth:`MPOEnvironmentBuilder.init_LP_RP_iterative`
                - Recommended for general use
            'TM': :meth:`MPOTransferMatrix.find_init_LP_RP`
                - In case 'iter' cannot be applied
                - Faster for small bond dimension (chi < 150)

        Parameters
        ----------
        init_LP, init_RP: ``None`` | :class:`~tenpy.linalg.np_conserved.Array`
            Initial very left part ``LP`` and very right part ``RP``.
            If ``None``, try to build (and converge) them as described above.
        age_LP, age_RP : int
            The number of physical sites involved into the contraction of `init_LP` and `init_RP`.
        start_env_sites : int | None
            Number of sites over which to converge the environment for infinite systems.
            See above.
        force_init_method : {None, 'iter', 'TM'}
            Force method 'TM' or 'iter' as described above for **infinite** MPS.
        gmres_options : dict
            Further optional parameters for :class:`tenpy.linalg.krylov_based.GMRES`.
            Only relevant for **infinite** MPS if method 'iter' is used to get `init_LP`/`init_RP`.

        """
        if (
            not self.finite
            and (init_LP is None or init_RP is None)
            and start_env_sites is None
            and self.bra is self.ket
        ):
            norm_err = np.linalg.norm(self.ket.norm_test())
            if norm_err > 1.0e-10:
                warnings.warn(
                    'call psi.canonical_form() to regenerate MPO environments from psi'
                    f' with current norm error {norm_err:.2e}'
                )
                self.ket.canonical_form()

            # select method for initialization
            if not self.chinfo.trivial_shift:
                if force_init_method is None:
                    force_init_method = 'TM'
                if force_init_method == 'iter':
                    msg = (
                        'force_init_method="iter" is not yet supported with shift symmetry. '
                        'use force_init_method="TM" in the meantime.'
                    )
                    warnings.warn(msg, stacklevel=4)
                    force_init_method = 'TM'
            if force_init_method is None:
                if (max(self.ket.chi) <= 150) or (not _mpo_check_for_iter_LP_RP_infinite(self.H)):
                    force_init_method = 'TM'
                else:
                    force_init_method = 'iter'

            # call that method
            if force_init_method == 'iter':
                _env_init = MPOEnvironmentBuilder(self.H, self.ket)
                env_data, _ = _env_init.init_LP_RP_iterative('both', gmres_options=gmres_options)
            elif force_init_method == 'TM':
                env_data = MPOTransferMatrix.find_init_LP_RP(self.H, self.ket, 0, self.L - 1)
            else:
                raise ValueError(f'Invalid {force_init_method=}')

            init_LP = env_data['init_LP']
            init_RP = env_data['init_RP']
            start_env_sites = 0
        if start_env_sites is None:
            start_env_sites = 0 if self.finite else self.L
        if self.finite and start_env_sites != 0:
            warnings.warn('setting `start_env_sites` to 0 for finite MPS')
            start_env_sites = 0
        init_LP, init_RP = self._check_compatible_legs(init_LP, init_RP, start_env_sites)
        if self.ket.bc == 'segment' and (init_LP is None or init_RP is None):
            raise ValueError('Environments with segment b.c. need explicit environments!')
        super().init_first_LP_last_RP(init_LP, init_RP, age_LP, age_RP, start_env_sites)

    def _check_compatible_legs(self, init_LP, init_RP, start_env_sites):
        if init_LP is not None:
            try:
                i = -start_env_sites
                init_LP.get_leg('wR').test_contractible(self.H.get_W(i).get_leg('wL'))
            except ValueError:
                warnings.warn('dropping `init_LP` with incompatible MPO legs')
                init_LP = None
        if init_RP is not None:
            try:
                j = self.L - 1 + start_env_sites
                init_RP.get_leg('wL').test_contractible(self.H.get_W(j).get_leg('wR'))
            except ValueError:
                warnings.warn('dropping `init_RP` with incompatible MPO legs')
                init_RP = None
        return super()._check_compatible_legs(init_LP, init_RP, start_env_sites)

    def test_sanity(self):
        """Sanity check, raises ValueErrors, if something is wrong."""
        super().test_sanity()
        assert self.bra.finite == self.ket.finite == self.H.finite == self.finite
        # check that the physical legs are contractable
        for b_s, H_s, k_s in zip(self.bra.sites, self.H.sites, self.ket.sites):
            b_s.leg.test_equal(k_s.leg)
            b_s.leg.test_equal(H_s.leg)
        assert any(key in self.cache for key in self._LP_keys)
        assert any(key in self.cache for key in self._RP_keys)

    def init_LP(self, i, start_env_sites=0):
        r"""Build an initial left part ``LP``.

        For `start_env_sites` > 0, assume that `bra` is the same as `ket`
        and in canonical form, and that H is a Hamiltonian with the following block-form
        (up to a permutation of MPO indices; this is the case for any model defined in TeNPy),

        .. math ::

            W = \begin{pmatrix} 1 & C & D  \\
                                0 & A & B  \\
                                0 & 0 & 1  \end{pmatrix}

        Given that, we can converge the environment even in the thermodynamic limit:
        ``LP[IdR, :, :]`` contains the extensive energy contribution for the left part of
        the Hamiltonian, which we can ignore (since we only look at relative energies)
        ``LP[IdL, :, :] = eye(:, :)`` is just the MPS environment.

        For the remaining part we need to converge $C + CA + CAA + CAAA + ... $ sandwiched
        between the MPS. If H has finite range, `A` is nil-potent, and it is sufficient
        to contract the environment a few times from the left.

        For infinite/long range interactions it just limits the number of iterations.
        In this case, the exact environment can still be obtained  by solving the
        geometric series.

        .. note ::

            For `start_env_sites` > 0, this function provides a fast approximation of
            the environment. In general, we recommend using
            :meth:`MPOEnvironmentBuilder.init_LP_RP_iterative` which solves the geometric
            series exactly.



        Parameters
        ----------
        i : int
            Build ``LP`` left of site `i`.
        start_env_sites : int
            How many sites to contract to converge the `init_LP`; the initial `age_LP`.

        Returns
        -------
        init_LP : :class:`~tenpy.linalg.np_conserved.Array`
            Environment left of site `i` with labels ``'vR*', 'wR', 'vR'``.

        """
        i0 = i - start_env_sites
        IdL = self.H.get_IdL(i0)
        if IdL is None:
            raise RuntimeError(f'Need to set IdL at i0={i0} for the MPO self.H')
        init_LP = super().init_LP(i0, 0)
        leg_mpo = self.H.get_W(i0).get_leg('wL').conj()
        init_LP = init_LP.add_leg(leg_mpo, IdL, axis=1, label='wR')
        for j in range(i0, i):
            init_LP = self._contract_LP(j, init_LP)
        return init_LP

    def init_RP(self, i, start_env_sites=0):
        """Build initial right part ``RP`` for an MPS/MPOEnvironment.

        Parameters
        ----------
        i : int
            Build ``RP`` right of site `i`.
        start_env_sites : int
            How many sites to contract to converge the `init_RP`; the initial `age_RP`.

        Returns
        -------
        init_RP : :class:`~tenpy.linalg.np_conserved.Array`
            Environment right of site `i` with labels ``'vL*', 'wL', 'vL'``.

        """
        i0 = i + start_env_sites
        IdR = self.H.get_IdR(i0)
        if IdR is None:
            raise RuntimeError(f'Need to set IdR at i0={i0} for the MPO self.H')
        init_RP = super().init_RP(i0, 0)
        leg_mpo = self.H.get_W(i0).get_leg('wR').conj()
        init_RP = init_RP.add_leg(leg_mpo, IdR, axis=1, label='wL')
        for j in range(i0, i, -1):
            init_RP = self._contract_RP(j, init_RP)
        return init_RP

    def get_LP(self, i, store=True):
        """Calculate LP at given site from nearest available one (including `i`).

        The returned ``LP_i`` corresponds to the following contraction,
        where the M's and the N's are in the 'A' form::

            |     .-------M[0]--- ... --M[i-1]--->-   'vR'
            |     |       |             |
            |     LP[0]---W[0]--- ... --W[i-1]--->-   'wR'
            |     |       |             |
            |     .-------N[0]*-- ... --N[i-1]*--<-   'vR*'


        Parameters
        ----------
        i : int
            The returned `LP` will contain the contraction *strictly* left of site `i`.
        store : bool
            Whether to store the calculated `LP` in `self` (``True``) or discard them (``False``).

        Returns
        -------
        LP_i : :class:`~tenpy.linalg.np_conserved.Array`
            Contraction of everything left of site `i`,
            with labels ``'vR*', 'wR', 'vR'`` for `bra`, `H`, `ket`.

        """
        # actually same as MPSEnvironment, just updated the labels in the doc string.
        return super().get_LP(i, store)

    def get_RP(self, i, store=True):
        """Calculate RP at given site from nearest available one (including `i`).

        The returned ``RP_i`` corresponds to the following contraction,
        where the M's and the N's are in the 'B' form::

            |     'vL'  ->---M[i+1]-- ... --M[L-1]----.
            |                |              |         |
            |     'wL'  ->---W[i+1]-- ... --W[L-1]----RP[-1]
            |                |              |         |
            |     'vL*' -<---N[i+1]*- ... --N[L-1]*---.

        Parameters
        ----------
        i : int
            The returned `RP` will contain the contraction *strictly* right of site `i`.
        store : bool
            Whether to store the calculated `RP` in `self` (``True``) or discard them (``False``).

        Returns
        -------
        RP_i : :class:`~tenpy.linalg.np_conserved.Array`
            Contraction of everything right of site `i`,
            with labels ``'vL*', 'wL', 'vL'`` for `bra`, `H`, `ket`.

        """
        # actually same as MPSEnvironment, just updated the labels in the doc string.
        return super().get_RP(i, store)

    def full_contraction(self, i0):
        """Calculate the energy by a full contraction of the network.

        The full contraction of the environments gives the value
        ``<bra|H|ket> `` ignoring the :attr:`~tenpy.networks.mps.MPS.norm` of the `bra` and `ket`,
        i.e. the total energy (even if bra and ket are not normalized).
        For this purpose, this function contracts ``get_LP(i0+1, store=False)`` and
        ``get_RP(i0, store=False)`` with appropriate singular values in between.

        Parameters
        ----------
        i0 : int
            Site index.

        """
        # same as MPSEnvironment.full_contraction, but also contract 'wL' with 'wR'
        LP, RP = self._full_contraction_LP_RP(i0)
        res = npc.inner(LP, RP, axes=[['vR*', 'wR', 'vR'], ['vL*', 'wL', 'vL']], do_conj=False)
        if self.H.explicit_plus_hc:
            res = res + np.conj(res)
        return res

    def _contract_LP(self, i, LP):
        """Contract LP with the tensors on site `i` to form ``self._LP[i+1]``"""
        # same as MPSEnvironment._contract_LP, but also contract with `H.get_W(i)`
        LP = npc.tensordot(LP, self.ket.get_B(i, form='A'), axes=('vR', 'vL'))
        LP = npc.tensordot(self.H.get_W(i), LP, axes=(['p*', 'wL'], ['p', 'wR']))
        axes = (self.bra._get_p_label('*') + ['vL*'], self.ket._p_label + ['vR*'])
        # for a usual MPS, axes = (['p*', 'vL*'], ['p', 'vR*'])
        LP = npc.tensordot(self.bra.get_B(i, form='A').conj(), LP, axes=axes)
        return LP  # labels 'vR*', 'wR', 'vR'

    def _contract_RP(self, i, RP):
        """Contract RP with the tensors on site `i` to form ``self._RP[i-1]``"""
        # same as MPSEnvironment._contract_RP, but also contract with `H.get_W(i)`
        RP = npc.tensordot(self.ket.get_B(i, form='B'), RP, axes=('vR', 'vL'))
        RP = npc.tensordot(RP, self.H.get_W(i), axes=(['p', 'wL'], ['p*', 'wR']))
        axes = (self.ket._p_label + ['vL*'], self.ket._get_p_label('*') + ['vR*'])
        # for a usual MPS, axes = (['p', 'vL*'], ['p*', 'vR*'])
        RP = npc.tensordot(RP, self.bra.get_B(i, form='B').conj(), axes=axes)
        return RP  # labels 'vL', 'wL', 'vL*'

    def _contract_LHeff(self, i, label_p='p0', pipe=None):
        LP = self.get_LP(i)
        p, ps = label_p, label_p + '*'
        W = self.H.get_W(i).replace_labels(['p', 'p*'], [p, ps])
        LHeff = npc.tensordot(LP, W, axes=['wR', 'wL'])
        if pipe is None:
            pipe = LHeff.make_pipe(['vR*', p], qconj=+1)

        LHeff = LHeff.combine_legs([['vR*', p], ['vR', ps]], pipes=[pipe, pipe.conj()], new_axes=[0, 2])
        return LHeff

    def _contract_RHeff(self, i, label_p='p1', pipe=None):
        RP = self.get_RP(i)
        p, ps = label_p, label_p + '*'
        W = self.H.get_W(i).replace_labels(['p', 'p*'], [p, ps])
        RHeff = npc.tensordot(W, RP, axes=['wR', 'wL'])
        if pipe is None:
            pipe = RHeff.make_pipe([p, 'vL*'], qconj=-1)
        RHeff = RHeff.combine_legs([[p, 'vL*'], [ps, 'vL']], pipes=[pipe, pipe.conj()], new_axes=[2, 1])
        return RHeff


class MPOEnvironmentBuilder:
    r"""Construct boundary environments for periodic MPOEnvironments.

    This class implement the construction scheme from :cite:`phien2012` to construct
    `LP[0]` and `RP[self.L-1]` for a periodic :class:`MPOEnvironment`::

        |             - - > - - - - - 'vR*'
        |            |          |
        |   LP[0] = E[0] ->- T_H**n - 'wR' (index `j`)
        |            |          |
        |             - - > - - - - - 'vR'

    where T_H has the structure of a corresponding :class:`MPOTransferMatrix`
    and the limit :math:`n → \infty` has to be taken.

    The above equation does generally not converge to a fixpoint due to the
    extensive energy contribution of the Hamiltonian.

    However, for an MPO `H` that is upper triangular up to permutations,
    `LP[0]` can be constructed iteratively in index `j`::

        E[n+1][:,j,:] = \sum_{i<=j} E[n][:,i,:]T_H[:,i,:][:,j,:]

    with::

        E[n][:,j=IdL,:]=Id

    The last environment E[n][:,j=IdR,:] requires solving the geometric series::

        E[n+1][:,j=D-1,:] = \sum_{k=0,...,n-1} T_H[:,j,:][:,j,:]**k (C)

    which is singular due to the identity and density matrix as eigenvector
    pair with eigenvalue 1. To avoid this,::

        E[n+1][:,j=D-1,:] = c0_j + epsilon * n * Id

    can be decomposed into a constant term and an extensive contribution. The
    latter captures the energy per site of the environment.

    Here, we also generalize the scheme to higher powers of MPOs, where
    the environment more generally is decomposed into terms proportional
    to different powers of `n`::

        LP[0] = e_0 * n**0 * LP[0][0] + e_1 * n**1 LP[0][1] + ...

    The largest needed `n` is given by the number of identities on the diagonal
    of the MPO.

    .. todo ::

        - Currently, we only allow simple loops. E.g. a double loop
            Outer[j] -> A1 -> Outer[j] AND Outer[j] -> A2 -> Outer[j] is not allowed.
            This can in principle be adjusted by grouping nodes of the graph.
        - Currently, we assume identities along the loops. In principle, we can allow
            arbitrary operators if we compute the dominant eigenvectors explicitly.
    """

    def __init__(self, H, psi):
        self.H = H
        self.ket = psi
        self.L = psi.L
        self.dtype = np.result_type(self.ket.dtype, self.H.dtype)
        self._p_label = self.ket._p_label
        self.test_sanity()
        # MPS tensors as needed by transfer matrices during init_LP_RP
        self._Ms = None
        self._Ns = None

    def test_sanity(self):
        """Sanity check, raises ValueErrors, if something is wrong."""
        assert self.ket.bc == self.H.bc == 'infinite'
        # check that the physical legs are contractable
        assert self.L == self.ket.L == self.H.L
        for H_s, k_s in zip(self.H.sites, self.ket.sites):
            k_s.leg.test_equal(H_s.leg)

    def _contract_cL(self, cL, i, op):
        """Partial contraction cL=(A-op-A*)="""
        cL = npc.tensordot(self.ket.get_B(i, form='A'), cL, axes=('vL', 'vR'))
        cL = npc.tensordot(cL, op, axes=[self.ket._p_label, 'p*'])
        axes = (['p', 'vR*'], self.ket._get_p_label('*') + ['vL*'])
        cL = npc.tensordot(cL, self.ket.get_B(i, form='A').conj(), axes=axes)
        return cL

    def _contract_cR(self, cR, i, op):
        """Contract =(B-op-B*)=cR"""
        cR = npc.tensordot(self.ket.get_B(i, form='B'), cR, axes=('vR', 'vL'))
        cR = npc.tensordot(cR, op, axes=[self.ket._p_label, 'p*'])
        axes = (['p', 'vL*'], self.ket._get_p_label('*') + ['vR*'])
        cR = npc.tensordot(cR, self.ket.get_B(i, form='B').conj(), axes=axes)
        return cR

    def _determine_cycles(self, tol=1e-12):
        """Determine the cycles of `self.H` with norm 1

        .. note ::
            - Cycles are only allowed to contain identities with positive prefactor at the moment.
            - Can be generalized to allow arbitrary operators. Requires adjusting self._c0_rho
        """
        ones = []
        for j_outer, loop in self.H._cycles.items():
            norm = 1.0
            for j in range(self.H.L):
                op = self.H._graph[j][(loop[j], loop[j + 1])]  # (i,j)
                factor = npc.norm(op, ord=1) / op.shape[0]
                if norm * factor < tol:
                    norm = 0.0
                    break  # norm close to zero
                # op == factor*id with factor>0
                is_id = npc.norm(op - factor * npc.diag(1.0, op.get_leg('p')), ord=1) < tol
                if not is_id:
                    raise ValueError(f'W[{j}][{loop[j]},{loop[j + 1]}] != a*Id with a>0')
                norm *= factor
            if norm >= 1.0 + tol:
                raise ValueError(f'self.H contains cycle with norm larger than one at outer index {loop[0]}')
            if abs(norm - 1.0) < 1e-13:
                ones.append(j_outer)
        return ones

    def _left_grid(self, remove=[]):
        r"""Construct a grid representing partial contractions of `self` as graph.

        We can view contractions of an :class:`MPOEnvironment` as graph with the same
        structure as the underlying MPOGraph: For example,
        `LP[i+1]['wR'=k] = \sum_j LP[i]['wR'=j]*(B*[i]*W[j,k]*B[i])`
        corresponds to summing the links W[j,k] between layer (i,i+1)
        given by the MPS sites.

        Contracting the MPOEnvironment in this way is usually slower than the naive
        way, but is beneficial for :meth:`init_LP_RP_iterative`.

        We represent a grid as:
            list of {list of { [None | :class:`~tenpy.linalg.np_conserved.Array`, set of int] }}
            - `grid[j_site][j_virtual][0]` contains the partial contractions (including site j_site)
            - `grid[j_site][j_virtual][1]` contains the remaining ingoing indices not yet summed

        Parameters
        ----------
        remove : list of int
            Remove edges from left nodes `LP[0]['wR'=j]=0 for j in remove` that are zero.

        Returns
        -------
        grid : list of {list of { [None | :class:`~tenpy.linalg.np_conserved.Array`, set of int] }}
            As described above

        """
        grid = []
        for chi in self.H.chi[1:]:
            # cPartial, ingoing inds - cPartial initialized as None
            layer = [[None, set()] for _ in range(chi)]
            grid.append(layer)
        for j_site, layer in enumerate(self.H._graph):
            for i, j in layer:
                grid[j_site][j][1].add(i)
        # remove connections of left nodes that are 0
        if len(remove) > 0:
            zero_nodes = remove
            for j_site in range(self.L):
                empty_nodes = []
                for iL in zero_nodes:
                    conns = [j for i, j in self.H._graph[j_site] if i == iL]
                    for j in conns:
                        grid[j_site][j][1].remove(iL)
                        if not grid[j_site][j][1]:  # ingoing indices sum to zero
                            empty_nodes.append(j)
                zero_nodes = empty_nodes
        return grid

    def _right_grid(self, remove=[]):
        """Same as left grid, but starting from `RP[self.L-1]`.

        Note: Layers are always indexed from the left, meaning
            right_grid[j] <-> `RP[j-1]`
        """
        grid = []
        for chi in self.H.chi[:-1]:
            layer = [[None, set()] for _ in range(chi)]
            grid.append(layer)
        for j_site, layer in enumerate(self.H._graph):
            for i, j in layer:
                grid[j_site][i][1].add(j)
        if len(remove) > 0:
            zero_nodes = remove
            for j_site in range(self.L - 1, -1, -1):
                empty_nodes = []
                for jR in zero_nodes:
                    conns = [i for i, j in self.H._graph[j_site] if j == jR]
                    for i in conns:
                        grid[j_site][i][1].remove(jR)
                        if not grid[j_site][i][1]:  # all ingoing indices sum to zero
                            empty_nodes.append(i)
                zero_nodes = empty_nodes
        return grid

    def _contract_left_grid(self, grid, c0_outer, j_outer):
        """Helper function.

        Carry out all possible contractions starting from the initial node `c0_outer` at the outer
        virtual index `j_outer`
        """
        ready_nodes = [[c0_outer, j_outer]]
        for j_site in range(self.L):
            finished_nodes = []
            for cL, iL in ready_nodes:
                conns = [j for i, j in self.H._graph[j_site] if i == iL]
                for j in conns:
                    res = self._contract_cL(cL, j_site, self.H._graph[j_site][(iL, j)])
                    if grid[j_site][j][0] is None:
                        grid[j_site][j][0] = res
                    else:
                        grid[j_site][j][0] += res
                    grid[j_site][j][1].remove(iL)
                    if not grid[j_site][j][1]:  # all ingoing indices summed up
                        finished_nodes.append((grid[j_site][j][0], j))
                # delete cL, not needed anymore & saves storage
                if j_site != 0:
                    # double check that set with ingoing elements for ready node is empty
                    assert not grid[j_site - 1][iL][1]
                    del grid[j_site - 1][iL][0]
            ready_nodes = finished_nodes

    def _contract_right_grid(self, grid, c0_outer, j_outer):
        ready_nodes = [[c0_outer, j_outer]]
        for j_site in range(self.L - 1, -1, -1):
            finished_nodes = []
            for cR, jR in ready_nodes:
                conns = [i for i, j in self.H._graph[j_site] if j == jR]
                for i in conns:
                    res = self._contract_cR(cR, j_site, self.H._graph[j_site][(i, jR)])
                    if grid[j_site][i][0] is None:
                        grid[j_site][i][0] = res
                    else:
                        grid[j_site][i][0] += res
                    grid[j_site][i][1].remove(jR)
                    if not grid[j_site][i][1]:  # all ingoing indices summed up
                        finished_nodes.append((grid[j_site][i][0], i))
                # delete cR, not needed anymore & saves storage
                if j_site != self.L - 1:
                    # double check that set with ingoing elements for ready node is empty
                    assert not grid[j_site + 1][jR][1]
                    del grid[j_site + 1][jR][0]  # won't be accessed anymore
            ready_nodes = finished_nodes

    def init_LP_RP_iterative(self, which='both', calc_E=False, tol_c0=None, gmres_options=None, tol_id=1e-12):
        """Construct boundary environments for periodic MPO environments.

            See class docstring for an explanation.

        Parameters
        ----------
        which : {'LP', 'RP', 'both'}
            Specifies which environments to compute.
        tol_c0 : float | None
            Tolerance for explicitly computing the dominant left and right eigenvectors
            of the :class:`MPSTransferMatrix` associated with :attr:`self.ket`, if numerical errors
            affect the MPS canonical form. Ignored if `None`. In this case uses `c0=Id`.
        calc_E : bool
            Whether to return the energy. Only permitted when the expectation value scales
            at most linearly with system size. For higher-order scaling,
            expectation values must be computed via explicit contractions.
        gmres_options : dict
            Further optional parameters passed to :class:`tenpy.linalg.krylov_based.GMRES`.
        tol_id : float
            Cycles with smaller norm are discarded.

        Returns
        -------
        init_env_data : dict
            Dictionary with `init_LP` and `init_RP` in the same format as
            :meth:`MPOTransferMatrix.find_init_LP_RP`.
        envs : dict of list
            All environments grouped by powers of `n`.
            envs['init_LP'][j]=`LP[0][j]` and envs['init_RP'][j]=`RP[self.L-1][j]`
        E : float
            Energy per site, only returned if `calc_E` is True.

        """
        if not self.H.chinfo.trivial_shift:
            raise NotImplementedError(
                'Iterative LP/RP initialization is not yet supported for shift-symmetry with infinite systems.'
            )
        if _mpo_check_for_iter_LP_RP_infinite(self.H) is False:
            raise ValueError('Iterative environment initialization failed: Hamiltonian cannot be ordered.')
        assert which == 'LP' or 'RP' or 'both', f'Invalid environment type "{which}"'
        ones = self._determine_cycles()
        n_terms = len(ones)
        # gmres defaults, set N_min=0 for states close to product states
        if gmres_options is None:
            gmres_options = {'N_min': 0, 'res': 1e-11}
        else:
            gmres_options['N_min'] = gmres_options.get('N_min', 0)
            gmres_options['res'] = gmres_options.get('res', 1e-11)
        legs_labels = {
            'init_LP': (
                [
                    self.H.get_W(0).get_leg('wL').conj(),
                    self.ket.get_B(0).get_leg('vL').conj(),
                    self.ket.get_B(0).get_leg('vL'),
                ],
                ['wR', 'vR', 'vR*'],
            ),
            'init_RP': (
                [
                    self.H.get_W(self.L - 1).get_leg('wR').conj(),
                    self.ket.get_B(self.L - 1).get_leg('vR').conj(),
                    self.ket.get_B(self.L - 1).get_leg('vR'),
                ],
                ['wL', 'vL', 'vL*'],
            ),
        }
        envs = {}
        # only for MPOEnvironments of extensive Hamiltonians, i.e. len(n_terms)==2
        Es = [1.0, 1.0]  # per unit cell

        # NOTE: main work starts here
        for name in ['init_LP', 'init_RP'] if which == 'both' else ['init_' + which]:
            # Ms, Ns as needed for TransferMatrix
            form = 'A' if name == 'init_LP' else 'B'
            self._Ms = [self.ket.get_B(i, form=form) for i in range(self.L)]
            self._Ns = [self.ket.get_B(i, form=form).conj() for i in range(self.L)]
            envs[name] = [
                npc.Array(legs_labels[name][0], dtype=self.dtype, labels=legs_labels[name][1]) for _ in range(n_terms)
            ]
            grids = self._make_grids(name, ones)
            last_site = self.L - 1 if name == 'init_LP' else 0

            # dominant eigenvector c0 = c0*TW_00, analytical prediction c0=Id
            # Unstable w.r.t. canonical form of the MPS <- fixed by checking explicitly
            # normalized via npc.inner(c0,rho) = 1 with rho = TW_00*rho the associated density
            c0_base, rho = self._c0_rho(name, legs_labels, tol_c0)

            m = 0
            for j_outer in self.H._outer_permutation if name == 'init_LP' else reversed(self.H._outer_permutation):
                cs = []
                eps_temp = []
                # NOTE: contributions ~ c0
                if j_outer in ones:
                    c0 = c0_base.copy()
                    if m != 0:  # compute next epsilon
                        Ctot = self._ctot_loop(grids[m - 1], self.H._cycles[j_outer], name)
                        next_eps = np.real(npc.inner(Ctot, rho) / m)
                        if m == 1:
                            index = 0 if name == 'init_LP' else 1
                            Es[index] = next_eps  # only meaningful if len(ones)==2
                        eps_temp.insert(0, next_eps)
                        c0 *= next_eps
                    envs[name][m][j_outer] = c0
                    cs.append(c0)
                    if m < len(grids):
                        self._contract_grid(grids[m], c0, j_outer, name)
                    # compute c0 contributions for lower envs and adjust epsilons
                    for gamma in range(m - 1, 0, -1):
                        Ctot_gamma = self._ctot_loop(grids[gamma - 1], self.H._cycles[j_outer], name)
                        eps_gamma = np.real(npc.inner(Ctot_gamma, rho))
                        for j_eps, alpha in enumerate(range(gamma + 1, m + 1)):
                            # print("eps_temp j_eps:",eps_temp[j_eps],comb(alpha,gamma-1))
                            eps_gamma -= eps_temp[j_eps] * comb(alpha, gamma - 1)
                        eps_gamma /= gamma
                        eps_temp.insert(0, eps_gamma)
                        # add to environment afterwards
                    m += 1
                # NOTE: contributions orthogonal to c0
                offset = 1 if j_outer in ones else 0
                for gamma in range(m - 1 - offset, -1, -1):
                    if j_outer in self.H._cycles:
                        Ctot = self._ctot_loop(grids[gamma], self.H._cycles[j_outer], name)
                    else:
                        Ctot = grids[gamma][last_site][j_outer][0]
                        if Ctot is None:  # nothing to sum up
                            Ctot = npc.zeros(
                                legs_labels[name][0][1:], dtype=self.dtype, labels=legs_labels[name][1][1:]
                            )
                    for j_cs, alpha in enumerate(range(gamma + 1, m)):
                        Ctot -= comb(alpha, gamma) * cs[j_cs]
                    if j_outer in self.H._cycles:
                        res = self._solve_cj(self.H._cycles[j_outer], name, Ctot, offset, gmres_options)
                        if offset == 1 and gamma != 0:
                            res += (eps_temp[gamma - 1] / eps_temp[-1]) * cs[-1]
                        cs.insert(0, res)
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            # ignore complex warning when self.dtype=float since GMRES internally uses complex
                            envs[name][gamma][j_outer] = res
                        self._contract_grid(grids[gamma], res, j_outer, name)
                    else:
                        cs.insert(0, Ctot)
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            # ignore complex warning when self.dtype=float since GMRES internally uses complex
                            envs[name][gamma][j_outer] = Ctot
                        self._contract_grid(grids[gamma], Ctot, j_outer, name)
        if calc_E and n_terms == 2 and which == 'both':
            return {k: envs[k][0] for k in envs.keys()}, envs, [E / self.L for E in Es]
        return {k: envs[k][0] for k in envs.keys()}, envs

    def _make_grids(self, name, ones):
        """Initialize grids for `self.init_LP_RP_iterative()`"""
        js_loops = sorted([self.H._outer_permutation.index(j) for j in ones])
        if name == 'init_LP':
            gs = [self._left_grid(self.H._outer_permutation[:j0]) for j0 in js_loops[:-1]]
            # last norm 1 cycle not on last index
            if self.H._outer_permutation[-1] not in ones:
                gs += [self._left_grid(self.H._outer_permutation[: js_loops[-1]])]
        else:
            gs = [self._right_grid(self.H._outer_permutation[j0 + 1 :]) for j0 in reversed(js_loops[1:])]
            if self.H._outer_permutation[0] not in ones:
                gs += [self._right_grid(self.H._outer_permutation[js_loops[0] + 1 :])]
        return gs

    def _contract_grid(self, grid, c0_outer, j_outer, name):
        """For `self.init_LP_RP_iterative()`"""
        if name == 'init_LP':
            self._contract_left_grid(grid, c0_outer, j_outer)
        else:
            self._contract_right_grid(grid, c0_outer, j_outer)

    def _ctot_loop(self, grid, cycle, name):
        """For `self.init_LP_RP_iterative()`

        Compute Ctot for indices with cycles
        """
        if name == 'init_LP':
            return self._ctot_loop_left(grid, cycle)
        else:
            return self._ctot_loop_right(grid, cycle)

    def _ctot_loop_left(self, grid, cycle):
        j_start = 0
        c_loop = None
        for j_site in range(self.L):
            # double check, should not trigger
            assert len(grid[j_site][cycle[j_site + 1]][1]) == 1 and cycle[j_site] in grid[j_site][cycle[j_site + 1]][1]
        for j_site in range(self.L):
            if grid[j_site][cycle[j_site + 1]][0] is not None:
                c_loop = grid[j_site][cycle[j_site + 1]][0]
                j_start = j_site
                break
        # unlikely but not accounted for beforehand
        assert c_loop is not None, 'Hamiltonian contains cycle that does not connect to other indices'
        # do contractions
        for j_site in range(j_start + 1, self.L):
            c_loop = self._contract_cL(c_loop, j_site, self.H._graph[j_site][(cycle[j_site], cycle[j_site + 1])])
            if grid[j_site][cycle[j_site + 1]][0] is not None:
                c_loop += grid[j_site][cycle[j_site + 1]][0]
        return c_loop

    def _ctot_loop_right(self, grid, cycle):
        j_start = self.L - 1
        c_loop = None
        for j_site in range(self.L - 1, -1, -1):
            # double check, should not trigger
            assert len(grid[j_site][cycle[j_site]][1]) == 1 and cycle[j_site + 1] in grid[j_site][cycle[j_site]][1]
        for j_site in range(self.L - 1, -1, -1):
            if grid[j_site][cycle[j_site]][0] is not None:
                c_loop = grid[j_site][cycle[j_site]][0]
                j_start = j_site
                break
        assert c_loop is not None, 'Hamiltonian contains cycle that does not connect to other indices'
        # do contractions
        for j_site in range(j_start - 1, -1, -1):
            c_loop = self._contract_cR(c_loop, j_site, self.H._graph[j_site][(cycle[j_site], cycle[j_site + 1])])
            if grid[j_site][cycle[j_site]][0] is not None:
                c_loop += grid[j_site][cycle[j_site]][0]
        return c_loop

    def _c0_rho(self, name, legs_labels, tol_c0):
        """For `self.init_LP_RP_iterative()`

        Determine dominant left and right eigenvectors of the `MPSTransferMatrix`
        associated with `self.ket`.
        """
        # Identity
        c0 = npc.diag(1.0, legs_labels[name][0][1], dtype=self.dtype, labels=legs_labels[name][1][1:])
        if tol_c0 is None:  # ignore canonical form errors
            S = self.ket.get_SR(self.L - 1) if name == 'init_LP' else self.ket.get_SL(0)
            if isinstance(S, npc.Array):
                rho = npc.tensordot(S, S.conj(), axes=['vR', 'vR*'] if name == 'init_LP' else ['vL', 'vL*'])
                rho.iset_leg_labels(legs_labels[name][1][1:])
            else:
                rho = npc.diag(S**2, legs_labels[name][0][1].conj(), labels=legs_labels[name][1][-1:-3:-1])
            return c0, rho
        # Compute dominant eigenvector pair if needed
        _TM = TransferMatrix.from_Ns_Ms(
            self._Ns,
            self._Ms,
            transpose=True if name == 'init_LP' else False,
            charge_sector=None,
            p_label=self._p_label,
            conjugate_Ns=False,
        )
        if npc.norm(_TM.matvec(c0) - c0) < tol_c0:
            S = self.ket.get_SR(self.L - 1) ** 2 if name == 'init_LP' else self.ket.get_SL(0) ** 2
            if isinstance(S, npc.Array):
                rho = npc.tensordot(S, S.conj(), axes=['vR', 'vR*'] if name == 'init_LP' else ['vL', 'vL*'])
                rho.iset_leg_labels(legs_labels[name][1][1:])
            else:
                rho = npc.diag(S**2, legs_labels[name][0][1].conj(), labels=legs_labels[name][1][-1:-3:-1])
            # NOTE: iMPS should always be normalized s.t. npc.inner(c0,rho)=1
            return c0, rho
        msg = f'Identity not dominant eigenvector of MPSTransferMatrix up to tol={tol_c0:.1e}. Computing explicitly...'
        warnings.warn(msg)
        c0 = _TM.eigenvectors()[1][0]
        c0 = c0.split_legs()
        c1 = TransferMatrix.from_Ns_Ms(
            self._Ns,
            self._Ms,
            transpose=False if name == 'init_LP' else True,
            charge_sector=None,
            p_label=self._p_label,
            conjugate_Ns=False,
        ).eigenvectors()[1][0]
        c1 = c1.split_legs()
        if name == 'init_LP':
            if npc.trace(c1) < 0.0:
                c1 *= -1.0  # fix possible negative sign
            c1._labels = legs_labels[name][1][-1:-3:-1]
            c0 /= npc.inner(c0, c1)  # normalization
            return c0, c1
        if npc.trace(c1) < 0:  # init_RP
            c1 *= -1.0
        c1._labels = legs_labels[name][1][-1:-3:-1]
        c0 /= npc.inner(c1, c0)
        return c0, c1

    def _solve_cj(self, loop, name, b, norm_one, options):
        """For `self._init_LP_RP_iterative()`: Solves c_gamma^j (1-TWjj) = b"""
        if npc.norm(b) == 0.0:
            # A has not full rank if Wjj=id, as Id(1-TWjj)=0
            # Contributions in the kernel are already subtracted though
            # we can thus assume norm(b)==0 => x=npc.zeros()
            return npc.zeros(b.legs, dtype=b.dtype, qtotal=b.qtotal, labels=b._labels)
        # TWjj
        transpose = True if name == 'init_LP' else False
        if norm_one == 1:  # skip Id contractions
            ket_M = self._Ms
        else:
            ops = [self.H._graph[j][(loop[j], loop[j + 1])] for j in range(self.L)]
            ket_M = [
                npc.tensordot(self._Ms[j], ops[j], axes=[self.ket._p_label, self.ket._get_p_label('*')])
                for j in range(self.L)
            ]
        TWjj = TransferMatrix.from_Ns_Ms(
            self._Ns,
            ket_M,
            transpose=transpose,
            charge_sector=None,
            p_label=self.ket._p_label,
            conjugate_Ns=False,
            unit_cell_width=self.ket.unit_cell_width,
        )
        # GMRES solver
        A = ShiftNpcLinearOperator(TWjj, -1.0)
        solver = GMRES(A, b, b, options=options)  # makes internal copy
        x_sol, res, _, _ = solver.run()
        if res > options['res']:
            msg = f'GMRES converged within tol={res} in environment initialization, requested was tol={options["res"]}.'
            warnings.warn(msg)
        # fix legs
        legs = ['vR', 'vR*'] if name == 'init_LP' else ['vL', 'vL*']
        x_sol.split_legs()
        x_sol.itranspose(legs)
        return -x_sol  # cancel global minus sign


class MPOTransferMatrix(NpcLinearOperator):
    """Transfermatrix of a Hamiltonian-like MPO sandwiched between canonicalized MPS.

    Given an MPS in canonical form, this class helps to find the correct initial MPO environment
    on the left or right by diagonalizing the transfer matrix.
    This is only needed for *infinite* range Hamiltonians; for finite range you can just use
    :meth:`MPOEnvironment.init_first_LP_last_RP` with ``start_env_sites=H.max_range``.
    This class **assumes** that `H` is the sum of local terms such that the transfer matrix has
    a Jordan Block form when the MPO leg is divided into :attr:`MPO.IdL`, :attr:`MPO.IdR` and the
    rest.

    Parameters
    ----------
    H : :class:`~tenpy.networks.mpo.MPO`
        The MPO sandwiched between `psi`.
        Should have 'IdL' and 'IdR'.
    psi : :class:`~tenpy.networks.mps.MPS`
        The MPS to project on. Should be given in usual 'ket' form;
        we call `conj()` on the matrices directly.
    transpose : bool
        Whether `self.matvec` acts on `RP` (``False``) or `LP` (``True``).
    guess : :class:`~tenpy.linalg.np_conserved.Array`
        Initial guess for the converged environment.

    Attributes
    ----------
    transpose : bool
        Whether `self.matvec` acts on `RP` (``False``) or `LP` (``True``).
    dtype :
        Common dtype of `H` and `psi`.
    IdL, IdR : int
        Indices of the MPO leg between unit cells, where only identities are to the left/right.
    _M, _M_conj, _W : list of :class:`~tenpy.linalg.np_conserved.Array`
        Tensors to be contracted into `vec` in :meth:`matvec`.
    guess : :class:`~tenpy.linalg.np_conserved.Array`
        Initial guess as npc Array.
    flat_linop : :class:`~tenpy.linalg.sparse.FlatLinearOperator`
        Wrapper to allow calling scipy sparse functions.
    flat_guess :
        Initial guess suitable for `flat_linop` in non-tenpy form.
    unit_cell_width : int
        See :attr:`~tenpy.models.lattice.Lattice.mps_unit_cell_width`.

    """

    def __init__(self, H, psi, transpose=False, guess=None, _subtraction_gauge='rho'):
        if psi.finite or H.bc != 'infinite':
            raise ValueError('Only makes sense for infinite MPS')
        self.L = L = lcm(H.L, psi.L)
        if np.linalg.norm(psi.norm_test()) > 1.0e-10:
            raise ValueError('psi should be in canonical form!')
        if psi._p_label != ['p']:
            raise NotImplementedError('What would the MPO act on...?')
        self.dtype = dtype = np.promote_types(psi.dtype, H.dtype)
        self.transpose = transpose
        self._M = []
        self._M_conj = []
        self._W = []
        self.IdL = H.get_IdL(0)
        self.IdR = H.get_IdR(-1)  # on bond between MPS unit cells
        self.unit_cell_width = H.unit_cell_width * (L // H.L)
        if self.IdL is None or self.IdR is None:
            raise ValueError('MPO needs to have structure with IdL/IdR')
        S = psi.get_SL(0)
        if not transpose:  # right to left
            wR = H.get_W(self.L - 1).get_leg('wR')
            wL = wR.conj()
            vR = psi.get_B(psi.L - 1, 'B').get_leg('vR')
            if isinstance(S, npc.Array):
                rho = npc.tensordot(S, S.conj(), axes=['vL', 'vL*'])
            else:
                S2 = S**2
                rho = npc.diag(S2, vR, labels=['vR', 'vR*'])

            self.acts_on = ['vL', 'wL', 'vL*']  # vec: vL wL vL*

            for i in reversed(range(self.L)):
                # optimize: transpose arrays to mostly avoid it in matvec
                B = psi.get_B(i, 'B').astype(dtype, False)
                self._M.append(B.transpose(['vL', 'p', 'vR']))
                self._W.append(H.get_W(i).transpose(['p*', 'wR', 'p', 'wL']).astype(dtype, False))
                self._M_conj.append(B.conj().itranspose(['vR*', 'p*', 'vL*']))

            # vR = self._M[0].get_leg('vR')
            self._chi0 = chi0 = vR.ind_len
            eye_R = npc.diag(1.0, vR.conj(), dtype=dtype, labels=['vL', 'vL*'])
            self._E_shift = eye_R.add_leg(wL, self.IdL, axis=1, label='wL')  # vL wL vL*
            self._proj_trace = self._E_shift.conj().iset_leg_labels(['vR', 'wR', 'vR*']) / chi0
            self._proj_norm = eye_R.add_leg(wL, self.IdR, axis=1, label='wL').conj()  # vL* wL* vL
            self._proj_rho = rho.add_leg(wR, self.IdL, axis=1, label='wR')  # vR wR vR*
        else:  # left to right
            wL = H.get_W(0).get_leg('wL')
            wR = wL.conj()
            vL = psi.get_B(0, 'A').get_leg('vL')
            if isinstance(S, npc.Array):
                rho = npc.tensordot(S.conj(), S, axes=['vR*', 'vR'])
            else:
                S2 = S**2
                rho = npc.diag(S2, vL.conj(), labels=['vL*', 'vL'])

            self.acts_on = ['vR*', 'wR', 'vR']  # labels of the vec

            for i in range(self.L):
                A = psi.get_B(i, 'A').astype(dtype, False)
                self._M.append(A.transpose(['vL', 'p', 'vR']))
                self._W.append(H.get_W(i).transpose(['wR', 'p', 'wL', 'p*']).astype(dtype, False))
                self._M_conj.append(A.conj().itranspose(['vR*', 'p*', 'vL*']))

            # vL = self._M[0].get_leg('vL')
            self._chi0 = chi0 = vL.ind_len
            eye_L = npc.diag(1.0, vL, dtype=dtype, labels=['vR*', 'vR'])
            self._E_shift = eye_L.add_leg(wR, self.IdR, axis=1, label='wR')  # vR* wR vR
            self._proj_trace = self._E_shift.conj().iset_leg_labels(['vL*', 'wL', 'vL']) / chi0
            self._proj_norm = eye_L.add_leg(wR, self.IdL, axis=1, label='wR').conj()  # vR wR* vR*
            self._proj_rho = rho.add_leg(wL, self.IdR, axis=1, label='wL')  # vL* wL vL
        if _subtraction_gauge == 'trace':
            self._proj_subtr = self._proj_trace
        elif _subtraction_gauge == 'rho':
            self._proj_subtr = self._proj_rho
        else:
            raise ValueError(f'unknown _subtraction_gauge={_subtraction_gauge!r}')
        # check guess for correctness
        if guess is not None:
            try:
                if not transpose:
                    guess.get_leg('wL').test_equal(wL)
                    guess.get_leg('vL').test_contractible(vR)
                    guess.get_leg('vL*').test_equal(vR)
                else:
                    guess.get_leg('wR').test_equal(wR)
                    guess.get_leg('vR').test_contractible(vL)
                    guess.get_leg('vR*').test_equal(vL)
            except ValueError:
                logger.warning('dropping guess for MPOTransferMatrix with incompatible legs')
                guess = None
        if guess is None:
            if not transpose:
                guess = eye_R.add_leg(wL, self.IdR, axis=1, label='wL')  # vL wL vL*
            else:
                guess = eye_L.add_leg(wR, self.IdL, axis=1, label='wR')  # vR* wR vR
            # no need to _project: E = 0
        else:
            if not transpose:
                guess = guess.transpose(['vL', 'wL', 'vL*'])  # copy!
            else:
                guess = guess.transpose(['vR*', 'wR', 'vR'])  # copy!
            self._project(guess)
        self.guess = guess
        self.flat_linop, self.flat_guess = FlatLinearOperator.from_guess_with_pipe(self.matvec, self.guess, dtype=dtype)
        self._explicit_plus_hc = H.explicit_plus_hc

    def matvec(self, vec, project=True):
        """One matvec-operation.

        Parameters
        ----------
        project : bool
            If True, project away the trace of the "IdL" part (transpose=False)
            or "IdR" part (transpose=True), respectively, to transform the Jordan-Block structure
            into something that is translation invariant.

        """
        if not self.transpose:  # right to left
            vec.itranspose(['vL', 'wL', 'vL*'])  # shouldn't do anything
            for Bc, W, B in zip(self._M_conj, self._W, self._M):
                # vec: vL wL vL*
                vec = npc.tensordot(B, vec, axes=['vR', 'vL'])  # vL p wL vL*
                vec = npc.tensordot(vec, W, axes=[['p', 'wL'], ['p*', 'wR']])  # vL vL* p wL
                vec = npc.tensordot(vec, Bc, axes=[['vL*', 'p'], ['vR*', 'p*']])  # vL wL vL*
            vec = vec.shift_charges_horizontal(dx_0=self.unit_cell_width)
        else:
            vec.itranspose(['vR*', 'wR', 'vR'])  # shouldn't do anything
            for Ac, W, A in zip(self._M_conj, self._W, self._M):
                vec = npc.tensordot(vec, A, axes=['vR', 'vL'])  # vR* wR p vR
                vec = npc.tensordot(W, vec, axes=[['wL', 'p*'], ['wR', 'p']])  # wR p vR* vR
                vec = npc.tensordot(Ac, vec, axes=[['p*', 'vL*'], ['p', 'vR*']])  # vR* wR vR
            vec = vec.shift_charges_horizontal(dx_0=-self.unit_cell_width)
        if project:
            self._project(vec)
        return vec

    def _project(self, vec):
        """Project out additive energy part from vec."""
        if not self.transpose:  # Acts to the right, T * RP = RP + e_R * I
            vec.itranspose(['vL', 'wL', 'vL*'])  # shouldn't do anything
            E = npc.inner(vec, self._proj_subtr, axes=[['vL', 'wL', 'vL*'], ['vR', 'wR', 'vR*']])
            vec -= self._E_shift * E
        else:  # Acts to the left, LP * T = LP + e_L * I
            vec.itranspose(['vR*', 'wR', 'vR'])  # shouldn't do anything
            E = npc.inner(vec, self._proj_subtr, axes=[['vR*', 'wR', 'vR'], ['vL*', 'wL', 'vL']])
            vec -= self._E_shift * E

    def dominant_eigenvector(self, **kwargs):
        """Find dominant eigenvector of self using :mod:`scipy.sparse`.

        Parameters
        ----------
        **kwargs :
            Keyword arguments for :meth:`~tenpy.linalg.sparse.FlatLinearOperator.eigenvectors`.

        Returns
        -------
        val : float
            Eigenvalue for the transfer matrix; should be (very) close to 1.
        vec :
            Eigenvector to be used as initial LP/RP for an :class:`MPOEnvironment`.

        """
        if 'v0_npc' not in kwargs:
            kwargs.setdefault('v0', self.flat_guess)
        vals, vecs = self.flat_linop.eigenvectors(**kwargs)
        val = vals[0]
        v0 = vecs[0]
        v0 = v0.split_legs()
        norm = npc.inner(self._proj_norm, v0, axes='range', do_conj=False) / self._chi0
        return val, v0 / norm

    def energy(self, dom_vec):
        """Given the dominant eigenvector, calculate the energy per MPS site.

        **Assumes** that `dominant_vec` is the result of :meth:`dominant_eigenvector`.

        Returns
        -------
        energy : float
            Energy *per site* of the MPS.

        """
        if not self.transpose:
            axes = (['vL', 'wL', 'vL*'], ['vR', 'wR', 'vR*'])
        else:
            axes = (['vR*', 'wR', 'vR'], ['vL*', 'wL', 'vL'])
        E0 = npc.inner(dom_vec, self._proj_rho, axes)
        vec = self.matvec(dom_vec, project=False)
        E = npc.inner(vec, self._proj_rho, axes)
        E = (E - E0) / self.L
        if self._explicit_plus_hc:
            E = np.real(E + np.conj(E))
        return E

    @classmethod
    def find_init_LP_RP(
        cls,
        H,
        psi,
        first=0,
        last=None,
        guess_init_env_data=None,
        calc_E=False,
        tol_ev0=1.0e-8,
        _subtraction_gauge='rho',
        **kwargs,
    ):
        """Find the initial LP and RP.

        .. note ::

            - In most cases :meth:`MPOEnvironmentBuilder.init_LP_RP_iterative` will provide
              better performance. Additionally, it allows to compute  :attr:`_subtraction_gauge`
              explicitly, therefore being stable against errors in the MPS canonical form.

        Parameters
        ----------
        H, psi :
            MPO and MPS, see class docstring.
        first, last : int
            Indices to the left/right of which to extract the environments.
        calc_E : bool
            Wether to calculate and return the energy.
        tol_ev0 : float
            Tolerance to trigger a warning about non-unit eigenvalue.
        guess : None | dict
            Possible `init_env_data` with the guess/result of DMRG updates.
            If some legs are incompatible, trigger a warning and ignore.
        _subtraction_gauge : string
            How the additive part of the generalized eigenvector is subtracted out.
            Possible values are 'rho' and 'trace'; see documentation for MPOTransferMatrix
            for more details.
        **kwargs :
            Further keyword arguments for
            :meth:`~tenpy.linalg.sparse.FlatLinearOperator.eigenvectors`.

        Returns
        -------
        init_env_data : dict
            Dictionary with `init_LP` and `init_RP` that can be given to :class:`MPOEnvironment`.
        E : float
            Energy per site. Only returned if `calc_E` is True.
        eps : float
            The contraction of ``<LP |SS|RP>`` for the environment

        """
        # first right to left
        envs = []
        Es = []
        if guess_init_env_data is None:
            guess_init_env_data = {}
        for transpose in [False, True]:
            guess = guess_init_env_data.get('init_LP' if transpose else 'init_RP', None)
            TM = cls(H, psi, transpose=transpose, guess=guess, _subtraction_gauge=_subtraction_gauge)
            val, vec = TM.dominant_eigenvector(**kwargs)
            if abs(1.0 - val) > tol_ev0:
                logger.warning('MPOTransferMatrix eigenvalue not 1: got %s', val)
            envs.append(vec)
            if calc_E:
                Es.append(TM.energy(vec))  # E_R, E_L
            L = TM.L
            del TM
        init_env_data = {'init_LP': envs[1], 'init_RP': envs[0], 'age_LP': 0, 'age_RP': 0}
        if first != 0 or (last is not None and last % L != L - 1):
            env = MPOEnvironment(psi, H, psi, **init_env_data)
            if first % L != 0:
                init_env_data['init_LP'] = env.get_LP(first, store=False)
            if last is not None and last % L != L - 1:
                init_env_data['init_RP'] = env.get_RP(last, store=False)
        if calc_E:
            # We need this for segment excitation energies.
            # TODO: this doesn't work for non-default first/last!?
            if first != 0 or last is not None:
                assert (last + 1) % L == first % L, (
                    'Need to have an integer number of unit cells for the bond to be the same.'
                )
            SL = psi.get_SL(first)
            if not isinstance(SL, npc.Array):
                vL = init_env_data['init_LP'].get_leg('vR').conj()
                SL = npc.diag(SL, vL, dtype=np.promote_types(psi.dtype, H.dtype), labels=['vL', 'vR'])
            E0 = npc.tensordot(init_env_data['init_LP'], SL, axes=(['vR'], ['vL']))
            E0 = npc.tensordot(E0, SL.conj(), axes=(['vR*'], ['vL*']))
            E0 = npc.tensordot(E0, init_env_data['init_RP'], axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
            # E0 = LP * s^2 * RP on site 0
            return init_env_data, Es, E0
        # else:
        return init_env_data


def grid_insert_ops(site, grid):
    """Replaces entries representing operators in a grid of ``W[i]`` with npc.Arrays.

    Parameters
    ----------
    site : :class:`~tenpy.networks.site`
        The site on which the grid acts.
    grid : list of list of `entries`
        Represents a single matrix `W` of an MPO, i.e. the lists correspond to the legs
        ``'vL', 'vR'``, and entries to onsite operators acting on the given `site`.
        `entries` may be ``None``, :class:`~tenpy.linalg.np_conserved.Array`, a single string
        or of the form ``[('opname', strength), ...]``, where ``'opname'`` labels an operator in
        the `site`.

    Returns
    -------
    grid : list of list of {None | :class:`~tenpy.linalg.np_conserved.Array`}
        Copy of `grid` with entries ``[('opname', strength), ...]`` replaced by
        ``sum([strength*site.get_op('opname') for opname, strength in entry])``
        and entries ``'opname'`` replaced by ``site.get_op('opname')``.

    """
    new_grid = [None] * len(grid)
    for i, row in enumerate(grid):
        new_row = new_grid[i] = list(row)
        for j, entry in enumerate(new_row):
            if entry is None or isinstance(entry, npc.Array):
                continue
            if isinstance(entry, str):
                new_row[j] = site.get_op(entry)
            else:
                opname, strength = entry[0]
                res = strength * site.get_op(opname)
                for opname, strength in entry[1:]:
                    res = res + strength * site.get_op(opname)
                new_row[j] = res  # replace entry
                # new_row[j] = sum([strength*site.get_op(opname) for opname, strength in entry])
    return new_grid


def _calc_grid_legs_finite(chinfo, grids, Ws_qtotal, leg0):
    """Calculate LegCharges from `grids` for a finite MPO.

    This is the easier case. We just gauge the very first leg to the left to zeros, then all other
    charges (hopefully) follow from the entries of the grid.
    """
    if leg0 is None:
        if len(grids[0]) != 1:
            raise ValueError('finite MPO with len of first bond != 1')
        q = chinfo.make_valid()
        leg0 = npc.LegCharge.from_qflat(chinfo, [q], qconj=+1)
    legs = [leg0]
    for i, gr in enumerate(grids):
        gr_legs = [legs[-1], None]
        gr_legs = npc.detect_grid_outer_legcharge(gr, gr_legs, qtotal=Ws_qtotal[i], qconj=-1, bunch=False)
        legs.append(gr_legs[1].conj())
    return legs


def _calc_grid_legs_infinite(chinfo, grids, Ws_qtotal, leg0, IdL_0):
    """Calculate LegCharges from `grids` for an iMPO.

    Similar like :func:`_calc_grid_legs_finite`, but the hard case.
    Initially, we do not know all charges of the first leg; and they have to
    be consistent with the final leg.

    The way this works: gauge 'IdL' on the very left leg to 0,
    then gradually calculate the charges by going along the edges of the graph
    (maybe also over the iMPO boundary).

    When initializing from an MPO graph directly, use :meth:`MPOGraph._calc_legcharges` directly.
    """
    if leg0 is not None:
        # have charges of first leg: simple case, can use the _calc_grid_legs_finite version.
        legs = _calc_grid_legs_finite(chinfo, grids, Ws_qtotal, leg0)
        legs[-1].test_contractible(legs[0])  # consistent?
        return legs
    L = len(grids)
    chis = [len(g) for g in grids]
    charges = [[None] * chi for chi in chis]
    charges.append(charges[0])  # the *same* list is shared for 0 and -1.

    charges[0][IdL_0] = chinfo.make_valid(None)  # default charge = 0.

    for _ in range(1000 * L):  # I don't expect interactions with larger range than that...
        for i in range(L):
            grid = grids[i]
            QsL, QsR = charges[i : i + 2]
            for vL, row in enumerate(grid):
                qL = QsL[vL]
                if qL is None:
                    continue  # don't know the charge on the left yet
                for vR, op in enumerate(row):
                    if op is None:
                        continue
                    # calculate charge qR from the entry of the grid
                    qR = chinfo.make_valid(qL + op.qtotal - Ws_qtotal[i])
                    if QsR[vR] is None:
                        QsR[vR] = qR
                    elif np.any(QsR[vR] != qR):
                        raise ValueError('incompatible charges while creating the MPO')
        if not any(q is None for Qs in charges for q in Qs):
            break
    else:  # no `break` in the for loop, i.e. we are unable to determine all grid legcharges.
        # this should not happen (if we have no bugs), but who knows ^_^
        # if it happens, there might be unconnected parts in the graph
        raise ValueError("Can't determine LegCharge for the MPO")
    legs = [npc.LegCharge.from_qflat(chinfo, qflat, qconj=+1) for qflat in charges[:-1]]
    legs.append(legs[0])
    return legs


def _mpo_graph_state_order(key):
    """Key-function for sorting they `states` of an MPO Graph.

    For standard TeNPy MPOs we expect keys of the form
    ``'IdL'``, ``'IdR'``, ``(i, op_i, opstr)`` and recursively ``key + (j, op_j, opstr)``,
    (Note that op_j can be opstr if ``j-i >= L``.)
    For multi coupling terms keys have the form ``("left",i, op_i, opstr)``

    The goal is to ensure that standard TeNPy MPOs yield an upper-right W for the MPO.
    """
    if isinstance(key, tuple):
        if key[0] == 'left':  # left states first
            return (-1, len(key)) + key[1:]
        elif key[0] == 'right':  # right states afterwards
            return (1, -len(key)) + key[1:]
        return key
    if isinstance(key, str):
        if key == 'IdL':  # should be first
            return (-2,)
        if key == 'IdR':  # should be last
            return (2,)
        # fallback: compare strings
        return (0, key)
    return (0, str(key))


def _mpo_check_for_iter_LP_RP_infinite(mpo):
    """Check that :meth:`MPOEnvironmentBuilder.init_LP_RP_iterative` works for an MPO

    Initializes the respective attributes on the fly if needed
    """
    if mpo._graph is None:
        mpo._make_graph()
    if mpo._outer_permutation is None:
        mpo._order_graph()
    return mpo._outer_permutation


def _partition_W(W, IdL_L, IdR_L, IdL_R, IdR_R):
    """Split MPO into blocks with respect to standard upper triangular form.

    1 C D
    0 A B
    0 0 1

    """
    DL, DR, d, d = W.shape
    proj_L = np.ones(DL, dtype=np.bool_)
    proj_L[IdL_L] = False
    proj_L[IdR_L] = False
    proj_R = np.ones(DR, dtype=np.bool_)
    proj_R[IdL_R] = False
    proj_R[IdR_R] = False

    # Extract (A, B, C, D)
    D_npc = W.copy()
    D_npc.iproject([IdL_L, IdR_R], ['wL', 'wR'])
    D_npc = D_npc.squeeze()  # remove dummy wL, wR legs
    C_npc = W.copy()
    C_npc.iproject([IdL_L, proj_R], ['wL', 'wR'])
    B_npc = W.copy()
    B_npc.iproject([proj_L, IdR_R], ['wL', 'wR'])
    A_npc = W.copy()
    A_npc.iproject([proj_L, proj_R], ['wL', 'wR'])
    return A_npc, B_npc, C_npc, D_npc
