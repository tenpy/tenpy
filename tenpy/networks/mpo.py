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
canoncial form. However, unlike for an MPS, this doesn't simplify calculations.
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
# Copyright 2018-2019 TeNPy Developers, GNU GPLv3

import numpy as np
import warnings

from ..linalg import np_conserved as npc
from .site import group_sites, Site
from ..tools.string import vert_join
from .mps import MPS as _MPS  # only for MPS._valid_bc
from .mps import MPSEnvironment
from .terms import OnsiteTerms, CouplingTerms, MultiCouplingTerms
from ..tools.misc import add_with_None_0
from ..tools.math import lcm

__all__ = ['MPO', 'MPOGraph', 'MPOEnvironment', 'grid_insert_ops']


class MPO:
    """Matrix product operator, finite (MPO) or infinite (iMPO).

    Parameters
    ----------
    sites : list of :class:`~tenpy.models.lattice.Site`
        Defines the local Hilbert space for each site.
    Ws : list of :class:`~tenpy.linalg.np_conserved.Array`
        The matrices of the MPO. Should have labels ``wL, wR, p, p*``.
    bc : {'finite' | 'segment' | 'infinite'}
        Boundary conditions as described in :mod:`~tenpy.networks.mps`.
        ``'finite'`` requires ``Ws[0].get_leg('wL').ind_len = 1``.
    IdL : (iterable of) {int | None}
        Indices on the bonds, which correpond to 'only identities to the left'.
        A single entry holds for all bonds.
    IdR : (iterable of) {int | None}
        Indices on the bonds, which correpond to 'only identities to the right'.
    max_range : int | np.inf | None
        Maximum range of hopping/interactions (in unit of sites) of the MPO. ``None`` for unknown.

    Attributes
    ----------
    L : int
        ``len(sites)``. For an iMPS, this is the number of sites in the MPS unit cell.
    chinfo : :class:`~tenpy.linalg.np_conserved.ChargeInfo`
        The nature of the charge.
    sites : list of :class:`~tenpy.models.lattice.Site`
        Defines the local Hilbert space for each site.
    dtype : type
        The data type of the `_W`.
    bc : {'finite' | 'segment' | 'infinite'}
        Boundary conditions as described in :mod:`~tenpy.networks.mps`.
        ``'finite'`` requires ``Ws[0].get_leg('wL').ind_len = 1``.
    IdL : list of {int | None}
        Indices on the bonds (length `L`+1), which correpond to 'only identities to the left'.
        ``None`` for bonds where it is not set.
        In standard form, this is `0` (except for unset bonds in finite case)
    IdR : list of {int | None}
        Indices on the bonds (length `L`+1), which correpond to 'only identities to the right'.
        ``None`` for bonds where it is not set.
        In standard form, this is the last index on the bond (except for unset bonds in finite case).
    max_range : int | np.inf | None
        Maximum range of hopping/interactions (in unit of sites) of the MPO. ``None`` for unknown.
    grouped : int
        Number of sites grouped together, see :meth:`group_sites`.
    _W : list of :class:`~tenpy.linalg.np_conserved.Array`
        The matrices of the MPO. Labels are ``'wL', 'wR', 'p', 'p*'``.
    _valid_bc : tuple of str
        Valid boundary conditions. The same as for an MPS.
    """

    _valid_bc = _MPS._valid_bc  # same valid boundary conditions as an MPS.

    def __init__(self, sites, Ws, bc='finite', IdL=None, IdR=None, max_range=None):
        self.sites = list(sites)
        self.chinfo = self.sites[0].leg.chinfo
        self.dtype = dtype = np.find_common_type([W.dtype for W in Ws], [])
        self._W = [W.astype(dtype, copy=True) for W in Ws]
        self.IdL = self._get_Id(IdL, len(sites))
        self.IdR = self._get_Id(IdR, len(sites))
        self.grouped = 1
        self.bc = bc
        self.max_range = max_range
        self.test_sanity()

    @classmethod
    def from_grids(cls,
                   sites,
                   grids,
                   bc='finite',
                   IdL=None,
                   IdR=None,
                   Ws_qtotal=None,
                   leg0=None,
                   max_range=None):
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
            Indices on the bonds, which correpond to 'only identities to the left'.
            A single entry holds for all bonds.
        IdR : (iterable of) {int | None}
            Indices on the bonds, which correpond to 'only identities to the right'.
        Ws_qtotal : (list of) total charge
            The `qtotal` to be used for each grid. Defaults to zero charges.
        leg0 : :class:`~tenpy.linalg.charge.LegCharge`
            LegCharge for 'wL' of the left-most `W`. By default, construct it.
        max_range : int | np.inf | None
            Maximum range of hopping/interactions (in unit of sites) of the MPO.
            ``None`` for unknown.

        See also
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
        if bc != 'infinite':
            if leg0 is None:
                # ensure that we have only a single entry in the first and last leg
                # i.e. project grids[0][:, :] -> grids[0][IdL[0], :]
                # and         grids[-1][:, :] -> grids[-1][:,IdR[-1], :]
                first_grid = grids[0]
                last_grid = grids[-1]
                if len(first_grid) > 1:
                    grids[0] = first_grid[IdL[0]]
                    IdL[0] = 0
                if len(last_grid[0]) > 1:
                    grids[0] = [row[IdR[-1]] for row in last_grid]
                    IdR[-1] = 0
            legs = _calc_grid_legs_finite(chinfo, grids, Ws_qtotal, leg0)
        else:
            legs = _calc_grid_legs_infinite(chinfo, grids, Ws_qtotal, leg0, IdL[0])
        # now build the `W` from the grid
        assert len(legs) == L + 1
        Ws = []
        for i in range(L):
            W = npc.grid_outer(grids[i], [legs[i], legs[i + 1].conj()], Ws_qtotal[i])
            W.iset_leg_labels(['wL', 'wR', 'p', 'p*'])
            Ws.append(W)
        return cls(sites, Ws, bc, IdL, IdR, max_range)

    def test_sanity(self):
        """Sanity check, raises ValueErrors, if something is wrong."""
        assert self.L == len(self.sites)
        if self.bc not in self._valid_bc:
            raise ValueError("invalid MPO boundary conditions: " + repr(self.bc))
        for i in range(self.L):
            S = self.sites[i]
            W = self._W[i]
            S.leg.test_equal(W.get_leg('p'))
            S.leg.test_contractible(W.get_leg('p*'))
            if self.bc == 'infinite' or i + 1 < self.L:
                W2 = self.get_W(i + 1)
                W.get_leg('wR').test_contractible(W2.get_leg('wL'))
        if not (len(self.IdL) == len(self.IdR) == self.L + 1):
            raise ValueError("wrong len of `IdL`/`IdR`")

    @property
    def L(self):
        """Number of physical sites; for an iMPO the len of the MPO unit cell."""
        return len(self.sites)

    @property
    def dim(self):
        """List of local physical dimensions."""
        return [site.dim for site in self.sites]

    @property
    def finite(self):
        """Distinguish MPO vs iMPO.

        True for an MPO (``bc='finite', 'segment'``), False for an iMPO (``bc='infinite'``).
        """
        assert (self.bc in self._valid_bc)
        return self.bc != 'infinite'

    @property
    def chi(self):
        """Dimensions of the virtual bonds."""
        return [W.get_leg('wL').ind_len for W in self._W] + [self._W[-1].get_leg('wR').ind_len]

    def get_W(self, i, copy=False):
        """Return `W` at site `i`."""
        i = self._to_valid_index(i)
        if copy:
            return self._W[i].copy()
        return self._W[i]

    def set_W(self, i, W):
        """Set `W` at site `i`."""
        i = self._to_valid_index(i)
        self._W[i] = W

    def get_IdL(self, i):
        """Return index of `IdL` at bond to the *left* of site `i`.

        May be ``None``.
        """
        i = self._to_valid_index(i)
        return self.IdL[i]

    def get_IdR(self, i):
        """Return index of `IdR` at bond to the *right* of site `i`.

        May be ``None``.
        """
        i = self._to_valid_index(i)
        return self.IdR[i + 1]

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
        if self.max_range is not None:
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
        # done

    def expectation_value(self, psi, tol=1.e-10, max_range=100):
        """Calculate ``<psi|self|psi>/<psi|psi>``.

        For a finite MPS, simply contract the network ``<psi|self|psi>``.
        For an infinite MPS, it assumes that `self` is the a of terms, with :attr:`IdL`
        and :attr:`IdR` defined on each site.  Under this assumption,
        it calculates the expectation value of terms with the left-most non-trivial
        operator inside the MPO unit cell and returns the average value per site.

        Parameters
        ----------
        psi : :class:`~tenpy.networks.mps.MPS`
            State for which the expectation value should be taken.
        tol : float
            Ignored for finite `psi`.
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
            return MPOEnvironment(psi, self, psi).full_contraction(0)
        env = MPOEnvironment(psi, self, psi)
        L = self.L
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

        for i in range(1, max_range * L):
            i0 = i % L
            W = self._W[i0]
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

            if i >= L:
                RP = env.init_RP(i)
                current_value = npc.inner(LP,
                                          RP,
                                          axes=[['vR*', 'wR', 'vR'], ['vL*', 'wL', 'vL']],
                                          do_conj=False)
                LP_converged = LP.copy()
                LP_converged.iproject(masks_R_no_IdRL[i0], 'wR')
                if npc.norm(LP_converged) < tol:
                    break  # no more terms left
        else:  # no break
            msg = "Tolerance {0:.2e} not reached within {1:d} sites".format(tol, max_range)
            warnings.warn(msg, stacklevel=2)
        return current_value / L

    def dagger(self):
        """Return hermition conjugate copy of self."""
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
        return MPO(self.sites, Ws, self.bc, self.IdL, self.IdR, self.max_range)

    def is_hermitian(self, eps=1.e-10, max_range=None):
        """Check if `self` is a hermitian MPO.

        Shorthand for ``self.is_equal(self.dagger(), eps, max_range)``.
        """
        return self.is_equal(self.dagger(), eps, max_range)

    def is_equal(self, other, eps=1.e-10, max_range=None):
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
            sites with indices ``range(self.L + max_range)``.
            None defaults to :attr:`max_range` (or :attr:`L` in case this is infinite or None).

        Returns
        -------
        equal : bool
            Whether `self` equals `other` to the desired precision.
        """
        if self.finite:
            max_i = self.L
        else:
            if max_range is None:
                if self.max_range is None or self.max_range == np.inf:
                    max_range = self.L
                else:
                    max_range = self.max_range
            max_i = self.L + max_range

        def overlap(A, B):
            """<A|B> on sites 0 to max_i."""
            wA = A.get_W(0).take_slice([A.get_IdL(0)], ['wL']).conj()
            wB = B.get_W(0).take_slice([B.get_IdL(0)], ['wL'])
            trAdB = npc.tensordot(wA, wB, axes=[['p*', 'p'], ['p', 'p*']])  # wR* wR
            i = 0
            for i in range(1, max_i):
                trAdB = npc.tensordot(trAdB, A.get_W(i).conj(), axes=['wR*', 'wL*'])
                trAdB = npc.tensordot(trAdB,
                                      B.get_W(i),
                                      axes=[['wR', 'p*', 'p'], ['wL', 'p', 'p*']])
            trAdB = trAdB.itranspose(['wR*', 'wR'])[A.get_IdR(i), B.get_IdR(i)]
            return trAdB

        self_other = 2. * np.real(overlap(other, self))
        norms = overlap(self, self) + overlap(other, other)
        return abs(norms - self_other) < eps * abs(norms)

    def _to_valid_index(self, i):
        """Make sure `i` is a valid index (depending on `self.bc`)."""
        if not self.finite:
            return i % self.L
        if i < 0:
            i += self.L
        if i >= self.L or i < 0:
            raise ValueError("i = {0:d} out of bounds for finite MPO".format(i))
        return i

    @staticmethod
    def _get_Id(Id, L):
        """parse the IdL or IdR argument of __init__"""
        if Id is None:
            return [None] * (L + 1)
        else:
            try:
                return list(Id)
            except TypeError:
                return [Id] * (L + 1)

    def get_grouped_mpo(self, blocklen):
        """Contract `blocklen` subsequent tensors into a single one and return result as a new MPO
        object."""
        msg = "Use functions from `tenpy.algorithms.exact_diag.ExactDiag.from_H_mpo` instead"
        warnings.warn(msg, FutureWarning, 2)
        from copy import deepcopy
        groupedMPO = deepcopy(self)
        groupedMPO.group_sites(n=blocklen)
        return (groupedMPO)

    def get_full_hamiltonian(self, maxsize=1e6):
        """extract the full Hamiltonian as a d**L x d**L matrix."""
        msg = "Use functions from `tenpy.algorithms.exact_diag.ExactDiag.from_H_mpo` instead"
        warnings.warn(msg, FutureWarning, 2)
        if (self.dim[0]**(2 * self.L) > maxsize):
            print('Matrix dimension exceeds maxsize')
            return np.zeros(1)
        singlesitempo = self.get_grouped_mpo(self.L)
        # Note: the trace works only for 'finite' boundary conditions
        # where the legs are trivial - otherwise it would give 0 or even raise an error!
        return npc.trace(singlesitempo.get_W(0), axes=[['wL'], ['wR']])

    def __add__(self, other):
        """Return an MPO representing `self + other`.

        Requires both `self` and `other` to be in standard sum form with `IdL` and `IdR` being set.

        Parameters
        ----------
        other : :class:`MPO`
            MPO to be added to `self`.

        Returns
        -------
        sum_mpo : :class:`MPO`
            The sum `self + other`.
        """
        L = self.L
        assert self.bc == other.bc
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

        # l/r = left/rigth,  s/o = self/other
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
            ]  # yapf: disable
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
        return MPO(self.sites, Ws, self.bc, IdL, IdR, max_range)

    def _get_block_projections(self, i):
        """projecteions onto (IdL, other, IdR) on bond `i` in range(0, L+1)"""
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


class MPOGraph:
    """Representation of an MPO by a graph, based on a 'finite state machine'.

    This representation is used for building H_MPO from the interactions.
    The idea is to view the MPO as a kind of 'finite state machine'.
    The **states** or **keys** of this finite state machine life on the MPO bonds *between* the
    `Ws`. They label the indices of the virtul bonds of the MPOs, i.e., the indices on legs
    ``wL`` and ``wR``. They can be anything hash-able like a ``str``, ``int`` or a tuple of them.

    The **edges** of the graph are the entries ``W[keyL, keyR]``, which itself are onsite operators
    on the local Hilbert space. The indices `keyL` and `keyR` correspond to the legs ``'wL', 'wR'``
    of the MPO. The entry ``W[keyL, keyR]`` connects the state ``keyL`` on bond ``(i-1, i)``
    with the state ``keyR`` on bond ``(i, i+1)``.

    The keys ``'IdR'`` (for 'idenity left') and ``'IdR'`` (for 'identity right') are reserved to
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

    Attributes
    ----------
    L
    sites : list of :class:`~tenpy.models.lattice.Site`
        Defines the local Hilbert space for each site.
    chinfo : :class:`~tenpy.linalg.np_conserved.ChargeInfo`
        The nature of the charge.
    bc : {'finite', 'infinite'}
        MPO boundary conditions.
    max_range : int | np.inf | None
        Maximum range of hopping/interactions (in unit of sites) of the MPO. ``None`` for unknown.
    states : list of set of keys
        ``states[i]`` gives the possible keys at the virtual bond ``(i-1, i)`` of the MPO.
    graph : list of dict of dict of list of tuples
        For each site `i` a dictionary ``{keyL: {keyR: [(opname, strength)]}}`` with
        ``keyL in vertices[i]`` and ``keyR in vertices[i+1]``.
    _grid_legs : None | list of LegCharge
        The charges for the MPO
    """
    def __init__(self, sites, bc='finite', max_range=None):
        self.sites = list(sites)
        self.chinfo = self.sites[0].leg.chinfo
        self.bc = bc
        self.max_range = max_range
        # empty graph
        self.states = [set() for _ in range(self.L + 1)]
        self.graph = [{} for _ in range(self.L)]
        self._ordered_states = None
        self.test_sanity()

    @classmethod
    def from_terms(cls, onsite_terms, coupling_terms, sites, bc):
        """Initialize an :class:`MPOGraph` from OnsiteTerms and CouplingTerms.

        Parameters
        ----------
        onsite_terms : :class:`~tenpy.networks.terms.OnsiteTerms`
            Onsite terms to be added to the new :class:`MPOGraph`.
        coupling_terms :class:`~tenpy.networks.terms.CouplingTerms` | :class:`~tenpy.networks.terms.MultiCouplingTerms`
            Coupling terms to be added to the new :class:`MPOGraph`.
        sites : list of :class:`~tenpy.networks.site.Site`
            Local sites of the Hilbert space.
        bc : ``'finite' | 'infinite'``
            MPO boundary conditions.

        Returns
        -------
        graph : :class:`MPOGraph`
            Initialized with the given terms.

        See also
        --------
        from_term_list : equivalent for other representation terms.
        """
        graph = cls(sites, bc, coupling_terms.max_range())
        onsite_terms.add_to_graph(graph)
        coupling_terms.add_to_graph(graph)
        graph.add_missing_IdL_IdR()
        return graph

    @classmethod
    def from_term_list(cls, term_list, sites, bc):
        """Initialize form a list of operator terms and prefactors.

        Parameters
        ----------
        term_list : :class:`~tenpy.networks.mps.TermList`
            Terms to be added to the MPOGraph.
        sites : list of :class:`~tenpy.networks.site.Site`
            Local sites of the Hilbert space.
        bc : ``'finite' | 'infinite'``
            MPO boundary conditions.

        Returns
        -------
        graph : :class:`MPOGraph`
            Initialized with the given terms.

        See also
        --------
        from_terms : equivalent for other representation of terms.
        """
        ot, ct = term_list.to_OnsiteTerms_CouplingTerms(sites)
        return cls.from_terms(ot, ct, sites, bc)

    def test_sanity(self):
        """Sanity check, raises ValueErrors, if something is wrong."""
        assert len(self.graph) == self.L
        assert len(self.states) == self.L + 1
        if self.bc not in MPO._valid_bc:
            raise ValueError("invalid MPO boundary conditions: " + repr(self.bc))
        for i, site in enumerate(self.sites):
            if site.leg.chinfo != self.chinfo:
                raise ValueError("invalid ChargeInfo for site {i:d}".format(i=i))
            stL, stR = self.states[i:i + 2]
            # check graph
            gr = self.graph[i]
            for keyL in gr:
                assert keyL in stL
                for keyR in gr[keyL]:
                    assert keyR in stR
                    for opname, strength in gr[keyL][keyR]:
                        assert site.valid_opname(opname)
        # done

    @property
    def L(self):
        """Number of physical sites; for infinite boundaries the length of the unit cell."""
        return len(self.sites)

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
                raise ValueError("operator {0!r} not existent on site {1:d}".format(opname, i))
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

    def add_string(self, i, j, key, opname='Id', check_op=True, skip_existing=True):
        r"""Insert a bunch of edges for an 'operator string' into the graph.

        Terms like :math:`S^z_i S^z_j` actually stand for
        :math:`S^z_i \otimes \prod_{i < k < j} \mathbb{1}_k \otimes S^z_j`.
        This function adds the :math:`\mathbb{1}` terms to the graph.

        Parameters
        ----------
        i, j: int
            An edge is inserted on all bonds between `i` and `j`, `i < j`.
            `j` can be larger than :attr:`L`, in which case the operators are supposed to act on
            different MPS unit cells.
        key: hashable
            The state at bond (i-1, i) to connect from and on bond (j-1, j) to connect to.
            Also used for the intermediate states.
            No operator is inserted on a site `i < k < j` if ``has_edge(k, key, key)``.
        opname : str
            Name of the operator to be used for the string.
            Useful for the Jordan-Wigner transformation to fermions.
        skip_existing : bool
            Whether existing graph nodes should be skipped.

        Returns
        -------
        label_j : hashable
            The `key` on the left of site j to connect to. Usually the same as the parameter `key`,
            except if ``j - i > self.L``, in which case we use the additional labels ``(key, 1)``,
            ``(key, 2)``, ... to generate couplings over multiple unit cells.
        """
        if j < i:
            raise ValueError("j < i not allowed")
        keyL = keyR = key
        for k in range(i + 1, j):
            if (k - i) % self.L == 0:
                # necessary to extend key because keyL is already in use at this bond
                keyR = keyL + (k, opname, opname)  # same structure as for other standard keys
                # (i, op_i, op_str_right_of_i) e.g. in MultiCouplingTerms.add_to_graph
            k = k % self.L
            if not self.has_edge(k, keyL, keyR):
                self.add(k, keyL, keyR, opname, 1., check_op=check_op, skip_existing=skip_existing)
            keyL = keyR
        return keyL

    def add_missing_IdL_IdR(self):
        """Add missing identity ('Id') edges connecting ``'IdL'->'IdL' and ``'IdR'->'IdR'``.

        For ``bc='infinite'``, insert missing identities at *all* bonds.
        For ``bc='finite' | 'segment'`` only insert
        ``'IdL'->'IdL'`` to the left of the rightmost existing 'IdL' and
        ``'IdR'->'IdR'`` to the right of the leftmost existing 'IdR'.

        This function should be called *after* all other operators have been inserted.
        """
        if self.bc == 'infinite':
            max_IdL = self.L  # add identities for all sites
            min_IdR = 0
        else:
            max_IdL = max([0] + [i for i, s in enumerate(self.states[:-1]) if 'IdL' in s])
            min_IdR = min([self.L] + [i for i, s in enumerate(self.states[:-1]) if 'IdR' in s])
        for k in range(0, max_IdL):
            if not self.has_edge(k, 'IdL', 'IdL'):
                self.add(k, 'IdL', 'IdL', 'Id', 1.)
        for k in range(min_IdR, self.L):
            if not self.has_edge(k, 'IdR', 'IdR'):
                self.add(k, 'IdR', 'IdR', 'Id', 1.)
        # done

    def has_edge(self, i, keyL, keyR):
        """True if there is an edge from `keyL` on bond (i-1, i) to `keyR` on bond (i, i+1)."""
        return keyR in self.graph[i].get(keyL, [])

    def build_MPO(self, Ws_qtotal=None, leg0=None):
        """Build the MPO represented by the graph (`self`).

        Parameters
        ----------
        Ws_qtotal : None | (list of) charges
            The `qtotal` for each of the Ws to be generated., default (``None``) means 0 charge.
            A single qtotal holds for each site.
        leg0 : None | :class:`npc.LegCharge`
            The charges to be used for the very first leg (which is a gauge freedom).
            If ``None`` (default), use zeros.

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
        H = MPO.from_grids(self.sites, grids, self.bc, IdL, IdR, Ws_qtotal, leg0, self.max_range)
        return H

    def __repr__(self):
        return "<MPOGraph L={L:d}>".format(L=self.L)

    def __str__(self):
        """string showing the graph for debug output."""
        res = []
        for i in range(self.L):
            G = self.graph[i]
            strs = []
            for keyL in self.states[i]:
                s = [repr(keyL)]
                s.append("-" * len(s[-1]))
                D = G.get(keyL, [])
                for keyR in D:
                    s.append(repr(keyR) + ":")
                    for optuple in D[keyR]:
                        s.append("  " + repr(optuple))
                strs.append("\n".join(s))
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
        """translate the graph dictionaries into grids for the `Ws`."""
        states = self._ordered_states
        assert (states is not None)  # make sure that _set_ordered_states was called
        grids = []
        for i in range(self.L):
            stL, stR = states[i:i + 2]
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


class MPOEnvironment(MPSEnvironment):
    """Stores partial contractions of :math:`<bra|H|ket>` for an MPO `H`.

    The network for a contraction :math:`<bra|H|ket>` of an MPO `H` bewteen two MPS looks like::

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

    To avoid recalculations of the whole network e.g. in the DMRG sweeps,
    we store the contractions up to some site index in this class.
    For ``bc='finite','segment'``, the very left and right part ``LP[0]`` and
    ``RP[-1]`` are trivial and don't change in the DMRG algorithm,
    but for iDMRG (``bc='infinite'``) they are also updated
    (by inserting another unit cell to the left/right).

    The MPS `bra` and `ket` have to be in canonical form.
    All the environments are constructed without the singular values on the open bond.
    In other words, we contract left-canonical `A` to the left parts `LP`
    and right-canonical `B` to the right parts `RP`.


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
    init_LP : ``None`` | :class:`~tenpy.linalg.np_conserved.Array`
        Initial very left part ``LP``. If ``None``, build trivial one with
        :meth`init_LP`.
    init_RP : ``None`` | :class:`~tenpy.linalg.np_conserved.Array`
        Initial very right part ``RP``. If ``None``, build trivial one with
        :meth:`init_RP`.
    age_LP : int
        The number of physical sites involved into the contraction yielding `firstLP`.
    age_RP : int
        The number of physical sites involved into the contraction yielding `lastRP`.

    Attributes
    ----------
    H : :class:`~tenpy.networks.mpo.MPO`
        The MPO sandwiched between `bra` and `ket`.
    """
    def __init__(self, bra, H, ket, init_LP=None, init_RP=None, age_LP=0, age_RP=0):
        if ket is None:
            ket = bra
        if ket is not bra:
            ket._gauge_compatible_vL_vR(bra)  # ensure matching charges
        self.bra = bra
        self.ket = ket
        self.H = H
        self.L = L = lcm(lcm(bra.L, ket.L), H.L)
        self._finite = bra.finite
        self.dtype = np.find_common_type([bra.dtype, ket.dtype, H.dtype], [])
        self._LP = [None] * L
        self._RP = [None] * L
        self._LP_age = [None] * L
        self._RP_age = [None] * L
        if init_LP is None:
            init_LP = self.init_LP(0)
        self.set_LP(0, init_LP, age=age_LP)
        if init_RP is None:
            init_RP = self.init_RP(L - 1)
        self.set_RP(L - 1, init_RP, age=age_RP)
        self.test_sanity()

    def test_sanity(self):
        """Sanity check, raises ValueErrors, if something is wrong."""
        assert (self.bra.finite == self.ket.finite == self.H.finite == self._finite)
        # check that the network is contractable
        for b_s, H_s, k_s in zip(self.bra.sites, self.H.sites, self.ket.sites):
            b_s.leg.test_equal(k_s.leg)
            b_s.leg.test_equal(H_s.leg)
        assert any([LP is not None for LP in self._LP])
        assert any([RP is not None for RP in self._RP])

    def init_LP(self, i):
        """Build initial left part ``LP``.

        Parameters
        ----------
        i : int
            Build ``LP`` left of site `i`.

        Returns
        -------
        init_LP : :class:`~tenpy.linalg.np_conserved.Array`
            Identity contractible with the `vL` leg of ``.ket.get_B(i)``,
            multiplied with a unit vector nonzero in ``H.IdL[i]``,
            with labels ``'vR*', 'wR', 'vR'``.
        """
        init_LP = super().init_LP(i)
        leg_mpo = self.H.get_W(i).get_leg('wL').conj()
        IdL = self.H.get_IdL(i)
        init_LP = init_LP.add_leg(leg_mpo, IdL, axis=1, label='wR')
        return init_LP

    def init_RP(self, i):
        """Build initial right part ``RP`` for an MPS/MPOEnvironment.

        Parameters
        ----------
        i : int
            Build ``RP`` right of site `i`.

        Returns
        -------
        init_RP : :class:`~tenpy.linalg.np_conserved.Array`
            Identity contractible with the `vR` leg of ``self.get_B(i)``,
            multiplied with a unit vector nonzero in ``H.IdR[i]``,
            with labels ``'vL*', 'wL', 'vL'``.
        """
        init_RP = super().init_RP(i)
        leg_mpo = self.H.get_W(i).get_leg('wR').conj()
        IdR = self.H.get_IdR(i)
        init_RP = init_RP.add_leg(leg_mpo, IdR, axis=1, label='wL')
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
            Wheter to store the calculated `LP` in `self` (``True``) or discard them (``False``).

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
            The returned `RP` will contain the contraction *strictly* rigth of site `i`.
        store : bool
            Wheter to store the calculated `RP` in `self` (``True``) or discard them (``False``).

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
        ``<bra|H|ket> / (norm(|bra>)*norm(|ket>))``,
        i.e. if `bra` is `ket` and normalized, the total energy.
        For this purpose, this function contracts
        ``get_LP(i0+1, store=False)`` and ``get_RP(i0, store=False)``.

        Parameters
        ----------
        i0 : int
            Site index.
        """
        # same as MPSEnvironment.full_contraction, but also contract 'wL' with 'wR'
        if self.ket.finite and i0 + 1 == self.L:
            # special case to handle `_to_valid_index` correctly:
            # get_LP(L) is not valid for finite b.c, so we use need to calculate it explicitly.
            LP = self.get_LP(i0, store=False)
            LP = self._contract_LP(i0, LP)
        else:
            LP = self.get_LP(i0 + 1, store=False)
        # multiply with `S`: a bit of a hack: use 'private' MPS._scale_axis_B
        S_bra = self.bra.get_SR(i0).conj()
        LP = self.bra._scale_axis_B(LP, S_bra, form_diff=1., axis_B='vR*', cutoff=0.)
        # cutoff is not used for form_diff = 1
        S_ket = self.ket.get_SR(i0)
        LP = self.bra._scale_axis_B(LP, S_ket, form_diff=1., axis_B='vR', cutoff=0.)
        RP = self.get_RP(i0, store=False)
        return npc.inner(LP, RP, axes=[['vR*', 'wR', 'vR'], ['vL*', 'wL', 'vL']], do_conj=False)

    def _contract_LP(self, i, LP):
        """Contract LP with the tensors on site `i` to form ``self._LP[i+1]``"""
        # same as MPSEnvironment._contract_LP, but also contract with `H.get_W(i)`
        LP = npc.tensordot(LP, self.ket.get_B(i, form='A'), axes=('vR', 'vL'))
        LP = npc.tensordot(self.H.get_W(i), LP, axes=(['p*', 'wL'], ['p', 'wR']))
        LP = npc.tensordot(self.bra.get_B(i, form='A').conj(),
                           LP,
                           axes=(['p*', 'vL*'], ['p', 'vR*']))
        return LP  # labels 'vR*', 'wR', 'vR'

    def _contract_RP(self, i, RP):
        """Contract RP with the tensors on site `i` to form ``self._RP[i-1]``"""
        # same as MPSEnvironment._contract_RP, but also contract with `H.get_W(i)`
        RP = npc.tensordot(self.ket.get_B(i, form='B'), RP, axes=('vR', 'vL'))
        RP = npc.tensordot(self.H.get_W(i), RP, axes=(['p*', 'wR'], ['p', 'wL']))
        RP = npc.tensordot(self.bra.get_B(i, form='B').conj(),
                           RP,
                           axes=(['p*', 'vR*'], ['p', 'vL*']))
        return RP  # labels 'vL', 'wL', 'vL*'


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
            raise ValueError("finite MPO with len of first bond != 1")
        q = chinfo.make_valid()
        leg0 = npc.LegCharge.from_qflat(chinfo, [q], qconj=+1)
    legs = [leg0]
    for i, gr in enumerate(grids):
        gr_legs = [legs[-1], None]
        gr_legs = npc.detect_grid_outer_legcharge(gr,
                                                  gr_legs,
                                                  qtotal=Ws_qtotal[i],
                                                  qconj=-1,
                                                  bunch=False)
        legs.append(gr_legs[1].conj())
    return legs


def _calc_grid_legs_infinite(chinfo, grids, Ws_qtotal, leg0, IdL_0):
    """calculate LegCharges from `grids` for an iMPO.

    The hard case. Initially, we do not know all charges of the first leg; and they have to
    be consistent with the final leg.

    The way this workso: gauge 'IdL' on the very left leg to 0,
    then gradually calculate the charges by going along the edges of the graph (maybe also over the iMPO boundary).
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
            QsL, QsR = charges[i:i + 2]
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
                        raise ValueError("incompatible charges while creating the MPO")
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

    The goal is to ensure that standard TeNPy MPOs yield an upper-right W for the MPO.
    """
    if isinstance(key, tuple):
        return key
    if isinstance(key, str):
        if key == 'IdL':  # should be first
            return (-2, )
        if key == 'IdR':  # should be last
            return (np.inf, )
    # fallback: compare strings
        return (-1, key)
    return (-1, str(key))
