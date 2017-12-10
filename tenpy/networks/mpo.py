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

.. todo ::
    transfermatrix for MPO
"""


import itertools
import numpy as np
from ..linalg import np_conserved as npc
from ..tools.string import vert_join
from .mps import MPS as _MPS  # only for MPS._valid_bc
from .mps import MPSEnvironment

__all__ = ['MPO', 'MPOGraph', 'MPOEnvironment']


class MPO(object):
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
        Indices on the bonds, which correpond to 'only identities to the left'.
        ``None`` for bonds where it is not set.
    IdR : list of {int | None}
        Indices on the bonds, which correpond to 'only identities to the right'.
        ``None`` for bonds where it is not set.
    _W : list of :class:`~tenpy.linalg.np_conserved.Array`
        The matrices of the MPO. Labels are ``'wL', 'wR', 'p', 'p*'``.
    _valid_bc : tuple of str
        Valid boundary conditions. The same as for an MPS.
    """

    _valid_bc = _MPS._valid_bc  # same valid boundary conditions as an MPS.

    def __init__(self, sites, Ws, bc='finite', IdL=None, IdR=None):
        self.sites = list(sites)
        self.chinfo = self.sites[0].leg.chinfo
        self.dtype = dtype = np.find_common_type([W.dtype for W in Ws], [])
        self._W = [W.astype(dtype, copy=True) for W in Ws]
        if IdL is None:
            self.IdL = [None] * (self.L + 1)
        else:
            try:
                self.IdL = list(IdL)
            except TypeError:
                self.IdL = [IdL] * (self.L + 1)
        if IdR is None:
            self.IdR = [None] * (self.L + 1)
        else:
            try:
                self.IdR = list(IdR)
            except TypeError:
                self.IdR = [IdR] * (self.L + 1)
        self.bc = bc
        self.test_sanity()

    def test_sanity(self):
        """Sanity check. Raises Errors if something is wrong."""
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
        if self.bc == 'finite':
            assert (self._W[0].get_leg('wL').ind_len == 1)
            assert (self._W[-1].get_leg('wR').ind_len == 1)
        if not (len(self.IdL) == len(self.IdR) == self.L + 1):
            raise ValueError("wrong len of `IdL`/`IdR`")

    @property
    def L(self):
        """Number of physical sites. For an iMPO the len of the MPO unit cell."""
        return len(self.sites)

    @property
    def dim(self):
        """List of local physical dimensions."""
        return [site.dim for site in self.sites]

    @property
    def finite(self):
        "Distinguish MPO (``True; bc='finite', 'segment'`` ) vs. iMPO (``False; bc='infinite'``)"
        assert (self.bc in self._valid_bc)
        return self.bc != 'infinite'

    @property
    def chi(self):
        """Dimensions of the (nontrivial) virtual bonds."""
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

        May be ``None``."""
        i = self._to_valid_index(i)
        return self.IdL[i]

    def get_IdR(self, i):
        """Return index of `IdR` at bond to the *right* of site `i`.

        May be ``None``."""
        i = self._to_valid_index(i)
        return self.IdR[i + 1]

    def _to_valid_index(self, i):
        """Make sure `i` is a valid index (depending on `self.bc`)."""
        if not self.finite:
            return i % self.L
        if i < 0:
            i += self.L
        if i >= self.L or i < 0:
            raise ValueError("i = {0:d} out of bounds for finite MPO".format(i))
        return i


class MPOGraph(object):
    """Representation of an MPO by a graph, based on a 'finite state machine'.

    This representation is used for building H_MPO from the interactions.
    The idea is to view the MPO as a kind of 'finite state machine'.
    The 'states' or **keys** of this finite state machine life on the MPO bonds *between* the `Ws`,
    and are the **vertices** of the graph. They label the indices of the virtul bonds
    of the MPOS (i.e. the indices on legs ``wL`` and ``wR``).
    They can be anything hash-able like a ``str``, ``int`` or a tuple of them.

    The **edges** of the graph are the entries ``W[wL, wR]``, which itself are onsite operators
    on the local Hilbert space. The entry ``W[wL, wR]`` connects the vertex ``wL`` on bond
    ``(i-1, i)`` with the vertex ``wR`` on bond ``(i, i+1)``.

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

    Attributes
    ----------
    L
    sites : list of :class:`~tenpy.models.lattice.Site`
        Defines the local Hilbert space for each site.
    chinfo : :class:`~tenpy.linalg.np_conserved.ChargeInfo`
        The nature of the charge.
    bc : {'finite', 'infinite'}
        MPO boundary conditions.
    states : list of set of keys
        ``states[i]`` gives the possible keys at the virtual bond ``(i-1, i)`` of the MPO.
    graph : list of dict of dict of list of tuples
        For each site `i` a dictionary ``{keyL: {keyR: [(opname, strength)]}}`` with
        ``keyL in vertices[i]`` and ``keyR in vertices[i+1]``.
    _grid_legs : None | list of LegCharge
        The charges for the MPO
    """

    def __init__(self, sites, bc='finite'):
        self.sites = list(sites)
        self.chinfo = self.sites[0].leg.chinfo
        self.bc = bc
        # empty graph
        self.states = [set() for _ in range(self.L + 1)]
        self.graph = [{} for _ in range(self.L)]
        self._ordered_states = None
        self._grids = None
        self._grid_legs = None
        self.test_sanity()

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
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
        """Number of physical sites. For an iMPS the length of the unit cell."""
        return len(self.sites)

    def add(self, i, keyL, keyR, opname, strength, check_op=True):
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
            Wheter to check that 'opname' exists on the given `site`.
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
            D[keyR].append((opname, strength))

    def add_string(self, i, j, key, opname='Id', check_op=True):
        """Insert a bunch of edges for an 'operator string' into the graph.

        Terms like :math:`S^z_i S^z_j` actually stand for
        :math:`S^z_i \otimes \prod_{i < k < j} \mathbb{1}_k \otimes S^z_j`.
        This function adds the :math:`\mathbb{1}` terms.

        Parameters
        ----------
        i, j: int
            An edge is inserted on all bonds between `i` and `j`.
        key: hashable
            The state at bond (i-1, i) to connect from and on bond (j-1, j) to connect to.
            Also used for the intermediate states.
            No operator is inserted on a site `i < k < j` if ``has_edge(k, key, key)``.
        opname : str
            Name of the operator to be used for the string.
            Useful for the Jordan-Wigner transformation to fermions.
        """
        if j < i:
            if self.bc != 'infinite':
                raise ValueError("j < i not allowed for finite boundary conditions")
            j += self.L
        for k in range(i + 1, j):
            k = k % self.L
            if not self.has_edge(k, key, key):
                self.add(k, key, key, opname, 1., check_op=check_op)

    def add_missing_IdL_IdR(self):
        """Add missing identity ('Id') edges connecting ``'IdL'->'IdL' and ``'IdR'->'IdR'``.

        For ``bc='infinite'``, insert missing identities at *all* bonds.
        For ``bc='finite' | 'segment'`` only insert
        ``'IdL'->'IdL'`` to the left of the rightmost existing 'IdL' and
        ``'IdR'->'IdR'`` to the right of the leftmost existing 'IdR'.

        This function should be called *after* all other operators have been inserted.
        """
        if self.bc == 'infinite':
            # infinite boundary connections: add for all sites
            self.add_string(-1, self.L, 'IdL')
            self.add_string(-1, self.L, 'IdR')
        else:
            max_IdL = max([0] + [i for i, s in enumerate(self.states[:-1]) if 'IdL' in s])
            self.add_string(-1, max_IdL, 'IdL', 'Id', 1.)
            min_IdR = min([self.L] + [i for i, s in enumerate(self.states[:-1]) if 'IdR' in s])
            self.add_string(min_IdR - 1, self.L, 'IdR', 'Id', 1.)
        # done

    def has_edge(self, i, keyL, keyR):
        """True if there is an edge from `keyL` on bond (i-1, i) to `keyR` on bond (i, i+1)."""
        return keyR in self.graph[i].get(keyL, [])

    def build_MPO(self, W_qtotal=None, leg0=None):
        """Build the MPO represented by the graph (`self`).

        Parameters
        ----------
        W_qtotal : None | charge
            A single qtotal used for *each* of the individual `W`.
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
        self._build_grids()
        self._calc_grid_legs(W_qtotal, leg0)
        # now build the `W` from the grid
        Ws = []
        for i in range(self.L):
            legs = [self._grid_legs[i], self._grid_legs[i + 1].conj()]
            W = npc.grid_outer(self._grids[i], legs, W_qtotal)
            W.iset_leg_labels(['wL', 'wR', 'p', 'p*'])
            Ws.append(W)
        IdL = [s.get('IdL', None) for s in self._ordered_states]
        IdR = [s.get('IdR', None) for s in self._ordered_states]
        return MPO(self.sites, Ws, self.bc, IdL, IdR)

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
            # try 'IdL'=0 and 'IdR'=-1.
            if 'IdL' in s:
                offset = 1
                d['IdL'] = 0
            else:
                offset = 0
            for i, key in enumerate(sorted(s - {'IdL', 'IdR'}, key=str)):
                d[key] = i + offset
            if 'IdR' in s:
                d['IdR'] = len(s) - 1
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
            for sL, a in stL.items():
                row = [None] * len(stR)
                for sR, lst in graph[sL].items():
                    b = stR[sR]
                    row[b] = lst
                grid[a] = row
            grid = self._grid_insert_ops(grid, i)  # replace `lst` by the actual operators
            grids.append(grid)
        self._grids = grids

    def _grid_insert_ops(self, grid, i):
        """Replaces ``[('opname', strength)]`` in a grid representing W[i] with actual Arrays.

        Parameters
        ----------
        grid : list of list of entries
            Entries may be ``None`` or like ``[(opname, strength)]``. Modified.
        i : int
            MPS index at which the grid is define

        Returns
        -------
        grid : list of list of {None | :class:`~tenpy.linalg.np_conserved.Array`}
            The grid W[i] with operators
        """
        site = self.sites[i]
        for row in grid:
            for i, entry in enumerate(row):
                if entry is None:
                    continue
                opname, strength = entry[0]
                res = strength * site.get_op(opname)
                for opname, strength in entry[1:]:
                    res = res + strength * site.get_op(opname)
                row[i] = res  # replace entry
        return grid

    def _calc_grid_legs(self, W_qtotal, leg0):
        """calculate LegCharges for the grids from self.grid"""
        grids = self._grids
        assert (grids is not None)  # make sure _grid_insert_ops was called
        if self.bc != 'infinite':
            self._calc_grid_legs_finite(grids, W_qtotal, leg0)
        else:
            self._calc_grid_legs_infinite(grids, W_qtotal, leg0)

    def _calc_grid_legs_finite(self, grids, W_qtotal, leg0):
        """calculate LegCharges from `self._grid` for a finite MPO.

        This is the easier case. We just gauge the very first leg to the left to zeros,
        then all other charges (hopefully) follow from the entries of the grid."""
        if leg0 is None:
            if len(grids[0]) != 1:
                raise ValueError("finite MPO with len of first bond != 1: how????")
            q = self.chinfo.make_valid()
            leg0 = npc.LegCharge.from_qflat(self.chinfo, [q], qconj=+1)
        legs = [leg0]
        for i, gr in enumerate(grids):
            gr_legs = [legs[-1], None]
            gr_legs = npc.detect_grid_outer_legcharge(
                gr, gr_legs, qtotal=W_qtotal, qconj=-1, bunch=False)
            legs.append(gr_legs[1].conj())
        self._grid_legs = legs

    def _calc_grid_legs_infinite(self, grids, W_qtotal, leg0):
        """calculate LegCharges from `self._grid` for an iMPO.

        The hard case. Initially, we do not know all charges of the first leg; and they have to
        be consistent with the final leg.

        The way to go: gauge 'IdL' on the very left leg to 0, then gradually calculate the charges
        by going along the edges of the graph (maybe also over the iMPO boundary).
        """
        if leg0 is not None:
            # have charges of first leg: simple case, can use the *_finite version.
            self._calc_grid_legs_finite(self, grids, W_qtotal, leg0)
            # just make sure, that everything is self consistent over the MPS boundary.
            self.legs[-1].test_contractible(self.legs[0])
            return
        chinfo = self.chinfo
        W_qtotal = chinfo.make_valid(W_qtotal)
        states = self._ordered_states
        assert (states is not None)  # make sure self._set_ordered_states() was called
        charges = [{} for _ in range(self.L)]
        charges.append(charges[0])  # the *same* dictionary is shared for 0 and -1.
        charges[0]['IdL'] = self.chinfo.make_valid(None)  # default charge = 0.
        chis = [len(s) for s in self.states]
        for _ in range(
                1000 * self.L):  # I don't expect interactions with larger range than that...
            for i in range(self.L):
                chL, chR = charges[i:i + 2]
                stL, stR = states[i:i + 2]
                graph = self.graph[i]
                grid = self._grids[i]
                for keyL, qL in chL.copy().items():  # copy: for L=1 infinite, chL is chR
                    for keyR in graph[keyL]:
                        # calculate charge qR from the entry of the grid
                        op = grid[stL[keyL]][stR[keyR]]
                        assert (op is not None)
                        qR = chinfo.make_valid(qL + op.qtotal - W_qtotal)
                        if keyR not in chR:
                            chR[keyR] = qR
                        elif any(chR[keyR] != qR):
                            raise ValueError("incompatible charges while creating the MPO")
            if all([len(qs) == chi for qs, chi in zip(charges, chis)]):
                break
        else:  # no `break` in the for loop, i.e. we are unable to determine all grid legcharges.
            # this should not happen (if we have no bugs), but who knows ^_^
            assert (False)  # maybe some unconnected parts in the graph?
        # finally generate LegCharge from the dictionaries
        self._grid_legs = []
        for qs, st in zip(charges, states):
            qfl = [None] * len(qs)
            for key, q in qs.items():
                qfl[st[key]] = q
            leg = npc.LegCharge.from_qflat(chinfo, qfl, qconj=+1)
            self._grid_legs.append(leg)


class MPOEnvironment(MPSEnvironment):
    """Stores partial contractions of :math:`<bra|H|ket>` for an MPO `H`.

    The network for a contraction :math:`<bra|H|ket>` of an MPO `H` bewteen two MPS looks like::

        |     .------>-M[0]-->-M[1]-->-M[2]-->- ...  ->--.
        |     |        |       |       |                 |
        |     |        ^       ^       ^                 |
        |     |        |       |       |                 |
        |     LP[0] ->-W[0]-->-W[1]-->-W[2]-->- ...  ->- Rp[-1]
        |     |        |       |       |                 |
        |     |        ^       ^       ^                 |
        |     |        |       |       |                 |
        |     .------>-N[0]*->-N[1]*->-N[2]*->- ...  ->--.

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
    ket : :class:`~tenpy.networks.mpo.MPO`
        The MPS on which `H` acts. May be identical with `bra`.
    firstLP : ``None`` | :class:`~tenpy.linalg.np_conserved.Array`
        Initial very left part. If ``None``, build trivial one.
    rightRP : ``None`` | :class:`~tenpy.linalg.np_conserved.Array`
        Initial very right part. If ``None``, build trivial one.
    age_LP : int
        The number of physical sites involved into the contraction yielding `firstLP`.
    age_RP : int
        The number of physical sites involved into the contraction yielding `lastRP`.

    Attributes
    ----------
    H : :class:`~tenpy.networks.mpo.MPO`
        The MPO sandwiched between `bra` and `ket`.
    """

    def __init__(self, bra, H, ket, firstLP=None, lastRP=None, age_LP=0, age_RP=0):
        if ket is None:
            ket = bra
        self.bra = bra
        self.ket = ket
        self.H = H
        self.L = L = bra.L
        self.finite = bra.finite
        self.dtype = np.find_common_type([bra.dtype, ket.dtype, H.dtype], [])
        self._LP = [None] * L
        self._RP = [None] * L
        self._LP_age = [None] * L
        self._RP_age = [None] * L
        if firstLP is None:
            # Build trivial verly first LP
            leg_bra = bra.get_B(0).get_leg('vL')
            leg_mpo = H.get_W(0).get_leg('wL').conj()
            leg_ket = ket.get_B(0).get_leg('vL').conj()
            leg_ket.test_contractible(leg_bra)
            firstLP = npc.zeros([leg_bra, leg_mpo, leg_ket], dtype=self.dtype)
            # should work for both finite and segment bc
            firstLP[:, H.IdL[0], :] = npc.diag(1., leg_bra, dtype=self.dtype)
            firstLP.iset_leg_labels(['vR*', 'wR', 'vR'])
        self.set_LP(0, firstLP, age=age_LP)
        if lastRP is None:
            # Build trivial verly last RP
            leg_bra = bra.get_B(L - 1).get_leg('vR')
            leg_mpo = H.get_W(L - 1).get_leg('wR').conj()
            leg_ket = ket.get_B(L - 1).get_leg('vR').conj()
            leg_ket.test_contractible(leg_bra)
            lastRP = npc.zeros([leg_bra, leg_mpo, leg_ket], dtype=self.dtype)
            lastRP[:, H.IdR[L], :] = npc.diag(1., leg_bra, dtype=self.dtype)
            lastRP.iset_leg_labels(['vL*', 'wL', 'vL'])
        self.set_RP(L - 1, lastRP, age=age_RP)
        self.test_sanity()

    def test_sanity(self):
        assert (self.bra.L == self.ket.L == self.H.L)
        assert (self.bra.finite == self.ket.finite == self.H.finite)
        # check that the network is contractable
        for b_s, H_s, k_s in zip(self.bra.sites, self.H.sites, self.ket.sites):
            b_s.leg.test_equal(k_s.leg)
            b_s.leg.test_equal(H_s.leg)
        assert any([LP is not None for LP in self._LP])
        assert any([RP is not None for RP in self._RP])

    def get_LP(self, i, store=True):
        """Calculate LP at given site from nearest available one (including `i`).

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
            with labels ``'vR*', 'vR'`` for `bra`, `ket`.
        """
        # actually same as MPSEnvironment, just updated the labels in the doc string.
        return super(MPOEnvironment, self).get_LP(i, store=True)

    def get_RP(self, i, store=True):
        """Calculate RP at given site from nearest available one (including `i`).

        Parameters
        ----------
        i : int
            The returned `RP` will contain the contraction *strictly* rigth of site `i`.
        store : bool
            Wheter to store the calculated `RP` in `self` (``True``) or discard them (``False``).

        Returns
        -------
        RP_i : :class:`~tenpy.linalg.np_conserved.Array`
            Contraction of everything left of site `i`,
            with labels ``'vL*', 'wL', 'vL'`` for `bra`, `H`, `ket`.
        """
        # actually same as MPSEnvironment, just updated the labels in the doc string.
        return super(MPOEnvironment, self).get_RP(i, store=True)

    def full_contraction(self, i0):
        """Calculate the energy by a full contraction of the network.

        The full contraction of the environments gives the value ``<bra|H|ket>``,
        i.e. if `bra` is `ket`, the total energy. For this purpose, this function contracts
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
        LP = npc.tensordot(
            self.bra.get_B(i, form='A').conj(), LP, axes=(['p*', 'vL*'], ['p', 'vR*']))
        return LP  # labels 'vR*', 'wR', 'vR'

    def _contract_RP(self, i, RP):
        """Contract RP with the tensors on site `i` to form ``self._RP[i-1]``"""
        # same as MPSEnvironment._contract_RP, but also contract with `H.get_W(i)`
        RP = npc.tensordot(self.ket.get_B(i, form='B'), RP, axes=('vR', 'vL'))
        RP = npc.tensordot(self.H.get_W(i), RP, axes=(['p*', 'wR'], ['p', 'wL']))
        RP = npc.tensordot(
            self.bra.get_B(i, form='B').conj(), RP, axes=(['p*', 'vR*'], ['p', 'vL*']))
        return RP  # labels 'vL', 'wL', 'vL*'
