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

from __future__ import division
import itertools

from ..linalg import np_conserved as npc
from ..tools.string import vert_join
from .mps import MPS as _MPS   # only for MPS._valid_bc

__all__ = ['MPO', 'MPOGraph']


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
        The matrices of the MPO. Labels are ``wL, wR, p, p*``
    _valid_bc : tuple of str
        Valid boundary conditions. The same as for an MPS.
    """

    _valid_bc = _MPS._valid_bc   # same valid boundary conditions as an MPS.

    def __init__(self, sites, Ws, bc='finite', IdL=None, IdR=None):
        self.sites = list(sites)
        self.chinfo = self.sites[0].leg.chinfo
        self._W = list(Ws)
        if IdL is None:
            self.IdL = [None]*(self.L+1)
        else:
            self.IdL = list(IdL)
        if IdR is None:
            self.IdR = [None]*(self.L+1)
        else:
            self.IdR = list(IdR)
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
                W2 = self.get_W(i+1)
                W.get_leg('wR').test_contractible(W2.get_leg('wL'))
        if self.bc == 'finite':
            assert(self._W[0].get_leg('wL').ind_len == 1)
            assert(self._W[-1].get_leg('wR').ind_len == 1)
        if not (len(self.IdL) == len(self.IdR) == self.L+1):
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
        i = self._valid_index(i)
        return self.IdL[i]

    def get_IdR(self, i):
        """Return index of `IdL` at bond to the *right* of site `i`.

        May be ``None``."""
        i = self._valid_index(i)
        return self.IdR[i+1]

    def _to_valid_index(self, i):
        """Make sure `i` is a valid index (depending on `self.bc`)."""
        if not self.finite:
            return i % self.L
        if i < 0:
            i += self.L
        if i >= self.L or i < 0:
            raise ValueError("i = {0:d} out of bounds for finite MPS".format(i))
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


    .. todo ::
        might be useful to add a "cleanup" function which removes operators cancelling each other
        and/or unused states. Or better use a 'compress' of the MPO?
    """
    def __init__(self, sites, bc='finite'):
        self.sites = list(sites)
        self.chinfo = self.sites[0].leg.chinfo
        self.bc = bc
        # empty graph
        self.states = [set() for _ in xrange(self.L+1)]
        self.graph = [{} for _ in xrange(self.L)]
        self._ordered_states = None
        self._grids = None
        self._grid_legs = None
        self.test_sanity()

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        assert len(self.graph) == self.L
        assert len(self.states) == self.L+1
        if self.bc not in MPO._valid_bc:
            raise ValueError("invalid MPO boundary conditions: " + repr(self.bc))
        for i, site in enumerate(self.sites):
            if site.leg.chinfo != self.chinfo:
                raise ValueError("invalid ChargeInfo for site {i:d}".format(i=i))
            stL, stR = self.states[i:i+2]
            # check graph
            gr = self.graph[i]
            for keyL in gr:
                assert keyL in stL
                for keyR in gr[keyL]:
                    assert keyR in stR
                    for opname, strength in gr[keyL][keyR]:
                        assert opname in site.opnames
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
            if opname not in self.sites[i].opnames:
                raise ValueError("operator {0!r} not existent on site {1:d}".format(opname, i))
        G = self.graph[i]
        if keyL not in self.states[i]:
            self.states[i].add(keyL)
        if keyR not in self.states[i+1]:
            self.states[i+1].add(keyR)
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
        for k in range(i+1, j):
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
        for i in xrange(self.L):
            legs = [self._grid_legs[i], self._grid_legs[i+1].conj()]
            W = npc.grid_outer(self._grids[i], legs, W_qtotal)
            W.set_leg_labels(['wL', 'wR', 'p', 'p*'])
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
                s.append("-"*len(s[-1]))
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
        assert(states is not None)   # make sure that _set_ordered_states was called
        grids = []
        for i in range(self.L):
            stL, stR = states[i:i+2]
            graph = self.graph[i]  # ``{keyL: {keyR: [(opname, strength)]}}``
            grid = [None]*len(stL)
            for sL, a in stL.iteritems():
                row = [None] * len(stR)
                for sR, lst in graph[sL].iteritems():
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
        assert(grids is not None)  # make sure _grid_insert_ops was called
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
                raise ValueError("finite MPO with first bond > 1: how????")
            q = self.chinfo.make_valid()
            leg0 = npc.LegCharge.from_qflat(self.chinfo, [q], qconj=+1)
        legs = [leg0]
        for i, gr in enumerate(grids):
            gr_legs = [legs[-1], None]
            gr_legs = npc.detect_grid_outer_legcharge(gr, gr_legs, qtotal=W_qtotal, qconj=-1,
                                                      bunch=False)
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
        assert(states is not None)  # make sure self._set_ordered_states() was called
        charges = [{} for _ in xrange(self.L)]
        charges.append(charges[0])  # the *same* dictionary is shared for 0 and -1.
        charges[0]['IdL'] = self.chinfo.make_valid(None)  # default charge = 0.
        chis = [len(s) for s in self.states]
        for _ in xrange(1000*self.L):  # I don't expect interactions with larger range than that...
            for i in xrange(self.L):
                chL, chR = charges[i:i+2]
                stL, stR = states[i:i+2]
                graph = self.graph[i]
                grid = self._grids[i]
                for keyL, qL in chL.iteritems():
                    for keyR in graph[keyL]:
                        # calculate charge qR from the entry of the grid
                        op = grid[stL[keyL]][stR[keyR]]
                        assert(op is not None)
                        qR = chinfo.make_valid(qL + op.qtotal - W_qtotal)
                        if keyR not in chR:
                            chR[keyR] = qR
                        elif any(chR[keyR] != qR):
                            raise ValueError("incompatible charges while creating the MPO")
            if all([len(qs) == chi for qs, chi in itertools.izip(charges, chis)]):
                break
        else:  # no `break` in the for loop, i.e. we are unable to determine all grid legcharges.
            # this should not happen (if we have no bugs), but who knows ^_^
            assert(False)  # maybe some unconnected parts in the graph?
        # finally generate LegCharge from the dictionaries
        self._grid_legs = []
        for qs, st in itertools.izip(charges, states):
            qfl = [None]*len(qs)
            for key, q in qs.iteritems():
                qfl[st[key]] = q
            leg = npc.LegCharge.from_qflat(chinfo, qfl, qconj=+1)
            self._grid_legs.append(leg)
