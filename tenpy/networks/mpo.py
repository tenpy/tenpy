"""Matrix product operator (MPO).

An MPO is the generalization of an MPS to operators. Graphically::

          ^        ^         ^
          |        |         |
     ->- Ws[0] ->- Ws[1] ->- Ws[2] ->- ...
          |        |         |
          ^        ^         ^

We use the following label convention (arrows indicate `qconj`)::

           p*
           ^
           |
    wL ->- W ->- wR
           |
           ^
           p


.. todo ::
    implement & test, test, test !!!
"""

from __future__ import division
import itertools

from ..linalg import np_conserved as npc
from ..tools.string import vert_join

__all__ = ['MPO', 'MPOGraph']


class MPO(object):
    """Matrix product operator, finite (MPO) or infinite (iMPO).

    Similar as an MPS, but each `matrix` has two physical legs ``p, p*`` instead of just one,
    i.e. the entries are local operators.
    An MPO can be applied to an MPS, which increases the `chi` of the MPS by a *factor* of the
    MPO bond dimension.

    In general, you can view the MPO as an MPS with larger physical space and bring it into
    canoncial form. However, unlike for an MPS, this doesn't simplify calculations.

    However, if an MPO describes a sum of local terms (as most Hamiltonians are),
    some bond indices correspond to 'only identities to the left/right'.
    We store these indices in `idL` and `idR` (if there are such indices).

    Parameters
    ----------
    sites : list of :class:`~tenpy.models.lattice.Site`
        Defines the local Hilbert space for each site.
    Ws : list of :class:`npc.Array`
        The matrices of the MPO. Should have labels ``wL, wR, p, p*``.
    idL : None | list of int
        Indices on the bonds, which correpond to 'only identities to the left'.
    idR : None | list of int
        Indices on the bonds, which correpond to 'only identities to the right'.

    Attributes
    ----------
    L : int
        ``len(sites)``. For an iMPS, this is the number of sites in the MPS unit cell.
    chinfo : class:`npc.ChargeInfo`
        The nature of the charge.
    sites : list of :class:`~tenpy.models.lattice.Site`
        Defines the local Hilbert space for each site.
    Ws : list of :class:`npc.Array``
        The matrices of the MPO. Labels are ``wL, wR, p, p*``
    bc : {'finite', 'infinite'}
        Boundary conditions.
    idL : list of {int | None}
        Indices on the bonds, which correpond to 'only identities to the left'.
        ``None`` for bonds where it is not set.
    idR : list of {int | None}
        Indices on the bonds, which correpond to 'only identities to the right'.
        ``None`` for bonds where it is not set.

    .. todo :
        how to best document the types as npc.Array ?
    """

    #: valid boundary conditions.
    _valid_bc = ('infinite', 'finite')

    def __init__(self, sites, Ws, bc='finite', idL=None, idR=None):
        self.sites = list(sites)
        self.chinfo = self.sites[0].leg.chinfo
        self.L = len(self.sites)
        self.Ws = list(Ws)
        self._set_chi_from_Ws()
        if idL is None:
            self.idL = [None]*len(self.L+1)
        else:
            self.idL = list(idL)
        if idR is None:
            self.idR = [None]*len(self.L+1)
        else:
            self.idR = list(idR)
        self.bc = bc
        self.test_sanity()

    def test_sanity(self):
        """Sanity check. Raises Errors if something is wrong."""
        assert self.L == len(self.sites)
        if self.bc not in self._valid_bc:
            raise ValueError("invalid MPO boundary conditions: " + repr(self.bc))
        for i in range(self.L):
            S = self.sites[i]
            W = self.Ws[i]
            S.leg.test_equal(W.get_leg('p'))
            S.leg.test_contractible(W.get_leg('p*'))
            if self.bc == 'infinite' or i < self.L:
                W2 = self.get_W(i+1)
                W.get_leg('wR').test_contractible(W2.get_leg('wL'))
        if self.bc == 'finite':
            assert(self.Ws[0].get_leg('wL').ind_len == 1)
            assert(self.Ws[-1].get_leg('wR').ind_len == 1)
        if not (len(self.idL) == len(self.idR) == self.L+1):
                raise ValueError("wrong len of `idL`/`idR`")

    def get_W(self, i):
        """return `W` at site `i`."""
        return self.W[i % self.L]

    def _set_chi_from_Ws(self):
        """set ``self.chi` from the ``Ws``."""
        chis = [W.get_leg('wL').ind_len for W in self.Ws]
        chis.append(self.Ws[-1].get_leg('wR').ind_len)
        self.chi = chis


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

    The keys ``'idR'`` (for 'idenity left') and ``'idR'`` (for 'identity right') are reserved to
    represent only ``'Id'`` (=identity) operators to the left and right of the bond, respectively.

    Parameters
    ----------
    sites : list of :class:`~tenpy.models.lattice.Site`
        Local sites of the Hilbert space.
    bc : {'finite', 'infinite'}
        MPO boundary conditions.
    add_id : bool
        Wheter to add identities to generate the states 'idL' and 'idR' on each site.

    Attributes
    ----------
    L
    sites : list of :class:`~tenpy.models.lattice.Site`
        Defines the local Hilbert space for each site.
    chinfo : :class:`npc.ChargeInfo`
        The nature of the charge.
    bc : {'finite', 'infinite'}
        MPO boundary conditions.
    states : list of set of keys
        ``edges[i]`` gives the possible keys at the virtual bond ``(i-1, i)`` of the MPO.
    graph : list of dict of dict of list of tuples
        For each site `i` a dictionary ``{keyL: {keyR: [(opname, strength)]}}`` with
        ``keyL in vertices[i]`` and ``keyR in vertices[i+1]``.
    _grid_legs : None | list of LegCharge
        The charges for the MPO
    """
    def __init__(self, sites, bc='finite', add_id=True):
        self.sites = list(sites)
        self.chinfo = self.sites[0].leg.chinfo
        self.bc = bc
        # empty graph
        self.states = [set() for _ in xrange(self.L+1)]
        self.graph = [{} for _ in xrange(self.L)]
        # add usual entries for 'idL' and 'idR'
        if add_id:
            for i in range(self.L):
                self.add(i, 'idL', 'idL', 'Id', 1., check_op=False)
                self.add(i, 'idR', 'idR', 'Id', 1., check_op=False)
        self.test_sanity()
        self._ordered_states = None
        self._grids = None
        self._grid_legs = None

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        assert len(self.graph) == self.L
        assert len(self.states) == self.L+1
        if self.bc not in MPO._valid_bc:
            raise ValueError("invalid MPO boundary conditions: " + repr(self.bc))
        # TODO: much more checks
        for i, site in enumerate(self.sites):
            if site.leg.chinfo != self.chinfo:
                raise ValueError("invalid ChargeInfo for site {i:d}".format(i=i))

    @property
    def L(self):
        """number of physical sites. For an iMPS the length of the unit cell."""
        return len(self.sites)

    def add(self, i, keyL, keyR, opname, strength, check_op=True):
        """insert an edge into the graph.

        Parameters
        ----------
        i : int
            site index at which the edge of the graph is to be inserted.
        keyL : hashable
            The state at bond (i-1, i) to connect from.
        keyR : hashable
            The state at bond (i, i+1) to connect to.
        opname : str
            name of the operator
        strength : str
            prefactor of the operator to be inserted.
        check_op : bool
            wheter to check that 'opname' exists on the given `site`.
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

    def build_MPO(self, W_qtotal=None, leg0=None):
        """build the MPO represented by the graph (`self`).

        Parameters
        ----------
        W_qtotal : None | charges
            charges for *each* of the individual `W`.
        leg0 : None | :class:`npc.LegCharge`
            The charges to be used for the very first leg.

        Returns
        -------
        mpo : :class:`MPO`
            the MPO which self represents.
        """
        self.test_sanity()
        # TODO : remove `unused` states !!!
        # pre-work: generate the grid
        self._set_ordered_states()
        self._build_grid()
        self._calc_grid_legs(W_qtotal)
        # now build the `W` from the grid
        Ws = []
        for i in xrange(self.L):
            legs = [self._grid_legs[i], self._grid_legs[i+1]]
            W = npc.grid_outer(self._grids[i], legs, W_qtotal)
            W.set_leg_labels(['wL', 'wR', 'p', 'p*'])
            Ws.append(W)
        idL = [s.get('idL', None) for s in self._ordered_states]
        idR = [s.get('idR', None) for s in self._ordered_states]
        return MPO(self.sites, Ws, self.bc, idL, idR)

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
        """define an ordering of the 'states' on each MPO bond.

        Set ``self._ordered_sites`` to a list of dictionaries ``{state: index}``.
        """
        res = []
        for s in self.states:
            d = {}
            # try 'idL'=0 and 'idR'=-1.
            if 'idL' in s:
                offset = 1
                d['idL'] = 0
            else:
                offset = 0
            for i, key in enumerate(sorted(s - {'idL', 'idR'}, key=str)):
                d[key] = i + offset
            if 'idR' in s:
                d['idR'] = len(s) - 1
            res.append(d)
        return res

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
        grid : list of list of {None|Array}
            The grid W[i] with operators
        """
        site = self.sites[i]
        for row in grid:
            for i, entry in enumerate(row):
                if entry is None:
                    continue
                opname, strength = entry[0]
                # TODO: allow ``0 + npc.Array``....
                res = strength * site.get_op(opname)
                for opname, strength in entry[1:]:
                    res = res + strength * site.get_op(opname)
                row[i] = res  # replace entry
        return grid

    def _calc_grid_legs(self, W_qtotal, leg0):
        """calculate LegCharges for the grids from self.grid"""
        grids = self._grids
        assert(grids is not None)  # make sure _grid_insert_ops was called
        if self.bc == 'finite':
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
            legs.append(gr_legs[1])
        self._grid_legs = legs

    def _calc_grid_legs_inifinte(self, grids, W_qtotal, leg0):
        """calculate LegCharges from `self._grid` for an iMPO.

        The hard case. Initially, we do not know all charges of the first leg; and they have to
        be consistent with the final leg.

        The way to go: gauge 'idL' on the very left leg to 0, then gradually calculate the charges
        by going along the edges of the graph (maybe also over the iMPO boundary).
        """
        if leg0 is not None:
            # have charges of first leg: simple case, can use the *_finite version.
            self._calc_grid_legs_finite(self, grids, W_qtotal, leg0)
            # just make sure, that everything is self consistent over the MPS boundary.
            self.legs[-1].test_contractible(self.legs[0])
            return
        chinfo = self.chinfo
        states = self._ordered_states
        assert(states is not None)  # make sure self._set_ordered_states() was called
        charges = [{} for _ in xrange(self.L)]
        charges.append(charges[0])  # the *same* dictionary is shared for 0 and -1.
        charges['idL'] = self.chinfo.make_valid(None)  # default charge = 0.
        chis = [len(s) for s in self.states]
        for _ in xrange(1000*self.L):  # I don't expect interactions with larger range than that...
            for i in xrange(self.L):
                chL, chR = self.charges[i:i+2]
                stL, stR = states[i:i+2]
                graph = self.graph[i]
                grid = self._grid[i]
                for keyL, qL in chL.iteritems():
                    for keyR in graph[keyL]:
                        # calculate charge qR from the entry of the grid
                        op = grid[stL[keyL], stR[keyR]]
                        assert(op is not None)
                        qR = chinfo.make_valid(qL + op.qtotal - W_qtotal)
                        if keyR not in chR:
                            chR[keyR] = qR
                        elif any(chR[keyR] != qR):
                            raise ValueError("incompatible charges while creating the MPO")
            if all([len(qs) == chi for qs, chi in itertools.izip(charges, chis)]):
                break
        else:  # no break
            # this should not happen (if we have no bugs), but who knows :(
            # if it happens, it means that some
            raise ValueError("Unable to determine the grid legcharges.")
        # finally generate LegCharge from the dictionaries
        self._grid_legs = []
        for qs, st in itertools.izip(charges, states):
            qfl = [None]*len(qs)
            for key, q in qs.iteritems():
                qfl[states[key]] = q
            leg = npc.LegCharge.from_qflat(chinfo, qfl, qconj=+1)
            self._grid_legs.append(leg)
