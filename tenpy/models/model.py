"""This module contains a base class for a model.

A 'model' is supposed to represent a Hamiltonian in a generalized way.
Beside a :class:`~tenpy.models.lattice.Lattice` specifying the geometry and
underlying Hilbert space, it thus needs some way to represent the different terms
of the Hamiltonian.

Different algorithms require different representations of the Hamiltonian.
For example, if you only want to do DRMG, it is enough to specify the Hamiltonian
as an MPO with a :class:`MPOModel`.
On the other hand, TEBD needs the model to be 'nearest neighbor' and thus
a representation by nearest-neighbor terms.

The `CouplingModel` is the attempt to generalize the representation of `H`
by explicitly specifying the couplings of onsite-terms, and providing functionality
for converting the specified couplings into an MPO or nearest-neighbor bonds.
This allows to quickly generate new 'models' for a broad class of Hamiltonians.
However, this class is (at least for now) limited to interactions involving
only two sites. For other cases (e.g. exponentially decaying long-range interactions in 1D),
it might be simpler to specify the MPO explicitly.

Of course, we also provide ways to transform a :class:`NearestNeighborModel` into a
:class:`MPOModel` and vice versa, as far as this is possible.

See also the introduction :doc:`../intro_model`.
"""

import numpy as np

from ..linalg import np_conserved as npc
from ..tools.misc import to_array
from ..networks import mpo  # used to construct the Hamiltonian as MPO

__all__ = ['CouplingModel', 'NearestNeighborModel', 'MPOModel']

_bc_coupling_choices = {'open': True, 'periodic': False}


class CouplingModel(object):
    """Base class for a general model of a Hamiltonian consisting of two-site couplings.

    In this class, the terms of the Hamiltonian are specified explicitly as onsite or coupling
    terms.

    Parameters
    ----------
    lattice : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the geometry and the local Hilbert space(s).
    bc_coupling : (iterable of) {'open' | 'periodic'}
        Boundary conditions of the couplings in each direction of the lattice. Defines how the
        couplings are added in :meth:`add_coupling`. A singe string holds for all directions.

    Attributes
    ----------
    lat : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the geometry and the local Hilbert space(s).
    bc_coupling : bool ndarray
        Boundary conditions of the couplings in each direction of the lattice,
        translated into a bool array with the global `_bc_coupling_choices`.
    onsite_terms : list of dict
        For each MPS index `i` a dictionary ``{'opname': strength}`` defining the onsite terms.
        Filled by :meth:`add_onsite`.
    coupling_terms : list of dict
        For each MPS index `i` a dictionary of the form
        ``{('opname_i', 'opname_string'): {j: {'opname_j': strength}}}``.
        Entries with ``j < i`` are only allowed for ``lat.bc_MPS == 'infinite'``, in which case
        they indicate couplings over the iMPS boundary, i.e., between the sites
        ``(i, j+lat.N_sites)`` and between the sites ``(i-lat.N_sites, j)``.
        Filled by :meth:`add_coupling`.
    H_onsite : list of :class:`npc.Array`
        For each site (in MPS order) the onsite part of the Hamiltonian.
    """

    def __init__(self, lattice, bc_coupling='open'):
        self.lat = lattice
        global _bc_coupling_choices
        if bc_coupling in list(_bc_coupling_choices.keys()):
            bc_coupling = [_bc_coupling_choices[bc_coupling]] * self.lat.dim
        else:
            bc_coupling = [_bc_coupling_choices[bc] for bc in bc_coupling]
        self.bc_coupling = np.array(bc_coupling)
        self.onsite_terms = [dict() for _ in range(self.lat.N_sites)]
        self.coupling_terms = [dict() for _ in range(self.lat.N_sites)]
        self.H_onsite = None
        CouplingModel.test_sanity(self)
        # like self.test_sanity(), but use the version defined below even for derived class

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        if self.bc_coupling.shape != (self.lat.dim, ):
            raise ValueError("Wrong len of bc_coupling")
        assert self.bc_coupling.dtype == np.bool
        assert int(_bc_coupling_choices['open']) == 1  # this is used explicitly
        assert int(_bc_coupling_choices['periodic']) == 0
        sites = self.lat.mps_sites()
        for site, terms in zip(sites, self.onsite_terms):
            for opname, strength in terms.items():
                if not site.valid_opname(opname):
                    raise ValueError("Operator {op!r} not in site".format(op=opname))
        for site_i, d1 in zip(sites, self.coupling_terms):
            for (op_i, opstring), d2 in d1.items():
                if not site_i.valid_opname(op_i):
                    raise ValueError("Operator {op!r} not in site".format(op=op_i))
                for j, d3 in d2.items():
                    for op_j in d3.keys():
                        if not sites[j].valid_opname(op_j):
                            raise ValueError("Operator {op!r} not in site".format(op=op_j))
        # done

    def add_onsite(self, strength, u, opname):
        """Add onsite terms to self.

        Adds ``sum_{x_0, ..., x_{dim-1}} strength[x_0, ..., x_{dim-1}] * lat.unit_cell[u].opname``,
        where the operator acts on the site given by a lattice index ``(x_0, ..., x_{dim-1}, u)``,
        to the represented Hamiltonian.
        The necessary terms are just added to ``self.onsite_terms``; doesn't rebuild the MPO.

        Parameters
        ----------
        strength : scalar | array
            Prefactor of the onsite term. May vary spatially and is tiled to shape ``lat.Ls``.
        u : int
            Picks a :class:`~tenpy.model.lattice.Site` ``lat.unit_cell[u]`` out of the unit cell.
        opname : str
            valid operator name of an onsite operator in ``lat.unit_cell[u]``.
        """
        strength = to_array(strength, self.lat.Ls)  # tile to lattice shape
        if not np.any(strength != 0.):
            return  # nothing to do: can even accept non-defined `opname`.
        if not self.lat.unit_cell[u].valid_opname(opname):
            raise ValueError("unknown onsite operator {0!r} for u={1:d}\n"
                             "{2!r}".format(opname, u, self.lat.unit_cell[u]))
        for i, i_lat in zip(*self.lat.mps_lat_idx_fix_u(u)):
            term = self.onsite_terms[i]
            term[opname] = term.get(opname, 0) + strength[tuple(i_lat)]

    def add_coupling(self,
                     strength,
                     u1,
                     op1,
                     u2,
                     op2,
                     dx,
                     op_string='Id',
                     str_on_first=True,
                     raise_op2_left=False):
        """Add twosite coupling terms to the Hamiltonian.

        Represents couplings of the form
        ``sum_{x_0, ..., x_{dim-1}} strength[x_0, ... x_{dim-1}] * OP1 * OP2``
        where ``OP1 := lat.unit_cell[u1].opname1`` acts on the site ``(x_0, ..., x_{dim-1}, u1)``
        and ``OP2 := lat.unit_cell[u2].opname2`` acts on the site
        ``(x_0+dx[0], ..., x_{dim-1}+dx[dim-1], u2)``.
        For periodic boundary conditions (``bc_coupling[a] == False``)
        the index ``x_a`` is taken modulo ``lat.Ls[a]`` and runs through ``range(lat.Ls[a])``.
        For open boundary conditions, ``x_a`` is limited to ``0 <= x_a < Ls[a]`` and
        ``0 <= x_a+dx[a] < lat.Ls[a]``.

        Parameters
        ----------
        strength : scalar | array
            Prefactor of the coupling. May vary spatially and is tiled to shape ``lat.Ls``.
        u1 : int
            Picks the site ``lat.unit_cell[u]`` for OP1.
        op1 : str
            Valid operator name of an onsite operator in ``lat.unit_cell[u1]`` for OP1.
        u2 : int
            Picks the site ``lat.unit_cell[u]`` for OP2.
        op2 : str
            Valid operator name of an onsite operator in ``lat.unit_cell[u2]`` for OP2.
        dx : iterable of int
            Translation vector (of the unit cell) between OP1 and OP2.
            For a 1D lattice, a single int is also fine.
        op_string : str
            Name of an operator to be used between OP1 and OP2 *and* on the smaller of the two sites.
            Typical use case is the phase for a Jordan-Wigner transformation.
            (This operator should be defined on all sites in the unit cell.)
        str_on_first : bool
            This option should be chosen as ``True`` for Jordan-Wigner strings.
            When handling Jordan-Wigner strings we need to extend the `op_string` to also act on
            the 'left', first site (in the sense of the MPS ordering of the sites given by the
            lattice). In this case, there is a well-defined ordering of the operators in the
            physical sense (i.e. which of `op1` or `op2` acts first on a given state).
            We follow the convention that `op2` acts first (in the physical sense),
            independent of the MPS ordering.
        raise_op2_left : bool
            Raise an error when `op2` appears left of `op1`
            (in the sense of the MPS ordering given by the lattice).
        """
        dx = np.array(dx, np.intp).reshape([self.lat.dim])
        strength = to_array(strength, self._coupling_shape(dx))  # tile to correct shape
        if not np.any(strength != 0.):
            return  # nothing to do: can even accept non-defined onsite operators
        luc = len(self.lat.unit_cell)
        for op, u in [(op1, u1), (op2, u2)] + [(op_string, u) for u in range(luc)]:
            if not self.lat.unit_cell[u].valid_opname(op):
                raise ValueError("unknown onsite operator {0!r} for u={1:d}\n"
                                 "{2:!r}".format(op, u, self.lat.unit_cell[u]))
        if np.all(dx == 0) and u1 % luc == u2 % luc:
            raise ValueError("Coupling shouldn't be onsite!")
        idx_i, idx_i_lat = self.lat.mps_lat_idx_fix_u(u1)
        idx_j_lat_shifted = idx_i_lat + dx
        idx_j_lat = idx_j_lat_shifted % np.array(self.lat.Ls)
        keep = np.all(
            np.logical_or(
                idx_j_lat_shifted == idx_j_lat,  # not accross the boundary
                ~self.bc_coupling),  # direction has periodic bound. cond.
            axis=1)
        idx_i = idx_i[keep]
        idx_i_lat = idx_i_lat[keep]
        idx_j_lat = idx_j_lat[keep]
        idx_j = self.lat.lat2mps_idx(np.hstack([idx_j_lat, [[u2]] * len(idx_i_lat)]))
        for i, i_lat, j in zip(idx_i, idx_i_lat, idx_j):
            o1, o2 = op1, op2
            if self.lat.bc_MPS == 'infinite':
                d_in = abs(i - j)  # distance within the chain
                d_out = self.lat.N_sites - d_in  # distance over the boundary
                if d_in < d_out:
                    swap = (j < i)  # ensure coupling within the chain
                elif d_in > d_out:
                    swap = (i < j)  # ensure coupling over the iMPS boundary
                else:  # d_in == d_out
                    swap = False  # don't change the order
                    # this is necessary for correct TEBD for iMPS with L=2
            else:  # finite MPS: allow periodic boundary conditions
                swap = (j < i)  # ensure i <= j
            if swap:
                if raise_op2_left:
                    raise ValueError("Op2 is left")
                i, o1, j, o2 = j, op2, i, op1  # swap OP1 <-> OP2
            # now o1 is the "left" operator;
            # if j < i, o2 acts one unit cell "right" of o1.
            if str_on_first and op_string != 'Id':
                if swap:
                    o1 = op_string + ' ' + o1  # o1==op2 should act first
                else:
                    o1 = o1 + ' ' + op_string  # o1==op1 should act second
            d1 = self.coupling_terms[i]
            # form of d1: ``{('opname_i', 'opname_string'): {j: {'opname_j': strength}}}``
            d2 = d1.setdefault((o1, op_string), dict())
            d3 = d2.setdefault(j, dict())
            d3[o2] = d3.get(o2, 0) + strength[tuple(i_lat)]

    def calc_H_onsite(self, tol_zero=1.e-15):
        """Calculate `H_onsite` from `self.onsite_terms`.

        Parameters
        ----------
        tol_zero : float
            prefactors with ``abs(strength) < tol_zero`` are considered to be zero.

        Returns
        -------
        H_onsite : list of npc.Array
            onsite terms of the Hamiltonian.
        """
        self._remove_onsite_terms_zeros(tol_zero)
        res = []
        for i, terms in enumerate(self.onsite_terms):
            s = self.lat.site(i)
            H = npc.zeros([s.leg, s.leg.conj()])
            for opname, strength in terms.items():
                H = H + strength * s.get_op(opname)  # (can't use ``+=``: may change dtype)
            res.append(H)
        return res

    def calc_H_bond(self, tol_zero=1.e-14):
        """calculate `H_bond` from `self.coupling_terms` and `self.H_onsite`.

        If ``self.H_onsite is None``, it is calculated with :meth:`self.calc_H_onsite`.

        Parameters
        ----------
        tol_zero : float
            prefactors with ``abs(strength) < tol_zero`` are considered to be zero.

        Returns
        -------
        H_bond : list of :class:`~tenpy.linalg.np_conserved.Array`
            Bond terms as required by the constructor of :class:`NearestNeighborModel`.
            Legs are ``['p0', 'p0*', 'p1', 'p1*']``

        Raises
        ------
        ValueError : if the Hamiltonian contains longer-range terms.
        """
        self._remove_coupling_terms_zeros(tol_zero)
        if self.H_onsite is None:
            self.H_onsite = self.calc_H_onsite(tol_zero)
        finite = (self.lat.bc_MPS != 'infinite')
        res = [None] * self.lat.N_sites
        for i, d1 in enumerate(self.coupling_terms):
            j = (i + 1) % self.lat.N_sites
            d1 = self.coupling_terms[i]
            site_i = self.lat.site(i)
            site_j = self.lat.site(j)
            strength_i = 1. if finite and i == 0 else 0.5
            strength_j = 1. if finite and j == self.lat.N_sites - 1 else 0.5
            if finite and j == 0:  # over the boundary
                strength_i, strength_j = 0., 0.  # just to make the assert below happy
            H = npc.outer(strength_i * self.H_onsite[i], site_j.Id)
            H = H + npc.outer(site_i.Id, strength_j * self.H_onsite[j])
            for (op1, op_str), d2 in d1.items():
                for j2, d3 in d2.items():
                    # i, j in terms are defined such that we expect j = j2,
                    # (including the case N_sites = 2 and iMPS
                    if j != j2:
                        raise ValueError("Can't give H_bond for long-range: {i:d} {j:d}".format(
                            i=i, j=j2))
                    for op2, strength in d3.items():
                        H = H + strength * npc.outer(site_i.get_op(op1), site_j.get_op(op2))
            H.iset_leg_labels(['p0', 'p0*', 'p1', 'p1*'])
            res[j] = H
        if finite:
            assert (res[0].norm(np.inf) <= tol_zero)
        return res

    def calc_H_MPO(self, tol_zero=1.e-15):
        """Calculate MPO representation of the Hamiltonian.

        Uses :attr:`onsite_terms` and :attr:`coupling_terms` to build an MPO graph
        (and then an MPO).

        Parameters
        ----------
        tol_zero : float
            prefactors with ``abs(strength) < tol_zero`` are considered to be zero.

        Returns
        -------
        H_MPO : :class:`~tenpy.networks.mpo.MPO`
            MPO representation of the Hamiltonian.
        """
        graph = mpo.MPOGraph(self.lat.mps_sites(), self.lat.bc_MPS)
        # onsite terms
        self._remove_onsite_terms_zeros(tol_zero)
        for i, terms in enumerate(self.onsite_terms):
            for opname, strength in terms.items():
                graph.add(i, 'IdL', 'IdR', opname, strength)
        # coupling terms
        self._remove_coupling_terms_zeros(tol_zero)
        for i, d1 in enumerate(self.coupling_terms):
            for (opname_i, op_string), d2 in d1.items():
                label = (i, opname_i, op_string)
                graph.add(i, 'IdL', label, opname_i, 1.)
                for j, d3 in d2.items():
                    j2 = j if j > i else j + graph.L
                    graph.add_string(i, j2, label, op_string)
                    for opname_j, strength in d3.items():
                        graph.add(j, label, 'IdR', opname_j, strength)
        # add 'IdL' and 'IdR' and convert the graph to an MPO
        graph.add_missing_IdL_IdR()
        self.H_MPO_graph = graph
        H_MPO = graph.build_MPO()
        return H_MPO

    def _remove_onsite_terms_zeros(self, tol_zero=1.e-15):
        """remove entries of strength `0` from ``self.onsite_terms``."""
        for term in self.onsite_terms:
            for op in list(term.keys()):
                if abs(term[op]) < tol_zero:
                    del term[op]
        # done

    def _remove_coupling_terms_zeros(self, tol_zero=1.e-15):
        """remove entries of strength `0` from ``self.coupling_terms``."""
        for d1 in self.coupling_terms:
            # d1 = ``{('opname_i', 'opname_string'): {j: {'opname_j': strength}}}``
            for op_i_op_str, d2 in list(d1.items()):
                for j, d3 in list(d2.items()):
                    for op_j, st in list(d3.items()):
                        if abs(st) < tol_zero:
                            del d3[op_j]
                    if len(d3) == 0:
                        del d2[j]
                if len(d2) == 0:
                    del d1[op_i_op_str]
        # done

    def _coupling_shape(self, dx):
        """calculate correct shape of the strengths for each coupling."""
        return tuple(
            [La - dxa * int(bca) for La, dxa, bca in zip(self.lat.Ls, dx, self.bc_coupling)])


class NearestNeighborModel(object):
    """Base class for a model of nearest neigbor interactions (w.r.t. the MPS index).

    Suitable for TEBD.

    Parameters
    ----------
    lattice : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the geometry and the local Hilbert space(s).
    H_bond : list of :class:`~tenpy.linalg.np_conserved.Array`
        The Hamiltonian rewritten as ``sum_i H_bond[i]`` for MPS indices ``i``.
        ``H_bond[i]`` acts on sites ``(i-1, i)``; we require ``len(H_bond) == lat.N_sites``.

    Attributes
    ----------
    lat : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the geometry and the local Hilbert space(s).
    H_bond : list of :class:`npc.Array`
        The Hamiltonian rewritten as ``sum_i H_bond[i]`` for MPS indices ``i``.
        ``H_bond[i]`` acts on sites ``(i-1, i)``.
    """

    def __init__(self, lat, H_bond):
        self.lat = lat
        self.H_bond = list(H_bond)
        if self.lat.bc_MPS != 'infinite':
            self.H_bond[0] = None
        NearestNeighborModel.test_sanity(self)
        # like self.test_sanity(), but use the version defined below even for derived class

    def test_sanity(self):
        if len(self.H_bond) != self.lat.N_sites:
            raise ValueError("wrong len of H_bond")

    def trivial_like_NNModel(self):
        """Return a NearestNeighborModel with same lattice, but trivial (H=0) bonds."""
        triv_H = [H.zeros_like() if H is not None else None for H in self.H_bond]
        return NearestNeighborModel(self.lat, triv_H)

    def bond_energies(self, psi):
        """Calculate bond energies <psi|H_bond|psi>.

        Parameters
        ----------
        psi : :class:`~tenpy.networks.mps.MPS`
            The MPS for which the bond energies should be calculated.

        Returns
        -------
        E_bond : 1D ndarray
            List of bond energies: for finite bc, ``E_Bond[i]`` is the energy of bond ``i, i+1``.
            (i.e. we omit bond 0 between sites L-1 and 0);
            for infinite bc ``E_bond[i]`` is the energy of bond ``i-1, i``.
        """
        if self.lat.bc_MPS == 'infinite':
            return psi.expectation_value(self.H_bond, axes=(['p0', 'p1'], ['p0*', 'p1*']))
        # else
        return psi.expectation_value(self.H_bond[1:], axes=(['p0', 'p1'], ['p0*', 'p1*']))


class MPOModel(object):
    """Base class for a model with an MPO representation of the Hamiltonian.

    Suitable for MPO-based algorithms, e.g. DMRG and MPO time evolution.

    .. todo ::
        implement: provide (function to calculate) the MPO for time evolution.
        Also, provide function to get H_MPO from H_bond

    Parameters
    ----------
    lattice : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the geometry and the local Hilbert space(s).
    H_MPO : :class:`~tenpy.networks.mpo.MPO`
        The Hamiltonian rewritten as an MPO.

    Attributes
    ----------
    lat : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the geometry and the local Hilbert space(s).
    H_MPO : :class:`tenpy.tn.mpo.MPO`
        MPO representation of the Hamiltonian.
    """

    def __init__(self, lat, H_MPO):
        self.lat = lat
        self.H_MPO = H_MPO
        MPOModel.test_sanity(self)
        # like self.test_sanity(), but use the version defined below even for derived class

    def test_sanity(self):
        if self.H_MPO.sites != self.lat.mps_sites():
            raise ValueError("lattice incompatible with H_MPO.sites")
