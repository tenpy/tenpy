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

.. todo ::
    User guide: introduction how to create a new model
"""

from __future__ import division
import numpy as np

from ..linalg import np_conserved as npc
from ..tools.misc import to_array
from ..networks import mpo  # used to construct the Hamiltonian as MPO
from ..tools.params import get_parameter

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

    .. todo ::
        implement ...
        Some way to generalize to couplings involving multiple sites?
    """

    def __init__(self, lattice, bc_coupling='open'):
        self.lat = lattice
        global _bc_coupling_choices
        if bc_coupling in _bc_coupling_choices.keys():
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
            for opname, strength in terms.iteritems():
                if opname not in site.opnames:
                    raise ValueError("Operator {op!r} not in site".format(op=opname))
        for site_i, d1 in zip(sites, self.coupling_terms):
            for (op_i, opstring), d2 in d1.iteritems():
                if op_i not in site_i.opnames:
                    raise ValueError("Operator {op!r} not in site".format(op=op_i))
                for j, d3 in d2.iteritems():
                    for op_j in d3.keys():
                        if op_j not in sites[j].opnames:
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
        if opname not in self.lat.unit_cell[u].opnames:
            raise ValueError("unknown onsite operator {1!r} for u={2:d}".format(opname, u))
        strength = to_array(strength, self.lat.Ls)  # tile to lattice shape
        for i, i_lat in zip(*self.lat.mps_lat_idx_fix_u(u)):
            term = self.onsite_terms[i]
            term[opname] = term.get(opname, 0) + strength[tuple(i_lat)]

    def add_coupling(self, strength, u1, op1, u2, op2, dx, op_string='Id'):
        """Add twosite coupling terms to the Hamiltonian.

        Represents couplings of the form
        ``sum_{x_0, ..., x_{dim-1}} strength[x_0, ... x_{dim-1}] * OP1 * OP2``
        where ``OP1 := lat.unit_cell[u1].opname1`` acts on the site ``(x_0, ..., x_{dim-1}, u1)``
        and ``OP2 := lat.unit_cell[u2].opname2`` acts on the site
        ``(x_0+dx[0], ..., x_{dim-1}+dx[dim-1], u2)``.
        For periodic boundary conditions (``bc_coupling[a] == False``)
        the index ``x_a`` is taken modulo ``lat.Ls[a]`` and runs through ``range(lat.Ls[a])``.
        For open boundary conditions, ``x_a`` is limited to ``0 <= x_a < Ls[a]`` and
        ``0 <= x_a+dx < lat.Ls[a]``.

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
            Name of an operator to be used between OP1 and OP2.
            Typical use case is the phase for a Jordan-Wigner transformation.
            (This operator should be defined on all sites in the unit cell.)
        """
        for op, u in [(op1, u1), (op2, u2)]:
            if op not in self.lat.unit_cell[u].opnames:
                raise ValueError("unknown onsite operator {1!r} for u={2:d}".format(op, u))
        dx = np.array(dx, np.intp).reshape([self.lat.dim])
        strength = to_array(strength, self._coupling_shape(dx))  # tile to correct shape
        luc = len(self.lat.unit_cell)
        if np.all(dx == 0) and u1 % luc == u2 % luc:
            raise ValueError("Coupling shouldn't be onsite!")
        idx_i, idx_i_lat = self.lat.mps_lat_idx_fix_u(u1)
        idx_j_lat_shifted = idx_i_lat + dx
        idx_j_lat = idx_j_lat_shifted % np.array(self.lat.Ls)
        keep = np.all(np.logical_or(idx_j_lat_shifted == idx_j_lat,  # not accross the boundary
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
            else:  # finite MPS
                swap = (j < i)  # ensure i < j
            if swap:
                i, o1, j, o2 = j, op2, i, op1  # swap OP1 <-> OP2
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
            for opname, strength in terms.iteritems():
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
            Legs are ``['pL', 'pL*', 'pR', 'pR*']``

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
            for (op1, op_str), d2 in d1.iteritems():
                for j2, d3 in d2.iteritems():
                    # i, j in terms are defined such that we expect j = j2,
                    # (including the case N_sites = 2 and iMPS
                    if j != j2:
                        raise ValueError("Can't give H_bond for long-range: {i:d} {j:d}".format(
                            i=i, j=j2))
                    for op2, strength in d3.iteritems():
                        H = H + strength * npc.outer(site_i.get_op(op1), site_j.get_op(op2))
            H.set_leg_labels(['pL', 'pL*', 'pR', 'pR*'])
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
            for opname, strength in terms.iteritems():
                graph.add(i, 'IdL', 'IdR', opname, strength)
        # coupling terms
        self._remove_coupling_terms_zeros(tol_zero)
        for i, d1 in enumerate(self.coupling_terms):
            for (opname_i, op_string), d2 in d1.iteritems():
                label = (i, opname_i, op_string)
                graph.add(i, 'IdL', label, opname_i, 1.)
                for j, d3 in d2.iteritems():
                    j2 = j if j > i else j + graph.L
                    graph.add_string(i, j2, label, op_string)
                    for opname_j, strength in d3.iteritems():
                        graph.add(j, label, 'IdR', opname_j, strength)
        # add 'IdL' and 'IdR' and convert the graph to an MPO
        graph.add_missing_IdL_IdR()
        self.H_MPO_graph = graph
        H_MPO = graph.build_MPO()
        return H_MPO

    def _remove_onsite_terms_zeros(self, tol_zero=1.e-15):
        """remove entries of strength `0` from ``self.onsite_terms``."""
        for term in self.onsite_terms:
            for op in term.keys():
                if abs(term[op]) < tol_zero:
                    del term[op]
        # done

    def _remove_coupling_terms_zeros(self, tol_zero=1.e-15):
        """remove entries of strength `0` from ``self.coupling_terms``."""
        for d1 in self.coupling_terms:
            # d1 = ``{('opname_i', 'opname_string'): {j: {'opname_j': strength}}}``
            for op_i_op_str, d2 in d1.items():
                for j, d3 in d2.items():
                    for op_j, st in d3.items():
                        if abs(st) < tol_zero:
                            del d3[op_j]
                    if len(d3) == 0:
                        del d2[j]
                if len(d2) == 0:
                    del d1[op_i_op_str]
        # done

    def _coupling_shape(self, dx):
        """calculate correct shape of the strengths for each coupling."""
        return tuple([La - dxa * int(bca)
                      for La, dxa, bca in zip(self.lat.Ls, dx, self.bc_coupling)])


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
    bond_eig_vals : list of 1D arrays
        eigenvalues for each entry of H_bond
    bond_eig_vecs : list of npc.Array
        eigenvectors for each entry of H_bond
    U_bond: None | list of tuples
        exp(i dt H_bond) depe for TEBD parameters given by `U_parameters`
    U_param : dict
        TEBD parameters for which `U_bond` was calculated.

    .. todo ::
        implement
    """

    def __init__(self, lat, H_bond):
        self.lat = lat
        self.H_bond = list(H_bond)
        if self.lat.bc_MPS != 'infinite':
            self.H_bond[0] = None
        self.calc_bond_eig()
        self.U_bond = None
        self.U_param = dict()
        NearestNeighborModel.test_sanity(self)
        # like self.test_sanity(), but use the version defined below even for derived class

    def test_sanity(self):
        if len(self.H_bond) != self.lat.N_sites:
            raise ValueError("wrong len of H_bond")

    def calc_bond_eig(self):
        """calculate ``self.bond_eig_{vals,vecs}`` from ``self.H_bond``.

        Raises ValueError is 2-site Hamiltonian could not be diagonalized.
        """
        self.bond_eig_vals = []
        self.bond_eig_vecs = []
        for h in self.H_bond:
            #TODO: Check if hermitian?!
            if h == None:
                w = v = None
            else:
                H2 = h.combine_legs([('pL', 'pR'), ('pL*', 'pR*')], qconj=[+1, -1])
                w, v = npc.eigh(H2)
            self.bond_eig_vals.append(w)
            self.bond_eig_vecs.append(v)

            # check if the diagonalisation worked, as in TenPy
            if h != None:
                Hnp = H2.to_ndarray()
                vnp = v.to_ndarray()
                Hp = np.dot(np.dot(vnp, np.diag(w)), np.conj(vnp.T))
                if np.allclose(Hnp, Hp) != True:
                    raise ValueError("Diagonalisation of bond did not work!")

    def calc_U(self, param):
        #TODO: Old TenPy has E_offset
        #TODO: Document!!
        if (param == self.U_param):
            return
        self.U_param = param

        TrotterOrder = get_parameter(param, 'order', 2, 'TrotterDecomp')
        dt = get_parameter(param, 'dt', 0.1, 'TrotterDecomp')
        type = get_parameter(param, 'type', 'REAL', 'TrotterDecomp')

        if TrotterOrder == 1:
            self.U_bond = [[None] * len(self.H_bond)]
            for i_bond in range(len(self.H_bond)):
                if self.bond_eig_vals[i_bond] != None:
                    if (type == 'IMAG'):
                        s = np.exp(-dt * self.bond_eig_vals[i_bond])
                    elif (type == 'REAL'):
                        s = np.exp(-1j * dt * (self.bond_eig_vals[i_bond]))
                    else:
                        raise ValueError(
                            "Need to have either real time (REAL) or imaginary time (IMAG)")
                    V = self.bond_eig_vecs[i_bond]
                    #U = V s V^dag, s = e^(- tau E )
                    U = V.scale_axis(s, axis=1)
                    U = npc.tensordot(U, V.conj(), axes=(1, 1))
                    labels = self.H_bond[i_bond].combine_legs(
                        [('pL', 'pR'), ('pL*', 'pR*')], qconj=[+1, -1]).get_leg_labels()
                    U.set_leg_labels(labels)
                    #TODO: Not nice!!
                    self.U_bond[0][i_bond] = U.split_legs()
        elif TrotterOrder == 2:
            raise NotImplementedError()
        elif TrotterOrder == 4:
            raise NotImplementedError()
        # TODO ....
        else:
            raise NotImplementedError('Only 4th order Trotter has been implemented')


class MPOModel(object):
    """Base class for a model with an MPO representation of the Hamiltonian.

    Suitable for MPO-based algorithms, e.g. DMRG and MPO time evolution.

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

    .. todo ::
        Should the class host the environment, similar as NearestNeighborModel hosts U?
        implement: provide (function to calculate) the MPO for time evolution.
    """

    def __init__(self, lat, H_MPO):
        self.lat = lat
        self.H_MPO = H_MPO
        MPOModel.test_sanity(self)
        # like self.test_sanity(), but use the version defined below even for derived class

    def test_sanity(self):
        if self.H_MPO.sites != self.lat.mps_sites():
            raise ValueError("lattice incompatible with H_MPO.sites")
