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

The `TwoSiteModel` is the attempt to generalize the representation of `H`
by explicitly specifying the couplings of onsite-terms, and providing functionality
for converting the specified couplings into an MPO or nearest-neighbor bonds.
This allows to quickly generate new 'models' for a broad class of Hamiltonians.
However, as the name suggests, this class is (at least for now) limited to interactions involving
only two sites. For other cases (e.g. exponentially decaying long-range interactions in 1D),
it might be simpler to specify MPO explicitly.

Of course, we also provide ways to transform a :class:`NearestNeighborModel` into a
:class:`MPOModel` and vice versa, as far as this is possible.

.. todo :
    User guide: introduction how to create a new model
"""

from __future__ import division
import numpy as np

from ..linalg import np_conserved as npc
from ..tools.misc import to_array

__all__ = ['TwoSiteModel', 'NearestNeighborModel', 'MPOModel']

class TwoSiteModel(object):
    """Base class for a general Model consisting of two-site couplings.

    A `model` is the general form of an Hamiltonian. It consists of
    1) The underlying hilbert space given by the :class:`~tenpy.model.lattice.Lattice`.
    2) The Hamiltonian's onsite terms.
    3) The Hamiltonian's couplings between different sites.

    This information specifies the Hamiltonian completely, so it is always possible to
    construct an MPO representing the Hamiltonian. This is done during initialization
    and saved in `self.H_MPO`.

    Furthermore, the model can hold two-site bond-operators, e.g. for the application of TEBD.

    Parameters
    ----------
    lattice : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the geometry and the local Hilbert space(s).
    bc_coupling : (iterable of) {'open' | 'periodic'}
        Boundary conditions (of the couplings) in each direction of the lattice.
        A singe {'open' | 'periodic'} holds for all directions.
    H_onsite_terms : list of tuples
        Entries ``(strength, u, opname)`` are given as arguments to :meth:`add_onsite`.
    H_couplings : list of tuples
        Entries ``(strength, u, opname, u2, opname2, direction)`` (& possible optional arguments)
        are given as arguments to :meth:`add_twosite_terms`.

    Attributes
    ----------
    lat : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the geometry and the local Hilbert space(s).
    bc_coupling : list of {'open', 'periodic'}
        Boundary conditions (of the couplings) in each direction of the lattice.
    onsite_terms : list of dict
        For each mps index `i` a dictionary ``{'opname': strength}`` defining the onsite terms.
        Filled by :meth:`add_onsite`.
    coupling_terms : list of dict
        For each mps index `i` a dictionary of the form
        ``{('opname_i', 'opname_string'): {j: {'opname_j': strength}}}``.
        If ``lat.bc_MPS == 'infinite'`` it may have entries with `j` < `i` going over the
        iMPS boundary. all other terms have `i` < `j`. Filled by :meth:`add_coupling_term`.
    H_onsite : list of :class:`npc.Array`
        For each site (in MPS order) the onsite part of the Hamiltonian.

    .. todo :
        implement ...
    """
    def __init__(self, lattice, bc_coupling='open', H_onsite_terms=[], H_coupling_terms=[]):
        self.lat = lattice
        if bc_coupling in ['open', 'periodic']:
            bc_coupling = [bc_coupling] * self.lat.dim
        self.bc_coupling = list(bc_coupling)
        self.onsite_terms = [dict() for _ in range(self.lattice.N_sites)]
        self.coupling_terms = [dict() for _ in range(self.lattice.N_sites)]
        self.H_onsite = None
        self.test_sanity()

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        if len(self.bc_coupling) != self.lat.dim:
            raise ValueError("Wrong len of bc_coupling")
        for bc in self.bc_coupling:
            if bc not in ['open', 'periodic']:
                raise ValueError("invalid value for boundary conditions: " + repr(bc))
        raise NotImplementedError()  # TODO

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
        If ``bc_coupling[a] == 'periodic'``, the index ``x_a`` is taken modulo ``lattice.Ls[a]``
        and runs through ``range(lattice.Ls[a])``; for open boundary conditions
        it runs only up to ``range(lattic.Ls[a]-dx[a])``.

        Parameters
        ----------
        strength : scalar | array
            Prefactor of the coupling. May vary spatially and is tiled to shape ``lat.Ls``.
        u1 : int
            Picks the site ``lat.unit_cell[u]`` for OP1.
        op1 : str
            valid operator name of an onsite operator in ``lat.unit_cell[u1]`` for OP1.
        u2 : int
            Picks the site ``lat.unit_cell[u]`` for OP2.
        op2 : str
            valid operator name of an onsite operator in ``lat.unit_cell[u2]`` for OP2.
        dx : iterable of int
            translation vector between OP1 and OP2.
            For a 1D lattice, a single int is also fine.
        op_string : str
            name of an operator to be used inbewteen OP1 and OP2.
            Typical use case is the phase for a Jordan-Wigner transformation.
        """
        for op, u in [(op1, u1), (op2, u2)]:
            if op not in self.lat.unit_cell[u].opnames:
                raise ValueError("unknown onsite operator {1!r} for u={2:d}".format(op, u))
        strength = to_array(strength, self._coupling_shape(dx))  # tile to correct shape
        dx = np.array(dx, np.intp).reshape([self.lat.dim])
        luc = len(self.lat.unit_cell)
        if np.all(dx == 0) and u1 % luc == u2 % luc:
            raise ValueError("Coupling shouldn't be onsite!")
        idx_i, idx_i_lat = self.lat.mps_lat_idx_fix_u(u1)
        idx_j_lat = (idx_i_lat + dx) % np.array(self.lat.Ls)
        idx_j = self.lat.lat2mps_idx(np.hstack(idx_j_lat, [[u2]]*len(idx_i_lat)))
        for i, i_lat, j in zip(idx_i, idx_i_lat, idx_j):
            o1, o2 = op1, op2
            if j < i and (self.lat.bc_MPS != 'infinite' or (i - j > j + self.N_sites - i)):
                # the last condition checks wether we better go over the iMPS boundary.
                # TODO check the last condition for special cases.... Does it work as expected?
                i, o1, j, o2 = j, op2, i, op1  # swap OP1 <-> OP2
            d1 = self.coupling_terms[i]
            # form of d1: ``{('opname_i', 'opname_string'): {j: {'opname_j': strength}}}``
            d2 = d1.setdefault((o1, op_string), dict())
            d3 = d2.setdefault(j, dict())
            d3[op2] = d3.get(o2, 0) + strength[i_lat]

    def calc_H_onsite(self, tol_zero=1.e-15):
        """Calculate `H_onsite` from `self.onsite_terms`.

        Parameters
        ----------
        tol_zero : float
            prefactors ``abs(strength) < tol_zero`` are considered to be zero.

        Returns
        -------
        H_onsite : list of npc.Array
            onsite terms of the Hamiltonian.
        """
        self._remove_onsite_terms_zeros(tol_zero)
        res = []
        for i, terms in enumerate(self.onsite_terms):
            s = self.lat.site(i)
            H = npc.zeros(s.leg.chinfo, [s.leg, s.leg.conj()])
            for opname, strength in terms.iteritems():
                H = H + getattr(s, terms[0])  # (can't use ``+=``: may change dtype)
            res.append(H)
        return res

    def calc_H_bond(self, tol_zero=1.e-15):
        """calculate `H_bond` from `self.coupling_terms` and `self.H_onsite`.

        If ``self.H_onsite is None``, it is calculated with :meth:`self.calc_H_onsite`.

        Parameters
        ----------
            prefactors ``abs(strength) < tol_zero``  are considered to be zero.
        tol_zero : float

        Returns
        -------
        H_bond : list of npc.Array
            Bond terms. Legs are ``['pL', 'pL*', 'pR', 'pR*']``

        Raises
        ------
        ValueError : if the Hamiltonian contains longer-range terms.
        """
        if self.H_onsite is None:
            self.H_onsite = self.calc_H_onsite(tol)
        finite = (self.lat.bc_MPS != 'infinte')
        res = []
        for i, d1 in enumerate(self.coupling_terms):
            j = i + 1 % self.lat.N_sites
            site_i = self.lat.site(i)
            site_j = self.lat.site(j)
            strength = 0.5 if i > 0 or i == 0 and not finite else 1.
            H = npc.outer(strength * self.H_onsite[i], site_j.Id)
            strength = 0.5 if j < self.lat.N_sites - 1 or (j == self.lat.N_sites - 1
                                                           and not finite) else 1.
            H = H + npc.outer(site_i.Id, strength * self.H_onsite[j])
            for (op1, op_str), d2 in d1.iteritems():
                for j2, d3 in d2.iteritems():
                    # we expect j = j2 even for N_sites = 1, 2 and PBC
                    # TODO: also for special cases? (L=1, L=2, ...)
                    if j != j2:
                        raise ValueError("Can't give H_bond for long-range: {i:d} {j:d}".format(
                            i=i, j=j2))
                    for op2, strength in d3.iteritems():
                        H = H + strength * npc.outer(op1, op2)
            H.set_leg_labels(['pL', 'pL*', 'pR', 'pR*'])
            res.append(H)
        if finite:
            assert(res[-1].norm(np.inf) <= tol)
        return res

    def calc_H_MPO(self):
        """calculate MPO representation of self."""
        return None   # TODO. (Just pass on for now to allow testing with XXZChain.
        raise NotImplementedError()  # TODO

    def _remove_onsite_terms_zeros(self, tol=1.e-15):
        """remove entries of strength `0` from ``self.onsite_terms``."""
        for term in self.onsite_terms:
            for op in term.keys():
                if abs(term[op]) < tol:
                    del term[op]
        # done

    def _remove_coupling_terms_zeros(self, tol=1.e-15):
        """remove entries of strength `0` from ``self.onsite_terms``."""
        for d1 in self.coupling_terms:
            # d1 = ``{('opname_i', 'opname_string'): {j: {'opname_j': strength}}}``
            for op_i_op_str, d2 in d1.iteritems():
                for j, d3 in d2.iteritems():
                    for op_j, st in d3:
                        if abs(st) < tol:
                            del d3[op_j]
                    if len(d3) == 0:
                        del d2[j]
                if len(d2) == 0:
                    del d1[op_i_op_str]
        # done

    def _coupling_shape(self, dx):
        """calculate correct shape of the strengths for each coupling."""
        return tuple([La - dxa * int(bca == 'periodic')
                      for La, dxa, bca in zip(self.lat.Ls, dx, self.bc_coupling)])


class NearestNeighborModel(object):
    """Base class for a model of nearest neigbor interactions (w.r.t. the MPS index).

    Suitable for TEBD.

    Attributes
    ----------
    lat : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the geometry and the local Hilbert space(s).
    H_bond : list of :class:`npc.Array`
        The Hamiltonian rewritten as ``sum_i H_bond[i]`` for MPS indices ``i``.
        ``H_bond[i]`` acts on sites ``(i, i+1)``.
    bond_eig_vals : list of 1D arrays
        eigenvalues for each entry of H_bond
    bond_eig_vecs : list of npc.Array
        eigenvectors for each entry of H_bond
    U_bond: None | list of tuples
        exp(i dt H_bond) depe for TEBD parameters given by `U_parameters`
    U_param : dict
        TEBD parameters for which `U_bond` was calculated.

    .. todo :
        implement
    """
    def __init__(self, lattice, H_bond):
        self.lat = lattice
        self.H_bond = list(H_bond)
        self.calc_bond_eig()
        self.U_bond = None
        self.U_param = dict()
        raise NotImplementedError()  # TODO


class MPOModel(object):
    """Base class for a model with an MPO representation of the Hamiltonian.

    Suitable for MPO-based algorithms, e.g. DMRG and MPO time evolution.

    Attributes
    ----------
    lat : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the geometry and the local Hilbert space(s).
    H_MPO : :class:`tenpy.tn.mpo.MPO`
        MPO representation of the Hamiltonian.

    .. todo :
        implement. Host environment?
    """
    def __init__(self, H_MPO):
        self.H_MPO = H_MPO
        self.lat = H_MPO.lat
