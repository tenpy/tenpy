"""This module contains a base class for a model"""

from __future__ import division

import numpy as np

from ..tools.misc import to_array


class Model:
    """Base class for a general `model`.

    A `model` is the general form of an Hamiltonian. It consists of
    1) The underlying hilbert space given by the :class:`tenpy.model.lattice.Lattice`.
    2) The Hamiltonian's onsite terms.
    3) The Hamiltonian's couplings between different sites.

    This information specifies the Hamiltonian completely, so it is always possible to
    construct an MPO representing the Hamiltonian. This is done during initialization
    via an `MPO_graph` and saved in `self.H_MPO`.

    From this information, the model is always able to construct an MPO representing
    the Hamiltonian. This is already done during initialization (via an MPO_graph).

    Furthermore, the model can hold two-site bond-operators, e.g. for the application of TEBD.

    Parameters
    ----------
    lat : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the underlying Hilbert space.
    bc_coupling : list of {'open', 'periodic'}
        boundary conditions (of the couplings) in each direction of the lattice.
    H_onsite_terms : list of ({scalar | array}, int, str)
        Entries are ``(strength, u, opname)``, where `u` picks a cite of the unit cell.
        `strength` is tiled to the lattice size.
    coupling_terms : list of (
        Entries are ``(strength, u, opname, u2, opname2, direction)``

    Attributes
    ----------
    lattice : :class:`tenpy.model.lattice.Lattice`
        An instance of a lattice.
    bc_coupling : list of {'open', 'periodic'}
        boundary conditions (of the couplings) in each direction of the lattice.
    H_MPO : :class:`tenpy.tn.mpo.MPO`
        MPO representation of the Hamiltonian.
    H_onsite : list of :class:`npc.Array`
        for each site the onsite Hamiltonian.
    terms : (H_onsite, H_couplings)
        the explicit onsite and coupling terms specified to the initialization.
    H_bond : None | list of :class:`npc.Array`
        The Hamiltonian rewritten as ``sum_i H_bond[i]`` for MPS indices ``i``, if such a rewriting
        is possible (i.e. if the hamiltonian has only nearest neighbour couplings with respect to
        ``lattice.order``). ``None``, if such a rewriting is not possible.
    eig_bond : None | list of (1D ndarray, npc.Array)
        eigenvalues and eigenvectors of H_bond. Only set if H_bond is not None.
    U_bond : None | list of list of npc.Array
        exp(H_bond) for TEBD parameters given by `U_parameters`
    U_parameters : None | dict
        parameters for which `U_bond` was calculated.

    .. todo :
        implement ...
    """
    def __init__(self, lattice, bc_coupling='open', H_onsite_terms=[], H_coupling_terms=[]):
        self.lattice = lattice
        self.bc_coupling = bc_coupling
        H_onsite_terms = self._valid_onsite_terms(H_onsite_terms)
        H_coupling_terms = self._valid_coupling_terms(H_coupling_terms)
        self.terms = (H_onsite_terms, H_coupling_terms)
        self.H_onsite_terms = H_onsite_terms  # TODO : parse, cast values to lattice.
        self.H_coupling_terms = H_coupling_terms  # TODO : parse, cast values to lattice.
        self.H_bond = None
        self.eig_bond = None
        self.U_bond = None
        self.U_parameters = None

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        raise NotImplementedError()  # TODO

    def calc_H_onsite(self, H_onsite):
        """calculate `self.H_onsite` from the onsite terms."""
        raise NotImplementedError()  # TODO

    def calc_H_MPO(self):
        """calculate `self.H_MPO` from ``self.H_coupling_terms``."""
        raise NotImplementedError()  # TODO

    def calc_H_bond(self, H_couplings, H_onsite=None):
        """calculate H_bond from the coupling terms."""
        raise NotImplementedError()

    def _convert_onsite_terms(self, H_onsite_terms):
        """convert H_onsite_terms into different format & check for incompatibility."""
        uc = self.lat.unit_cell
        luc = len(uc)
        terms = [dict() for _ in xrange(self.lat.N_sites)]
        for strength, u, opname in H_onsite_terms:
            if u < 0:  # TODO make a function in `Lattice` for that
                u += luc
            if u < 0 or u >= luc:
                raise IndexError("Index of unit_cell `u` out of bonds")
            if opname not in uc[u].opnames:
                raise ValueError("unknown onsite operator {1!r} for u={2:d}".format(opname, u))
            strength = to_array(strength, self.lat.Ls)
            for i, i_lat in zip(*self.lat.mps_lat_idx_fix_u(u)):
                terms[i] = terms[i].get(opname, 0) + strength[i_lat]  # TODO XXX
        for d in terms:  # remove zeros
            for k in d.keys():
                if d[k] == 0:
                    del d[k]
        return terms

    def _convert_coupling_terms(self, H_coupling_terms):
        """convert H_coupling_terms into valid form & check for incompatibility."""
        raise NotImplementedError()  # TODO
