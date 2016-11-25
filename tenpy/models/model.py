"""This module contains a base class for a model"""


class Model:
    """Base class for a general `model`.

    A `model` is the general form of an Hamiltonian. It consists of
    1) The underlying hilbert space given by the :class:`tenpy.model.lattice.Lattice`.
    2) The Hamiltonian's onsite terms.
    3) The Hamiltonian's couplings between different sites.

    This information specifies the Hamiltonian completely, so it is always possible to
    construct the MPO representing the Hamiltonian. This is usually done during initialization
    via an MPO_graph and saved in `H`.

    From this information, the model is always able to construct an MPO representing
    the Hamiltonian. This is already done during initialization (via an MPO_graph).

    Furthermore, the model can hold two-site bond-operators, e.g. for the application of TEBD.

    Parameters
    ----------
    lattice : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the underlying Hilbert space.
    bc : list of {'open', 'periodic'}
        boundary conditions (of the couplings) in each direction of the lattice.
    onsite_terms : list of (opname, unitcell_index, opstrength)
        opstrength is broadcasted to the lattice size
    coupling_terms :

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
        only set if the model consists only of nearest neighbour couplings
        (with respect to ``lattice.order``).

    .. todo :
        implement ...
    """
    def __init__(self, lattice, bc_coupling, H_onsite=[], H_couplings=[]):
        self.lattice = lattice
        self.bc_coupling = bc_coupling
        raise NotImplementedError()

    def calc_H_onsite(self, H_onsite):
        """calculate H_onsite from the onsite terms"""
        raise NotImplementedError()

    def calc_H_bond(self, H_couplings, H_onsite=None):
        """calculate H_bond from the coupling terms."""
        raise NotImplementedError()
