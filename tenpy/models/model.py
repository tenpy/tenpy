"""This module contains a base class for a model"""


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
    lattice : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the underlying Hilbert space.
    bc : list of {'open', 'periodic'}
        boundary conditions (of the couplings) in each direction of the lattice.
    onsite_terms : list of (opname, u, strength)
        opstrength is broadcasted to the lattice size
    coupling_terms : list of (opname1, u1, opname2, u2, direction, strength)

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

    def cast_1_per_site(self, value):
        """broadcast/tile `value` to a numpy array with one value per site.

        .. todo :
            implement. Need further cast funcitons (1 per bond respecting boundary conditions...)
            General `cast` should exist in tools.misc."""
        value = np.asarray(value)
        if value.ndim > self.dim + 1:
            raise ValueError("too many dimensions")



