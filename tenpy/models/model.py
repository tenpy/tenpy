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

from ..tools.misc import to_array


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
        For each mps index `i` a dictionary
        ``{'opname_i': {j: {'opname_j': (strength, string_opname)}}}``.
        If ``lat.bc_MPS == 'infinite'`` it may have entries with `j` < `i` going over the
        iMPS boundary, all other terms have `i` < `j`. Filled by :meth:`add_coupling_term`.
    H_onsite : list of :class:`npc.Array`
        For each site (in mps order) the onsite part of the Hamiltonian.

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
        self.H_bond = None
        self.eig_bond = None
        self.U_bond = None
        self.U_parameters = None
        self.test_sanity()

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
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
            May vary spatially and is tiled to shape ``lat.Ls``.
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

    def add_coupling(self, strength, u1, op1, u2, op2, dx, JWstring='Id'):
        """Add twosite coupling terms to the Hamiltonian.

                ({scalar | array}, int, str, int, str, 1D array)

        Entries are
        are given to
        Represents couplings of the form
        ``sum_{x_0, ..., x_{dim-1}} strength[x_0, ... x_{dim-1}] * OP1 * OP2``
        where ``OP1 := lat.unit_cell[u1].opname1`` acts on the site ``(x_0, ..., x_{dim-1}, u1)``
        and ``OP2 := lat.unit_cell[u2].opname2`` acts on the site
        ``(x_0+direction[0], ..., x_{dim-1}+direction{dim-1}, u2)``.
        The indices ``x_a`` is taken modulo ``lattice.Ls[a]`` and runs through
        ``range(lattice.Ls[a])`` if ``bc_coupling[a] == 'periodic'``.
        Else (if ``bc_coupling[a] == 'open') ``x_a`` runs only through
        ``range(lattic.Ls[a]-direction[a])``.
        The `strength` may vary spatially and is tiled to the appropriate shape.
        """

    def calc_H_onsite(self):
        """calculate `self.H_onsite` from `self.onsite_terms`."""
        raise NotImplementedError()  # TODO

    def calc_H_bond(self, H_couplings, H_onsite=None):
        """calculate and set `self.H_bond` from `self.coupling terms."""
        raise NotImplementedError()

    def calc_H_MPO(self):
        """calculate MPO representation of self."""
        raise NotImplementedError()  # TODO

    def _remove_onsite_terms_zeros(self, tol=1.e-15):
        """remove entries of strength `0` from ``self.onsite_terms``."""
        for term in self.onsite_terms:
            for op in term.keys():
                if abs(term[op]) < tol:
                    del term[op]

    def _remove_coupling_terms_zeros(self, tol=1.e-15):
        """remove entries of strength `0` from ``self.onsite_terms``."""
        raise NotImplementedError()  # TODO


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
