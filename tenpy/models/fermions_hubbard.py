"""Fermionic Hubbard model: (spin-full fermions with interactions).

.. todo ::
    Should we redefine the hopping to have a negative prefactor by default if t = 1?
    How to notify users of the class???
"""
# Copyright 2018 TeNPy Developers

import numpy as np

from .model import CouplingMPOModel, NearestNeighborModel
from ..tools.params import get_parameter
from ..networks.site import SpinHalfFermionSite


class FermionicHubbardModel(CouplingMPOModel):
    r"""Spin-1/2 fermionic Hubbard model in 1D.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i, j \rangle, i < j, \sigma} t (c^{\dagger}_{\sigma, i} c_{\sigma j} + h.c.)
            + \sum_i U n_{\uparrow, i} n_{\downarrow, i}
            + \sum_i \mu ( n_{\uparrow, i} + n_{\downarrow, i} )
            +  \sum_{\langle i, j \rangle, i< j, \sigma} V
                       (n_{\uparrow,i} + n_{\downarrow,i})(n_{\uparrow,j} + n_{\downarrow,j})


    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs.
    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    .. warning ::
        Using the Jordan-Wigner string (``JW``) is crucial to get correct results!
        See :doc:`/intro_JordanWigner` for details.

    Parameters
    ----------
    cons_N : {'N' | 'parity' | None}
        Whether particle number is conserved,
        see :class:`~tenpy.networks.site.SpinHalfFermionSite` for details.
    cons_Sz : {'Sz' | 'parity' | None}
        Whether spin is conserved,
        see :class:`~tenpy.networks.site.SpinHalfFermionSite` for details.
    t, U, mu : float | array
        Parameters as defined for the Hamiltonian above
    lattice : str | :class:`~tenpy.models.lattice.Lattice`
        Instance of a lattice class for the underlaying geometry.
        Alternatively a string being the name of one of the Lattices defined in
        :mod:`~tenpy.models.lattice`, e.g. ``"Chain", "Square", "HoneyComb", ...``.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions along the x-direction.
        For 'infinite' boundary conditions, repeat the unit cell in x-direction.
        Coupling boundary conditions in x-direction are chosen accordingly.
        Only used if `lattice` is a string.
    order : string
        Ordering of the sites in the MPS, e.g. 'default', 'snake';
        see :meth:`~tenpy.models.lattice.Lattice.ordering`.
        Only used if `lattice` is a string.
    L : int
        Lenght of the lattice.
        Only used if `lattice` is the name of a 1D Lattice.
    Lx, Ly : int
        Length of the lattice in x- and y-direction.
        Only used if `lattice` is the name of a 2D Lattice.
    bc_y : 'ladder' | 'cylinder'
        Boundary conditions in y-direction.
        Only used if `lattice` is the name of a 2D Lattice.
    """

    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        cons_N = get_parameter(model_params, 'cons_N', 'N', self.name)
        cons_Sz = get_parameter(model_params, 'cons_Sz', 'Sz', self.name)
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site

    def init_terms(self, model_params):
        # 0) Read out/set default parameters.
        t = get_parameter(model_params, 't', 1., self.name, True)
        U = get_parameter(model_params, 'U', 0, self.name, True)
        V = get_parameter(model_params, 'V', 0, self.name, True)
        mu = get_parameter(model_params, 'mu', 0., self.name, True)

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(mu, 0, 'Ntot')
            self.add_onsite(U, 0, 'NuNd')
        for u1, u2, dx in self.lat.nearest_neighbors:
            self.add_coupling(t, u1, 'Cdu', u2, 'Cu', dx)
            self.add_coupling(np.conj(t), u2, 'Cdu', u1, 'Cu', -dx)  # h.c.
            self.add_coupling(t, u1, 'Cdd', u2, 'Cd', dx)
            self.add_coupling(np.conj(t), u2, 'Cdd', u1, 'Cd', -dx)  # h.c.
            self.add_coupling(V, u1, 'Ntot', u2, 'Ntot', dx)


class FermionicHubbardChain(FermionicHubbardModel, NearestNeighborModel):
    """The :class:`FermionicHubbardModel` on a Chain, suitable for TEBD.

    See the :class:`FermionicHubbardModel` for the documentation of parameters.
    """

    def __init__(self, model_params):
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)
