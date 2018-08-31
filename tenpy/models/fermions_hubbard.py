"""Fermionic Hubbard model: (spin-full fermions with interactions).

.. todo ::
    Work in checks for common errors and raise some exceptions?
"""
# Copyright 2018 TeNPy Developers

from .lattice import Chain
from .model import CouplingModel, NearestNeighborModel, MPOModel
from ..tools.params import get_parameter, unused_parameters
from tenpy.networks.site import SpinHalfFermionSite


class FermionicHubbardChain(CouplingModel, NearestNeighborModel, MPOModel):
    r"""Spin-1/2 fermionic Hubbard model in 1D.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i, j \rangle, \sigma} t (c^{\dagger}_{\sigma, i} c_{\sigma j} + h.c.)
            + \sum_i U n_{\uparrow, i} n_{\downarrow, i}
            + \sum_i \mu ( n_{\uparrow, i} + n_{\downarrow, i} )
            +  \sum_{\langle i, j \rangle, \sigma} V
                       (n_{\uparrow,i} + n_{\downarrow,i})(n_{\uparrow,j} + n_{\downarrow,j})


    All parameters are collected in a single dictionary `model_param` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    Parameters
    ----------
    L : int
        Length of the chain
    t, U, mu : float | array
        Parameters as defined for the Hamiltonian above
    cons_N : {'N' | 'parity' | None}
        Whether particle number is conserved,
        see :class:`~tenpy.networks.site.SpinHalfFermionSite` for details.
    cons_Sz : {'Sz' | 'parity' | None}
        Whether spin is conserved,
        see :class:`~tenpy.networks.site.SpinHalfFermionSite` for details.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions. Coupling boundary conditions are chosen appropriately.
    """

    def __init__(self, model_param):
        # 0) Read out/set default parameters.
        L = get_parameter(model_param, 'L', 2, self.__class__)
        t = get_parameter(model_param, 't', 1., self.__class__)
        U = get_parameter(model_param, 'U', 0, self.__class__)
        V = get_parameter(model_param, 'V', 0, self.__class__)
        mu = get_parameter(model_param, 'mu', 0., self.__class__)
        bc_MPS = get_parameter(model_param, 'bc_MPS', 'finite', self.__class__)
        cons_N = get_parameter(model_param, 'cons_N', 'N', self.__class__)
        cons_Sz = get_parameter(model_param, 'cons_Sz', 'Sz', self.__class__)
        unused_parameters(model_param, self.__class__)

        # 1) Define the site and the lattice.
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        bc = 'periodic' if bc_MPS == 'infinite' else 'open'
        lat = Chain(L, site, bc=bc, bc_MPS=bc_MPS)
        # 2) Initialize CouplingModel
        CouplingModel.__init__(self, lat)

        # 3) Add terms of the hamiltonian.
        # 3a) On-site terms
        self.add_onsite(mu, 0, 'Ntot')
        self.add_onsite(U, 0, 'NuNd')

        # 3b) Coupling terms
        self.add_coupling(t, 0, 'Cdu', 0, 'Cu', 1, 'JW', True)
        self.add_coupling(t, 0, 'Cdu', 0, 'Cu', -1, 'JW', True)
        self.add_coupling(t, 0, 'Cdd', 0, 'Cd', 1, 'JW', True)
        self.add_coupling(t, 0, 'Cdd', 0, 'Cd', -1, 'JW', True)
        self.add_coupling(V, 0, 'Ntot', 0, 'Ntot', 1)
        # 4) Initialize MPO and bonds (order does not matter).
        MPOModel.__init__(self, lat, self.calc_H_MPO())
        NearestNeighborModel.__init__(self, lat, self.calc_H_bond())
