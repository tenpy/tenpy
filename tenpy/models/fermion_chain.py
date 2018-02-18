# Copyright 2018 TeNPy Developers

import numpy as np

from .lattice import Chain
from .model import CouplingModel, NearestNeighborModel, MPOModel
from ..tools.params import get_parameter, unused_parameters

from tenpy.networks.site import FermionSite


class FermionChain(CouplingModel, NearestNeighborModel, MPOModel):
    r"""Spinless fermion chain with N conservation.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i,j\rangle, i<j}
              - \mathtt{J} (c^{\dagger}_i c_j + c^{\dagger}_j c_i) + \mathtt{V} n_i n_j \\
            - \sum_i
              \mathtt{mu} n_{i}

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs.
    All parameters are collected in a single dictionary `model_param` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    .. warning ::
        Using the Jordan-Wigner string (``JW``) is crucial to get correct results!
        See :doc:`../intro_JordanWigner` for details.

    Parameters
    ----------
    L : int
        Length of the chain
    J, V, mu : float | array
        Hopping, interaction and chemical potential as defined for the
        Hamiltonian above.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions. Coupling boundary conditions are chosen
        appropriately.
    """

    def __init__(self, model_param):
        # 0) read out/set default parameters
        L = get_parameter(model_param, 'L', 2, self.__class__)
        J = get_parameter(model_param, 'J', 1., self.__class__)
        V = get_parameter(model_param, 'V', 1., self.__class__)
        mu = get_parameter(model_param, 'mu', 0., self.__class__)
        bc_MPS = get_parameter(model_param, 'bc_MPS', 'finite', self.__class__)
        conserve = get_parameter(model_param, 'conserve', 'N', self.__class__)
        unused_parameters(model_param, self.__class__)
        # 1) - 3)
        site = FermionSite(conserve=conserve)
        # 4) lattice
        lat = Chain(L, site, bc_MPS=bc_MPS)
        bc_coupling = 'periodic' if bc_MPS == 'infinite' else 'open'
        # 5) initialize CouplingModel
        CouplingModel.__init__(self, lat, bc_coupling)
        # 6) add terms of the Hamiltonian
        # (u is always 0 as we have only one site in the unit cell)
        self.add_onsite(-np.asarray(mu), 0, 'N')
        J = np.asarray(J)  # convert to array: allow `array_like` J
        self.add_coupling(-J, 0, 'Cd', 0, 'C', 1, 'JW', True)  # (for a nearest neighbor model, we
        self.add_coupling(-J, 0, 'Cd', 0, 'C', -1, 'JW', True)  # could leave the `JW` away)
        self.add_coupling(V, 0, 'N', 0, 'N', 1)
        # 7) initialize MPO
        MPOModel.__init__(self, lat, self.calc_H_MPO())
        # 8) initialize bonds (the order of 7/8 doesn't matter)
        NearestNeighborModel.__init__(self, lat, self.calc_H_bond())
