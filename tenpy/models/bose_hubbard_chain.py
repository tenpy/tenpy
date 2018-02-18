"""Simple Bose-Hubbard chain model.

.. todo::
    Work in checks for common errors and raise some exceptions?
    Run some tests, and perhaps benchmarks comparing to old TenPy?
    Write example simulation code?
"""
# Copyright 2018 TeNPy Developers

import numpy as np

from .lattice import Chain
from ..networks.site import BosonSite
from .model import CouplingModel, NearestNeighborModel, MPOModel
from ..tools.params import get_parameter, unused_parameters
from ..tools.misc import any_nonzero


class BoseHubbardChain(CouplingModel, MPOModel, NearestNeighborModel):
    r"""Spinless Bose-Hubbard model on a chain.

    The Hamiltonian is:

    .. math ::
        H = t \sum_i (b_i^{\dagger} b_{i+1} + b_{i+1}^{\dagger} b_i)
            + \frac{U}{2} \sum_i n_i (n_i - 1) + \mu \sum_i n_i

    Note that the signs of all parameters as defined in the Hamiltonian are positive.

    All parameters are collected in a single dictionary `model_param` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    Parameters
    ----------
    L : int
        Length of the chain
    n_max : int
        Maximum number of bosons per site.
    filling : float
        Average filling.
    conserve: {'N' | 'parity' | None}
        What should be conserved. See :class:`~tenpy.networks.Site.BosonSite`.
    t, U, Mu : float | array
        Couplings as defined in the Hamiltonian above.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions. Coupling boundary conditions are chosen appropriately.
    verbose : int
        Level of verbosity
    """

    def __init__(self, model_param):
        # 0) Read and set parameters.
        verbose = get_parameter(model_param, 'verbose', 1, self.__class__)
        L = get_parameter(model_param, 'L', 1, self.__class_)
        n_max = get_parameter(model_param, 'n_max', 3, self.__class__)
        filling = get_parameter(model_param, 'filling', 0.5, self.__class__)
        bc_MPS = get_parameter(model_param, 'bc_MPS', 'finite', self.__class__)
        t = get_parameter(model_param, 't', 1., self.__class__)
        U = get_parameter(model_param, 'U', 0, self.__class__)
        mu = get_parameter(model_param, 'mu', 0, self.__class__)
        conserve = get_parameter(model_param, 'conserve', 'N', self.__class__)
        unused_parameters(model_param, self.__class__)

        # 1) Sites and lattice.
        site = BosonSite(Nmax=n_max, conserve=conserve, filling=filling)
        lat = Chain(L, site, bc_MPS=bc_MPS)

        # 2) Initialize CouplingModel
        if bc_MPS == 'infinite' or bc_MPS == 'periodic':  #TODO Check if this is correct for mps 'periodic'
            bc_coupling = 'periodic'
        else:
            bc_coupling = 'open'
        CouplingModel.__init__(self, lat, bc_coupling)

        # 3) Build the Hamiltonian.
        # 3a) on-site terms.
        self.add_onsite(mu, 0, 'N')
        self.add_onsite(U, 0, 'NN')

        # 3b) coupling terms.
        self.add_coupling(t, 0, 'Bd', 0, 'B', 1)
        self.add_coupling(t, 0, 'B', 0, 'Bd', 1)

        # 4) Initialize MPO
        MPOModel.__init__(self, lat, self.calc_H_MPO())

        # 5) Initialize H bond  # LS: what does this mean?
        NearestNeighborModel.__init__(self, self.lat, self.calc_H_bond())
