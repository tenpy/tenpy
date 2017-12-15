"""Simple bosonic chain model.

Uniform lattice of bosons, including:
- Single-particle hopping
- Nearest-neighbour pair hopping
- U(1) symmetry-breaking fields
- density-density ineraction

.. todo ::
    Work out the hamiltonian, give example Hamiltonian in docstrings.
    Add correct Hamiltonian entries using add_onsite() and add_coupling()
    Work in checks for common errors and raise some exceptions.
    Clean up.
    Run some tests, and perhaps benchmarks comparing to old TenPy?
    Write example simulation code?
"""

# Original TenPy Hamiltonian: H = - sum_{j, r > 0} t_{j r} psi^D_{j+r} psi_{j} + h.c.
# This includes only nn-hopping and nothing else...

# H_{one-body} = (mu * N) + ( U * N (N - 1) )

import numpy as np

from .lattice import Chain
from ..networks.site import BosonSite
from .model import CouplingModel, NearestNeighborModel, MPOModel
from ..tools.params import get_parameter, unused_parameters
from ..tools.misc import any_nonzero


class BosonChain(CouplingModel, MPOModel, NearestNeighborModel):
    def __init__(self, model_param):
        # 0) Read and set parameters.
        verbose = get_parameter(model_param, 'verbose', 1, self.__class__)
        L = get_parameter(model_param, 'L', 1, self.__class_)
        nu = get_parameter(model_param, 'nu', 1, self.__class__)  #What is this?
        n_max = get_parameter(model_param, 'n_max', 3, self.__class__)
        filling = get_parameter(model_param, 'filling', 0.5, self.__class__)
        bc_MPS = get_parameter(model_param, 'bc_MPS', 'finite', self.__class__)
        t = get_parameter(model_param, 't1', 1., self.__class__)
        U = get_parameter(model_param, 'U', 0, self.__class__)
        mu = get_parameter(model_param, 'mu', 0, self.__class__)
        conserve = get_parameter(model_param, 'conserve', 'N',
                                 self.__class__)  # Supported: 'N', 'parity' or None
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
        self.add_onsite(U, 0, 'NN')  # Do we want longer-range dens-dens or just onsite/

        # 3b) coupling terms.
        self.add_coupling(t, 0, 'Bd', 0, 'B', 1)  # Single-particle hopping
        self.add_coupling(t, 0, 'B', 0, 'Bd', 1)

        # 4) Initialize MPO
        MPOModel.__init__(self, lat, self.calc_H_MPO())

        # 5) Initialize H bond  # LS: what does this mean?
        NearestNeighborModel.__init__(self, self.lat, self.calc_H_bond())
