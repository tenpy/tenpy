"""Fermionic Hubbard model. 

''todo::
	Decide between on-site or nn density-density (mediated by U)
	Work in checks for common errors and raise some exceptions?
	Run some tests, and perhaps benchmarks comparing to old TenPy?
	Write example simulation code?
"""

from __future__ import division
import numpy as np

from .lattice import Chain
from .model import CouplingModel, NearestNeighborModel, MPOModel
from ..tools.params import get_parameter, unused_parameters
from tenpy.networks.site import SpinHalfFermionSite


class FermionicHubbardChain(CouplingModel, NearestNeighborModel, MPOModel):
    r"""Spin-1/2 fermionic Hubbard model in 1D.

	The Hamiltonian reads:

	.. math :: 
		H = \sum_{\langle i, j \rangle, \sigma} t (c^{\dagger}_{\uparrow, i} c_{\uparrow j} + h.c.)
			+ \sum_i U n_{\uparrow, i} n_{\downarrow, i} 
			+ \sum_i \mu ( n_{\uparrow, i} + n_{\downarrow, i} )

	All parameters are collected in a single dictionary `model_param` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    Parameters
    ----------
    L : int
        Length of the chain
    t, U, mu : float | array
    	Parameters as defined for the Hamiltonian above
    cons_N : {'N' | 'parity' | None}
    	Whether particle number is conserved, see :class:`SpinHalfFermionSite` for details.
    cons_Sz : {'Sz' | 'parity' | None}
        Whether spin is conserved, see :class:`SpinHalfFermionSite` for details.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions. Coupling boundary conditions are chosen appropriately.
	"""

    def __init__(self, model_param):
        # 0) Read out/set default parameters.
        L = get_parameter(model_param, 'L', 2, self.__class__)
        t = get_parameter(model_param, 't', 1., self.__class__)
        U = get_parameter(model_param, 'U', 0, self.__class__)
        mu = get_parameter(model_param, 'mu', 0., self.__class__)
        bc_MPS = get_parameter(model_param, 'bc_MPS', 'finite', self.__class__)
        cons_N = get_parameter(model_param, 'cons_N', 'N', self.__class__)
        cons_Sz = get_parameter(model_param, 'cons_Sz', 'Sz', self.__class__)
        unused_parameters(model_param, self.__class__)

        # 1) Define the site and the lattice.
        site = SpinHalfFermionSite(conserve=conserve)
        lat = Chain(L, site, bc_MPS=bc_MPS)

        # 2) Initialize CouplingModel
        if bc_MPS == 'periodic' or bc_MPS == 'infinite':  #TODO Is this correct for mps 'periodic'?
            bc_coupling == 'periodic'
        else:
            bc_coupling == 'open'
        CouplingModel.__init__(self, lat, bc_coupling)

        # 3) Add terms of the hamiltonian.
        # 3a) On-site terms
        self.add_onsite(mu, 0, 'Ntot')

        # 3b) Coupling terms
        self.add_coupling(t, 0, 'Cdu', 0, 'Cu', 1, 'JW', True)
        self.add_coupling(t, 0, 'Cu', 0, 'Cdu', 1, 'JW', True)
        self.add_coupling(t, 0, 'Cdd', 0, 'Cd', 1, 'JW', True)
        self.add_coupling(t, 0, 'Cd', 0, 'Cdd', 1, 'JW', True)
        self.add_coupling(U, 0, 'Nu', 0, 'Nd', 0)  #TODO Should this be done as onsite?

        # 4) Initialize MPO and bonds (order does not matter).
        MPOModel.__init__(self, lat, self.calc_H_MPO())
        NearestNeighborModel.__init__(self, lat, self.calc_H_bond())
