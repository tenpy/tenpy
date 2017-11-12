"""Fermionic Hubbard model. 

''todo::
	Find the exact Hamiltonian to implement. Write in docstring.
	Class docstring.
	Add all correct terms from the Hamiltonian.
	Decide between on-site or nn density-density (mediated by U)
	Work in checks for common errors and raise some exceptions.
	Clean up.
	Run some tests, and perhaps benchmarks comparing to old TenPy?
	Write example simulation code?
	Should get_parameter or something else check for correct types? I.e. L = int etc.
"""

from __future__ import division
import numpy as np

from .lattice import Chain
from .model import CouplingModel, NearestNeighborModel, MPOModel
from ..tools.params import get_parameter, unused_parameters
from tenpy.networks.site import FermionSite


class FermionicHubbardChain(CouplingModel, NearestNeighborModel, MPOModel):

	def __init__(self, model_param):
		# 0) Read out/set default parameters.
		L = get_parameter(model_param, 'L', 2, self.__class__)
		J = get_parameter(model_param, 'J', 1., self.__class__)
		U = get_parameter(model_param, 'U', 0, self.__class__)
		mu = get_parameter(model_param, 'mu', 0., self.__class__)
		bc_MPS = get_parameter(model_param, 'bc_MPS', 'finite', self.__class__)
		conserve = get_parameter(model_param, 'conserve', 'N', self.__class__)  # Supported: 'N' or 'parity' or None
		verbose = get_parameter(model_param, 'verbose', 1, self.__class__)
		unused_parameters(model_param, self.__class__)

		# 1) Define the site and the lattice.
		site = FermionSite(conserve=conserve)
		lat = Chain(L ,site, bc_MPS=bc_MPS)
 
		# 2) Initialize CouplingModel
		if bc_MPS == 'periodic' or bc_MPS == 'infinite':  #TODO Check if this is correct for mps 'periodic'
			bc_coupling == 'periodic'
		else:
			bc_coupling == 'open'
		CouplingModel.__init__(self, lat, bc_coupling)

		# 3) Add terms of the hamiltonian.
		# 3a) On-site terms
		self.add_onsite(mu, 0, 'N')
		self.add_onsite(U, 0, 'NN')

		# 3b) Coupling terms
		self.add_coupling(J, 0, 'Cd', 0, 'C', 1)
		self.add_coupling(J, 0, 'C', 0, 'Cd', 1)  # Could also define 'Cd', 'C' with dx = -1
		self.add_coupling(U, 0, 'N', 0, 'N', 1)  # Should this be on-site?

		# 4) Initialize MPO and bonds (order does not matter).
		MPOModel.__init__(self, lat, self.calc_H_MPO())
		NearestNeighborModel.__init__(self, lat, self.calc_H_bond())