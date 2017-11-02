"""A TenPyLight implementation of the cold atomic (Harper-)Hofstadter model on a strip or cylinder.

.. todo::
	Implement the Hamiltonian.
"""

from __future__ import division
import numpy as np

from .lattice import Lattice
from tenpy.networks.site import BosonSite
from .model import CouplingModel, NearestNeighborModel, MPOModel
from ..linalg import np_conserved as npc
from ..tools.params import get_parameter, unused_parameters


__all__ = ['cold_hof_model']


class cold_hof_model(CouplingModel, NearestNeighborModel, MPOModel):
    """To be implemented.

    Parameters (model_pars)
    ----------
    Lx, Ly : int
        size of the simulation unit cell (in terms of magnetic unit cells)
    mx, my : int
    	size of the magnetic unit cell
    N_max : int
    	maximum number of bosons per site
    filling : float
    	average number of bosons per site
	Jx, Jy, phi_ext, kappa, omega, delta, u : float
		Hamiltonian parameters
	bc : {0 | 1}
		boundary conditions along the circumference. 0 = open, 1 = periodic.
	conserve : {'N' | 'parity' | None'}
		What quantum number to conserve. Right now, BosonSite cannot conserve N and parity at the
		same time.
	verbose : int
    """

    def __init__(self, model_pars):
    	# 0) read out/set default parameters
    	Lx = get_parameter('Lx', 3, self.__class__)
    	Ly = get_parameter('Ly', 1, self.__class__)
    	mx = get_parameter('mx', 4, self.__class__)
    	my = get_parameter('my', 1, self.__class__)
    	N_max = get_parameter('N_max', 3, self.__class__)  # max no. of on-site bosons.
    	filling = get_parameter('filling', 0.125, self.__class__)
    	Jx = get_parameter('Jx', 1., self.__class__)
    	Jy = get_parameter('Jy', 1., self.__class__)
    	phi_ext = get_parameter('phi_ext', 0, self.__class__)
    	kappa = get_parameter('kappa', 0, self.__class__)
    	omega = get_parameter('omega', 0, self.__class__)
    	delta = get_parameter('delta', 0, self.__class__)
    	u = get_parameter('u', 0, self.__class__)
    	bc = get_parameter('bc', 1, self.__class__)  # = bc around the circumference
    	conserve = get_parameter('conserve', 'N', self.__class__)
    	verbose = get_parameter('verbose', 1, self.__class__)
    	unused_parameters(model_pars, self.__class__)

    	# 1-3) Define the sites.
        site = BosonSite(Nmax=N_max, conserve='N', filling=filling)

        # 4) Lattice
        lat = Lattice

