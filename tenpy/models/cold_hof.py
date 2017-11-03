"""A TenPyLight implementation of the cold atomic (Harper-)Hofstadter model on a strip or cylinder.

.. todo::
	Sort out the lattice (including correct bc handling for cylinder/strip).
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
	Jx, Jy, phi_0, phi_ext, kappa, omega, delta, u, mu : float
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
    	phi_0 = get_parameter('phi_0', 0., self.__class__)
    	phi_ext = get_parameter('phi_ext', 0, self.__class__)
    	kappa = get_parameter('kappa', 0, self.__class__)
    	omega = get_parameter('omega', 0, self.__class__)
    	delta = get_parameter('delta', 0, self.__class__)
    	u = get_parameter('u', 0, self.__class__)
    	mu = get_parameter('mu', 0, self.__class__)
    	bc = get_parameter('bc', 1, self.__class__)  # = bc around the circumference
    	conserve = get_parameter('conserve', 'N', self.__class__)
    	verbose = get_parameter('verbose', 1, self.__class__)
    	unused_parameters(model_pars, self.__class__)

    	# 1-3) Define the sites.
        site = BosonSite(Nmax=N_max, conserve='N', filling=filling)

        # 4) Lattice
        lat = Lattice([Lx * mx, Ly * my], unit_cell, order='default', bc_MPS='finite', basis=None, positions=None)
        bc_coupling = # TODO

		# 5) initialize CouplingModel
        CouplingModel.__init__(self, lat, bc_coupling)

        # 6) add terms of the Hamiltonian
        # TODO, self.add_onsite(), self.add_coupling()
        self.add_onsite(u/2, 0, 'NN')
        self.add_onsite((delta/2)*(-1), 0, 'N')  # Can we make this staggered?

        hop_y = - Jy * ( 1 + mu * (kappa / (2 * omega)) ** 2 )
        hop_x = - Jx * np.exp( 1.j * (np.pi / 2) * (index) - phi_0)  # TODO figure out index
        hop_x_hc = - Jx * np.exp( 1.j * (np.pi / 2) * (index) - phi_0)  # TODO figure out index
        self.add_coupling(hop_y, u1, 'B', u2, 'Bd', dx)  # TODO figure out u1, u2 and dx
        self.add_coupling(hop_y, u1, 'Bd', u2, 'B', dx)
        self.add_coupling(hop_x, u1, 'B', u2, 'Bd', dx)
        self.add_coupling(hop_x_hc, u1, 'Bd', u2, 'B', dx)

        # 7) initialize MPO
        MPOModel.__init__(self, lat, self.calc_H_MPO())
        # 8) initialize bonds (the order of 7/8 doesn't matter)
        NearestNeighborModel.__init__(self, lat, self.calc_H_bond())