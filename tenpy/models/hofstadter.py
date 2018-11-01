"""Cold atomic (Harper-)Hofstadter model on a strip or cylinder.

.. todo ::
    WARNING: These models are still under development and not yet tested for correctness.
    Use at your own risk!

.. todo ::
    switch to using the CouplingMPOModel
"""
# Copyright 2018 TeNPy Developers

import numpy as np

from .lattice import Square
from ..networks.site import BosonSite, FermionSite
from .model import CouplingModel, MPOModel
from ..tools.params import get_parameter, unused_parameters

__all__ = ['HofstadterBosons', 'HofstadterFermions']


class HofstadterFermions(CouplingModel, MPOModel):
    r"""Fermions on a square lattice with magnetic flux.

    For now, the Hamiltonian reads:

    .. math ::
        H = - \sum_{x, y} \mathtt{Jx} (c^\dagger_{x,y} c_{x+1,y} + h.c.)   \\
            - \sum_{x, y} \mathtt{Jy} (e^{i \mathtt{phi} x} c^\dagger_{x,y} c_{x,y+1} + h.c.)


    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    .. todo :
        More complicated hopping amplitudes...


    Parameters
    ----------
    Lx, Ly : int
        Size of the simulation unit cell in terms of lattice sites.
    filling : float
        Average number of fermions per site.
        Changes the definition of ``'dN'`` in the :class:`~tenpy.networks.site.FermionSite`.
    Jx, Jy, phi: float
        Hamiltonian parameters as defined above.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions along the x-direction.
        For 'infinite' boundary conditions, repeat the unit cell in x-direction.
        Coupling boundary conditions in x-direction are chosen accordingly.
    bc_y : 'ladder' | 'cylinder'
        Boundary conditions in y-direction.
    conserve : {'N' | 'parity' | None}
        What quantum number to conserve.
    order : string
        Ordering of the sites in the MPS, e.g. 'default', 'snake';
        see :meth:`~tenpy.models.lattice.Lattice.ordering`.
    """

    def __init__(self, model_params):
        # 0) read out/set default parameters
        Lx = get_parameter(model_params, 'Lx', 4, self.__class__)
        Ly = get_parameter(model_params, 'Ly', 2, self.__class__)
        filling = get_parameter(model_params, 'filling', 0.125, self.__class__)
        Jx = get_parameter(model_params, 'Jx', 1., self.__class__)
        Jy = get_parameter(model_params, 'Jy', 1., self.__class__)
        phi = get_parameter(model_params, 'phi', 2. * np.pi / Lx, self.__class__)
        bc_MPS = get_parameter(model_params, 'bc_MPS', 'infinite', self.__class__)
        bc_y = get_parameter(model_params, 'bc_y', 'cylinder', self.__class__)
        conserve = get_parameter(model_params, 'conserve', 'N', self.__class__)
        order = get_parameter(model_params, 'order', 'default', self.__class__)
        unused_parameters(model_params, self.__class__)

        assert bc_y in ['cylinder', 'ladder']

        # 1-4) Define the sites and the lattice.
        site = FermionSite(conserve=conserve, filling=filling)
        bc_x = 'periodic' if bc_MPS == 'infinite' else 'open'
        bc_y = 'periodic' if bc_y == 'cylinder' else 'open'
        lat = Square(Lx, Ly, site, order, bc=[bc_x, bc_y], bc_MPS=bc_MPS)
        # 5) initialize CouplingModel
        CouplingModel.__init__(self, lat)

        # 6) add terms of the Hamiltonian
        #self.add_onsite(np.asarray(U)/2, 0, 'NN')
        #self.add_onsite(-np.asarray(U)/2 - np.asarray(mu), 0, 'N')

        # hopping in x-direction: uniform
        self.add_coupling(-Jx, 0, 'Cd', 0, 'C', (1, 0), 'JW', True)
        self.add_coupling(-Jx, 0, 'Cd', 0, 'C', (-1, 0), 'JW', True)
        #self.add_coupling(Jx, 0, 'C', 0, 'Cd', (1, 0), 'JW', True)
        # hopping in y-direction:
        # The hopping amplitudes depend on position -> use an array for couplings.
        # If the array is smaller than the actual number of couplings,
        # it is 'tiled', i.e. repeated periodically, see also tenpy.tools.to_array().
        # (Lx, 1) can be tiled to (Lx,Ly-1) for 'ladder' and (Lx, Ly) for 'cylinder' bc.
        hop_y = -Jy * np.exp(1.j * phi * np.arange(Lx)[:, np.newaxis])  # has shape (Lx, 1)
        self.add_coupling(hop_y, 0, 'Cd', 0, 'C', [0, 1], 'JW', True)
        self.add_coupling(np.conj(hop_y), 0, 'Cd', 0, 'C', [0, -1], 'JW', True)

        # 7) initialize MPO
        MPOModel.__init__(self, lat, self.calc_H_MPO())


class HofstadterBosons(CouplingModel, MPOModel):
    r"""Bosons on a square lattice with magnetic flux.

    For now, the Hamiltonian reads:

    .. math ::
        H = - \sum_{x, y} \mathtt{Jx} (a^\dagger_{x+1,y} a_{x,y} + h.c.)   \\
            - \sum_{x, y} \mathtt{Jy} (e^{i \mathtt{phi} x} a^\dagger_{x,y+1} a_{x,y} + h.c.)   \\
            + \sum_{x, y} \frac{\mathtt{U}}{2} n_{x,y} (n_{x,y} - 1) - \mathtt{mu} n_{x,y}


    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    .. todo :
        More complicated hopping amplitudes...


    Parameters
    ----------
    Lx, Ly : int
        Size of the simulation unit cell in terms of lattice sites.
    N_max : int
        Maximum number of bosons per site.
    filling : float
        Average number of bosons per site.
        Changes the definition of ``'dN'`` in the :class:`~tenpy.networks.site.BosonSite`.
    Jx, Jy, phi, mu, U: float
        Hamiltonian parameters as defined above.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions along the x-direction.
        For 'infinite' boundary conditions, repeat the unit cell in x-direction.
        Coupling boundary conditions in x-direction are chosen accordingly.
    bc_y : 'ladder' | 'cylinder'
        Boundary conditions in y-direction.
    conserve : {'N' | 'parity' | None}
        What quantum number to conserve.
    order : string
        Ordering of the sites in the MPS, e.g. 'default', 'snake';
        see :meth:`~tenpy.models.lattice.Lattice.ordering`.
    """

    def __init__(self, model_params):
        # 0) read out/set default parameters
        Lx = get_parameter(model_params, 'Lx', 4, self.__class__)
        Ly = get_parameter(model_params, 'Ly', 2, self.__class__)
        N_max = get_parameter(model_params, 'N_max', 3, self.__class__)
        filling = get_parameter(model_params, 'filling', 0.125, self.__class__)
        Jx = get_parameter(model_params, 'Jx', 1., self.__class__)
        Jy = get_parameter(model_params, 'Jy', 1., self.__class__)
        phi = get_parameter(model_params, 'phi', 2. * np.pi / Lx, self.__class__)
        mu = get_parameter(model_params, 'mu', 0, self.__class__)
        U = get_parameter(model_params, 'U', 0, self.__class__)
        bc_MPS = get_parameter(model_params, 'bc_MPS', 'infinite', self.__class__)
        bc_y = get_parameter(model_params, 'bc_y', 'cylinder', self.__class__)
        conserve = get_parameter(model_params, 'conserve', 'N', self.__class__)
        order = get_parameter(model_params, 'order', 'default', self.__class__)
        unused_parameters(model_params, self.__class__)

        assert bc_y in ['cylinder', 'ladder']

        # 1-4) Define the sites and the lattice.
        site = BosonSite(Nmax=N_max, conserve=conserve, filling=filling)
        bc_x = 'periodic' if bc_MPS == 'infinite' else 'open'
        bc_y = 'periodic' if bc_y == 'cylinder' else 'open'
        lat = Square(Lx, Ly, site, order=order, bc=[bc_x, bc_y], bc_MPS=bc_MPS)
        # 5) initialize CouplingModel
        CouplingModel.__init__(self, lat)

        # 6) add terms of the Hamiltonian
        self.add_onsite(np.asarray(U) / 2, 0, 'NN')
        self.add_onsite(-np.asarray(U) / 2 - np.asarray(mu), 0, 'N')

        # hopping in x-direction: uniform
        self.add_coupling(-Jx, 0, 'B', 0, 'Bd', [1, 0])
        self.add_coupling(-Jx, 0, 'Bd', 0, 'B', [1, 0])
        # hopping in y-direction:
        # The hopping amplitudes depend on position -> use an array for couplings.
        # If the array is smaller than the actual number of couplings,
        # it is 'tiled', i.e. repeated periodically, see also tenpy.tools.to_array().
        # (Lx, 1) can be tiled to (Lx,Ly-1) for 'ladder' and (Lx, Ly) for 'cylinder' bc.
        hop_y = -Jy * np.exp(1.j * phi * np.arange(Lx)[:, np.newaxis])  # has shape (Lx, 1)
        self.add_coupling(hop_y, 0, 'Bd', 0, 'B', [0, 1])
        self.add_coupling(np.conj(hop_y), 0, 'B', 0, 'Bd', [0, 1])

        # 7) initialize MPO
        MPOModel.__init__(self, lat, self.calc_H_MPO())
