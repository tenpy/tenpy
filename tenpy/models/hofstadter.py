"""Cold atomic (Harper-)Hofstadter model on a strip or cylinder.

.. todo ::
    WARNING: These models are still under development and not yet tested for correctness.
    Use at your own risk!
    Gauge-based hopping parametres currently assume Lx, Ly to be the size of the magnetic unit cell. MUC needs to be independently defined!
    Replicate known results to confirm models work correctly.
    Implement different gauges (landau_y, symmetric/periodic, ...)
    Add assertions for consistency between gauge and lattice
    Move hop_x, hop_y computation based on gauge to helper function outside classes?
    Long term: implement different lattices
"""
# Copyright 2018 TeNPy Developers

import numpy as np

from .lattice import Square
from ..networks.site import BosonSite, FermionSite
from .model import CouplingModel, MPOModel, CouplingMPOModel
from ..tools.params import get_parameter, unused_parameters

__all__ = ['HofstadterBosons', 'HofstadterFermions']


class HofstadterFermions(CouplingMPOModel):
    r"""Fermions on a square lattice with magnetic flux.

    For now, the Hamiltonian reads:

    .. math ::
        H = - \sum_{x, y} \mathtt{Jx} (c^\dagger_{x,y} c_{x+1,y} + h.c.)   \\
            - \sum_{x, y} \mathtt{Jy} (e^{i \mathtt{phi} x} c^\dagger_{x,y} c_{x,y+1} + h.c.)
            - \sum_{x, y} \mathtt{mu} n_{x,y}

    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    .. todo :
        Add different gauges
        HofstadterFermions passes nosetesting. However, HofstadterFermions threw errors, the fixes for which are not present in HofstadterFermions.

    Parameters
    ----------
    Lx, Ly : int
        Size of the simulation unit cell in terms of lattice sites.
    filling : tuple
        Average number of fermions per site, defined as a fraction (numerator, denominator)
        Changes the definition of ``'dN'`` in the :class:`~tenpy.networks.site.FermionSite`.
    Jx, Jy, phi: float
        Hamiltonian parameters as defined above.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions along the x-direction.
        For 'infinite' boundary conditions, repeat the unit cell in x-direction.
        Coupling boundary conditions in x-direction are chosen accordingly.
    bc_x : 'periodic' | 'infinite'
        Lattice boundary conditions in x-direction
    bc_y : 'ladder' | 'cylinder'
        Lattice boundary conditions in y-direction.
    conserve : {'N' | 'parity' | None}
        What quantum number to conserve.
    order : string
        Ordering of the sites in the MPS, e.g. 'default', 'snake';
        see :meth:`~tenpy.models.lattice.Lattice.ordering`.
    phi : tuple
        Magnetic flux density, defined as a fraction (numerator, denominator)
    phi_ext : float
        External magnetic flux 'threaded' through the cylinder.
    gauge : 'landau_x' | 'landau_y' | 'symmetric'
        Choice of the gauge used for the magnetic field. This changes the magnetic unit cell.
    """

    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        conserve = get_parameter(model_params, 'conserve', 'N', self.name)
        filling = get_parameter(model_params, 'filling', (1, 8), self.name)
        filling = filling[0] / filling[1]
        site = FermionSite(conserve=conserve, filling=filling)
        return site

    def init_lattice(self, model_params):
        bc_MPS = get_parameter(model_params, 'bc_MPS', 'infinite', self.name)
        order = get_parameter(model_params, 'order', 'default', self.name)
        site = self.init_sites(model_params)
        Lx = get_parameter(model_params, 'Lx', 3, self.name)
        Ly = get_parameter(model_params, 'Ly', 4, self.name)
        bc_x = 'periodic' if bc_MPS == 'infinite' else 'open'
        bc_x = get_parameter(model_params, 'bc_x', bc_x, self.name)
        bc_y = get_parameter(model_params, 'bc_y', 'cylinder', self.name)
        assert bc_y in ['cylinder', 'ladder']
        bc_y = 'periodic' if bc_y == 'cylinder' else 'open'
        if bc_MPS == 'infinite' and bc_x == 'open':
            raise ValueError("You need to use 'periodic' `bc_x` for infinite systems!")
        lat = Square(Lx, Ly, site, order=order, bc=[bc_x, bc_y], bc_MPS=bc_MPS)  # LS are Lx, Ly, site defined?
        return lat

    def init_terms(self, model_params):
        Jx = get_parameter(model_params, 'Jx', 1., self.name)
        Jy = get_parameter(model_params, 'Jy', 1., self.name)
        phi_pq = get_parameter(model_params, 'phi', (1, 3), self.name)
        phi = 2 * np.pi * phi_pq[0] / phi_pq[1]
        phi_ext = get_parameter(model_params, 'phi_ext', 0., self.name)
        mu = get_parameter(model_params, 'mu', 1., self.name, True)
        gauge = get_parameter(model_params, 'gauge', 'landau_x', self.name)

        # 6) add terms of the Hamiltonian
        self.add_onsite(-mu, 0, 'N')
        Lx = self.lat.shape[0]
        Ly = self.lat.shape[1]

        if gauge == 'landau_x':
            assert Lx % phi_pq[1] == 0, "Flux density inconsistent with Lx in Landau-x gauge."
            # hopping in x-direction: uniform
            hop_x = -Jx
            # hopping in y-direction:
            # The hopping amplitudes depend on position -> use an array for couplings.
            # If the array is smaller than the actual number of couplings,
            # it is 'tiled', i.e. repeated periodically, see also tenpy.tools.to_array().
            # (Lx, 1) can be tiled to (Lx,Ly-1) for 'ladder' and (Lx, Ly) for 'cylinder' bc.
            hop_y = -Jy * np.exp(1.j * phi * np.arange(Lx)[:, np.newaxis])  # has shape (Lx, 1)
        elif gauge == 'landau_y':
            assert Ly % phi_pq[1] == 0, "Flux density inconsistent with Ly in Landau-y gauge."
            # hopping in y-direction: uniform
            hop_y = -Jy
            # hopping in x-direction:
            # The hopping amplitudes depend on position -> use an array for couplings.
            # If the array is smaller than the actual number of couplings,
            # it is 'tiled', i.e. repeated periodically, see also tenpy.tools.to_array().
            # (1, Ly) can be tiled to (Lx,Ly-1) for 'ladder' and (Lx, Ly) for 'cylinder' bc.
            hop_x = -Jx * np.exp(1.j * phi * np.arange(Ly)[np.newaxis, :])  # has shape (1, Ly)
        elif gauge == 'symmetric':
            assert Lx % phi_pq[1] == 0, "Flux density inconsistent with Lx in symmetric gauge."
            assert Ly % phi_pq[1] == 0, "Flux density inconsistent with Ly in symmetric gauge."
            hop_x = -Jx * np.exp(1.j * (phi/2) * np.arange(Ly)[:, np.newaxis])
            hop_y = -Jy * np.exp(1.j * (phi/2) * np.arange(Lx)[np.newaxis, :])
        else:
            raise NotImplementedError()
        self.add_coupling(hop_x, 0, 'Cd', 0, 'C', (1, 0))
        self.add_coupling(np.conj(hop_x), 0, 'Cd', 0, 'C', (-1, 0))  # h.c.
        dy = np.array([0, 1])
        hop_y = self.coupling_strength_add_ext_flux(hop_y, dy, [0, phi_ext])
        self.add_coupling(hop_y, 0, 'Cd', 0, 'C', dy)
        self.add_coupling(np.conj(hop_y), 0, 'Cd', 0, 'C', -dy)  # h.c.


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
        Add different gauges


    Parameters
    ----------
    Lx, Ly : int
        Size of the simulation unit cell in terms of lattice sites.
    N_max : int
        Maximum number of bosons per site.
    filling : tuple
        Average number of fermions per site, defined as a fraction (numerator, denominator)
        Changes the definition of ``'dN'`` in the :class:`~tenpy.networks.site.BosonSite`.
    Jx, Jy, phi, mu, U: float
        Hamiltonian parameters as defined above.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions along the x-direction.
        For 'infinite' boundary conditions, repeat the unit cell in x-direction.
        Coupling boundary conditions in x-direction are chosen accordingly.
    bc_x : 'periodic' | 'infinite'
        Boundary conditions in x-direction
    bc_y : 'ladder' | 'cylinder'
        Boundary conditions in y-direction.
    conserve : {'N' | 'parity' | None}
        What quantum number to conserve.
    order : string
        Ordering of the sites in the MPS, e.g. 'default', 'snake';
        see :meth:`~tenpy.models.lattice.Lattice.ordering`
    phi : tuple
        Magnetic flux density, defined as a fraction (numerator, denominator)
    phi_ext : float
        External magnetic flux 'threaded' through the cylinder.
    gauge : 'landau_x' | 'landau_y' | 'symmetric'
        Choice of the gauge used for the magnetic field. This changes the magnetic unit cell..
    """

    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        Nmax = get_parameter(model_params, 'Nmax', 3, self.__class__)
        conserve = get_parameter(model_params, 'conserve', 'N', self.name)
        filling = get_parameter(model_params, 'filling', (1, 8), self.name)
        filling = filling[0] / filling[1]
        site = BosonSite(Nmax=Nmax, conserve=conserve, filling=filling)
        return site

    def init_lattice(self, model_params):
        bc_MPS = get_parameter(model_params, 'bc_MPS', 'infinite', self.name)
        order = get_parameter(model_params, 'order', 'default', self.name)
        site = self.init_sites(model_params)
        Lx = get_parameter(model_params, 'Lx', 4, self.name)
        Ly = get_parameter(model_params, 'Ly', 6, self.name)
        bc_x = 'periodic' if bc_MPS == 'infinite' else 'open'  # Next line needs default
        bc_x = get_parameter(model_params, 'bc_x', bc_x, self.name)
        bc_y = get_parameter(model_params, 'bc_y', 'cylinder', self.name)
        assert bc_y in ['cylinder', 'ladder']
        bc_y = 'periodic' if bc_y == 'cylinder' else 'open'
        if bc_MPS == 'infinite' and bc_x == 'open':
            raise ValueError("You need to use 'periodic' `bc_x` for infinite systems!")
        lat = Square(Lx, Ly, site, bc=[bc_x, bc_y], bc_MPS=bc_MPS)
        return lat

    def init_terms(self, model_params):
        # TODO Lx, Ly now get a default twice, which is ugly (done for debugging reasons). Figure out best way to avoid this.
        Lx = get_parameter(model_params, 'Lx', 4, self.name)
        Ly = get_parameter(model_params, 'Ly', 6, self.name)
        Jx = get_parameter(model_params, 'Jx', 1., self.name)
        Jy = get_parameter(model_params, 'Jy', 1., self.name)
        phi_pq = get_parameter(model_params, 'phi', (1, 4), self.name)
        phi = 2 * np.pi * phi_pq[0] / phi_pq[1]
        phi_ext = get_parameter(model_params, 'phi_ext', 0., self.name)
        mu = get_parameter(model_params, 'mu', 1., self.name, True)
        U = get_parameter(model_params, 'U', 0, self.name)
        gauge = get_parameter(model_params, 'gauge', 'landau_x', self.name)

        # 6) add terms of the Hamiltonian
        self.add_onsite(np.asarray(U) / 2, 0, 'NN')
        self.add_onsite(-np.asarray(U) / 2 - np.asarray(mu), 0, 'N')

        if gauge == 'landau_x':
            assert Lx % phi_pq[1] == 0, "Flux density inconsistent with Lx in Landau-x gauge."
            # hopping in x-direction: uniform
            # hopping in y-direction:
            # The hopping amplitudes depend on position -> use an array for couplings.
            # If the array is smaller than the actual number of couplings,
            # it is 'tiled', i.e. repeated periodically, see also tenpy.tools.to_array().
            # (Lx, 1) can be tiled to (Lx,Ly-1) for 'ladder' and (Lx, Ly) for 'cylinder' bc.
            hop_x = -Jx  
            hop_y = -Jy * np.exp(1.j * phi * np.arange(Lx)[:, np.newaxis])  # has shape (Lx, 1)
        elif gauge == 'landau_y':
            assert Ly % phi_pq[1] == 0, "Flux density inconsistent with Ly in Landau-y gauge."
            # hopping in y-direction: uniform
            hop_y = -Jy
            # hopping in x-direction:
            # The hopping amplitudes depend on position -> use an array for couplings.
            # If the array is smaller than the actual number of couplings,
            # it is 'tiled', i.e. repeated periodically, see also tenpy.tools.to_array().
            # (1, Ly) can be tiled to (Lx,Ly-1) for 'ladder' and (Lx, Ly) for 'cylinder' bc.
            hop_x = -Jx * np.exp(1.j * phi * np.arange(Ly)[np.newaxis, :])  # has shape (1, Ly)
        elif gauge == 'symmetric':
            assert Lx % phi_pq[1] == 0, "Flux density inconsistent with Lx in symmetric gauge."
            assert Ly % phi_pq[1] == 0, "Flux density inconsistent with Ly in symmetric gauge."
            hop_x = -Jx * np.exp(1.j * (phi/2) * np.arange(Ly)[:, np.newaxis])
            hop_y = -Jy * np.exp(1.j * (phi/2) * np.arange(Lx)[np.newaxis, :])
        else:
            raise NotImplementedError()

        self.add_coupling(hop_x, 0, 'Bd', 0, 'B', [1, 0])
        self.add_coupling(np.conj(hop_x), 0, 'Bd', 0, 'B', [-1, 0])  # h.c.
        dy = np.array([0, 1])
        hop_y = self.coupling_strength_add_ext_flux(hop_y, dy, [0, phi_ext])
        self.add_coupling(hop_y, 0, 'Bd', 0, 'B', dy)
        self.add_coupling(np.conj(hop_y), 0, 'Bd', 0, 'B', -dy)  # h.c.