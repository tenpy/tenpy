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
import warnings

from .lattice import Square
from ..networks.site import BosonSite, FermionSite
from .model import CouplingModel, MPOModel, CouplingMPOModel
from ..tools.params import get_parameter, unused_parameters

__all__ = ['HofstadterBosons', 'HofstadterFermions']


def gauge_hopping(gauge, mx, my, Jx, Jy, phi, phi_pq):
    r"""Compute hopping amplitudes for the Hofstadter models based on a gauge choice.

    In the Hofstadter model, the magnetic field enters as an Aharonov-Bohm phase.
    This phase is dependent on a choice of gauge, which simultaneously defines a
    'magnetic unit cell' (MUC). 

    The magnetic unit cell is the smallest set of lattice plaquettes that 
    encloses an integer number of flux quanta. It can be user-defined by setting
    mx and my, but for common gauge choices is computed based on the flux 
    density.

    The gauge choices are:
        * 'landau_x': Landau gauge along the x-axis. The magnetic unit cell will
          have shape :math`(\mathtt{mx}, 1)`. For flux densities :math:`p/q`, mx will default to q.
          Example: at a flux density :math:`1/3`, the magnetic unit cell will have shape 
          :math:`(3,1)`, so it encloses exactly 1 flux quantum.
        * 'landau_y': Landau gauge along the y-axis. The magnetic unit cell will
          have shape :math`(1, \mathtt{my})`. For flux densities :math`p/q`, my will default to q.
          Example: at a flux density :math:`3/7`, the magnetic unit cell will have shape
          :math:`(1,7)`, so it encloses axactly 3 flux quanta.
        * 'symmetric': symmetric gauge. The magnetic unit cell will have shape
          :math:`(\mathtt{mx}, \mathtt{my})`, with :math:`mx = my`. For flux densities :math:`p/q`, 
          mx and my will default to :math:`\sqrt{q}`
          Example: at a flux density 4/9, the magnetic unit cell will have shape
          (3,3), so it encloses exactly 4 flux quanta.

    .. todo :
        Add periodic gauge (generalization of symmetric with mx, my unequal)
    
    Parameters
    ----------
    gauge : 'landau_x' | 'landau_y' | 'symmetric'
        Choice of the gauge
    mx, my : int
        Dimensions of the magnetic unit cell in terms of lattice sites.
    Jx, Jy: float
        `Bare' hopping amplitudes (without phase)
    phi : float
        Magnetic flux.
    phi : tuple
        Magnetic flux as a fraction p/q, defined as (p, q)
    """
    if gauge == 'landau_x':
        # hopping in x-direction: uniform
        # hopping in y-direction:
        # The hopping amplitudes depend on position -> use an array for couplings.
        # If the array is smaller than the actual number of couplings,
        # it is 'tiled', i.e. repeated periodically, see also tenpy.tools.to_array().
        # (mx, 1) can be tiled to (Lx,Ly-1) for 'ladder' and (Lx, Ly) for 'cylinder' bc.
        # If no magnetic unit cell size is defined, minimal size will be used.
        if mx == 0: mx = phi_pq[1]
        hop_x = -Jx
        hop_y = -Jy * np.exp(1.j * phi * np.arange(mx)[:, np.newaxis])  # has shape (Lx, 1)
    elif gauge == 'landau_y':
        # hopping in y-direction: uniform
        # hopping in x-direction:
        # The hopping amplitudes depend on position -> use an array for couplings.
        # If the array is smaller than the actual number of couplings,
        # it is 'tiled', i.e. repeated periodically, see also tenpy.tools.to_array().
        # (1, my) can be tiled to (Lx,Ly-1) for 'ladder' and (Lx, Ly) for 'cylinder' bc.
        # If no magnetic unit cell size is defined, minimal size will be used.
        if my == 0: my = phi_pq[1]
        hop_y = -Jy
        hop_x = -Jx * np.exp(1.j * phi * np.arange(my)[np.newaxis, :])  # has shape (1, Ly)
    elif gauge == 'symmetric':
        # hopping in x-direction depends on y-coordinate. Hopping in y-direction depends on 
        # x-coordinate.
        # If no magnetic unit cell size is defined, minimal size will be used.
        if mx == 0 or my == 0:
            # TODO Rework so minimal MUC always contains integer number of flux quanta (i.e. not sqrt).
            warnings.warn("Magnetic unit cell not (fully) specified.")
            mx = my = np.sqrt(phi_pq[1])
            assert np.issubdtype(mx, int)
            assert np.issubdtype(my, int)
        hop_x = -Jx * np.exp(1.j * (phi/2) * np.arange(my)[:, np.newaxis])
        hop_y = -Jy * np.exp(1.j * (phi/2) * np.arange(mx)[np.newaxis, :])
    else:
        raise NotImplementedError()
    return hop_x, hop_y


class HofstadterFermions(CouplingMPOModel):
    r"""Fermions on a square lattice with magnetic flux.

    For now, the Hamiltonian reads:

    .. math ::
        H = - \sum_{x, y} \mathtt{Jx} (e^{i \mathtt{phi}_{x,y} } c^\dagger_{x,y} c_{x+1,y} + h.c.)   \\
            - \sum_{x, y} \mathtt{Jy} (e^{i \mathtt{phi}_{x,y} } c^\dagger_{x,y} c_{x,y+1} + h.c.)
            - \sum_{x, y} \mathtt{mu} n_{x,y},

    where :math:`e^{i \mathtt{phi}_{x,y} }` is a complex Aharonov-Bohm hopping
    phase, depending on lattice coordinates and gauge choice (see 
    :func:`tenpy.models.hofstadter.gauge_hopping`).

    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    Parameters
    ----------
    Lx, Ly : int
        Size of the simulation unit cell in terms of lattice sites.
    mx, my : int
        Size of the magnetic unit cell in terms of lattice sites.
    filling : tuple
        Average number of fermions per site, defined as a fraction (numerator, denominator)
        Changes the definition of ``'dN'`` in the :class:`~tenpy.networks.site.FermionSite`.
    Jx, Jy, mu: float
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
        Choice of the gauge used for the magnetic field. This changes the 
        magnetic unit cell.
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
        lat = Square(Lx, Ly, site, order=order, bc=[bc_x, bc_y], bc_MPS=bc_MPS)
        return lat

    def init_terms(self, model_params):
        Lx = self.lat.shape[0]
        Ly = self.lat.shape[1]
        mx = get_parameter(model_params, 'mx', 0, self.name)
        my = get_parameter(model_params, 'my', 0, self.name)
        Jx = get_parameter(model_params, 'Jx', 1., self.name)
        Jy = get_parameter(model_params, 'Jy', 1., self.name)
        phi_pq = get_parameter(model_params, 'phi', (1, 3), self.name)
        phi = 2 * np.pi * phi_pq[0] / phi_pq[1]
        phi_ext = get_parameter(model_params, 'phi_ext', 0., self.name)
        mu = get_parameter(model_params, 'mu', 1., self.name, True)
        gauge = get_parameter(model_params, 'gauge', 'landau_x', self.name)
        hop_x, hop_y = gauge_hopping(gauge, mx, my, Jx, Jy, phi, phi_pq)

        # 6) add terms of the Hamiltonian
        self.add_onsite(-mu, 0, 'N')
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
        H = - \sum_{x, y} \mathtt{Jx} (e^{i \mathtt{phi}_{x,y} } a^\dagger_{x+1,y} a_{x,y} + h.c.)   \\
            - \sum_{x, y} \mathtt{Jy} (e^{i \mathtt{phi}_{x,y} } a^\dagger_{x,y+1} a_{x,y} + h.c.)   \\
            + \sum_{x, y} \frac{\mathtt{U}}{2} n_{x,y} (n_{x,y} - 1) - \mathtt{mu} n_{x,y}

    where :math:`e^{i \mathtt{phi}_{x,y} }` is a complex Aharonov-Bohm hopping
    phase, depending on lattice coordinates and gauge choice (see 
    :func:`tenpy.models.hofstadter.gauge_hopping`).

    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    Parameters
    ----------
    Lx, Ly : int
        Size of the simulation unit cell in terms of lattice sites.
    mx, my : int
        Size of the magnetic unit cell in terms of lattice sites.
    N_max : int
        Maximum number of bosons per site.
    filling : tuple
        Average number of fermions per site, defined as a fraction (numerator, denominator)
        Changes the definition of ``'dN'`` in the :class:`~tenpy.networks.site.BosonSite`.
    Jx, Jy, mu, U: float
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
        Choice of the gauge used for the magnetic field. This changes the 
        magnetic unit cell. 
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
        lat = Square(Lx, Ly, site, order=order, bc=[bc_x, bc_y], bc_MPS=bc_MPS)
        return lat

    def init_terms(self, model_params):
        Lx = self.lat.shape[0]
        Ly = self.lat.shape[1]
        mx = get_parameter(model_params, 'mx', 0, self.name)
        my = get_parameter(model_params, 'my', 0, self.name)
        Jx = get_parameter(model_params, 'Jx', 1., self.name)
        Jy = get_parameter(model_params, 'Jy', 1., self.name)
        phi_pq = get_parameter(model_params, 'phi', (1, 4), self.name)
        phi = 2 * np.pi * phi_pq[0] / phi_pq[1]
        phi_ext = get_parameter(model_params, 'phi_ext', 0., self.name)
        mu = get_parameter(model_params, 'mu', 1., self.name, True)
        U = get_parameter(model_params, 'U', 0, self.name)
        gauge = get_parameter(model_params, 'gauge', 'landau_x', self.name)
        hop_x, hop_y = gauge_hopping(gauge, mx, my, Jx, Jy, phi, phi_pq)

        # 6) add terms of the Hamiltonian
        self.add_onsite(np.asarray(U) / 2, 0, 'NN')
        self.add_onsite(-np.asarray(U) / 2 - np.asarray(mu), 0, 'N')
        self.add_coupling(hop_x, 0, 'Bd', 0, 'B', [1, 0])
        self.add_coupling(np.conj(hop_x), 0, 'Bd', 0, 'B', [-1, 0])  # h.c.
        dy = np.array([0, 1])
        hop_y = self.coupling_strength_add_ext_flux(hop_y, dy, [0, phi_ext])
        self.add_coupling(hop_y, 0, 'Bd', 0, 'B', dy)
        self.add_coupling(np.conj(hop_y), 0, 'Bd', 0, 'B', -dy)  # h.c.