"""Cold atomic (Harper-)Hofstadter model on a strip or cylinder.

.. todo ::
    Long term: implement different lattices.
    Long term: implement variable hopping strengths Jx, Jy.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import warnings

from .lattice import Square
from ..networks.site import BosonSite, FermionSite
from .model import CouplingModel, MPOModel, CouplingMPOModel

__all__ = ['HofstadterBosons', 'HofstadterFermions', 'gauge_hopping']


def gauge_hopping(model_params):
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
          have shape :math`(\mathtt{mx}, 1)`. For flux densities :math:`p/q`, `mx` will default to q.
          Example: at a flux density :math:`1/3`, the magnetic unit cell will have shape
          :math:`(3,1)`, so it encloses exactly 1 flux quantum.
        * 'landau_y': Landau gauge along the y-axis. The magnetic unit cell will
          have shape :math`(1, \mathtt{my})`. For flux densities :math`p/q`, `my` will default to q.
          Example: at a flux density :math:`3/7`, the magnetic unit cell will have shape
          :math:`(1,7)`, so it encloses axactly 3 flux quanta.
        * 'symmetric': symmetric gauge. The magnetic unit cell will have shape
          :math:`(\mathtt{mx}, \mathtt{my})`, with :math:`mx = my`. For flux densities :math:`p/q`,
          `mx` and `my` will default to :math:`q`
          Example: at a flux density 4/9, the magnetic unit cell will have shape
          (9,9).

    .. todo :
        Add periodic gauge (generalization of symmetric with mx, my unequal).

    Parameters
    ----------
    gauge : 'landau_x' | 'landau_y' | 'symmetric'
        Choice of the gauge, see table above.
    mx, my : int | None
        Dimensions of the magnetic unit cell in terms of lattice sites.
        ``None`` defaults to the minimal choice compatible with `gauge` and `phi_pq`.
    Jx, Jy: float
        'Bare' hopping amplitudes (without phase).
        Without any flux we have ``hop_x = -Jx`` and ``hop_y = -Jy``.
    phi_pq : tuple (int, int)
        Magnetic flux as a fraction p/q, defined as (p, q)

    Returns
    -------
    hop_x, hop_y : float | array
        Hopping amplitudes to be used as prefactors for :math:`c^\dagger_{x,y} c_{x+1,y}` (`hop_x`)
        and :math:`c^\dagger_{x,y} c_{x,y+1}` (`hop_x`), respectively, with the necessary phases
        for the gauge.
    """
    # The hopping amplitudes depend on position -> use an array for couplings.
    # If the array is smaller than the actual number of couplings,
    # it is 'tiled', i.e. repeated periodically, see also tenpy.tools.to_array().
    # If no magnetic unit cell size is defined, minimal size will be used.
    gauge = model_params.get('gauge', 'landau_x')
    mx = model_params.get('mx', None)
    my = model_params.get('my', None)
    Jx = model_params.get('Jx', 1.)
    Jy = model_params.get('Jy', 1.)
    phi_p, phi_q = model_params.get('phi', (1, 3))
    phi = 2 * np.pi * phi_p / phi_q

    if gauge == 'landau_x':
        # hopping in x-direction: uniform
        # hopping in y-direction: depends on x, shape (mx, 1)
        # can be tiled to (Lx,Ly-1) for 'ladder' and (Lx, Ly) for 'cylinder' bc.
        if mx is None:
            mx = phi_q
        hop_x = -Jx
        hop_y = -Jy * np.exp(1.j * phi * np.arange(mx)[:, np.newaxis])  # has shape (mx, 1)
    elif gauge == 'landau_y':
        # hopping in x-direction: depends on y, shape (1, my)
        # hopping in y-direction: uniform
        # can be tiled to (Lx,Ly-1) for 'ladder' and (Lx, Ly) for 'cylinder' bc.
        if my is None:
            my = phi_q
        hop_y = -Jy
        hop_x = -Jx * np.exp(-1.j * phi * np.arange(my)[np.newaxis, :])  # has shape (1, my)
    elif gauge == 'symmetric':
        # hopping in x-direction: depends on y, shape (mx, my)
        # hopping in y-direction: depends on x, shape (mx, my)
        if mx is None or my is None:
            mx = my = phi_q
        hop_x = -Jx * np.exp(-1.j * (phi / 2) * np.arange(my)[np.newaxis, :])  # shape (1, my)
        hop_y = -Jy * np.exp(1.j * (phi / 2) * np.arange(mx)[:, np.newaxis])  # shape (mx, 1)
    else:
        raise ValueError("Undefinied gauge " + repr(gauge))
    return hop_x, hop_y


class HofstadterFermions(CouplingMPOModel):
    r"""Fermions on a square lattice with magnetic flux.

    For now, the Hamiltonian reads:

    .. math ::
        H = - \sum_{x, y} \mathtt{Jx} (e^{i \mathtt{phi}_{x,y} } c^\dagger_{x,y} c_{x+1,y} + h.c.)   \\
            - \sum_{x, y} \mathtt{Jy} (e^{i \mathtt{phi}_{x,y} } c^\dagger_{x,y} c_{x,y+1} + h.c.)   \\
            + \sum_{x, y} \mathtt{v} ( n_{x, y} n_{x, y + 1} + n_{x, y} n_{x + 1, y}   \\
            - \sum_{x, y} \mathtt{mu} n_{x,y},

    where :math:`e^{i \mathtt{phi}_{x,y} }` is a complex Aharonov-Bohm hopping
    phase, depending on lattice coordinates and gauge choice (see
    :func:`tenpy.models.hofstadter.gauge_hopping`).

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`HofstadterFermions` below.

    Options
    -------
    .. cfg:config :: HofstadterFermions
        :include: CouplingMPOModel

        Lx, Ly : int
            Length of the lattice in x- and y-direction.
        mx, my : int
            Size of the magnetic unit cell along x and y directions, in terms of lattice sites.
        filling : tuple
            Average number of fermions per site, defined as a fraction (numerator, denominator)
            Changes the definition of ``'dN'`` in the :class:`~tenpy.networks.site.FermionSite`.
        Jx, Jy, mu, v : float
            Hamiltonian parameter as defined above.
        conserve : {'N' | 'parity' | None}
            What quantum number to conserve.
        phi : tuple
            Magnetic flux density, defined as a fraction ``(numerator, denominator)``
        phi_ext : float
            External magnetic flux 'threaded' through the cylinder.
        gauge : 'landau_x' | 'landau_y' | 'symmetric'
            Choice of the gauge used for the magnetic field. This changes the
            magnetic unit cell. See :func:`gauge_hopping` for details.

    """
    default_lattice = Square
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'N')
        filling = model_params.get('filling', (1, 8))
        filling = filling[0] / filling[1]
        site = FermionSite(conserve=conserve, filling=filling)
        return site

    def init_terms(self, model_params):
        Lx = self.lat.shape[0]
        Ly = self.lat.shape[1]
        phi_ext = model_params.get('phi_ext', 0.)
        mu = np.asarray(model_params.get('mu', 0.))
        v = np.asarray(model_params.get('v', 0))
        hop_x, hop_y = gauge_hopping(model_params)

        # 6) add terms of the Hamiltonian
        self.add_onsite(-mu, 0, 'N')
        dx = np.array([1, 0])
        self.add_coupling(hop_x, 0, 'Cd', 0, 'C', dx)
        self.add_coupling(np.conj(hop_x), 0, 'Cd', 0, 'C', -dx)  # h.c.
        dy = np.array([0, 1])
        hop_y = self.coupling_strength_add_ext_flux(hop_y, dy, [0, phi_ext])
        self.add_coupling(hop_y, 0, 'Cd', 0, 'C', dy)
        self.add_coupling(np.conj(hop_y), 0, 'Cd', 0, 'C', -dy)  # h.c.
        self.add_coupling(v, 0, 'N', 0, 'N', dx)
        self.add_coupling(v, 0, 'N', 0, 'N', dy)


class HofstadterBosons(CouplingMPOModel):
    r"""Bosons on a square lattice with magnetic flux.

    For now, the Hamiltonian reads:

    .. math ::
        H = - \sum_{x, y} \mathtt{Jx} (e^{i \mathtt{phi}_{x,y} } a^\dagger_{x+1,y} a_{x,y} + h.c.)   \\
            - \sum_{x, y} \mathtt{Jy} (e^{i \mathtt{phi}_{x,y} } a^\dagger_{x,y+1} a_{x,y} + h.c.)   \\
            + \sum_{x, y} \frac{\mathtt{U}}{2} n_{x,y} (n_{x,y} - 1) - \mathtt{mu} n_{x,y}

    where :math:`e^{i \mathtt{phi}_{x,y} }` is a complex Aharonov-Bohm hopping
    phase, depending on lattice coordinates and gauge choice (see
    :func:`tenpy.models.hofstadter.gauge_hopping`).

    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`HofstadterBosons` below.

    Options
    -------
    .. cfg:config :: HofstadterBosons
        :include: CouplingMPOModel

        Lx, Ly : int
            Length of the lattice in x- and y-direction.
        mx, my : int
            Size of the magnetic unit cell along x and y, in terms of lattice sites.
        Nmax : int
            Maximum number of bosons per site.
        filling : tuple
            Average number of fermions per site, defined as a fraction (numerator, denominator)
            Changes the definition of ``'dN'`` in the :class:`~tenpy.networks.site.BosonSite`.
        Jx, Jy, mu, U : float
            Hamiltonian parameter as defined above.
        conserve : {'N' | 'parity' | None}
            What quantum number to conserve.
        phi : tuple
            Magnetic flux density, defined as a fraction (numerator, denominator)
        phi_ext : float
            External magnetic flux 'threaded' through the cylinder.
        gauge : 'landau_x' | 'landau_y' | 'symmetric'
            Choice of the gauge used for the magnetic field. This changes the
            magnetic unit cell.
    """
    default_lattice = Square
    force_default_lattice = True

    def init_sites(self, model_params):
        Nmax = model_params.get('Nmax', 3)
        conserve = model_params.get('conserve', 'N')
        filling = model_params.get('filling', (1, 8))
        filling = filling[0] / filling[1]
        site = BosonSite(Nmax=Nmax, conserve=conserve, filling=filling)
        return site

    def init_terms(self, model_params):
        Lx = self.lat.shape[0]
        Ly = self.lat.shape[1]
        phi_ext = model_params.get('phi_ext', 0.)
        mu = np.asarray(model_params.get('mu', 0.))
        U = np.asarray(model_params.get('U', 0))
        hop_x, hop_y = gauge_hopping(model_params)

        # 6) add terms of the Hamiltonian
        self.add_onsite(U / 2, 0, 'NN')
        self.add_onsite(-U / 2 - mu, 0, 'N')
        dx = np.array([1, 0])
        self.add_coupling(hop_x, 0, 'Bd', 0, 'B', dx)
        self.add_coupling(np.conj(hop_x), 0, 'Bd', 0, 'B', -dx)  # h.c.
        dy = np.array([0, 1])
        hop_y = self.coupling_strength_add_ext_flux(hop_y, dy, [0, phi_ext])
        self.add_coupling(hop_y, 0, 'Bd', 0, 'B', dy)
        self.add_coupling(np.conj(hop_y), 0, 'Bd', 0, 'B', -dy)  # h.c.
