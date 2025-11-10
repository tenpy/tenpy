"""Cold atomic (Harper-)Hofstadter model on a strip or cylinder.

.. todo ::
    Long term: implement different lattices.
    Long term: implement variable hopping strengths Jx, Jy.
"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np

from ..networks.site import BosonSite, FermionSite
from .lattice import Square
from .model import CouplingMPOModel

__all__ = ['HofstadterBosons', 'HofstadterFermions', 'gauge_hopping', 'hopping_phases']


def hopping_phases(p: int, q: int, Lx: int, Ly: int, pbc_x: bool, pbc_y: bool, gauge):
    r"""Calculate the complex hopping phases for Hofstadter models.

    To achieve a uniform magnetic flux density per plaquette of ``phi = p / q``, we use
    complex hopping phases :math:`e^{2 \pi i a_x(x, y)}` for hopping to the left,
    from ``(x + 1, y)`` to ``(x, y)`` and :math:`e^{2 \pi i a_y(x, y)}` for hopping down from
    ``(x, y + 1)`` to ``(x, y)``. For hopping in the opposite directions on a given bond, we get
    the conjugate prefactor, i.e. the negative phase.

    The phase picked up when hopping around a plaquette clockwise (i.e. in mathematically negative
    orientation, matching the negative charge of the electron), must add up to the flux density.
    Let us add up the phases for a hopping loop, starting at the bottom left of a plaquette,
    at ``(x, y)`` and going up, right, down, left. We then must have

    .. math ::
        - a_y(x, y) - a_x(x, y + 1) + a_y(x + 1, y) + a_x(x, y) = phi

    There are many gauge choices that achieve this. We support the following choices::

        ===========  ===============  ==============  ====================
        gauge        a_x(x, y)        a_y(x, y)       magnetic unit cell
        ===========  ===============  ==============  ====================
        landau_x     0                phi * x         (q, 1)
        -----------  ---------------  --------------  --------------------
        landau_y     -phi * y         0               (1, q)
        -----------  ---------------  --------------  --------------------
        symmetric    -.5 * phi * y    .5 * phi * x    (2 * q, 2 * q)
        ===========  ===============  ==============  ====================

    .. warning ::
        Note how the size of the  "magnetic unit cell" after which the phase factors repeat
        (s.t. the :math:`a_i` repeat modulo :math:`2\pi`) depends on the gauge choice.
        In any direction with periodic boundaries, we need this unit cell of hopping phases to
        commensurately tile the lattice unit cell. This also guarantees that the plaquettes that
        cross the periodic boundary have the correct flux. For directions with open boundaries,
        this technical aspect of commensuration is not relevant, and we can write down the model
        for any system size.

    Parameters
    ----------
    p, q : int
        Specifies the flux per plaquette as a fraction ``phi = p / q``
    lx, ly : int
        System size (for finite systems) or unit cell size (for infinite systems)
    pbc_x, pbc_y : int
        If the boundary conditions in the particular direction are periodic, else open.
    gauge : 'landau_x' | 'landau_y' | 'symmetric' | None
        Choices for the gauge, see table above. If ``None``, we try them in order and use the first
        that is commensurate with all periodic boundaries.

    Returns
    -------
    phases_x, phases_y : 2D array
        Complexes phases :math:`\mathtt{phases_j[x, y]} = e^{2 \pi i a_j(x, y)}``.
        Shape matches the bonds of the orientation in the given system, i.e. ``(lx, ly)`` or
        reduced by one at open boundaries.

    """
    assert isinstance(p, int) and p != 0, f'Expected non-zero integer. Got {p=}'
    assert isinstance(q, int) and q > 0, f'Expected positive integer. Got {q=}'
    phi = p / q
    # reduce the fraction p / q
    gcd = int(np.gcd(p, q))
    p = p // gcd
    q = q // gcd

    if gauge is None:
        # try the supported gauge choices in order
        errs = []
        for g in ['landau_x', 'landau_y', 'symmetric', 'periodic']:
            try:
                return hopping_phases(p=p, q=q, Lx=Lx, Ly=Ly, pbc_x=pbc_x, pbc_y=pbc_y, gauge=g)
            except ValueError as e:
                errs.append(e)
        raise ValueError(
            'None of the supported gauge choices could be applied. Error message for the default gauge choice above. '
        ) from errs[0]

    num_bonds_x = Lx if pbc_x else Lx - 1
    num_bonds_y = Ly if pbc_y else Ly - 1

    if gauge == 'landau_x':
        mx, my = (q, 1)
        phase_x = np.ones((num_bonds_x, Ly), complex)
        phase_y = np.tile(np.exp(2.0j * np.pi * phi * np.arange(Lx))[:, None], [1, num_bonds_y])
    elif gauge == 'landau_y':
        mx, my = (1, q)
        phase_x = np.tile(np.exp(-2.0j * np.pi * phi * np.arange(Ly))[None, :], [num_bonds_x, 1])
        phase_y = np.ones((Lx, num_bonds_y), complex)
    elif gauge == 'symmetric':
        mx, my = (2 * q, 2 * q)
        phase_x = np.tile(np.exp(-1.0j * np.pi * phi * np.arange(Ly))[None, :], [num_bonds_x, 1])
        phase_y = np.tile(np.exp(1.0j * np.pi * phi * np.arange(Lx))[:, None], [1, num_bonds_y])
    else:
        raise ValueError(f'Invalid gauge : "{gauge}"')

    # check commensuration with unit cell along any pbc direction
    if pbc_x and Lx % mx != 0:
        msg = (
            f'Magnetic unit cell is incommensurate with lattice unit cell in x-direction. '
            f'Expected `Lx` to be a multiple of ``{mx}``.'
        )
        raise ValueError(msg)
    if pbc_y and Ly % my != 0:
        msg = (
            f'Magnetic unit cell is incommensurate with lattice unit cell in y-direction. '
            f'Expected `Ly` to be a multiple of ``{my}``.'
        )
        raise ValueError(msg)

    # sanity check for the periodicity of the phases
    if pbc_x:
        assert np.allclose(np.roll(phase_x, mx, axis=0), phase_x)
        assert np.allclose(np.roll(phase_y, mx, axis=0), phase_y)
    else:
        assert np.allclose(phase_x[mx:, :], phase_x[:-mx, :])
        assert np.allclose(phase_y[mx:, :], phase_y[:-mx, :])
    if pbc_y:
        assert np.allclose(np.roll(phase_x, my, axis=1), phase_x)
        assert np.allclose(np.roll(phase_y, my, axis=1), phase_y)
    else:
        assert np.allclose(phase_x[:, my:], phase_x[:, :-my])
        assert np.allclose(phase_y[:, my:], phase_y[:, :-my])

    return phase_x, phase_y


class HofstadterFermions(CouplingMPOModel):
    r"""Fermions on a square lattice with uniform magnetic flux.

    For now, the Hamiltonian reads:

    .. math ::
        H = - \sum_{x, y} \mathtt{Jx} (e^{2 \pi i a_x(x, y)} c^\dagger_{x,y} c_{x+1,y} + h.c.)   \\
            - \sum_{x, y} \mathtt{Jy} (e^{2 \pi i a_y(x, y)} c^\dagger_{x,y} c_{x,y+1} + h.c.)   \\
            + \sum_{x, y} \mathtt{v} ( n_{x, y} n_{x, y + 1} + n_{x, y} n_{x + 1, y}   \\
            - \sum_{x, y} \mathtt{mu} n_{x,y},

    where :math:`e^{2 \pi i a_{x/y}(x, y)` is an Aharonov-Bohm hopping phase, that gives a uniform
    flux density per plaquette. The concrete form of the phases depends on the gauge choice,
    see :func:`~tenpy.models.hofstadter.hopping_phases`.

    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

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
        filling : tuple
            Average number of fermions per site, defined as a fraction ``(numerator, denominator)``
            Changes the definition of ``'dN'`` in the :class:`~tenpy.networks.site.FermionSite`.
            Default ``(1, 8)``, i.e. one particle per eight sites.
        Jx, Jy, mu, v : float
            Hamiltonian parameter as defined above. Defaults are ``Jx = Jy = 1``, ``mu = v = 0``.
        conserve : {'N' | 'parity' | None}
            What quantum number to conserve.
        phi : tuple
            Magnetic flux per plaquette, defined as a fraction ``(numerator, denominator)``.
            Default ``(1, 3)``, i.e. one flux quantum per three plaquettes.
        phi_ext : float
            External magnetic flux 'threaded' through the cylinder. Hopping amplitudes for bonds
            'across' the periodic boundary are modified such that particles hopping around the
            circumference of the cylinder acquire a phase ``2 pi phi_ext``.
        gauge : 'landau_x' | 'landau_y' | 'symmetric'
            Choice of the gauge used for the magnetic field. This affects the size and shape of
            the magnetic unit cell (the unit cell for the hopping phases), which in turn restricts
            the allowed MPS unit cell sizes. See :func:`hopping_phases` for details.

    """

    default_lattice = Square
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'N', str)
        filling = model_params.get('filling', (1, 8))
        filling = filling[0] / filling[1]
        site = FermionSite(conserve=conserve, filling=filling)
        return site

    def init_terms(self, model_params):
        phi_ext = model_params.get('phi_ext', 0.0, 'real')
        mu = np.asarray(model_params.get('mu', 0.0, 'real_or_array'))
        v = np.asarray(model_params.get('v', 0, 'real_or_array'))
        p, q = model_params.get('phi', (1, 3))
        gauge = model_params.get('gauge', None)
        Jx = model_params.get('Jx', 1.0, 'real')
        Jy = model_params.get('Jy', 1.0, 'real')
        model_params.deprecated_ignore('mx', 'my', extra_msg='This option did not affect the behavior anyway.')

        phases_x, phases_y = hopping_phases(
            p,
            q,
            Lx=self.lat.shape[0],
            Ly=self.lat.shape[1],
            pbc_x=not self.lat.bc[0],
            pbc_y=not self.lat.bc[1],
            gauge=gauge,
        )
        hop_x = -Jx * phases_x
        hop_y = -Jy * phases_y

        # 6) add terms of the Hamiltonian
        self.add_onsite(-mu, 0, 'N')
        dx = np.array([1, 0])
        self.add_coupling(hop_x, 0, 'Cd', 0, 'C', dx)
        self.add_coupling(np.conj(hop_x), 0, 'Cd', 0, 'C', -dx)  # h.c.
        dy = np.array([0, 1])
        hop_y = self.coupling_strength_add_ext_flux(hop_y, dy, [0, 2.0 * np.pi * phi_ext])
        self.add_coupling(hop_y, 0, 'Cd', 0, 'C', dy)
        self.add_coupling(np.conj(hop_y), 0, 'Cd', 0, 'C', -dy)  # h.c.
        self.add_coupling(v, 0, 'N', 0, 'N', dx)
        self.add_coupling(v, 0, 'N', 0, 'N', dy)


class HofstadterBosons(CouplingMPOModel):
    r"""Bosons on a square lattice with uniform magnetic flux.

    For now, the Hamiltonian reads:

    .. math ::
        H = - \sum_{x, y} \mathtt{Jx} (e^{2 \pi i a_x(x, y)} a^\dagger_{x+1,y} a_{x,y} + h.c.)   \\
            - \sum_{x, y} \mathtt{Jy} (e^{2 \pi i a_y(x, y)} a^\dagger_{x,y+1} a_{x,y} + h.c.)   \\
            + \sum_{x, y} \frac{\mathtt{U}}{2} n_{x,y} (n_{x,y} - 1) - \mathtt{mu} n_{x,y}

    where :math:`e^{2 \pi i a_{x/y}(x, y)` is an Aharonov-Bohm hopping phase, that gives a uniform
    flux density per plaquette. The concrete form of the phases depends on the gauge choice,
    see :func:`~tenpy.models.hofstadter.hopping_phases`.

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
        Nmax : int
            Maximum number of bosons per site. Default ``3``.
        filling : tuple
            Average number of bosons per site, defined as a fraction ``(numerator, denominator)``
            Changes the definition of ``'dN'`` in the :class:`~tenpy.networks.site.BosonSite`.
            Default ``(1, 8)``, i.e. one particle per eight sites.
        Jx, Jy, mu, U : float
            Hamiltonian parameter as defined above. Defaults are ``Jx = Jy = 1``, ``mu = U = 0``.
        conserve : {'N' | 'parity' | None}
            What quantum number to conserve.
        phi : tuple
            Magnetic flux per plaquette, defined as a fraction ``(numerator, denominator)``.
            Default ``(1, 3)``, i.e. one flux quantum per three plaquettes.
        phi_ext : float
            External magnetic flux 'threaded' through the cylinder. Hopping amplitudes for bonds
            'across' the periodic boundary are modified such that particles hopping around the
            circumference of the cylinder acquire a phase ``2 pi phi_ext``.
        gauge : 'landau_x' | 'landau_y' | 'symmetric'
            Choice of the gauge used for the magnetic field. This affects the size and shape of
            the magnetic unit cell (the unit cell for the hopping phases), which in turn restricts
            the allowed MPS unit cell sizes. See :func:`hopping_phases` for details.

    """

    default_lattice = Square
    force_default_lattice = True

    def init_sites(self, model_params):
        Nmax = model_params.get('Nmax', 3, int)
        conserve = model_params.get('conserve', 'N', str)
        filling = model_params.get('filling', (1, 8))
        filling = filling[0] / filling[1]
        site = BosonSite(Nmax=Nmax, conserve=conserve, filling=filling)
        return site

    def init_terms(self, model_params):
        phi_ext = model_params.get('phi_ext', 0.0, 'real')
        mu = np.asarray(model_params.get('mu', 0.0, 'real_or_array'))
        U = np.asarray(model_params.get('U', 0, 'real_or_array'))
        p, q = model_params.get('phi', (1, 3))
        Jx = model_params.get('Jx', 1.0, 'real')
        Jy = model_params.get('Jy', 1.0, 'real')
        gauge = model_params.get('gauge', None)
        model_params.deprecated_ignore('mx', 'my', extra_msg='This option did not affect the behavior anyway.')

        phases_x, phases_y = hopping_phases(
            p,
            q,
            Lx=self.lat.shape[0],
            Ly=self.lat.shape[1],
            pbc_x=not self.lat.bc[0],
            pbc_y=not self.lat.bc[1],
            gauge=gauge,
        )
        hop_x = -Jx * phases_x
        hop_y = -Jy * phases_y

        # 6) add terms of the Hamiltonian
        self.add_onsite(U / 2, 0, 'NN')
        self.add_onsite(-U / 2 - mu, 0, 'N')
        dx = np.array([1, 0])
        self.add_coupling(hop_x, 0, 'Bd', 0, 'B', dx)
        self.add_coupling(np.conj(hop_x), 0, 'Bd', 0, 'B', -dx)  # h.c.
        dy = np.array([0, 1])
        hop_y = self.coupling_strength_add_ext_flux(hop_y, dy, [0, 2 * np.pi * phi_ext])
        self.add_coupling(hop_y, 0, 'Bd', 0, 'B', dy)
        self.add_coupling(np.conj(hop_y), 0, 'Bd', 0, 'B', -dy)  # h.c.


def gauge_hopping(*a, **kw):
    raise RuntimeError('Deprecated. Use ``hopping_phases`` instead.')
