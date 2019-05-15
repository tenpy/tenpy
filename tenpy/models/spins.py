"""Nearest-neighbour spin-S models.

Uniform lattice of spin-S sites, coupled by nearest-neighbour interactions.
"""
# Copyright 2018 TeNPy Developers

import numpy as np

from ..networks.site import SpinSite
from .model import CouplingMPOModel, NearestNeighborModel
from ..tools.params import get_parameter
from ..tools.misc import any_nonzero

__all__ = ['SpinModel', 'SpinChain']


class SpinModel(CouplingMPOModel):
    r"""Spin-S sites coupled by nearest neighbour interactions.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i,j\rangle, i < j}
              (\mathtt{Jx} S^x_i S^x_j + \mathtt{Jy} S^y_i S^y_j + \mathtt{Jz} S^z_i S^z_j
            + \mathtt{muJ} i/2 (S^{-}_i S^{+}_j - S^{+}_i S^{-}_j))  \\
            - \sum_i (\mathtt{hx} S^x_i + \mathtt{hy} S^y_i + \mathtt{hz} S^z_i) \\
            + \sum_i (\mathtt{D} (S^z_i)^2 + \mathtt{E} ((S^x_i)^2 - (S^y_i)^2))

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs.
    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    Parameters
    ----------
    S : {0.5, 1, 1.5, 2, ...}
        The 2S+1 local states range from m = -S, -S+1, ... +S.
    conserve : 'best' | 'Sz' | 'parity' | None
        What should be conserved. See :class:`~tenpy.networks.Site.SpinSite`.
        For ``'best'``, we check the parameters what can be preserved.
    Jx, Jy, Jz, hx, hy, hz, muJ, D, E: float | array
        Couplings as defined for the Hamiltonian above.
    lattice : str | :class:`~tenpy.models.lattice.Lattice`
        Instance of a lattice class for the underlaying geometry.
        Alternatively a string being the name of one of the Lattices defined in
        :mod:`~tenpy.models.lattice`, e.g. ``"Chain", "Square", "HoneyComb", ...``.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions along the x-direction.
        For 'infinite' boundary conditions, repeat the unit cell in x-direction.
        Coupling boundary conditions in x-direction are chosen accordingly.
        Only used if `lattice` is a string.
    order : string
        Ordering of the sites in the MPS, e.g. 'default', 'snake';
        see :meth:`~tenpy.models.lattice.Lattice.ordering`.
        Only used if `lattice` is a string.
    L : int
        Lenght of the lattice.
        Only used if `lattice` is the name of a 1D Lattice.
    Lx, Ly : int
        Length of the lattice in x- and y-direction.
        Only used if `lattice` is the name of a 2D Lattice.
    bc_y : 'ladder' | 'cylinder'
        Boundary conditions in y-direction.
        Only used if `lattice` is the name of a 2D Lattice.
    """

    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        S = get_parameter(model_params, 'S', 0.5, self.name)
        conserve = get_parameter(model_params, 'conserve', 'best', self.name)
        if conserve == 'best':
            # check how much we can conserve
            if not any_nonzero(model_params, [('Jx', 'Jy'), 'hx', 'hy', 'E'],
                               "check Sz conservation"):
                conserve = 'Sz'
            elif not any_nonzero(model_params, ['hx', 'hy'], "check parity conservation"):
                conserve = 'parity'
            else:
                conserve = None
            if self.verbose >= 1.:
                print(self.name + ": set conserve to", conserve)
        site = SpinSite(S, conserve)
        return site

    def init_terms(self, model_params):
        Jx = get_parameter(model_params, 'Jx', 1., self.name, True)
        Jy = get_parameter(model_params, 'Jy', 1., self.name, True)
        Jz = get_parameter(model_params, 'Jz', 1., self.name, True)
        hx = get_parameter(model_params, 'hx', 0., self.name, True)
        hy = get_parameter(model_params, 'hy', 0., self.name, True)
        hz = get_parameter(model_params, 'hz', 0., self.name, True)
        D = get_parameter(model_params, 'D', 0., self.name, True)
        E = get_parameter(model_params, 'E', 0., self.name, True)
        muJ = get_parameter(model_params, 'muJ', 0., self.name, True)

        # (u is always 0 as we have only one site in the unit cell)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hx, u, 'Sx')
            self.add_onsite(-hy, u, 'Sy')
            self.add_onsite(-hz, u, 'Sz')
            self.add_onsite(D, u, 'Sz Sz')
            self.add_onsite(E * 0.5, u, 'Sp Sp')
            self.add_onsite(E * 0.5, u, 'Sm Sm')
        # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
        # Sx.Sx = 0.25 ( Sp.Sm + Sm.Sp + Sp.Sp + Sm.Sm )
        # Sy.Sy = 0.25 ( Sp.Sm + Sm.Sp - Sp.Sp - Sm.Sm )
        for u1, u2, dx in self.lat.nearest_neighbors:
            self.add_coupling((Jx + Jy) / 4., u1, 'Sp', u2, 'Sm', dx)
            self.add_coupling(np.conj((Jx + Jy) / 4.), u2, 'Sp', u1, 'Sm', -dx)  # h.c.
            self.add_coupling((Jx - Jy) / 4., u1, 'Sp', u2, 'Sp', dx)
            self.add_coupling(np.conj((Jx - Jy) / 4.), u2, 'Sm', u1, 'Sm', -dx)  # h.c.
            self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx)
            self.add_coupling(muJ * 0.5j, u1, 'Sm', u2, 'Sp', dx)
            self.add_coupling(muJ * -0.5j, u1, 'Sp', u2, 'Sm', dx)
        # done


class SpinChain(SpinModel, NearestNeighborModel):
    """The :class:`SpinModel` on a Chain, suitable for TEBD.

    See the :class:`SpinModel` for the documentation of parameters.
    """

    def __init__(self, model_params):
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)
