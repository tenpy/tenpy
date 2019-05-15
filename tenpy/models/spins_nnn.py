"""Next-Nearest-neighbour spin-S models.

Uniform lattice of spin-S sites, coupled by next-nearest-neighbour interactions.
We have two variants implementing the same hamiltonian. The first uses the
:class:`~tenpy.networks.site.GroupedSite` to keep it a
:class:`~tenpy.models.model.NearestNeighborModel` suitable for TEBD,
while the second one just involves longer-range couplings in the MPO.
The second one is preferable for pure DMRG calculations.
"""
# Copyright 2018 TeNPy Developers

import numpy as np

from .lattice import Chain
from ..networks.site import SpinSite, GroupedSite
from .model import CouplingMPOModel, NearestNeighborModel
from ..tools.params import get_parameter, unused_parameters
from ..tools.misc import any_nonzero

__all__ = ['SpinChainNNN', 'SpinChainNNN2']


class SpinChainNNN(CouplingMPOModel, NearestNeighborModel):
    r"""Spin-S sites coupled by (next-)nearest neighbour interactions on a `GroupedSite`.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i,j \rangle, i < j}
                \mathtt{Jx} S^x_i S^x_j + \mathtt{Jy} S^y_i S^y_j + \mathtt{Jz} S^z_i S^z_j \\
            + \sum_{\langle \langle i,j \rangle \rangle, i< j}
                \mathtt{Jxp} S^x_i S^x_j + \mathtt{Jyp} S^y_i S^y_j + \mathtt{Jzp} S^z_i S^z_j \\
            - \sum_i
              \mathtt{hx} S^x_i + \mathtt{hy} S^y_i + \mathtt{hz} S^z_i

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbors and
    :math:`\langle \langle i,j \rangle \rangle, i < j` denotes next nearest neighbors.
    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    Parameters
    ----------
    L : int
        Length of the chain in terms of :class:`~tenpy.networks.site.GroupedSite`,
        i.e. we have ``2*L`` spin sites.
    S : {0.5, 1, 1.5, 2, ...}
        The 2S+1 local states range from m = -S, -S+1, ... +S.
    conserve : 'best' | 'Sz' | 'parity' | None
        What should be conserved. See :class:`~tenpy.networks.Site.SpinSite`.
    Jx, Jy, Jz, Jxp, Jyp, Jzp, hx, hy, hz : float | array
        Couplings as defined for the Hamiltonian above.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions. Coupling boundary conditions are chosen appropriately.
    """

    def __init__(self, model_params):
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        S = get_parameter(model_params, 'S', 0.5, self.name)
        conserve = get_parameter(model_params, 'conserve', 'best', self.name)
        if conserve == 'best':
            # check how much we can conserve
            if not any_nonzero(model_params, [('Jx', 'Jy'), ('Jxp', 'Jyp'), 'hx', 'hy'],
                               "check Sz conservation"):
                conserve = 'Sz'
            elif not any_nonzero(model_params, ['hx', 'hy'], "check parity conservation"):
                conserve = 'parity'
            else:
                conserve = None
            if self.verbose >= 1.:
                print(self.name + ": set conserve to", conserve)
        spinsite = SpinSite(S, conserve)
        site = GroupedSite([spinsite, spinsite], charges='same')
        return site

    def init_terms(self, model_params):
        Jx = get_parameter(model_params, 'Jx', 1., self.name, True)
        Jy = get_parameter(model_params, 'Jy', 1., self.name, True)
        Jz = get_parameter(model_params, 'Jz', 1., self.name, True)
        Jxp = get_parameter(model_params, 'Jxp', 1., self.name, True)
        Jyp = get_parameter(model_params, 'Jyp', 1., self.name, True)
        Jzp = get_parameter(model_params, 'Jzp', 1., self.name, True)
        hx = get_parameter(model_params, 'hx', 0., self.name, True)
        hy = get_parameter(model_params, 'hy', 0., self.name, True)
        hz = get_parameter(model_params, 'hz', 0., self.name, True)

        # Only valid for self.lat being a Chain...
        self.add_onsite(-hx, 0, 'Sx0')
        self.add_onsite(-hy, 0, 'Sy0')
        self.add_onsite(-hz, 0, 'Sz0')
        self.add_onsite(-hx, 0, 'Sx1')
        self.add_onsite(-hy, 0, 'Sy1')
        self.add_onsite(-hz, 0, 'Sz1')
        # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
        # Sx.Sx = 0.25 ( Sp.Sm + Sm.Sp + Sp.Sp + Sm.Sm )
        # Sy.Sy = 0.25 ( Sp.Sm + Sm.Sp - Sp.Sp - Sm.Sm )
        # nearest neighbors
        self.add_onsite((Jx + Jy) / 4., 0, 'Sp0 Sm1')
        self.add_onsite(np.conj((Jx + Jy) / 4.), 0, 'Sp1 Sm0')  # h.c.
        self.add_onsite((Jx - Jy) / 4., 0, 'Sp0 Sp1')
        self.add_onsite(np.conj((Jx - Jy) / 4.), 0, 'Sm1 Sm0')  # h.c.
        self.add_onsite(Jz, 0, 'Sz0 Sz1')
        self.add_coupling((Jx + Jy) / 4., 0, 'Sp1', 0, 'Sm0', 1)
        self.add_coupling(np.conj((Jx + Jy) / 4.), 0, 'Sp0', 0, 'Sm1', -1)  # h.c.
        self.add_coupling((Jx - Jy) / 4., 0, 'Sp1', 0, 'Sp0', 1)
        self.add_coupling(np.conj((Jx - Jy) / 4.), 0, 'Sp0', 0, 'Sm1', -1)  # h.c.
        self.add_coupling(Jz, 0, 'Sz1', 0, 'Sz0', 1)
        # next nearest neighbors
        self.add_coupling((Jxp + Jyp) / 4., 0, 'Sp0', 0, 'Sm0', 1)
        self.add_coupling(np.conj((Jxp + Jyp) / 4.), 0, 'Sp0', 0, 'Sm0', -1)  # h.c.
        self.add_coupling((Jxp - Jyp) / 4., 0, 'Sp0', 0, 'Sp0', 1)
        self.add_coupling(np.conj((Jxp - Jyp) / 4.), 0, 'Sm0', 0, 'Sm0', -1)  # h.c.
        self.add_coupling(Jzp, 0, 'Sz0', 0, 'Sz0', 1)
        self.add_coupling((Jxp + Jyp) / 4., 0, 'Sp1', 0, 'Sm1', 1)
        self.add_coupling(np.conj((Jxp + Jyp) / 4.), 0, 'Sp1', 0, 'Sm1', -1)  # h.c.
        self.add_coupling((Jxp - Jyp) / 4., 0, 'Sp1', 0, 'Sp1', 1)
        self.add_coupling(np.conj((Jxp - Jyp) / 4.), 0, 'Sm1', 0, 'Sm1', -1)  # h.c.
        self.add_coupling(Jzp, 0, 'Sz1', 0, 'Sz1', 1)


class SpinChainNNN2(CouplingMPOModel):
    r"""Spin-S sites coupled by next-nearest neighbour interactions.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i,j \rangle, i < j}
                \mathtt{Jx} S^x_i S^x_j + \mathtt{Jy} S^y_i S^y_j + \mathtt{Jz} S^z_i S^z_j \\
            + \sum_{\langle \langle i,j \rangle \rangle, i< j}
                \mathtt{Jxp} S^x_i S^x_j + \mathtt{Jyp} S^y_i S^y_j + \mathtt{Jzp} S^z_i S^z_j \\
            - \sum_i
              \mathtt{hx} S^x_i + \mathtt{hy} S^y_i + \mathtt{hz} S^z_i

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbors and
    :math:`\langle \langle i,j \rangle \rangle, i < j` denotes next nearest neighbors.
    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    Parameters
    ----------
    S : {0.5, 1, 1.5, 2, ...}
        The 2S+1 local states range from m = -S, -S+1, ... +S.
    conserve : 'best' | 'Sz' | 'parity' | None
        What should be conserved. See :class:`~tenpy.networks.Site.SpinSite`.
        For ``'best'``, we check the parameters what can be preserved.
    Jx, Jy, Jz, Jxp, Jyp, Jzp, hx, hy, hz : float | array
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
            if not any_nonzero(model_params, [('Jx', 'Jy'), ('Jxp', 'Jyp'), 'hx', 'hy'],
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
        # 0) read out/set default parameters
        Jx = get_parameter(model_params, 'Jx', 1., self.name, True)
        Jy = get_parameter(model_params, 'Jy', 1., self.name, True)
        Jz = get_parameter(model_params, 'Jz', 1., self.name, True)
        Jxp = get_parameter(model_params, 'Jxp', 1., self.name, True)
        Jyp = get_parameter(model_params, 'Jyp', 1., self.name, True)
        Jzp = get_parameter(model_params, 'Jzp', 1., self.name, True)
        hx = get_parameter(model_params, 'hx', 0., self.name, True)
        hy = get_parameter(model_params, 'hy', 0., self.name, True)
        hz = get_parameter(model_params, 'hz', 0., self.name, True)

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hx, u, 'Sx')
            self.add_onsite(-hy, u, 'Sy')
            self.add_onsite(-hz, u, 'Sz')
        # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
        # Sx.Sx = 0.25 ( Sp.Sm + Sm.Sp + Sp.Sp + Sm.Sm )
        # Sy.Sy = 0.25 ( Sp.Sm + Sm.Sp - Sp.Sp - Sm.Sm )
        for u1, u2, dx in self.lat.nearest_neighbors:
            self.add_coupling((Jx + Jy) / 4., u1, 'Sp', u2, 'Sm', dx)
            self.add_coupling(np.conj((Jx + Jy) / 4.), u2, 'Sp', u1, 'Sm', -dx)  # h.c.
            self.add_coupling((Jx - Jy) / 4., u1, 'Sp', u2, 'Sp', dx)
            self.add_coupling(np.conj((Jx - Jy) / 4.), u2, 'Sm', u1, 'Sm', -dx)  # h.c.
            self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx)
        for u1, u2, dx in self.lat.next_nearest_neighbors:
            self.add_coupling((Jxp + Jyp) / 4., u1, 'Sp', u2, 'Sm', dx)
            self.add_coupling(np.conj((Jxp + Jyp) / 4.), u2, 'Sp', u1, 'Sm', -dx)  # h.c.
            self.add_coupling((Jxp - Jyp) / 4., u1, 'Sp', u2, 'Sp', dx)
            self.add_coupling(np.conj((Jxp - Jyp) / 4.), u2, 'Sm', u1, 'Sm', -dx)  # h.c.
            self.add_coupling(Jzp, u1, 'Sz', u2, 'Sz', dx)
