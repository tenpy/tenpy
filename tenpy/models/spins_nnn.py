"""Next-Nearest-neighbour spin-S models.

Uniform lattice of spin-S sites, coupled by next-nearest-neighbour interactions.
We have two variants implementing the same hamiltonian.
The :class:`SpinChainNNN` uses the
:class:`~tenpy.networks.site.GroupedSite` to keep it a
:class:`~tenpy.models.model.NearestNeighborModel` suitable for TEBD,
while the :class:`SpinChainNNN2` just involves longer-range couplings in the MPO.
The latter is preferable for pure DMRG calculations and avoids having to add each of the short
range couplings twice for the grouped sites.

Note that you can also get a :class:`~tenpy.models.model.NearestNeighborModel` for TEBD from the
latter by using :meth:`~tenpy.models.model.MPOModel.group_sites` and
:meth:`~tenpy.models.model.NearestNeighbormodel.from_MPOModel`.
An example for such a case is given in the file ``examples/c_tebd.py``.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np

from .lattice import Chain
from ..networks.site import SpinSite, GroupedSite
from .model import CouplingMPOModel, NearestNeighborModel
from ..tools.params import asConfig

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
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`SpinChainNNN` below.

    Options
    -------
    .. cfg:config :: SpinChainNNN
        :include: CouplingMPOModel

        L : int
            Length of the chain in terms of :class:`~tenpy.networks.site.GroupedSite`,
            i.e. we have ``2*L`` spin sites.
        S : {0.5, 1, 1.5, 2, ...}
            The 2S+1 local states range from m = -S, -S+1, ... +S.
        conserve : 'best' | 'Sz' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinSite`.
        Jx, Jy, Jz, Jxp, Jyp, Jzp, hx, hy, hz : float | array
            Coupling as defined for the Hamiltonian above.
        bc_MPS : {'finite' | 'infinte'}
            MPS boundary conditions. Coupling boundary conditions are chosen appropriately.

    """
    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params):
        S = model_params.get('S', 0.5)
        conserve = model_params.get('conserve', 'best')
        if conserve == 'best':
            # check how much we can conserve
            if not model_params.any_nonzero([('Jx', 'Jy'),
                                             ('Jxp', 'Jyp'), 'hx', 'hy'], "check Sz conservation"):
                conserve = 'Sz'
            elif not model_params.any_nonzero(['hx', 'hy'], "check parity conservation"):
                conserve = 'parity'
            else:
                conserve = None
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        spinsite = SpinSite(S, conserve)
        site = GroupedSite([spinsite, spinsite], charges='same')
        return site

    def init_terms(self, model_params):
        Jx = model_params.get('Jx', 1.)
        Jy = model_params.get('Jy', 1.)
        Jz = model_params.get('Jz', 1.)
        Jxp = model_params.get('Jxp', 1.)
        Jyp = model_params.get('Jyp', 1.)
        Jzp = model_params.get('Jzp', 1.)
        hx = model_params.get('hx', 0.)
        hy = model_params.get('hy', 0.)
        hz = model_params.get('hz', 0.)

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
        self.add_onsite((Jx + Jy) / 4., 0, 'Sp0 Sm1', plus_hc=True)
        self.add_onsite((Jx - Jy) / 4., 0, 'Sp0 Sp1', plus_hc=True)
        self.add_onsite(Jz, 0, 'Sz0 Sz1')
        self.add_coupling((Jx + Jy) / 4., 0, 'Sp1', 0, 'Sm0', 1, plus_hc=True)
        self.add_coupling((Jx - Jy) / 4., 0, 'Sp1', 0, 'Sp0', 1, plus_hc=True)
        self.add_coupling(Jz, 0, 'Sz1', 0, 'Sz0', 1)
        # next nearest neighbors
        self.add_coupling((Jxp + Jyp) / 4., 0, 'Sp0', 0, 'Sm0', 1, plus_hc=True)
        self.add_coupling((Jxp - Jyp) / 4., 0, 'Sp0', 0, 'Sp0', 1, plus_hc=True)
        self.add_coupling(Jzp, 0, 'Sz0', 0, 'Sz0', 1)
        self.add_coupling((Jxp + Jyp) / 4., 0, 'Sp1', 0, 'Sm1', 1, plus_hc=True)
        self.add_coupling((Jxp - Jyp) / 4., 0, 'Sp1', 0, 'Sp1', 1, plus_hc=True)
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
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`SpinChainNNN2` below.

    Options
    -------
    .. cfg:config :: SpinChainNNN2
        :include: CouplingMPOModel

        S : {0.5, 1, 1.5, 2, ...}
            The 2S+1 local states range from m = -S, -S+1, ... +S.
        conserve : 'best' | 'Sz' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinSite`.
            For ``'best'``, we check the parameters what can be preserved.
        Jx, Jy, Jz, Jxp, Jyp, Jzp, hx, hy, hz : float | array
            Coupling as defined for the Hamiltonian above.
    """
    def init_sites(self, model_params):
        S = model_params.get('S', 0.5)
        conserve = model_params.get('conserve', 'best')
        if conserve == 'best':
            # check how much we can conserve
            if not model_params.any_nonzero([('Jx', 'Jy'),
                                             ('Jxp', 'Jyp'), 'hx', 'hy'], "check Sz conservation"):
                conserve = 'Sz'
            elif not model_params.any_nonzero(['hx', 'hy'], "check parity conservation"):
                conserve = 'parity'
            else:
                conserve = None
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        site = SpinSite(S, conserve)
        return site

    def init_terms(self, model_params):
        # 0) read out/set default parameters
        Jx = model_params.get('Jx', 1.)
        Jy = model_params.get('Jy', 1.)
        Jz = model_params.get('Jz', 1.)
        Jxp = model_params.get('Jxp', 1.)
        Jyp = model_params.get('Jyp', 1.)
        Jzp = model_params.get('Jzp', 1.)
        hx = model_params.get('hx', 0.)
        hy = model_params.get('hy', 0.)
        hz = model_params.get('hz', 0.)

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hx, u, 'Sx')
            self.add_onsite(-hy, u, 'Sy')
            self.add_onsite(-hz, u, 'Sz')
        # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
        # Sx.Sx = 0.25 ( Sp.Sm + Sm.Sp + Sp.Sp + Sm.Sm )
        # Sy.Sy = 0.25 ( Sp.Sm + Sm.Sp - Sp.Sp - Sm.Sm )
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling((Jx + Jy) / 4., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling((Jx - Jy) / 4., u1, 'Sp', u2, 'Sp', dx, plus_hc=True)
            self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx)
        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
            self.add_coupling((Jxp + Jyp) / 4., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling((Jxp - Jyp) / 4., u1, 'Sp', u2, 'Sp', dx, plus_hc=True)
            self.add_coupling(Jzp, u1, 'Sz', u2, 'Sz', dx)
