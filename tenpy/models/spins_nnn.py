"""Next-Nearest-neighbour spin-S models.

Uniform lattice of spin-S sites, coupled by nearest-neighbour interactions.
"""

from __future__ import division
import numpy as np

from .lattice import Chain
from ..networks.site import SpinSite, DoubleSite
from .model import CouplingModel, NearestNeighborModel, MPOModel
from ..tools.params import get_parameter, unused_parameters
from ..tools.misc import any_nonzero

__all__ = ['SpinChainNNN']


class SpinChainNNN(CouplingModel, MPOModel, NearestNeighborModel):
    r"""Spin-S sites coupled by (next-)nearest neighbour interactions on a `DoubleSite`.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{<i,j>}
              \mathtt{Jx} S^x_i S^x_j
            + \mathtt{Jy} S^y_i S^y_j
            + \mathtt{Jz} S^z_i S^z_j
            + \sum_{<<i,j>>}
              \mathtt{Jx} S^x_i S^x_j
            + \mathtt{Jy} S^y_i S^y_j
            + \mathtt{Jz} S^z_i S^z_j
            \\
            + \sum_i
              \mathtt{hx} S^x_i
            + \mathtt{hy} S^y_i
            + \mathtt{hz} S^z_i

    All parameters are collected in a single dictionary `model_param` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    Parameters
    ----------
    L : int
        Length of the chain
    S : {0.5, 1, 1.5, 2, ...}
        The 2S+1 local states range from m = -S, -S+1, ... +S.
    conserve : 'best' | 'Sz' | 'parity' | None
        What should be conserved. See :class:`~tenpy.networks.Site.SpinSite`.
    Jx, Jy, Jz, Jxp, Jyp, Jzp, hx, hy, hz : float | array
        Couplings as defined for the Hamiltonian above.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions. Coupling boundary conditions are chosen appropriately.
    """

    def __init__(self, model_param):
        # 0) read out/set default parameters
        verbose = get_parameter(model_param, 'verbose', 1, self.__class__)
        L = get_parameter(model_param, 'L', 2, self.__class__)
        Jx = get_parameter(model_param, 'Jx', 1., self.__class__)
        Jy = get_parameter(model_param, 'Jy', 1., self.__class__)
        Jz = get_parameter(model_param, 'Jz', 1., self.__class__)
        Jxp = get_parameter(model_param, 'Jxp', 1., self.__class__)
        Jyp = get_parameter(model_param, 'Jyp', 1., self.__class__)
        Jzp = get_parameter(model_param, 'Jzp', 1., self.__class__)
        hx = get_parameter(model_param, 'hx', 0., self.__class__)
        hy = get_parameter(model_param, 'hy', 0., self.__class__)
        hz = get_parameter(model_param, 'hz', 0., self.__class__)
        bc_MPS = get_parameter(model_param, 'bc_MPS', 'finite', self.__class__)
        S = get_parameter(model_param, 'S', 0.5, self.__class__)
        conserve = get_parameter(model_param, 'conserve', 'best', self.__class__)
        # check what we can conserve
        if conserve == 'best':
            # check how much we can conserve:
            if not any_nonzero(model_param, [('Jx', 'Jy'), ('Jxp', 'Jyp'), 'hx', 'hy'],
                               "check Sz conservation"):
                conserve = 'Sz'
            elif not any_nonzero(model_param, ['hx', 'hy'], "check parity conservation"):
                conserve = 'parity'
            else:
                conserve = None
            if verbose >= 1:
                print str(self.__class__) + ": set conserve to ", conserve
        unused_parameters(model_param, self.__class__)
        # 1) define Site and lattice
        spinsite = SpinSite(S, conserve)
        site = DoubleSite(spinsite, spinsite)
        lat = Chain(L, site, bc_MPS=bc_MPS)
        # 2) initialize CouplingModel
        bc_coupling = 'periodic' if bc_MPS == 'infinite' else 'open'
        CouplingModel.__init__(self, lat, bc_coupling)
        # 3) add terms of the Hamiltonian
        # (u is always 0 as we have only one site in the unit cell)
        self.add_onsite(hx, 0, 'Sx0')
        self.add_onsite(hy, 0, 'Sy0')
        self.add_onsite(hz, 0, 'Sz0')
        self.add_onsite(hx, 0, 'Sx1')
        self.add_onsite(hy, 0, 'Sy1')
        self.add_onsite(hz, 0, 'Sz1')
        Jx = np.asarray(Jx)
        Jy = np.asarray(Jy)
        # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
        # Sx.Sx = 0.25 ( Sp.Sm + Sm.Sp + Sp.Sp + Sm.Sm )
        # Sy.Sy = 0.25 ( Sp.Sm + Sm.Sp - Sp.Sp - Sm.Sm )
        self.add_onsite((Jx + Jy)/4., 0, 'Sp0 Sm1')
        self.add_onsite((Jx + Jy)/4., 0, 'Sm0 Sp1')
        self.add_onsite((Jx - Jy)/4., 0, 'Sp0 Sp1')
        self.add_onsite((Jx - Jy)/4., 0, 'Sm0 Sm1')
        self.add_onsite(Jz, 0, 'Sz0 Sz1')
        self.add_coupling((Jx + Jy)/4., 0, 'Sp1', 0, 'Sm0', 1)
        self.add_coupling((Jx + Jy)/4., 0, 'Sm1', 0, 'Sp0', 1)
        self.add_coupling((Jx - Jy)/4., 0, 'Sp1', 0, 'Sp0', 1)
        self.add_coupling((Jx - Jy)/4., 0, 'Sm1', 0, 'Sm0', 1)
        self.add_coupling(Jz, 0, 'Sz1', 0, 'Sz0', 1)
        # next-nearest neighbor couplings
        Jxp = np.asarray(Jxp)
        Jyp = np.asarray(Jyp)
        self.add_coupling((Jxp + Jyp)/4., 0, 'Sp0', 0, 'Sm0', 1)
        self.add_coupling((Jxp + Jyp)/4., 0, 'Sm0', 0, 'Sp0', 1)
        self.add_coupling((Jxp - Jyp)/4., 0, 'Sp0', 0, 'Sp0', 1)
        self.add_coupling((Jxp - Jyp)/4., 0, 'Sm0', 0, 'Sm0', 1)
        self.add_coupling(Jzp, 0, 'Sz0', 0, 'Sz0', 1)
        self.add_coupling((Jxp + Jyp)/4., 0, 'Sp1', 0, 'Sm1', 1)
        self.add_coupling((Jxp + Jyp)/4., 0, 'Sm1', 0, 'Sp1', 1)
        self.add_coupling((Jxp - Jyp)/4., 0, 'Sp1', 0, 'Sp1', 1)
        self.add_coupling((Jxp - Jyp)/4., 0, 'Sm1', 0, 'Sm1', 1)
        self.add_coupling(Jzp, 0, 'Sz1', 0, 'Sz1', 1)
        # 4) initialize MPO
        MPOModel.__init__(self, lat, self.calc_H_MPO())
        # 5) initialize H_bond
        NearestNeighborModel.__init__(self, self.lat, self.calc_H_bond())
