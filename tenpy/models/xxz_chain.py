"""Prototypical example of a 1D quantum model: the spin-1/2 XXZ chain.

Although the XXZ chain is contained in the more general `SpinChain`,
the idea of this class is to serve as a pedagogical example for a 'model'.

.. todo ::
    `SpinChain` is not implemented yet. Add crossref in sphinx format.

.. todo ::
    Add something like the old tenpy.models.model.set_var to tools and use it.
"""

from __future__ import division
import numpy as np

from .lattice import Site, Chain
from .model import CouplingModel, NearestNeighborModel, MPOModel
from ..linalg import np_conserved as npc
from ..tools.params import get_parameter

__all__ = ['XXZChain']

class XXZChain(CouplingModel, NearestNeighborModel, MPOModel):
    r"""Spin-1/2 XXZ chain with Sz conservation.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{i} Jxx/2 (S^{+}_i S^{-}_{i+1} + S^{-}_i S^{+}_{i+1})
            + Jz S^Z_i S^Z_{i+1}
            - hz Sz_i

    Parameters
    ----------
    L : int
        Length of the chain
    Jxx, Jz, hz : float | array
        Couplings as defined for the Hamiltonian above.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions. Coupling boundary conditions are chosen appropriately.

    All parameters are
    """
    def __init__(self, model_param):
        # 0) read out/set default parameters
        L = get_parameter(model_param, 'L', 2, self.__class__)
        Jxx = get_parameter(model_param, 'Jxx', 1., self.__class__)
        Jz = get_parameter(model_param, 'Jz', 1., self.__class__)
        hz = get_parameter(model_param, 'hz', 0., self.__class__)
        bc_MPS = get_parameter(model_param, 'bc_MPS', 'finite', self.__class__)
        # 1) charges of the physical Leg. The only time that we actually define charges!
        leg = npc.LegCharge.from_qflat(npc.ChargeInfo([1], ['2*Sz']), [1, -1])
        # 2) onsite operators
        Sp = [[0., 1.], [0., 0.]]
        Sm = [[0., 0.], [1., 0.]]
        Sz = [[0.5, 0.], [0., -0.5]]
        # (Can't define Sx and Sy as onsite operators: they are incompatible with Sz charges.)
        # 3) local physical site
        site = Site(leg, ['up', 'down'], Sp=Sp, Sm=Sm, Sz=Sz)
        # NOTE: the most common `site` are pre-defined in tenpy.networks.site.
        # you could replace step 1)-3) by::
        # from tenpy.networks.site import spin_half_site
        # site = spin_half_site(conserve='Sz')
        # 4) lattice
        lat = Chain(L, site, bc_MPS=bc_MPS)
        bc_coupling = 'periodic' if bc_MPS == 'infinite' else 'open'
        # 5) initialize CouplingModel
        CouplingModel.__init__(self, lat, bc_coupling)
        # 6) add terms of the Hamiltonian
        # (u is always 0 as we have only one site in the unit cell)
        self.add_onsite(hz, 0, 'Sz')
        Jxx_half = np.asarray(Jxx)*0.5  # convert to array: allow `array_like` Jxx
        self.add_coupling(Jxx_half, 0, 'Sp', 0, 'Sm', 1)
        self.add_coupling(Jxx_half, 0, 'Sm', 0, 'Sp', 1)
        self.add_coupling(Jz, 0, 'Sz', 0, 'Sz', 1)
        # 7) initialize MPO
        MPOModel.__init__(self, lat, self.calc_H_MPO())
        # 8) initialize bonds (the order of 7/8 doesn't matter)
        NearestNeighborModel.__init__(self, lat, self.calc_H_bond())
