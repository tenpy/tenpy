"""Prototypical example of a 1D quantum model: the spin-1/2 XXZ chain.

The XXZ chain is contained in the more general :class:`~tenpy.models.spins.SpinChain`;
the idea of this module is more to serve as a pedagogical example for a model.
"""
# Copyright 2018 TeNPy Developers

import numpy as np

from .lattice import Site, Chain
from .model import CouplingModel, NearestNeighborModel, MPOModel, CouplingMPOModel
from ..linalg import np_conserved as npc
from ..tools.params import get_parameter, unused_parameters
from ..networks.site import SpinHalfSite  # if you want to use the predefined site

__all__ = ['XXZChain', 'XXZChain2']


class XXZChain(CouplingModel, NearestNeighborModel, MPOModel):
    r"""Spin-1/2 XXZ chain with Sz conservation.

    The Hamiltonian reads:

    .. math ::
        H = \sum_i \mathtt{Jxx}/2 (S^{+}_i S^{-}_{i+1} + S^{-}_i S^{+}_{i+1})
                 + \mathtt{Jz} S^z_i S^z_{i+1} \\
            - \sum_i \mathtt{hz} S^z_i

    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    Parameters
    ----------
    L : int
        Length of the chain.
    Jxx, Jz, hz : float | array
        Couplings as defined for the Hamiltonian above.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions. Coupling boundary conditions are chosen appropriately.
    """

    def __init__(self, model_params):
        # 0) read out/set default parameters
        name = "XXZChain"
        L = get_parameter(model_params, 'L', 2, name)
        Jxx = get_parameter(model_params, 'Jxx', 1., name, asarray=True)
        Jz = get_parameter(model_params, 'Jz', 1., name, True)
        hz = get_parameter(model_params, 'hz', 0., name, True)
        bc_MPS = get_parameter(model_params, 'bc_MPS', 'finite', name)
        unused_parameters(model_params, name)  # checks for mistyped parameters
        # 1-3):
        USE_PREDEFINED_SITE = False
        if not USE_PREDEFINED_SITE:
            # 1) charges of the physical leg. The only time that we actually define charges!
            leg = npc.LegCharge.from_qflat(npc.ChargeInfo([1], ['2*Sz']), [1, -1])
            # 2) onsite operators
            Sp = [[0., 1.], [0., 0.]]
            Sm = [[0., 0.], [1., 0.]]
            Sz = [[0.5, 0.], [0., -0.5]]
            # (Can't define Sx and Sy as onsite operators: they are incompatible with Sz charges.)
            # 3) local physical site
            site = Site(leg, ['up', 'down'], Sp=Sp, Sm=Sm, Sz=Sz)
        else:
            # there is a site for spin-1/2 defined in TeNPy, so just we can just use it
            # replacing steps 1-3)
            site = SpinHalfSite(conserve='Sz')
        # 4) lattice
        bc = 'periodic' if bc_MPS == 'infinite' else 'open'
        lat = Chain(L, site, bc=bc, bc_MPS=bc_MPS)
        # 5) initialize CouplingModel
        CouplingModel.__init__(self, lat)
        # 6) add terms of the Hamiltonian
        # (u is always 0 as we have only one site in the unit cell)
        self.add_onsite(-hz, 0, 'Sz')
        self.add_coupling(Jxx * 0.5, 0, 'Sp', 0, 'Sm', 1)
        self.add_coupling(np.conj(Jxx * 0.5), 0, 'Sp', 0, 'Sm', -1)  # h.c.
        self.add_coupling(Jz, 0, 'Sz', 0, 'Sz', 1)
        # 7) initialize H_MPO
        MPOModel.__init__(self, lat, self.calc_H_MPO())
        # 8) initialize H_bond (the order of 7/8 doesn't matter)
        NearestNeighborModel.__init__(self, lat, self.calc_H_bond())


class XXZChain2(CouplingMPOModel, NearestNeighborModel):
    """Another implementation of the Spin-1/2 XXZ chain with Sz conservation.

    This implementation takes the same parameters as the :class:`XXZChain`, but is implemented
    based on the :class:`~tenpy.models.model.CouplingMPOModel`.
    """

    def __init__(self, model_params):
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        return SpinHalfSite(conserve='Sz')  # use predefined Site

    def init_terms(self, model_params):
        # read out parameters
        Jxx = get_parameter(model_params, 'Jxx', 1., self.name, True)
        Jz = get_parameter(model_params, 'Jz', 1., self.name, True)
        hz = get_parameter(model_params, 'hz', 0., self.name, True)
        # add terms
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hz, u, 'Sz')
        for u1, u2, dx in self.lat.nearest_neighbors:
            self.add_coupling(Jxx * 0.5, u1, 'Sp', u2, 'Sm', dx)
            self.add_coupling(np.conj(Jxx * 0.5), u2, 'Sp', u1, 'Sm', -dx)  # h.c.
            self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx)
