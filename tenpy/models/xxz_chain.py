"""Prototypical example of a 1D quantum model: the spin-1/2 XXZ chain.

The XXZ chain is contained in the more general :class:`~tenpy.models.spins.SpinChain`; the idea of
this module is more to serve as a pedagogical example for a model.
"""
# Copyright (C) TeNPy Developers, Apache license

from .lattice import Site, Chain
from .model import CouplingModel, NearestNeighborModel, MPOModel, CouplingMPOModel
from ..linalg import np_conserved as npc
from ..tools.params import asConfig
from ..networks.site import SpinHalfSite  # if you want to use the predefined site

__all__ = ['XXZChain', 'XXZChain2']


class XXZChain(CouplingModel, NearestNeighborModel, MPOModel):
    r"""Spin-1/2 XXZ chain with Sz conservation.

    The Hamiltonian reads:

    .. math ::
        H = \sum_i \mathtt{Jxx}/2 (S^{+}_i S^{-}_{i+1} + S^{-}_i S^{+}_{i+1})
                 + \mathtt{Jz} S^z_i S^z_{i+1} \\
            - \sum_i \mathtt{hz} S^z_i

    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`XXZChain` below.

    Options
    -------
    .. cfg:config :: XXZChain
        :include: CouplingMPOModel

        L : int
            Length of the chain.
        conserve : 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
        Jxx, Jz, hz : float | array
            Coupling as defined for the Hamiltonian above.
            Defaults to ``Jxx=Jz=1`` without field ``hz=0``.
        bc_MPS : {'finite' | 'infinite'}
            MPS boundary conditions. Coupling boundary conditions are chosen appropriately.
        sort_charge : bool
            Whether to sort by charges of physical legs. `True` by default.

    """
    def __init__(self, model_params):
        # 0) read out/set default parameters
        model_params = asConfig(model_params, "XXZChain")
        L = model_params.get('L', 2, int)
        Jxx = model_params.get('Jxx', 1., 'real_or_array')
        Jz = model_params.get('Jz', 1., 'real_or_array')
        hz = model_params.get('hz', 0., 'real_or_array')
        bc_MPS = model_params.get('bc_MPS', 'finite', str)
        conserve = model_params.get('conserve', 'best', str)
        if conserve == 'best':
            conserve = 'Sz'
        sort_charge = model_params.get('sort_charge', True, bool)
        # 1-3):
        USE_PREDEFINED_SITE = False
        if not USE_PREDEFINED_SITE:
            # 1) charges of the physical leg. The only time that we actually define charges!
            if conserve == 'Sz':
                leg = npc.LegCharge.from_qflat(npc.ChargeInfo([1], ['2*Sz']), [1, -1])
            elif conserve == 'parity':
                leg = npc.LegCharge.from_qflat(npc.ChargeInfo([2], ['parity_Sz']), [1, 0])
            else:
                leg = npc.LegCharge.from_trivial(2)
            # 2) onsite operators
            Sp = [[0., 1.], [0., 0.]]
            Sm = [[0., 0.], [1., 0.]]
            Sz = [[0.5, 0.], [0., -0.5]]
            # (Can't define Sx and Sy as onsite operators: they are incompatible with Sz charges.)
            # 3) local physical site
            site = Site(leg, ['up', 'down'], sort_charge=sort_charge, Sp=Sp, Sm=Sm, Sz=Sz)
        else:
            # there is a site for spin-1/2 defined in TeNPy, so just we can just use it
            # replacing steps 1-3)
            site = SpinHalfSite(conserve=conserve, sort_charge=sort_charge)
        # 4) lattice
        bc = 'open' if bc_MPS == 'finite' else 'periodic'
        lat = Chain(L, site, bc=bc, bc_MPS=bc_MPS)
        # 5) initialize CouplingModel
        CouplingModel.__init__(self, lat)
        # 6) add terms of the Hamiltonian
        # (u is always 0 as we have only one site in the unit cell)
        self.add_onsite(-hz, 0, 'Sz')
        self.add_coupling(Jxx * 0.5, 0, 'Sp', 0, 'Sm', 1, plus_hc=True)
        # the `plus_hc=True` adds the h.c. term
        # see also the examples tenpy.models.model.CouplingModel.add_coupling
        self.add_coupling(Jz, 0, 'Sz', 0, 'Sz', 1)
        # 7) initialize H_MPO
        MPOModel.__init__(self, lat, self.calc_H_MPO())
        # 8) initialize H_bond (the order of 7/8 doesn't matter)
        NearestNeighborModel.__init__(self, lat, self.calc_H_bond())


class XXZChain2(CouplingMPOModel, NearestNeighborModel):
    """Another implementation of the Spin-1/2 XXZ chain with Sz conservation.

    This implementation takes the same parameters as the :class:`XXZChain`, but is implemented
    based on the :class:`~tenpy.models.model.CouplingMPOModel`.

    Parameters
    ----------
    model_params : dict | :class:`~tenpy.tools.params.Config`
        See :cfg:config:`XXZChain`
    """
    default_lattice = "Chain"
    force_default_lattice = True

    def init_sites(self, model_params):
        sort_charge = model_params.get('sort_charge', True, bool)
        conserve = model_params.get('conserve', 'best', str)
        if conserve == 'best':
            conserve = 'Sz'
        return SpinHalfSite(conserve=conserve, sort_charge=sort_charge)  # use predefined Site

    def init_terms(self, model_params):
        # read out parameters
        Jxx = model_params.get('Jxx', 1., 'real_or_array')
        Jz = model_params.get('Jz', 1., 'real_or_array')
        hz = model_params.get('hz', 0., 'real_or_array')
        # add terms
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hz, u, 'Sz')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(Jxx * 0.5, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx)
