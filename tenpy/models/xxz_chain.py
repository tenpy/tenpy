"""Prototypical example of a 1D quantum model: the spin-1/2 XXZ chain.

The XXZ chain is contained in the more general :class:`~tenpy.models.spins.SpinChain`; the idea of
this module is more to serve as a pedagogical example for a model.
"""
# Copyright (C) TeNPy Developers, Apache license

from cyten.models.couplings import spin_field_coupling, spin_spin_coupling
from cyten.models.sites import SpinSite

from ..tools.misc import to_array
from ..tools.params import asConfig
from .lattice import Chain
from .model import CouplingModel, CouplingMPOModel, MPOModel, NearestNeighborModel

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
            What should be conserved. See :class:`~cyten.models.sites.SpinSite`.
        Jxx, Jz, hz : float | array
            Coupling as defined for the Hamiltonian above.
            Defaults to ``Jxx=Jz=1`` without field ``hz=0``.
        bc_MPS : {'finite'}
            MPS boundary conditions. Only ``'finite'`` is currently supported by the
            cyten-native ``add_coupling`` path.

    """

    def __init__(self, model_params):
        # 0) read out/set default parameters
        model_params = asConfig(model_params, 'XXZChain')
        L = model_params.get('L', 2, int)
        Jxx = model_params.get('Jxx', 1.0, 'real_or_array')
        Jz = model_params.get('Jz', 1.0, 'real_or_array')
        hz = to_array(model_params.get('hz', 0.0, 'real_or_array'), [L])
        bc_MPS = model_params.get('bc_MPS', 'finite', str)
        conserve = model_params.get('conserve', 'best', str)
        if conserve == 'best':
            conserve = 'Sz'
        # 1-3) local physical site
        site = SpinSite(S=0.5, conserve=conserve)
        # 4) lattice
        bc = 'open' if bc_MPS == 'finite' else 'periodic'
        lat = Chain(L, site, bc=bc, bc_MPS=bc_MPS)
        # 5) initialize CouplingModel
        CouplingModel.__init__(self, lat)
        # 6) add terms of the Hamiltonian
        # (u is always 0 as we have only one site in the unit cell)
        field = spin_field_coupling([site], hz=1.0)
        for i in range(L):
            self.add_coupling(field, [i], strength=-hz[i])
        # Jxx/2 (Sp_i Sm_j + Sm_i Sp_j) = Jxx (Sx_i Sx_j + Sy_i Sy_j)
        coupling_xy = spin_spin_coupling([site, site], Jx=1.0, Jy=1.0)
        coupling_z = spin_spin_coupling([site, site], Jz=1.0)
        mps_i, mps_j, Jxx_vals = self.lat.possible_couplings(0, 0, [1], Jxx)
        for i, j, strength in zip(mps_i, mps_j, Jxx_vals):
            self.add_coupling(coupling_xy, [int(i), int(j)], strength=strength)
        mps_i, mps_j, Jz_vals = self.lat.possible_couplings(0, 0, [1], Jz)
        for i, j, strength in zip(mps_i, mps_j, Jz_vals):
            self.add_coupling(coupling_z, [int(i), int(j)], strength=strength)
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

    default_lattice = 'Chain'
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best', str)
        if conserve == 'best':
            conserve = 'Sz'
        return SpinSite(S=0.5, conserve=conserve)  # use predefined Site

    def init_terms(self, model_params):
        # read out parameters
        Jxx = model_params.get('Jxx', 1.0, 'real_or_array')
        Jz = model_params.get('Jz', 1.0, 'real_or_array')
        hz = to_array(model_params.get('hz', 0.0, 'real_or_array'), self.lat.Ls)
        # add terms
        for u in range(len(self.lat.unit_cell)):
            site = self.lat.unit_cell[u]
            field = spin_field_coupling([site], hz=1.0)
            for i, i_lat in zip(*self.lat.mps_lat_idx_fix_u(u)):
                self.add_coupling(field, [i], strength=-hz[tuple(i_lat)])
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            site1, site2 = self.lat.unit_cell[u1], self.lat.unit_cell[u2]
            coupling_xy = spin_spin_coupling([site1, site2], Jx=1.0, Jy=1.0)
            coupling_z = spin_spin_coupling([site1, site2], Jz=1.0)
            mps_i, mps_j, Jxx_vals = self.lat.possible_couplings(u1, u2, dx, Jxx)
            for i, j, strength in zip(mps_i, mps_j, Jxx_vals):
                self.add_coupling(coupling_xy, [int(i), int(j)], strength=strength)
            mps_i, mps_j, Jz_vals = self.lat.possible_couplings(u1, u2, dx, Jz)
            for i, j, strength in zip(mps_i, mps_j, Jz_vals):
                self.add_coupling(coupling_z, [int(i), int(j)], strength=strength)
