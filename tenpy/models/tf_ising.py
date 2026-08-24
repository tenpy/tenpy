"""Prototypical example of a quantum model: the transverse field Ising model.

Like the :class:`~tenpy.models.xxz_chain.XXZChain`, the transverse field ising chain
:class:`TFIChain` is contained in the more general :class:`~tenpy.models.spins.SpinChain`;
the idea is more to serve as a pedagogical example for a 'model'.

We choose the field along z to allow to conserve the parity, if desired.
"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
from cyten.models.couplings import spin_field_coupling, spin_spin_coupling
from cyten.models.sites import SpinSite

from ..tools.misc import to_array
from .lattice import Chain
from .model import CouplingMPOModel, NearestNeighborModel

__all__ = ['TFIModel', 'TFIChain']


class TFIModel(CouplingMPOModel):
    r"""Transverse field Ising model on a general lattice.

    The Hamiltonian reads:

    .. math ::
        H = - \sum_{\langle i,j\rangle, i < j} \mathtt{J} \sigma^x_i \sigma^x_{j}
            - \sum_{i} \mathtt{g} \sigma^z_i

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs, each pair appearing
    exactly once.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`TFIModel` below.

    Options
    -------
    .. cfg:config :: TFIModel
        :include: CouplingMPOModel

        conserve : None | 'parity'
            What should be conserved. See :class:`~cyten.models.sites.SpinSite`.
        J, g : float | array
            Coupling as defined for the Hamiltonian above.
            Defaults to ``J=g=1``

    """

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'parity', str)
        assert conserve != 'Sz'
        if conserve == 'best':
            conserve = 'parity'
            self.logger.info('%s: set conserve to %s', self.name, conserve)
        site = SpinSite(S=0.5, conserve=conserve)
        return site

    def init_terms(self, model_params):
        J = np.asarray(model_params.get('J', 1.0, 'real_or_array'))
        g = to_array(model_params.get('g', 1.0, 'real_or_array'), self.lat.Ls)
        for u in range(len(self.lat.unit_cell)):
            site = self.lat.unit_cell[u]
            # sigma^z = 2 * S^z
            field = spin_field_coupling([site], hz=2.0)
            for i, i_lat in zip(*self.lat.mps_lat_idx_fix_u(u)):
                self.add_coupling(field, [i], strength=-g[tuple(i_lat)])
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            site1, site2 = self.lat.unit_cell[u1], self.lat.unit_cell[u2]
            # sigma^x_i sigma^x_j = 4 * S^x_i S^x_j
            coupling = spin_spin_coupling([site1, site2], Jx=4.0)
            mps_i, mps_j, strength_vals = self.lat.possible_couplings(u1, u2, dx, J)
            for i, j, strength in zip(mps_i, mps_j, strength_vals):
                self.add_coupling(coupling, [int(i), int(j)], strength=-strength)
        # done


class TFIChain(TFIModel, NearestNeighborModel):
    """The :class:`TFIModel` on a Chain, suitable for TEBD.

    See the :class:`TFIModel` for the documentation of parameters.
    """

    default_lattice = Chain
    force_default_lattice = True
