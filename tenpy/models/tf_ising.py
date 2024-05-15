"""Prototypical example of a quantum model: the transverse field Ising model.

Like the :class:`~tenpy.models.xxz_chain.XXZChain`, the transverse field ising chain
:class:`TFIChain` is contained in the more general :class:`~tenpy.models.spins.SpinChain`;
the idea is more to serve as a pedagogical example for a 'model'.

We choose the field along z to allow to conserve the parity, if desired.
"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np

from .model import CouplingMPOModel, NearestNeighborModel
from .lattice import Chain
from ..networks.site import SpinHalfSite

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
            What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
        sort_charge : bool
            Whether to sort by charges of physical legs. `True` by default.
        J, g : float | array
            Coupling as defined for the Hamiltonian above.

    """
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'parity', str)
        assert conserve != 'Sz'
        if conserve == 'best':
            conserve = 'parity'
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        sort_charge = model_params.get('sort_charge', True, bool)
        site = SpinHalfSite(conserve=conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):
        J = np.asarray(model_params.get('J', 1., 'real_or_array'))
        g = np.asarray(model_params.get('g', 1., 'real_or_array'))
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g, u, 'Sigmaz')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'Sigmax', u2, 'Sigmax', dx)
        # done


class TFIChain(TFIModel, NearestNeighborModel):
    """The :class:`TFIModel` on a Chain, suitable for TEBD.

    See the :class:`TFIModel` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True
