"""Prototypical example of a quantum model: the transverse field Ising model.

Like the :class:`~tenpy.models.xxz_chain.XXZChain`, the transverse field ising chain
:class:`TFIChain` is contained in the more general :class:`~tenpy.models.spins.SpinChain`;
the idea is more to serve as a pedagogical example for a 'model'.

We choose the field along z to allow to conserve the parity, if desired.
"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
from lasyten.models.sites import SpinSite

from ..tools import to_array
from .lattice import Chain
from .model import ModelBuilder, NearestNeighborModel, SpinCouplingDefinition

__all__ = ['TFIModel', 'TFIChain']


class TFIModel(ModelBuilder):
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
            What should be conserved. See :class:`~lasyten.models.sites.SpinSite`.
        J, g : float | array
            Coupling as defined for the Hamiltonian above.
            Defaults to ``J=g=1``

    """

    def init_coupling_definitions(self, model_params):
        return [SpinCouplingDefinition]

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
        g = to_array(model_params.get('g', 1.0, 'real_or_array'), self.lattice.Ls)
        # sigma^z = 2 * S^z
        self.add_onsite_operator(-2 * g, 'Sz', x='sum_over', u='sum_over')
        for u1, u2, dx in self.lattice.pairs['nearest_neighbors']:
            # sigma^x_i sigma^x_j = 4 * S^x_i S^x_j
            self.add_two_site_coupling(-4 * J, 'Sx.Sx', dx, x1='sum_over', u1=u1, u2=u2)
        return


class TFIChain(TFIModel, NearestNeighborModel):
    """The :class:`TFIModel` on a Chain, suitable for TEBD.

    See the :class:`TFIModel` for the documentation of parameters.
    """

    default_lattice = Chain
    force_default_lattice = True
