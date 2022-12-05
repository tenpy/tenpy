"""Quantum Clock model, generalization of transverse field Ising model to higher dimensional on-site Hilbert space.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
from .model import CouplingMPOModel, NearestNeighborModel
from .lattice import Chain
from ..networks.site import ClockSite


class ClockModel(CouplingMPOModel):
    r"""Quantum clock model on a general lattice

    The Hamiltonian reads:

    .. math ::
        H = - \sum_{\langle i,j\rangle, i < j} \mathtt{J} (X_i X_j^\dagger + \mathrm{ h.c.})
            - \sum_{i} \mathtt{g} (Z_i + \mathrm{ h.c.})

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
    .. cfg:config :: ClockModel
        :include: CouplingMPOModel

        conserve : None | 'Zq'
            What should be conserved. See :class:`~tenpy.networks.Site.ClockSite`.
        sort_charge : bool | None
            Whether to sort by charges of physical legs.
            See change comment in :class:`~tenpy.networks.site.Site`.
        J, g : float | array
            Couplings as defined for the Hamiltonian above.

    """

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'Zq')
        q = model_params.get('q', None)
        if q is None:
            raise ValueError('Need to specify q.')
        if conserve == 'best':
            conserve = 'Zq'
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        sort_charge = model_params.get('sort_charge', None)
        site = ClockSite(q=q, conserve=conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):
        J = np.asarray(model_params.get('J', 1.))
        g = np.asarray(model_params.get('g', 1.))
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g, u, 'Zphc')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'X', u2, 'Xhc', dx)
            self.add_coupling(-J, u1, 'Xhc', u2, 'X', dx)


class ClockChain(ClockModel, NearestNeighborModel):
    """The :class:`ClockModel` on a Chain, suitable for TEBD.

    See the :class:`ClockModel` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True
