"""Quantum Clock model.

Generalization of transverse field Ising model to higher dimensional on-site Hilbert space.
"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
from .model import CouplingMPOModel, NearestNeighborModel
from .lattice import Chain
from ..networks.site import ClockSite


__all__ = ['ClockModel', 'ClockChain']


class ClockModel(CouplingMPOModel):
    r"""q-state Quantum clock model on a general lattice

    The Hamiltonian reads:

    .. math ::
        H = - \sum_{\langle i,j\rangle, i < j} \mathtt{J} (X_i X_j^\dagger + \mathrm{ h.c.})
            - \sum_{i} \mathtt{g} (Z_i + \mathrm{ h.c.})

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs, each pair appearing
    exactly once.
    The operators :math:`X_i` and :math:`Z_i` are :math:`N \times N` generalizations of
    the Pauli X and Z operators, see :class:`~tenpy.networks.site.ClockSite`.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`ClockModel` below.

    Options
    -------
    .. cfg:config :: ClockModel
        :include: CouplingMPOModel

        conserve : None | 'Z'
            What should be conserved. See :class:`~tenpy.networks.Site.ClockSite`.
        sort_charge : bool
            Whether to sort by charges of physical legs. `True` by default.
        q : int
            The number of states per site.
        J, g : float | array
            Couplings as defined for the Hamiltonian above.
            Defaults to ``J=g=1``.

    """

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'Z', str)
        if conserve == 'best':
            conserve = 'Z'
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        q = model_params.get('q', None, int)
        if q is None:
            raise ValueError('Need to specify q.')
        sort_charge = model_params.get('sort_charge', True, bool)
        return ClockSite(q=q, conserve=conserve, sort_charge=sort_charge)

    def init_terms(self, model_params):
        J = np.asarray(model_params.get('J', 1., 'real_or_array'))
        g = np.asarray(model_params.get('g', 1., 'real_or_array'))
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g, u, 'Z', plus_hc=True)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'X', u2, 'Xhc', dx, plus_hc=True)


class ClockChain(ClockModel, NearestNeighborModel):
    """The :class:`ClockModel` on a Chain, suitable for TEBD.

    See the :class:`ClockModel` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True
