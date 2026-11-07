"""Quantum Clock model.

Generalization of transverse field Ising model to higher dimensional on-site Hilbert space.
"""

# Copyright (C) TeNPy Developers, Apache license
from typing import Literal

import lasyten as lasy
import numpy as np

from .lattice import Chain
from .model import ModelBuilder

__all__ = ['ClockModel', 'ClockChain']


class ClockModel(ModelBuilder):
    r"""q-state Quantum clock model on a general lattice

    The Hamiltonian reads:

    .. math ::
        H = - \sum_{\langle i,j\rangle, i < j} \mathtt{J} (X_i X_j^\dagger + \mathrm{ h.c.})
            - \sum_{i} \mathtt{g} (Z_i + \mathrm{ h.c.})

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs, each pair appearing
    exactly once.
    The operators :math:`X_i` and :math:`Z_i` are :math:`N \times N` generalizations of
    the Pauli X and Z operators, see :class:`lasyten.sites.ClockSite`.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`ClockModel` below.

    Options
    -------
    .. cfg:config :: ClockModel
        :include: ModelBuilder

        conserve : None | 'Z_N'
            What should be conserved. See :class:`lasyten.sites.ClockSite`.
        q : int
            The number of states per site.
        J, g : float | array
            Couplings as defined for the Hamiltonian above.
            Defaults to ``J=g=1``.

    """

    conserve: Literal['Z_N', None]

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'Z_N', str)
        if conserve == 'best':
            conserve = 'Z_N'
            self.logger.info('%s: set conserve to %s', self.name, conserve)
        self.conserve = conserve
        q = model_params.get('q', None, int)
        if q is None:
            raise ValueError('Need to specify q.')
        return lasy.sites.ClockSite(q=q, conserve=conserve, backend=self.backend)

    def init_couplings(self, model_params):
        site, *_ = self.lattice.unit_cell
        self.define_coupling(
            lasy.models.clock_clock_coupling([site, site], Jx=1, plus_hc=False, name='X.Xh'),
            hc='Xh.X',
            permute='swap_hc',
        )

    def init_terms(self, model_params):
        J = np.asarray(model_params.get('J', 1.0, 'real_or_array'))
        g = np.asarray(model_params.get('g', 1.0, 'real_or_array'))
        self.add_onsite_operator(-g, 'Z', plus_hc=True)
        for u1, u2, dx in self.lattice.pairs['nearest_neighbors']:
            self.add_two_site_coupling(-J, 'X.Xh', dx, u1=u1, u2=u2, plus_hc=True)


class ClockChain(ClockModel):
    """The :class:`ClockModel` on a Chain, suitable for TEBD.

    See the :class:`ClockModel` for the documentation of parameters.
    """

    default_lattice = Chain
    force_default_lattice = True
