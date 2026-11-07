"""Spinless fermions with hopping and interaction.

.. todo ::     add further terms (e.g. c^dagger c^dagger + h.c.) to the Hamiltonian.
"""

# Copyright (C) TeNPy Developers, Apache license
from typing import Literal

import lasyten as lasy

from .lattice import Chain
from .model import ModelBuilder

__all__ = ['FermionModel', 'FermionChain']


class FermionModel(ModelBuilder):
    r"""Spinless fermions with particle number conservation.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i,j\rangle, i<j}
              - \mathtt{J}~(c^{\dagger}_i c_j + c^{\dagger}_j c_i) + \mathtt{V}~n_i n_j \\
            - \sum_i
              \mathtt{mu}~n_{i}

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    .. warning ::
        Using the Jordan-Wigner string (``JW``) is crucial to get correct results!
        See :doc:`/intro/JordanWigner` for details.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`FermionModel` below.

    Options
    -------
    .. cfg:config :: FermionModel
        :include: CouplingMPOModel

        conserve : 'best' | 'N' | 'parity'
            What should be conserved. See :class:`lasyten.sites.SpinlessFermionSite`.
            For ``'best'``, we check the parameters what can be preserved.
        J, V, mu : float | array
            Hopping, interaction and chemical potential as defined for the Hamiltonian above.
            Defaults to ``J=V=1`` and ``mu=0``.
        phi_ext : float
            For 2D lattices and periodic y boundary conditions only.
            External magnetic flux 'threaded' through the cylinder.
            Hopping amplitudes for bonds 'across' the periodic boundary are modified such that
            particles hopping around the circumference of the cylinder acquire a phase `phi_ext`.

    """

    conserve: Literal['N', 'parity']

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'N', str)
        if conserve == 'best':
            conserve = 'N'
            self.logger.info('%s: set conserve to %s', self.name, conserve)
        self.conserve = conserve
        self._check_fermion_conserve(conserve)
        return lasy.sites.SpinlessFermionSite(conserve=conserve)

    def init_couplings(self, model_params):
        site, *_ = self.lattice.unit_cell
        self.define_coupling(
            lasy.models.hopping([site, site], t_left=1, t_right=0, name='Cd.C'), hc='C.Cd', permute='swap_hc'
        )
        self.define_coupling(
            lasy.models.density_density_interaction([site, site], name='N.N'), hc='N.N', permute='invariant'
        )

    def init_terms(self, model_params):
        J = model_params.get('J', 1.0, 'real_or_array')
        V = model_params.get('V', 1.0, 'real_or_array')
        mu = model_params.get('mu', 0.0, 'real_or_array')
        phi_ext = model_params.get('phi_ext', None, 'real')
        self.add_onsite_operator(-mu, 'N')
        for u1, u2, dx in self.lattice.pairs['nearest_neighbors']:
            if phi_ext is None:
                hop = -J
            else:
                hop = self.coupling_strength_add_ext_flux(-J, dx, [0, phi_ext])
            self.add_two_site_coupling(hop, 'Cd.C', dx, u1=u1, u2=u2, plus_hc=True)
            self.add_two_site_coupling(V, 'N.N', dx, u1=u1, u2=u2)


class FermionChain(FermionModel):
    """The :class:`FermionModel` on a Chain, suitable for TEBD.

    See the :class:`FermionModel` for the documentation of parameters.
    """

    default_lattice = Chain
    force_default_lattice = True
