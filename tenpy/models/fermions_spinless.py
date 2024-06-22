"""Spinless fermions with hopping and interaction.

.. todo ::     add further terms (e.g. c^dagger c^dagger + h.c.) to the Hamiltonian.
"""
# Copyright (C) TeNPy Developers, GNU GPLv3


from .model import CouplingMPOModel, NearestNeighborModel
from ..networks.site import FermionSite
from .lattice import Chain

__all__ = ['FermionModel', 'FermionChain']


class FermionModel(CouplingMPOModel):
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

        conserve : 'best' | 'N' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.FermionSite`.
            For ``'best'``, we check the parameters what can be preserved.
        J, V, mu : float | array
            Hopping, interaction and chemical potential as defined for the Hamiltonian above.
        phi_ext : float
            For 2D lattices and periodic y boundary conditions only.
            External magnetic flux 'threaded' through the cylinder.
            Hopping amplitudes for bonds 'across' the periodic boundary are modified such that
            particles hopping around the circumference of the cylinder acquire a phase `phi_ext`.
    """
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'N', str)
        if conserve == 'best':
            conserve = 'N'
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        site = FermionSite(conserve=conserve)
        return site

    def init_terms(self, model_params):
        J = model_params.get('J', 1., 'real_or_array')
        V = model_params.get('V', 1., 'real_or_array')
        mu = model_params.get('mu', 0., 'real_or_array')
        phi_ext = model_params.get('phi_ext', None, 'real')
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-mu, u, 'N')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            if phi_ext is None:
                hop = -J
            else:
                hop = self.coupling_strength_add_ext_flux(-J, dx, [0, phi_ext])
            self.add_coupling(hop, u1, 'Cd', u2, 'C', dx, plus_hc=True)
            self.add_coupling(V, u1, 'N', u2, 'N', dx)


class FermionChain(FermionModel, NearestNeighborModel):
    """The :class:`FermionModel` on a Chain, suitable for TEBD.

    See the :class:`FermionModel` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True
