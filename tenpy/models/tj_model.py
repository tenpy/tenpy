"""tJ model"""
# Copyright (C) TeNPy Developers, GNU GPLv3

from .model import CouplingMPOModel, NearestNeighborModel
from .lattice import Chain
from ..networks.site import SpinHalfHoleSite

__all__ = ['tJModel', 'tJChain']


class tJModel(CouplingMPOModel):
    r"""Spin-1/2 t-J model.

    The Hamiltonian reads:

    .. math ::
        H = - \sum_{\langle i, j \rangle, i < j, \sigma} 
            t \mathcal{P}(c^{\dagger}_{\sigma, i} c_{\sigma j} + h.c.)\mathcal{P}
            + \sum_{\langle i, j \rangle, i < j, \sigma}  
            J (S^x_i S^x_j + S^y_i S^y_j + S^z_i S^z_j -\frac{1}{4}(n_{\uparrow,i} + n_{\downarrow,i})(n_{\uparrow,j} + n_{\downarrow,j}))


    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs and :math:`\mathcal{P}` is the Gutzwiller projector on empty and singly occupied sites.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    .. warning ::
        Using the Jordan-Wigner string (``JW``) is crucial to get correct results!
        See :doc:`/intro/JordanWigner` for details.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`tJModel` below.

    Options
    -------
    .. cfg:config :: tJModel
        :include: CouplingMPOModel

        cons_N : {'N' | 'parity' | None}
            Whether particle number is conserved,
            see :class:`~tenpy.networks.site.SpinHalfHoleSite` for details.
        cons_Sz : {'Sz' | 'parity' | None}
            Whether spin is conserved,
            see :class:`~tenpy.networks.site.SpinHalfHoleSite` for details.
        t, J: float | array
            Couplings as defined for the Hamiltonian above. Note the signs!
            Defaults to ``t=J=1``
    """

    def init_sites(self, model_params):
        cons_N = model_params.get('cons_N', 'N', str)
        cons_Sz = model_params.get('cons_Sz', 'Sz', str)
        site = SpinHalfHoleSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site

    def init_terms(self, model_params):
        # 0) Read out/set default parameters.
        t = model_params.get('t', 1., 'real_or_array')
        J = model_params.get('J', 1., 'real_or_array')

        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-t, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(-t, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
            self.add_coupling(J / 2., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)

            self.add_coupling(J, u1, 'Sz', u2, 'Sz', dx)
            self.add_coupling(-J / 4, u1, 'Ntot', u2, 'Ntot', dx)


class tJChain(tJModel, NearestNeighborModel):
    """The :class:`tJModel` on a Chain, suitable for TEBD.

    See the :class:`tJModel` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True
