"""Nearest-neighbor spin-S models.

Uniform lattice of spin-S sites, coupled by nearest-neighbor interactions.
"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
from lasyten.models.sites import SpinSite

from ..tools import to_array
from .lattice import Chain
from .model import ModelBuilder, SpinCouplingDefinition

__all__ = ['SpinModel', 'SpinChain', 'DipolarSpinChain']


def _onsite_matrix(site, hx, hy, hz, D, E):
    r"""Dense matrix for ``-hx Sx -hy Sy -hz Sz + D Sz^2 + E (Sx^2 - Sy^2)``.

    Built as a *single* matrix (rather than one term at a time) so that terms which
    individually break the site's conserved symmetry (e.g. Sx when Sz is conserved) can still
    combine into an overall-symmetric operator (e.g. when hx=hy=E=0, only diagonal Sz terms
    remain).
    """
    Sx, Sy, Sz = (site.spin_vector[:, :, k] for k in range(3))
    return -hx * Sx - hy * Sy - hz * Sz + D * (Sz @ Sz) + E * (Sx @ Sx - Sy @ Sy)


def _coupling_matrix(site1, site2, Jx, Jy, Jz, muJ):
    r"""Dense (unsplit) matrix for ``Jx Sx.Sx + Jy Sy.Sy + Jz Sz.Sz + muJ (Sy_i Sx_j - Sx_i Sy_j)``.

    Legs are ``[p0, p0*, p1, p1*]``; built as a single matrix for the same reason as
    :func:`_onsite_matrix`.
    """
    s1, s2 = site1.spin_vector, site2.spin_vector
    h = Jx * np.tensordot(s1[:, :, 0], s2[:, :, 0], axes=0)
    h = h + Jy * np.tensordot(s1[:, :, 1], s2[:, :, 1], axes=0)
    h = h + Jz * np.tensordot(s1[:, :, 2], s2[:, :, 2], axes=0)
    h = h + muJ * (np.tensordot(s1[:, :, 1], s2[:, :, 0], axes=0) - np.tensordot(s1[:, :, 0], s2[:, :, 1], axes=0))
    return h


class SpinModel(ModelBuilder):
    r"""Spin-S sites coupled by nearest neighbor interactions.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i,j\rangle, i < j}
              (\mathtt{Jx} S^x_i S^x_j + \mathtt{Jy} S^y_i S^y_j + \mathtt{Jz} S^z_i S^z_j
            + \mathtt{muJ} i/2 (S^{-}_i S^{+}_j - S^{+}_i S^{-}_j))  \\
            - \sum_i (\mathtt{hx} S^x_i + \mathtt{hy} S^y_i + \mathtt{hz} S^z_i) \\
            + \sum_i (\mathtt{D} (S^z_i)^2 + \mathtt{E} ((S^x_i)^2 - (S^y_i)^2))

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`SpinModel` below.

    Options
    -------
    .. cfg:config :: SpinModel
        :include: CouplingMPOModel

        S : {0.5, 1, 1.5, 2, ...}
            The 2S+1 local states range from m = -S, -S+1, ... +S.
        conserve : 'best' | 'Sz' | 'parity' | None
            What should be conserved. See :class:`~lasyten.models.sites.SpinSite`.
            For ``'best'``, we check the parameters what can be preserved.
        Jx, Jy, Jz, hx, hy, hz, muJ, D, E  : float | array
            Coupling as defined for the Hamiltonian above.
            Defaults to Heisenberg ``Jx=Jy=Jz=1.`` with other couplings 0.

    """

    def init_coupling_definitions(self, model_params):
        return [SpinCouplingDefinition]

    def init_sites(self, model_params):
        S = model_params.get('S', 0.5, 'real')
        conserve = model_params.get('conserve', 'best', str)
        if conserve == 'best':
            # check how much we can conserve
            if not model_params.any_nonzero([('Jx', 'Jy'), 'hx', 'hy', 'E'], 'check Sz conservation'):
                conserve = 'Sz'
            elif not model_params.any_nonzero(['hx', 'hy'], 'check parity conservation'):
                conserve = 'parity'
            else:
                conserve = None
            self.logger.info('%s: set conserve to %s', self.name, conserve)
        site = SpinSite(S=S, conserve=conserve)
        return site

    def init_terms(self, model_params):
        Jx = model_params.get('Jx', 1.0, 'real_or_array')
        Jy = model_params.get('Jy', 1.0, 'real_or_array')
        Jz = model_params.get('Jz', 1.0, 'real_or_array')
        hx = to_array(model_params.get('hx', 0.0, 'real_or_array'), self.lattice.Ls)
        hy = to_array(model_params.get('hy', 0.0, 'real_or_array'), self.lattice.Ls)
        hz = to_array(model_params.get('hz', 0.0, 'real_or_array'), self.lattice.Ls)
        D = to_array(model_params.get('D', 0.0, 'real_or_array'), self.lattice.Ls)
        E = to_array(model_params.get('E', 0.0, 'real_or_array'), self.lattice.Ls)
        muJ = model_params.get('muJ', 0.0, 'real_or_array')

        self.add_onsite_operator(-hx, 'Sx')
        self.add_onsite_operator(-hy, 'Sy')
        self.add_onsite_operator(-hz, 'Sz')
        self.add_onsite_operator(D, 'Sz Sz')
        self.add_onsite_operator(0.5 * E, 'Sp Sp', plus_hc=True)

        for u1, u2, dx in self.lattice.pairs['nearest_neighbors']:
            if np.all(Jx == Jy) and np.all(Jy == Jz):
                self.add_two_site_coupling(Jx, 'S.S', dx, x1='sum_over', u1=u1, u2=u2)
            else:
                self.add_two_site_coupling(0.25 * (Jx + Jy), 'Sp.Sm', dx, x1='sum_over', u1=u1, u2=u2, plus_hc=True)
                self.add_two_site_coupling(0.25 * (Jx - Jy), 'Sp.Sp', dx, x1='sum_over', u1=u1, u2=u2, plus_hc=True)
                self.add_two_site_coupling(Jz, 'Sz.Sz', dx, x1='sum_over', u1=u1, u2=u2)
            self.add_two_site_coupling(0.5j * muJ, 'Sp.Sm', dx, x1='sum_over', u1=u1, u2=u2, plus_hc=True)

        return


class SpinChain(SpinModel):
    """The :class:`SpinModel` on a Chain, suitable for TEBD.

    See the :class:`SpinModel` for the documentation of parameters.
    """

    default_lattice = Chain
    force_default_lattice = True


class DipolarSpinChain(ModelBuilder):
    r"""Dipole conserving H3-H4 spin-S chain.

    The Hamiltonian reads:

    .. math ::
        H = - \mathtt{J3} \sum_{i} (S^+_i (S^-_{i + 1})^2 S^+_{i + 2} + \mathrm{h.c.})
            - \mathtt{J4} \sum_{i} (S^+_i S^-_{i + 1} S^-_{i + 2} S^+_{i + 2} + \mathrm{h.c.})

    .. note ::
        Dipole ("dipole"-)conservation is not (yet) supported by the lasyten-based symmetry
        framework, so ``conserve='dipole'`` is currently unavailable. This class otherwise
        works with the symmetries that :class:`~lasyten.models.sites.SpinSite` does support.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`DipolarSpinChain` below.

    Options
    -------
    .. cfg:config :: DipolarSpinChain
        :include: CouplingMPOModel

        S : {0.5, 1, 1.5, 2, ...}
            The 2S+1 local states range from m = -S, -S+1, ... +S.
            Defaults to ``S=1``.
        conserve : 'best' | 'Sz' | 'parity' | None
            What should be conserved. See :class:`~lasyten.models.sites.SpinSite`.
            For ``'best'``, we preserve ``'Sz'``.
        J3, J4 : float | array
            Coupling as defined for the Hamiltonian above.

    """

    def init_coupling_definitions(self, model_params):
        return [SpinCouplingDefinition]

    def init_lattice(self, model_params):
        """Initialize a 1D lattice"""
        L = model_params.get('L', 64)
        S = model_params.get('S', 1)
        conserve = model_params.get('conserve', 'best')
        if conserve == 'best':
            conserve = 'Sz'
            self.logger.info('%s: set conserve to %s', self.name, conserve)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        bc = 'periodic' if bc_MPS in ['infinite', 'segment'] else 'open'
        bc = model_params.get('bc', bc)
        site = SpinSite(S=S, conserve=conserve)
        lattice = Chain(L, site, bc=bc, bc_MPS=bc_MPS)
        return lattice

    def init_terms(self, model_params):
        """Add the onsite and coupling terms to the model"""
        J3 = model_params.get('J3', 1)
        J4 = model_params.get('J4', 0)

        self.add_multi_site_coupling(-J3, 'Sp.SmSm.Sp', dx=[[0], [1], [2]], plus_hc=True)
        self.add_multi_site_coupling(-J4, 'Sp.Sm.Sm.Sp', dx=[[0], [1], [2], [3]], plus_hc=True)
