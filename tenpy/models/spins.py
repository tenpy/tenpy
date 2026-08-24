"""Nearest-neighbor spin-S models.

Uniform lattice of spin-S sites, coupled by nearest-neighbor interactions.
"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
from cyten.models.couplings import Coupling
from cyten.models.sites import SpinSite
from cyten.tensors import dagger

from ..tools.misc import to_array
from .lattice import Chain
from .model import CouplingMPOModel, NearestNeighborModel

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


class SpinModel(CouplingMPOModel):
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
            What should be conserved. See :class:`~cyten.models.sites.SpinSite`.
            For ``'best'``, we check the parameters what can be preserved.
        Jx, Jy, Jz, hx, hy, hz, muJ, D, E  : float | array
            Coupling as defined for the Hamiltonian above.
            Defaults to Heisenberg ``Jx=Jy=Jz=1.`` with other couplings 0.

    """

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
        hx = to_array(model_params.get('hx', 0.0, 'real_or_array'), self.lat.Ls)
        hy = to_array(model_params.get('hy', 0.0, 'real_or_array'), self.lat.Ls)
        hz = to_array(model_params.get('hz', 0.0, 'real_or_array'), self.lat.Ls)
        D = to_array(model_params.get('D', 0.0, 'real_or_array'), self.lat.Ls)
        E = to_array(model_params.get('E', 0.0, 'real_or_array'), self.lat.Ls)
        muJ = model_params.get('muJ', 0.0, 'real_or_array')

        # (u is always 0 as we have only one site in the unit cell)
        for u in range(len(self.lat.unit_cell)):
            site = self.lat.unit_cell[u]
            for i, i_lat in zip(*self.lat.mps_lat_idx_fix_u(u)):
                idx = tuple(i_lat)
                mat = _onsite_matrix(site, hx[idx], hy[idx], hz[idx], D[idx], E[idx])
                if not np.any(mat != 0.0):
                    continue
                coupling = Coupling.from_dense_block(mat, [site], understood_braiding=True)
                self.add_coupling(coupling, [i])
        # Jx Sx.Sx + Jy Sy.Sy + Jz Sz.Sz + muJ (Sy_i Sx_j - Sx_i Sy_j)
        # NB: don't call `possible_couplings` separately per coefficient: it filters out
        # positions where *that* coefficient is zero, which would misalign the four arrays
        # whenever e.g. muJ=0 but Jx/Jy/Jz aren't. Get the (unfiltered) positions once via
        # strength=None, then look up each coefficient at those lattice positions.
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            site1, site2 = self.lat.unit_cell[u1], self.lat.unit_cell[u2]
            mps_i, mps_j, lat_indices, coupling_shape = self.lat.possible_couplings(u1, u2, dx, None)
            Jx_arr = to_array(Jx, coupling_shape)
            Jy_arr = to_array(Jy, coupling_shape)
            Jz_arr = to_array(Jz, coupling_shape)
            muJ_arr = to_array(muJ, coupling_shape)
            for i, j, lat_idx in zip(mps_i, mps_j, lat_indices):
                idx = tuple(lat_idx)
                h = _coupling_matrix(site1, site2, Jx_arr[idx], Jy_arr[idx], Jz_arr[idx], muJ_arr[idx])
                if not np.any(h != 0.0):
                    continue
                h = np.transpose(h, [0, 2, 3, 1])
                coupling = Coupling.from_dense_block(h, [site1, site2], understood_braiding=True)
                self.add_coupling(coupling, [int(i), int(j)])
        # done


class SpinChain(SpinModel, NearestNeighborModel):
    """The :class:`SpinModel` on a Chain, suitable for TEBD.

    See the :class:`SpinModel` for the documentation of parameters.
    """

    default_lattice = Chain
    force_default_lattice = True


class DipolarSpinChain(CouplingMPOModel):
    r"""Dipole conserving H3-H4 spin-S chain.

    The Hamiltonian reads:

    .. math ::
        H = - \mathtt{J3} \sum_{i} (S^+_i (S^-_{i + 1})^2 S^+_{i + 2} + \mathrm{h.c.})
            - \mathtt{J4} \sum_{i} (S^+_i S^-_{i + 1} S^-_{i + 2} S^+_{i + 2} + \mathrm{h.c.})

    .. note ::
        Dipole ("dipole"-)conservation is not (yet) supported by the cyten-based symmetry
        framework, so ``conserve='dipole'`` is currently unavailable. This class otherwise
        works with the symmetries that :class:`~cyten.models.sites.SpinSite` does support.

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
            What should be conserved. See :class:`~cyten.models.sites.SpinSite`.
            For ``'best'``, we preserve ``'Sz'``.
        J3, J4 : float | array
            Coupling as defined for the Hamiltonian above.

    """

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
        site = self.lat.unit_cell[0]
        Sp = site.spin_vector[:, :, 0] + 1.0j * site.spin_vector[:, :, 1]
        Sm = site.spin_vector[:, :, 0] - 1.0j * site.spin_vector[:, :, 1]
        Sm2 = Sm @ Sm

        # -J3 (Sp_0 Sm_1^2 Sp_2 + h.c.)
        h3 = np.tensordot(np.tensordot(Sp, Sm2, axes=0), Sp, axes=0)  # [p0,p0*,p1,p1*,p2,p2*]
        h3 = np.transpose(h3, [0, 2, 4, 5, 3, 1])  # -> [p0,p1,p2,p2*,p1*,p0*]
        coupling3 = Coupling.from_dense_block(h3, [site, site, site], understood_braiding=True)
        hc_coupling3 = Coupling.from_tensor(dagger(coupling3.to_tensor()), [site, site, site])
        for i in range(self.lat.N_sites - 2):
            self.add_coupling(coupling3, [i, i + 1, i + 2], strength=-J3)
            self.add_coupling(hc_coupling3, [i, i + 1, i + 2], strength=-np.conj(J3))

        # -J4 (Sp_0 Sm_1 Sm_2 Sp_3 + h.c.)
        h4 = np.tensordot(np.tensordot(np.tensordot(Sp, Sm, axes=0), Sm, axes=0), Sp, axes=0)
        h4 = np.transpose(h4, [0, 2, 4, 6, 7, 5, 3, 1])  # -> [p0,p1,p2,p3,p3*,p2*,p1*,p0*]
        coupling4 = Coupling.from_dense_block(h4, [site, site, site, site], understood_braiding=True)
        hc_coupling4 = Coupling.from_tensor(dagger(coupling4.to_tensor()), [site, site, site, site])
        for i in range(self.lat.N_sites - 3):
            self.add_coupling(coupling4, [i, i + 1, i + 2, i + 3], strength=-J4)
            self.add_coupling(hc_coupling4, [i, i + 1, i + 2, i + 3], strength=-np.conj(J4))
