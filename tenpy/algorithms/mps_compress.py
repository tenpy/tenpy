r"""Compression of a MPS.

.. todo ::
    This is still a beta version, use with care!
    The interface will probably still change!
"""
# Copyright 2019-2020 TeNPy Developers, GNU GPLv3

import numpy as np
from scipy.linalg import expm

from ..linalg import np_conserved as npc
from .truncation import svd_theta
from ..networks import mps, mpo
from .mps_sweeps import Sweep

__all__ = ['MPSCompression', 'mps_compress', 'svd_two_site', 'apply_mpo']


class MPSCompression(Sweep):
    def __init__(self, psi, options):
        self.options = asConfig("MPSCompression", options)
        self.psi = psi
        super().__init__(psi, None, self.options)

    def init_env(self, _):
        init_env_data = self.options.get("init_env_data", {})
        old_psi = self.psi.copy()
        self.env = MPSEnvironment(self.psi, old_psi, **init_env_data)
        self.reset_stats()

    def update_local(self):
        raise NotImplementedError("TODO")

    def run(self):
        raise NotImplementedError("TODO")


def mps_compress(psi, trunc_par):
    r"""Takes an MPS and compresses it; in place.

    Parameters
    ----------
    psi : :class:`tenpy.networks.mps.MPS`
        MPS to be compressed.
    trunc_par : dict
        See :func:`~tenpy.algorithms.truncation.truncate`
    """
    bc = psi.bc
    L = psi.L
    if bc == 'finite':
        # TODO: could we simply replace this with MPS.canonical_form_finite()?
        # Do QR starting from the left
        B = psi.get_B(0, form='Th')
        for i in range(psi.L - 1):
            B = B.combine_legs(['vL', 'p'])
            q, r = npc.qr(B, inner_labels=['vR', 'vL'])
            B = q.split_legs()
            psi.set_B(i, B, form=None)
            B = psi.get_B((i + 1) % L, form='B')
            B = npc.tensordot(r, B, axes=('vR', 'vL'))
        # Do SVD from right to left, truncate the singular values according to trunc_par
        for i in range(psi.L - 1, 0, -1):
            B = B.combine_legs(['p', 'vR'])
            u, s, vh, err, norm_new = svd_theta(B, trunc_par)
            psi.norm *= norm_new
            vh = vh.split_legs()
            psi.set_B(i % L, vh, form='B')
            B = psi.get_B(i - 1, form=None)
            B = npc.tensordot(B, u, axes=('vR', 'vL'))
            B.iscale_axis(s, 'vR')
            psi.set_SL(i % L, s)
        psi.set_B(0, B, form='Th')
    if bc == 'infinite':
        for i in range(psi.L):
            svd_two_site(i, psi)
        for i in range(psi.L - 1, -1, -1):
            svd_two_site(i, psi, trunc_par)


def svd_two_site(i, mps, trunc_par=None):
    r"""Builds a theta and splits it using svd for an MPS.

    Parameters
    ----------
    i : int
        First site.
    mps : :class:`tenpy.networks.mps.MPS`
        MPS to use on.
    trunc_par : None|dict
       If None no truncation is done. Else dict as in :func:`~tenpy.algorithms.truncation.truncate`.
    """
    # TODO: this is already implemented somewhere else....
    theta = mps.get_theta(i, n=2)
    theta = theta.combine_legs([['vL', 'p0'], ['p1', 'vR']], qconj=[+1, -1])
    if trunc_par is None:
        trunc_par = {'chi_max': 10000, 'svd_min': 1.e-15, 'trunc_cut': 1.e-15}
    u, s, vh, err, renorm = svd_theta(theta, trunc_par)
    mps.norm *= renorm
    u = u.split_legs()
    vh = vh.split_legs()
    u.ireplace_label('p0', 'p')
    vh.ireplace_label('p1', 'p')
    mps.set_B(i, u, form='A')
    mps.set_B((i + 1) % mps.L, vh, form='B')
    mps.set_SR(i, s)


def apply_mpo(U_mpo, psi, trunc_par):
    """Applies an mpo and truncates the resulting MPS using SVD.

    Parameters
    ----------
    U_mpo : :class:`~tenpy.networks.mpo.MPO`
        MPO to apply. Usually one of :func:`make_U_I` or :func:`make_U_II()`.
        The approximation being made are uncontrolled for other mpos and infinite bc.
    psi : :class:`~tenpy.networks.mps.MPS`
        MPS to apply operator on
    trunc_par : dict
        Truncation parameters. See :func:`~tenpy.algorithms.truncation.truncate`

    Returns
    -------
    new_psi : :class:`~tenpy.networks.mps.MPS`
        Resulting new MPS representing `U_mpo |psi>`
    """
    bc = psi.bc
    if bc != U_mpo.bc:
        raise ValueError("Boundary conditions of MPS and MPO are not the same")
    if psi.L != U_mpo.L:
        raise ValueError("Length of MPS and MPO not the same")
    Bs = [
        npc.tensordot(psi.get_B(i, form='B'), U_mpo.get_W(i), axes=('p', 'p*'))
        for i in range(psi.L)
    ]
    if bc == 'finite':
        Bs[0] = npc.tensordot(psi.get_theta(0, 1), U_mpo.get_W(0), axes=('p0', 'p*'))
    for i in range(psi.L):
        if i == 0 and bc == 'finite':
            Bs[i] = Bs[i].take_slice(U_mpo.get_IdL(i), 'wL')
            Bs[i] = Bs[i].combine_legs(['wR', 'vR'], qconj=[-1])
            Bs[i].ireplace_labels(['(wR.vR)'], ['vR'])
            Bs[i].legs[Bs[i].get_leg_index('vR')] = Bs[i].get_leg('vR').to_LegCharge()
        elif i == psi.L - 1 and bc == 'finite':
            Bs[i] = Bs[i].take_slice(U_mpo.get_IdR(i), 'wR')
            Bs[i] = Bs[i].combine_legs(['wL', 'vL'], qconj=[1])
            Bs[i].ireplace_labels(['(wL.vL)'], ['vL'])
            Bs[i].legs[Bs[i].get_leg_index('vL')] = Bs[i].get_leg('vL').to_LegCharge()
        else:
            Bs[i] = Bs[i].combine_legs([['wL', 'vL'], ['wR', 'vR']], qconj=[+1, -1])
            Bs[i].ireplace_labels(['(wL.vL)', '(wR.vR)'], ['vL', 'vR'])
            Bs[i].legs[Bs[i].get_leg_index('vL')] = Bs[i].get_leg('vL').to_LegCharge()
            Bs[i].legs[Bs[i].get_leg_index('vR')] = Bs[i].get_leg('vR').to_LegCharge()

    if bc == 'infinite':
        #calculate good (rather arbitrary) guess for S[0] (no we don't like it either)
        weight = np.ones(U_mpo.get_W(0).shape[U_mpo.get_W(0).get_leg_index('wL')]) * 0.05
        weight[U_mpo.get_IdL(0)] = 1
        weight = weight / np.linalg.norm(weight)
        S = [np.kron(weight, psi.get_SL(0))]  # order dictated by '(wL,vL)'
    else:
        S = [np.ones(Bs[0].get_leg('vL').ind_len)]
    #Wrong S values but will be calculated in mps_compress
    for i in range(psi.L):
        S.append(np.ones(Bs[i].get_leg('vR').ind_len))

    forms = ['B' for i in range(psi.L)]
    if bc == 'finite':
        forms[0] = 'Th'
    new_mps = mps.MPS(psi.sites, Bs, S, form=forms, bc=psi.bc)
    mps_compress(new_mps, trunc_par)
    return new_mps
