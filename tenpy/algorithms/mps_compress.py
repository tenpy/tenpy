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
from ..networks.mps import MPSEnvironment
from ..networks.mpo import MPOEnvironment
from .mps_sweeps import Sweep, TwoSiteH
from ..tools.params import asConfig

__all__ = ['VariationalCompression', 'VariationalApplyMPO']


class VariationalCompression(Sweep):
    """Variational compression of an MPS.

    """
    EffectiveH = TwoSiteH

    def __init__(self, psi, options):
        self.options = asConfig(options, "VarMPSCompression")
        self.renormalize = []
        Sweep.__init__(self, psi, None, self.options)

    def run(self):
        options = self.options
        N_sweeps = options.get("N_sweeps", 5)
        for i in range(N_sweeps):
            self.renormalize = []
            max_trunc_err = self.sweep()
        # TODO: more fancy stopping criteria?
        if self.psi.finite:
            self.psi.norm *= max(self.renormalize)
        return max_trunc_err

    def init_env(self, _):
        init_env_data = self.options.get("init_env_data", {})
        old_psi = self.psi.copy()
        self.env = MPSEnvironment(self.psi, old_psi, **init_env_data)
        self.reset_stats()

    def update_local(self, _, optimize=True):
        i0 = self.i0
        th = self.env.ket.get_theta(i0, n=2)  # ket is old psi
        LP = self.env.get_LP(i0)
        RP = self.env.get_RP(i0 + 1)
        th = npc.tensordot(LP, th, ['vR', 'vL'])
        th = npc.tensordot(th, RP, ['vR', 'vL'])
        th.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
        th = th.combine_legs([['vL', 'p0'], ['p1', 'vR']], qconj=[+1, -1])
        return self.update_new_psi(th)

    def update_new_psi(self, theta):
        i0 = self.i0
        new_psi = self.psi
        qtotal_i0 = new_psi.get_B(i0, form=None).qtotal
        U, S, VH, err, renormalize = svd_theta(theta,
                                               self.trunc_params,
                                               qtotal_LR=[qtotal_i0, None],
                                               inner_labels=['vR', 'vL'])
        self.renormalize.append(renormalize)
        B0 = U.split_legs(['(vL.p0)']).replace_label('p0', 'p')
        B1 = VH.split_legs(['(p1.vR)']).replace_label('p1', 'p')
        new_psi.set_B(i0, B0, form='A')  # left-canonical
        new_psi.set_B(i0 + 1, B1, form='B')  # right-canonical
        new_psi.set_SR(i0, S)
        # the old stored environments are now invalid
        # => delete them to ensure that they get calculated again in :meth:`update_LP` / RP
        for o_env in self.ortho_to_envs:
            o_env.del_LP(i0 + 1)
            o_env.del_RP(i0)
        self.env.del_LP(i0 + 1)
        self.env.del_RP(i0)
        return {'U': U, 'VH': VH, 'err': err}

    def update_LP(self, _):
        self.env.get_LP(self.i0 + 1, store=True)

    def update_RP(self, _):
        self.env.get_RP(self.i0, store=True)


class VariationalApplyMPO(VariationalCompression):
    """Variational compression for applying an MPO to an MPS."""
    def __init__(self, psi, U_MPO, options):
        self.options = asConfig(options, "MpoMpsCompression")
        self.renormalize = [None] * (psi.L - int(psi.finite))
        Sweep.__init__(self, psi, U_MPO, self.options)

    def init_env(self, U_MPO):
        init_env_data = self.options.get("init_env_data", {})
        old_psi = self.psi.copy()
        self.env = MPOEnvironment(self.psi, U_MPO, old_psi, **init_env_data)
        self.reset_stats()

    def update_local(self, _, optimize=True):
        i0 = self.i0
        self.make_eff_H()
        th = self.env.ket.get_theta(i0, n=2)  # ket is old psi
        th = self.eff_H.combine_theta(th)
        th = self.eff_H.matvec(th)
        if not self.eff_H.combine:
            th = th.combine_legs([['vL', 'p0'], ['p1', 'vR']], qconj=[+1, -1])
        return self.update_new_psi(th)
