r"""Compression of a MPS.

.. todo ::
    This is still a beta version, use with care!
"""
# Copyright 2019-2020 TeNPy Developers, GNU GPLv3

import numpy as np
from scipy.linalg import expm

from ..linalg import np_conserved as npc
from .truncation import svd_theta, TruncationError
from ..networks.mps import MPSEnvironment
from ..networks.mpo import MPOEnvironment
from .mps_sweeps import Sweep, TwoSiteH
from ..tools.params import asConfig

__all__ = ['VariationalCompression', 'VariationalApplyMPO']


class VariationalCompression(Sweep):
    """Variational compression of an MPS (in place).

    To compress an MPS `psi`, use ``VariationalCompression(psi, options).run()``.

    The algorithm is the same as described in :class:`VariationalApplyMPO`,
    except that we dont have an MPO in the networks - one can think of the MPO being trivial.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        The state to be compressed.
    options : dict
        See :cfg:config:`VariationalCompression`.

    Options
    -------
    .. cfg:config :: VariationalCompression

        trunc_params : dict
            Truncation parameters as described in :cfg:config:`truncation`.
        N_sweeps : int
            Number of sweeps during each call to :class:`run`.

    Attributes
    ----------
    renormalize : list
        Used to keep track of renormalization in the last sweep for `psi.norm`.
    """
    EffectiveH = TwoSiteH

    def __init__(self, psi, options):
        self.options = asConfig(options, "VariationalCompression")
        self.renormalize = []
        Sweep.__init__(self, psi, None, self.options)

    def run(self):
        """Run the compression.

        The state :attr:`psi` is compressed in place.

        Returns
        -------
        max_trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The maximal truncation error of a two-site wave function.
        """
        N_sweeps = self.options.get("N_sweeps", 3)

        for i in range(N_sweeps):  # TODO: more fancy stopping criteria?
            self.renormalize = []
            max_trunc_err = self.sweep()

        if self.psi.finite:
            self.psi.norm *= max(self.renormalize)
        return TruncationError(max_trunc_err, 1. - 2. * max_trunc_err)

    def init_env(self, _):
        """Initialize the environment.

        The argument is not used and only there for compatibility with the Sweep class.
        """
        init_env_data = self.options.get("init_env_data", {})
        old_psi = self.psi.copy()
        self.env = MPSEnvironment(self.psi, old_psi, **init_env_data)
        self.reset_stats()

    def update_local(self, _, optimize=True):
        """Perform local update.

        This simply contracts the environments and `theta` from the `ket` to get an updated
        `theta` for the bra `self.psi` (to be changed in place).
        """
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
        # TODO: up to the `renormalize`, we could use `new_psi.set_svd_theta`.
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
    """Variational compression for applying an MPO to an MPS (in place).

    To apply an MPO `U_MPO` to an MPS `psi`, use
    ``VariationalApplyMPO(psi, U_MPO, options).run()``.

    The goal is to find a new MPS `phi` (with `N` tensors) which is optimally close
    to ``U_MPO|psi>``, i.e. it is normalized and maximizes ``| <phi|U_MPO|psi> |^2``.
    The network for this (with `M` tensors for `psi`) is given by


        |     .-------M[0]----M[1]----M[2]---- ...  ----.
        |     |       |       |       |                 |
        |     LP[0]---W[0]----W[1]----W[2]---- ...  --- RP[-1]
        |     |       |       |       |                 |
        |     .-------N[0]*---N[1]*---N[2]*--- ...  ----.

    Here `LP` and `RP` are the environments with partial contractions,
    see also :class:`~tenpy.networks.mpo.MPOEnvironment`.
    This algorithms sweeps through the sites, updating 2 `N` tensors in each :meth:`update_local`,
    say on sites `i0` and `i1`=`i0`+1. We need to maximize:

        |     .-------M[i0]---M[i1]---.
        |     |       |       |       |
        |     LP[i0]--W[i0]---W[i1]---RP[i1]
        |     |       |       |       |
        |     .-------N[i0]*--N[i1]*--.

    The optimal solution is given by::

        |                                     .-------M[i0]---M[i1]---.
        |   ---M[i0]---M[i1]---               |       |       |       |
        |      |       |          = SVD of    LP[i0]--W[i0]---W[i1]---RP[i1]
        |                                     |       |       |       |
        |                                     .-----                --.


    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        The state to which
    options : dict
        See :cfg:config:`VariationalCompression`.

    Options
    -------
    .. cfg:config :: VariationalApplyMPO
        :include: VariationalCompression

    Attributes
    ----------
    renormalize : list
        Used to keep track of renormalization in the last sweep for `psi.norm`.
    """
    def __init__(self, psi, U_MPO, options):
        self.options = asConfig(options, "VariationalApplyMPO")
        self.renormalize = [None] * (psi.L - int(psi.finite))
        Sweep.__init__(self, psi, U_MPO, self.options)

    def init_env(self, U_MPO):
        """Initialize the environment.

        Parameters
        ----------
        U_MPO : :class:`~tenpy.networks.mpo.MPO`
            The MPO to be applied to the sate.
        """
        init_env_data = self.options.get("init_env_data", {})
        old_psi = self.psi.copy()
        self.env = MPOEnvironment(self.psi, U_MPO, old_psi, **init_env_data)
        self.reset_stats()

    def update_local(self, _, optimize=True):
        """Perform local update.

        This simply contracts the environments and `theta` from the `ket` to get an updated
        `theta` for the bra `self.psi` (to be changed in place).
        """
        i0 = self.i0
        self.make_eff_H()
        th = self.env.ket.get_theta(i0, n=2)  # ket is old psi
        th = self.eff_H.combine_theta(th)
        th = self.eff_H.matvec(th)
        if not self.eff_H.combine:
            th = th.combine_legs([['vL', 'p0'], ['p1', 'vR']], qconj=[+1, -1])
        return self.update_new_psi(th)
