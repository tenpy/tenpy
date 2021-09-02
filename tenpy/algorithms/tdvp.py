"""Time Dependant Variational Principle (TDVP) with MPS (finite version only).

The TDVP MPS algorithm was first proposed by :cite:`haegeman2011`. However the stability of the
algorithm was later improved in :cite:`haegeman2016`, that we are following in this implementation.
The general idea of the algorithm is to project the quantum time evolution in the manyfold of MPS
with a given bond dimension. Compared to e.g. TEBD, the algorithm has several advantages:
e.g. it conserves the unitarity of the time evolution and the energy (for the single-site version),
and it is suitable for time evolution of Hamiltonian with arbitrary long range in the form of MPOs.
We have implemented the one-site formulation which **does not** allow for growth of the bond dimension,
and the two-site algorithm which does allow the bond dimension to grow - but requires truncation as in the TEBD case.
Much of the code is very similar as in DMRG, also based on the class :class:`~tenpy.algorithms.mps_common.Sweep`.

.. warning ::
    The interface changed compared to the previous version. Using :class:`TDVPEngine`
    will result in a error. Use :class:`SingleSiteTDVPEngine` or :class:`TwoSiteTDVPEngine` instead.

.. todo ::
    SingleSiteTDVPEngine is not working correctly

.. todo ::
    extend code to infinite MPS
    
.. todo ::
    allow for increasing bond dimension in SingleSiteTDVPEngine, similar to DMRG Mixer     
"""
# Copyright 2019-2021 TeNPy Developers, GNU GPLv3

from tenpy.linalg.lanczos import LanczosEvolution
from tenpy.algorithms.truncation import svd_theta, TruncationError
from tenpy.algorithms.mps_common import Sweep, ZeroSiteH, OneSiteH, TwoSiteH
from tenpy.algorithms.algorithm import TimeEvolutionAlgorithm
from tenpy.linalg import np_conserved as npc
import numpy as np
import time
import warnings
import logging
logger = logging.getLogger(__name__)


__all__ = ['TDVPEngine', 'SingleSiteTDVPEngine', 'TwoSiteTDVPEngine']


class TDVPEngine(TimeEvolutionAlgorithm, Sweep):
    """Time dependent variational principle algorithm for MPS.
    This class contains all methods that are generic between
    :class:`SingleSiteTDVPEngine` and :class:`TwoSiteTDVPEngine`.
    Use the latter two classes for actual TDVP runs.

    .. deprecated :: 0.6.0
        Renamed parameter/attribute `TDVP_params` to :attr:`options`.
    Parameters
    ----------
    psi, model, options, **kwargs:
        Same as for :class:`~tenpy.algorithms.algorithm.Algorithm`.
    Options
    -------
    .. cfg:config :: TDVP
        :include: TimeEvolutionAlgorithm
        trunc_params : dict
            Truncation parameters as described in :func:`~tenpy.algorithms.truncation.truncate`
        lanczos_options : dict
            Lanczos options as described in :cfg:config:`Lanczos`.
    Attributes
    ----------
    options: dict
        Optional parameters.
    evolved_time : float | complex
        Indicating how long `psi` has been evolved, ``psi = exp(-i * evolved_time * H) psi(t=0)``.
    psi : :class:`~tenpy.networks.mps.MPS`
        The MPS, time evolved in-place.
    env : :class:`~tenpy.networks.mpo.MPOEnvironment`
        The environment, storing the `LP` and `RP` to avoid recalculations.
    lanczos_options : :class:`~tenpy.tools.params.Config`
        Options passed on to :class:`~tenpy.linalg.lanczos.LanczosEvolution`.
    """
    EffectiveH = None

    def __init__(self, psi, model, options, **kwargs):
        # not sure if we really need this
        if psi.bc != 'finite':
            raise NotImplementedError("Only finite TDVP is implemented")
        assert psi.bc == model.lat.bc_MPS
        self.trunc_err = TruncationError()
        super().__init__(psi, model, options, **kwargs)

    def run(self):
        """Run TEBD real time evolution by `N_steps`*`dt`."""
        # initialize parameters
        self.dt = self.options.get('dt', 0.1)
        N_steps = self.options.get('N_steps', 2)

        Sold = np.mean(self.psi.entanglement_entropy())
        start_time = time.time()

        self.update(N_steps)

        S = self.psi.entanglement_entropy()
        logger.info(
            "--> time=%(t)3.3f, max(chi)=%(chi)d, max(S)=%(S).5f, "
            "avg DeltaS=%(dS).4e, since last update: %(wall_time).1fs", {
                't': self.evolved_time.real,
                'chi': max(self.psi.chi),
                'S': max(S),
                'dS': np.mean(S) - Sold,
                'wall_time': time.time() - start_time,
            })

    def update(self, N_steps):
        """Evolve by ``N_steps * dt``.
        Parameters
        ----------
        N_steps : int
            The number of steps for which the whole lattice should be updated.
        Returns
        -------
        trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The error of the represented state which is introduced due to the truncation during
            this sequence of update steps.
        """
        self.step_trunc_error = TruncationError()
        for _ in range(N_steps):
            self.sweep()
        self.evolved_time = self.evolved_time + N_steps * self.dt
        self.trunc_err = self.trunc_err + self.step_trunc_error  # not += : make a copy!
        # (this is done to avoid problems of users storing self.trunc_err after each `update`)
        return self.step_trunc_error

    def post_update_local(self, err, **update_data):
        self.step_trunc_error += err
        self.trunc_err_list.append(err.eps)


class TwoSiteTDVPEngine(TDVPEngine):
    """Engine for the two-site TDVP algorithm.
    Parameters
    ----------
    psi, model, options, **kwargs:
        Same as for :class:`~tenpy.algorithms.algorithm.Algorithm`.
    Options
    -------
    .. cfg:config :: TDVP
        :include: TimeEvolutionAlgorithm
        trunc_params : dict
            Truncation parameters as described in :func:`~tenpy.algorithms.truncation.truncate`
        lanczos_options : dict
            Lanczos options as described in :cfg:config:`Lanczos`.
    Attributes
    ----------
    options: dict
        Optional parameters.
    evolved_time : float | complex
        Indicating how long `psi` has been evolved, ``psi = exp(-i * evolved_time * H) psi(t=0)``.
    psi : :class:`~tenpy.networks.mps.MPS`
        The MPS, time evolved in-place.
    env : :class:`~tenpy.networks.mpo.MPOEnvironment`
        The environment, storing the `LP` and `RP` to avoid recalculations.
    lanczos_options : :class:`~tenpy.tools.params.Config`
        Options passed on to :class:`~tenpy.linalg.lanczos.LanczosEvolution`.
    """
    EffectiveH = TwoSiteH

    def get_sweep_schedule(self):
        """slightly different sweep schedule than DMRG"""
        L = self.psi.L
        if self.finite:
            i0s = list(range(0, L - 2)) + list(range(L - 2, -1, -1))
            move_right = [True] * (L - 2) + [False] * (L - 1)
            update_LP_RP = [[True, False]] * \
                (L - 2) + [[False, True]] * (L - 2) + [[False, False]]
        else:
            raise NotImplementedError("Only finite TDVP is implemented")
        return zip(i0s, move_right, update_LP_RP)

    def update_local(self, theta, **kwargs):
        dt = self.dt
        self.lanczos_options = self.options.subconfig('lanczos_options')

        i0 = self.i0
        if i0 == self.psi.L-2:  # instead of updating this twice, we can double the time
            dt2 = 2*dt
        else:
            dt2 = dt
        # update two-site wavefunction
        theta, N = LanczosEvolution(
            self.eff_H, theta, self.lanczos_options).run(-0.5j*dt2)
        theta = theta.combine_legs([['vL', 'p0'], ['p1', 'vR']],
                                   new_axes=[0, 1], qconj=[+1, -1])
        qtotal_i0 = self.env.bra.get_B(i0, form=None).qtotal
        U, S, VH, err, _ = svd_theta(theta,
                                     self.trunc_params,
                                     qtotal_LR=[qtotal_i0, None],
                                     inner_labels=['vR', 'vL'])
        B0 = U.split_legs(['(vL.p0)']).replace_label('p0', 'p')
        B1 = VH.split_legs(['(p1.vR)']).replace_label('p1', 'p')

        self.psi.set_B(i0, B0, form='A')  # left-canonical
        self.psi.set_B(i0 + 1, B1, form='B')  # right-canonical
        self.psi.set_SR(i0, S)
        if self.move_right and i0 != self.psi.L-2:  # right moving update
            self.eff_H.update_LP(self.env, i0+1, U)
            self.one_site_update(i0+1, theta)
        elif (not self.move_right) and i0 != 0:  # left moving update
            self.eff_H.update_RP(self.env, i0, VH)
            self.one_site_update(i0, theta)

        update_data = {
            'err': err,
            'N': N,
            'U': U,
            'VH': VH
        }
        return update_data

    def one_site_update(self, i, theta):
        H1 = OneSiteH(self.env, i)
        theta = self.psi.get_theta(i, n=1, cutoff=self.S_inv_cutoff)
        theta = H1.combine_theta(theta)
        theta, _ = LanczosEvolution(
            H1, theta, self.lanczos_options).run(0.5j*self.dt)
        self.psi.set_B(i, theta.replace_label('p0', 'p'), form='Th')


class SingleSiteTDVPEngine(TDVPEngine):
    """Engine for the single-site TDVP algorithm.
    Parameters
    ----------
    psi, model, options, **kwargs:
        Same as for :class:`~tenpy.algorithms.algorithm.Algorithm`.
    Options
    -------
    .. cfg:config :: TDVP
        :include: TimeEvolutionAlgorithm
        trunc_params : dict
            Truncation parameters as described in :func:`~tenpy.algorithms.truncation.truncate`
        lanczos_options : dict
            Lanczos options as described in :cfg:config:`Lanczos`.
    Attributes
    ----------
    options: dict
        Optional parameters.
    evolved_time : float | complex
        Indicating how long `psi` has been evolved, ``psi = exp(-i * evolved_time * H) psi(t=0)``.
    psi : :class:`~tenpy.networks.mps.MPS`
        The MPS, time evolved in-place.
    env : :class:`~tenpy.networks.mpo.MPOEnvironment`
        The environment, storing the `LP` and `RP` to avoid recalculations.
    lanczos_options : :class:`~tenpy.tools.params.Config`
        Options passed on to :class:`~tenpy.linalg.lanczos.LanczosEvolution`.
    """
    EffectiveH = OneSiteH

    def get_sweep_schedule(self):
        """slightly different sweep schedule than DMRG"""
        L = self.psi.L
        if self.finite:
            i0s = list(range(0, L - 1)) + list(range(L - 1, -1, -1))
            move_right = [True] * (L - 1) + [False] * (L)
            update_LP_RP = [[True, False]] * (L - 1) + \
                [[False, True]] * (L - 1) + [[False, False]]
        else:
            raise NotImplementedError("Only finite TDVP is implemented")
        return zip(i0s, move_right, update_LP_RP)

    def update_env(self, **update_data):
        if self.i0 == 0 and not self.move_right:
            # need to overwrite this special case because of the sweep_schedule
            # maybe need to do something different here?
            pass
        else:
            super().update_env(**update_data)

    def update_local(self, theta, **kwargs):
        dt = self.dt
        self.lanczos_options = self.options.subconfig('lanczos_options')

        i0 = self.i0
        if i0 == self.psi.L-1:  # instead of updating this twice, we can double the time
            dt2 = 2*dt
        else:
            dt2 = dt
        # update one-site wavefunction
        theta, N = LanczosEvolution(
            self.eff_H, theta, self.lanczos_options).run(-0.5j*dt2)
        if self.move_right:
            U, VH, err = self.right_moving_update(i0, theta)
        else:  # left moving
            U, VH, err = self.left_moving_update(i0, theta)
        update_data = {
            'err': err,
            'N': N,
            'U': U,
            'VH': VH
        }
        return update_data

    def right_moving_update(self, i0, theta):
        theta = theta.combine_legs(['vL', 'p0'], qconj=+1, new_axes=0)
        qtotal = [theta.qtotal, None]
        U, S, VH, err, _ = svd_theta(theta,
                                     self.trunc_params,
                                     qtotal_LR=qtotal,
                                     inner_labels=['vR', 'vL'])

        B0 = U.split_legs(['(vL.p0)']).replace_label('p0', 'p')
        self.psi.set_B(i0, B0, form='A')  # left-canonical
        self.psi.set_SR(i0, S)
        if i0 != self.psi.L-1:
            next_B = self.env.bra.get_B(i0 + 1, form='B')
            VH = npc.tensordot(VH, next_B, axes=['vR', 'vL'])
            self.psi.set_B(i0 + 1, VH, form='B')  # right-canonical
            self.eff_H.update_LP(self.env, i0+1, U)
            self.eff_H.update_RP(self.env, i0, VH)
            theta = self.zero_site_update(
                i0+1, S, (U.get_leg('vR'), VH.get_leg('vL')))
            theta_new = npc.tensordot(theta, VH, axes=['vR', 'vL'])
            self.psi.set_B(i0 + 1, theta_new, form='Th')
        return U, VH, err

    def left_moving_update(self, i0, theta):
        theta = theta.combine_legs(['p0', 'vR'], qconj=-1, new_axes=1)
        qtotal = [None, theta.qtotal]
        U, S, VH, err, _ = svd_theta(theta,
                                     self.trunc_params,
                                     qtotal_LR=qtotal,
                                     inner_labels=['vR', 'vL'])

        B1 = VH.split_legs(['(p0.vR)']).replace_label('p0', 'p')
        self.psi.set_B(i0, B1, form='B')  # right-canonical
        self.psi.set_SL(i0, S)
        if i0 != 0:
            next_B = self.env.bra.get_B(self.i0 - 1, form='A')
            U = npc.tensordot(next_B, U, axes=['vR', 'vL'])
            self.psi.set_B(i0 - 1, U, form='A')  # left-canonical
            self.eff_H.update_LP(self.env, i0, U)
            self.eff_H.update_RP(self.env, i0-1, VH)
            theta = self.zero_site_update(
                i0, S, (U.get_leg('vR'), VH.get_leg('vL')))
            theta_new = npc.tensordot(U, theta, axes=['vR', 'vL'])
            self.psi.set_B(i0-1, theta_new, form='Th')
        return U, VH, err

    def zero_site_update(self, i, S, legs):
        vR_U, vL_V = legs
        H0 = ZeroSiteH(self.env, i)
        s_ndarray = np.diag(S)
        s = npc.Array.from_ndarray(s_ndarray, [vR_U.conj(), vL_V.conj()],
                                   dtype=None,
                                   qtotal=None,
                                   cutoff=None)
        s.iset_leg_labels(['vL', 'vR'])
        theta, _ = LanczosEvolution(
            H0, s, self.lanczos_options).run(0.5j*self.dt)
        return theta
