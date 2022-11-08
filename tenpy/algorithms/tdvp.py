"""Time Dependant Variational Principle (TDVP) with MPS (finite version only).

The TDVP MPS algorithm was first proposed by :cite:`haegeman2011`. However the stability of the
algorithm was later improved in :cite:`haegeman2016`, that we are following in this implementation.
The general idea of the algorithm is to project the quantum time evolution in the manyfold of MPS
with a given bond dimension. Compared to e.g. TEBD, the algorithm has several advantages:
e.g. it conserves the unitarity of the time evolution and the energy (for the single-site version),
and it is suitable for time evolution of Hamiltonian with arbitrary long range in the form of MPOs.
We have implemented:

1. The one-site formulation following the TDVP princible in :class:`SingleSiteTDVPEngine`,
   which **does not** allow for growth of the bond dimension.

2. The two-site algorithm in the :class:`TwoSiteTDVPEngine`, which does allow the bond
   dimension to grow - but requires truncation as in the TEBD case, and is no longer strictly TDVP,
   i.e. it does *not* strictly preserve the energy.

Much of the code is very similar to DMRG, and also based on the
:class:`~tenpy.algorithms.mps_common.Sweep` class.

.. warning ::
    The interface changed compared to version 0.9.0: Using :class:`TDVPEngine` will result
    in a error. Use :class:`SingleSiteTDVPEngine` or :class:`TwoSiteTDVPEngine` instead.
    The old code is still around as :class:`OldTDVPEngine`.

.. todo ::
    extend code to infinite MPS

.. todo ::
    allow for increasing bond dimension in SingleSiteTDVPEngine, similar to DMRG Mixer
"""
# Copyright 2019-2021 TeNPy Developers, GNU GPLv3

from tenpy.linalg.lanczos import LanczosEvolution
from tenpy.algorithms.truncation import svd_theta, TruncationError
from tenpy.algorithms.mps_common import Sweep, ZeroSiteH, OneSiteH, TwoSiteH
from tenpy.algorithms.algorithm import TimeEvolutionAlgorithm, TimeDependentHAlgorithm
from tenpy.networks.mpo import MPOEnvironment
from tenpy.linalg import np_conserved as npc
import numpy as np
import time
import warnings
import logging

logger = logging.getLogger(__name__)

__all__ = ['TDVPEngine', 'SingleSiteTDVPEngine', 'TwoSiteTDVPEngine',
           'TimeDependentSingleSiteTDVP', 'TimeDependentTwoSiteTDVP', 'OldTDVPEngine', 'Engine',
           'H0_mixed', 'H1_mixed', 'H2_mixed']


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
        if self.__class__.__name__ == 'TDVPEngine':
            msg = ("TDVP interface changed. \n"
                   "The new TDVPEngine has subclasses SingleSiteTDVPEngine"
                   " and TwoSiteTDVPEngine that you can use.\n"
                   "For now, the previous version is still available as OldTDVPEngine."
                   )
            raise NameError(msg)
        if psi.bc != 'finite':
            raise NotImplementedError("Only finite TDVP is implemented")
        assert psi.bc == model.lat.bc_MPS
        super().__init__(psi, model, options, **kwargs)
        self.lanczos_options = self.options.subconfig('lanczos_options')

    # run() from TimeEvolutionAlgorithm

    def prepare_evolve(self, dt):
        "Do nothing."
        pass

    def evolve(self, N_steps, dt):
        """Evolve by ``N_steps * dt``.

        Parameters
        ----------
        N_steps : int
            The number of steps to evolve.
        """
        self.dt = dt
        trunc_err = TruncationError()
        for _ in range(N_steps):
            max_err = self.sweep()
            trunc_err += TruncationError(max_err, 1.-2*max_err)  # TODO update definition of TruncationError
        self.evolved_time = self.evolved_time + N_steps * self.dt
        return trunc_err


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

    def __init__(self, psi, model, options, **kwargs):
        super().__init__(psi, model, options, **kwargs)
        self.trunc_err = TruncationError()

    def get_sweep_schedule(self):
        """Slightly different sweep schedule than DMRG"""
        L = self.psi.L
        if self.finite:
            i0s = list(range(0, L - 2)) + list(range(L - 2, -1, -1))
            move_right = [True] * (L - 2) + [False] * (L - 1)
            update_LP_RP = [[True, False]] * (L - 2) + [[False, True]] * (L - 2) + [[False, False]]
        else:
            raise NotImplementedError("Only finite TDVP is implemented")
        return zip(i0s, move_right, update_LP_RP)

    def update_local(self, theta, **kwargs):
        i0 = self.i0
        L = self.psi.L

        dt = self.dt
        if i0 == L - 2:
            dt = 2. * dt  # instead of updating the last pair of sites twice, we double the time
        # update two-site wavefunction
        theta, N = LanczosEvolution(self.eff_H, theta, self.lanczos_options).run(-0.5j * dt)
        if self.combine:
            theta.itranspose(['(vL.p0)', '(p1.vR)'])  # shouldn't do anything
        else:
            theta = theta.combine_legs([['vL', 'p0'], ['p1', 'vR']], new_axes=[0, 1],
                                       qconj=[+1, -1])
        qtotal_i0 = self.psi.get_B(i0, form=None).qtotal
        U, S, VH, err, _ = svd_theta(theta,
                                     self.trunc_params,
                                     qtotal_LR=[qtotal_i0, None],
                                     inner_labels=['vR', 'vL'])
        B0 = U.split_legs(['(vL.p0)']).replace_label('p0', 'p')
        B1 = VH.split_legs(['(p1.vR)']).replace_label('p1', 'p')

        self.psi.set_B(i0, B0, form='A')  # left-canonical
        self.psi.set_B(i0 + 1, B1, form='B')  # right-canonical
        self.psi.set_SR(i0, S)
        update_data = {'err': err, 'N': N, 'U': U, 'VH': VH}
        # earlier update of environments, since they are needed for the one_site_update()
        super().update_env(**update_data)  # new environments, e.g. LP[i0+1] on right move.

        if self.move_right:
            # note that i0 == L-2 is left-moving
            self.one_site_update(i0 + 1, 0.5j * self.dt)
        elif i0 != 0:  # no one-site update on last update of the sweep
            self.one_site_update(i0, 0.5j * self.dt)
        return update_data

    def update_env(self, **update_data):
        """Do nothing; super().update_env() is called explicitly in :meth:`update_local`."""
        pass

    def one_site_update(self, i, dt):
        H1 = OneSiteH(self.env, i, combine=False)
        theta = self.psi.get_theta(i, n=1, cutoff=self.S_inv_cutoff)
        theta = H1.combine_theta(theta)
        theta, _ = LanczosEvolution(H1, theta, self.lanczos_options).run(dt)
        self.psi.set_B(i, theta.replace_label('p0', 'p'), form='Th')

    def post_update_local(self, err, **update_data):
        self.trunc_err = self.trunc_err + err
        self.trunc_err_list.append(err.eps)  # avoid error in return of sweep()


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
            move_right = [True] * (L - 1) + [False] * L
            update_LP_RP = [[True, False]] * (L - 1) + [[False, True]] * (L - 1) + [[False, False]]
        else:
            raise NotImplementedError("Only finite TDVP is implemented")
        return zip(i0s, move_right, update_LP_RP)

    def update_local(self, theta, **kwargs):
        i0 = self.i0
        L = self.psi.L

        dt = self.dt
        if i0 == L - 1:
            dt = 2. * dt  # instead of updating the last site twice, we double the time

        # update one-site wavefunction
        theta, N = LanczosEvolution(self.eff_H, theta, self.lanczos_options).run(-0.5j * dt)
        if self.move_right:
            self.right_moving_update(i0, theta)
        else:
            self.left_moving_update(i0, theta)
        return {}  # no truncation error in single-site TDVP!

    def right_moving_update(self, i0, theta):
        if self.combine:
            theta.itranspose(['(vL.p0)', 'vR'])
        else:
            theta = theta.combine_legs(['vL', 'p0'], qconj=+1, new_axes=0)
        U, S, VH = npc.svd(theta, qtotal_LR=[theta.qtotal, None], inner_labels=['vR', 'vL'])
        # no truncation
        A0 = U.split_legs(['(vL.p0)']).replace_label('p0', 'p')
        self.psi.set_B(i0, A0, form='A')  # left-canonical
        self.psi.set_SR(i0, S)

        if True:  # note that i0 == L - 1 is left moving, so we always do a zero-site update
            super().update_env(U=U)
            theta = VH.scale_axis(S, 'vL')
            theta, H0 = self.zero_site_update(i0 + 1, theta, 0.5j * self.dt)
            next_B = self.psi.get_B(i0 + 1, form='B')
            next_th = npc.tensordot(theta, next_B, axes=['vR', 'vL'])
            self.psi.set_B(i0 + 1, next_th, form='Th')  # used and updated for next i0

    def left_moving_update(self, i0, theta):
        if self.combine:
            theta.itranspose(['vL', '(p0.vR)'])
        else:
            theta = theta.combine_legs(['p0', 'vR'], qconj=-1, new_axes=1)
        U, S, VH = npc.svd(theta, qtotal_LR=[None, theta.qtotal], inner_labels=['vR', 'vL'])
        if i0 == 0:
            assert U.shape == (1, 1)
            VH *= U[0, 0]  # just a global phase, but better keep it!
        B1 = VH.split_legs(['(p0.vR)']).replace_label('p0', 'p')
        self.psi.set_B(i0, B1, form='B')  # right-canonical
        self.psi.set_SL(i0, S)

        if i0 != 0:  # left-moving, but not the last site of the update
            super().update_env(VH=VH)  # note: no update needed if i0=0!
            theta = U.iscale_axis(S, 'vR')
            theta, H0 = self.zero_site_update(i0, theta, 0.5j * self.dt)
            next_A = self.psi.get_B(i0 - 1, form='A')
            next_th = npc.tensordot(next_A, theta, axes=['vR', 'vL'])
            self.psi.set_B(i0 - 1, next_th, form='Th')  # used and updated for next i0
            # note: this zero-site update can change the singular values on the bond left of i0.
            # however, we *don't* save them in psi: it turns out that the right singular
            # values for correct expectation values/entropies are the ones set before the if above.
            # (Belive me - I had that coded up and spent days looking for the bug...)

    def update_env(self, **update_data):
        """Do nothing; super().update_env() is called explicitly in :meth:`update_local`."""
        pass

    def zero_site_update(self, i, theta, dt):
        """Zero-site update on the left of site `i`."""
        H0 = ZeroSiteH(self.env, i)
        theta, _ = LanczosEvolution(H0, theta, self.lanczos_options).run(dt)
        return theta, H0

    def post_update_local(self, **update_data):
        self.trunc_err_list.append(0.)  # avoid error in return of sweep()


class TimeDependentSingleSiteTDVP(TimeDependentHAlgorithm,SingleSiteTDVPEngine):
    """Variant of :class:`SingleSiteTDVPEngine` that can handle time-dependent Hamiltonians.

    See details in :class:`~tenpy.algorithms.algorithm.TimeDependentHAlgorithm` as well.
    """
    def reinit_model(self):
        # recreate model
        TimeDependentHAlgorithm.reinit_model(self)
        # and reinitializie environment accordingly
        self.init_env(self.model)


class TimeDependentTwoSiteTDVP(TimeDependentHAlgorithm,TwoSiteTDVPEngine):
    """Variant of :class:`TwoSiteTDVPEngine` that can handle time-dependent Hamiltonians.

    See details in :class:`~tenpy.algorithms.algorithm.TimeDependentHAlgorithm` as well.
    """

    def reinit_model(self):
        TimeDependentSingleSiteTDVP.reinit_model(self)


class OldTDVPEngine(TimeEvolutionAlgorithm):
    """Time dependent variational principle algorithm for MPS.

    .. deprecated :: 0.10.0
        Replace this engine with the new :class:`TDVPEngine`.

    .. deprecated :: 0.6.0
        Renamed parameter/attribute `TDVP_params` to :attr:`options`.

    Parameters
    ----------
    psi, model, options, **kwargs:
        Same as for :class:`~tenpy.algorithms.algorithm.Algorithm`.
    environment :
        Initial environment. If ``None`` (default), it will be calculated at the beginning.

    Options
    -------

    .. cfg:config :: TDVP
        :include: TimeEvolutionAlgorithm

        active_sites
            The number of active sites to be used for the time evolution.
            If set to 1, :meth:`run_one_site` is used. The bond dimension will not increase!
            If set to 2, :meth:`run_two_sites` is used.
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
    environment : :class:`~tenpy.networks.mpo.MPOEnvironment`
        The environment, storing the `LP` and `RP` to avoid recalculations.
    lanczos_options : :class:`~tenpy.tools.params.Config`
        Options passed on to :class:`~tenpy.linalg.lanczos.LanczosEvolution`.
    """
    def __init__(self, psi, model, options, environment=None, **kwargs):
        msg = "Deprecated `OldTDVPEngine`, use the new " \
            "`SingleSiteTDVPEngine` and `TwoSiteTDVPEngine` instead."
        warnings.warn(msg, category=FutureWarning, stacklevel=2)
        TimeEvolutionAlgorithm.__init__(self, psi, model, options, **kwargs)
        options = self.options
        if model.H_MPO.explicit_plus_hc:
            raise NotImplementedError("TDVP does not respect 'MPO.explicit_plus_hc' flag")
        self.lanczos_options = options.subconfig('lanczos_options')
        if environment is None:
            environment = MPOEnvironment(psi, model.H_MPO, psi)
        self.evolved_time = options.get('start_time', 0.)
        self.H_MPO = model.H_MPO
        self.environment = environment
        if psi.bc != 'finite':
            raise ValueError("TDVP is only implemented for finite boundary conditions")
        self.L = self.psi.L
        self.dt = options.get('dt', 2)
        self.N_steps = options.get('N_steps', 10)

    @property
    def TDVP_params(self):
        warnings.warn("renamed self.TDVP_params -> self.options", FutureWarning, stacklevel=2)
        return self.options

    def run(self):
        """(Real-)time evolution with TDVP."""
        active_sites = self.options.get('active_sites', 2)
        if active_sites == 1:
            self.run_one_site(self.N_steps)
        elif active_sites == 2:
            self.run_two_sites(self.N_steps)
        else:
            raise ValueError("TDVP can only use 1 or 2 active sites, not {}".format(active_sites))

    def run_one_site(self, N_steps=None):
        """Run the TDVP algorithm with the one site algorithm.

        .. warning ::
            Be aware that the bond dimension will not increase!

        Parameters
        ----------
        N_steps : integer. Number of steps
        """
        if N_steps != None:
            self.N_steps = N_steps
        D = self.H_MPO._W[0].shape[0]
        #Initialize in the correct order
        for i in range(self.L):
            self.psi.get_B(i).itranspose(('vL', 'p', 'vR'))
        for i in range(self.N_steps):
            self.sweep_left_right()
            self.sweep_right_left()
            self.evolved_time = self.evolved_time + self.dt

    def run_two_sites(self, N_steps=None):
        """Run the TDVP algorithm with two sites update.

        The bond dimension will increase. Truncation happens at every step of the
        sweep, according to the parameters set in trunc_params.

        Parameters
        ----------
        N_steps : integer. Number of steps
        """
        if N_steps != None:
            self.N_steps = N_steps
        D = self.H_MPO._W[0].shape[0]
        #Initialize in the correct order
        for i in range(self.L):
            self.psi.get_B(i).itranspose(('vL', 'p', 'vR'))
        for i in range(self.N_steps):
            self.sweep_left_right_two()
            self.sweep_right_left_two()
            self.evolved_time = self.evolved_time + self.dt

    def _del_correct(self, i):
        """Delete correctly the environment once the tensor at site i is updated.

        Parameters
        ----------
        i : int
            Site at which the tensor has been updated
        """

        if i + 1 < self.L:
            self.environment.del_LP(i + 1)
        if i - 1 > -1:
            self.environment.del_RP(i - 1)

    def sweep_left_right(self):
        """Performs the sweep left->right of the second order TDVP scheme with one site update.

        Evolve from 0.5*dt.
        """
        for j in range(self.L):
            # Get theta
            if j == 0:
                theta = self.psi.get_B(0, form='Th')
            else:
                B = self.psi.get_B(j, form='B')
                #theta[vL,p,vR]=s[vL,vR]*self.psi[p,vL,vR]
                theta = npc.tensordot(s, B, axes=('vR', 'vL'))
            Lp = self.environment.get_LP(j)
            Rp = self.environment.get_RP(j)
            W1 = self.environment.H.get_W(j)
            theta = self.update_theta_h1(Lp, Rp, theta, W1, -1j * 0.5 * self.dt)
            # SVD and update environment
            U, s, V = self.theta_svd_left_right(theta)
            self.psi.set_B(j, U, form='A')
            self.psi.set_SR(j, np.diag(s.to_ndarray()))
            self._del_correct(j)
            if j < self.L - 1:
                # Apply expm (-dt H) for 0-site

                B = self.psi.get_B(j + 1)
                B_jp1 = npc.tensordot(V, B, axes=['vR', 'vL'])
                self.psi.set_B(j + 1, B_jp1, form='B')
                Lpp = self.environment.get_LP(j + 1)
                Rp = npc.tensordot(Rp, V, axes=['vL', 'vR'])
                Rp = npc.tensordot(Rp, V.conj(), axes=['vL*', 'vR*'])
                H = H0_mixed(Lpp, Rp)

                s = self.update_s_h0(s, H, 1j * 0.5 * self.dt)
                s = s / np.linalg.norm(s.to_ndarray())
            else:
                self.psi.set_B(j, npc.tensordot(self.psi.get_B(j), V, axes=['vR', 'vL']), form='B')

    def sweep_left_right_two(self):
        """Performs the sweep left->right of the second order TDVP scheme with two sites update.

        Evolve from 0.5*dt
        """
        theta_old = self.psi.get_theta(0, 1)
        for j in range(self.L - 1):

            theta = npc.tensordot(theta_old, self.psi.get_B(j + 1), ('vR', 'vL'))
            theta.ireplace_label('p', 'p1')
            Lp = self.environment.get_LP(j)
            Rp = self.environment.get_RP(j + 1)
            W1 = self.environment.H.get_W(j)
            W2 = self.environment.H.get_W(j + 1)
            theta = self.update_theta_h2(Lp, Rp, theta, W1, W2, -0.5 * 1j * self.dt)
            theta = theta.combine_legs([['vL', 'p0'], ['vR', 'p1']], qconj=[+1, -1])
            # SVD and update environment
            U, s, V, err, renorm = svd_theta(theta, self.trunc_params)
            s = s / npc.norm(s)
            U = U.split_legs('(vL.p0)')
            U.ireplace_label('p0', 'p')
            V = V.split_legs('(vR.p1)')
            V.ireplace_label('p1', 'p')
            self.psi.set_B(j, U, form='A')
            self._del_correct(j)
            self.psi.set_SR(j, s)
            self.psi.set_B(j + 1, V, form='B')
            self._del_correct(j + 1)
            if j < self.L - 2:
                # Apply expm (-dt H) for 1-site
                theta = self.psi.get_theta(j + 1, 1)
                theta.ireplace_label('p0', 'p')
                Lp = self.environment.get_LP(j + 1)
                Rp = self.environment.get_RP(j + 1)
                theta = self.update_theta_h1(Lp, Rp, theta, W2, 1j * 0.5 * self.dt)
                theta_old = theta
                theta_old.ireplace_label('p', 'p0')

    def sweep_right_left(self):
        """Performs the sweep right->left of the second order TDVP scheme with one site update.

        Evolve from 0.5*dt
        """
        expectation_O = []
        for j in range(self.L - 1, -1, -1):
            B = self.psi.get_B(j, form='A')
            # Get theta
            if j == self.L - 1:
                theta = B
            else:
                theta = npc.tensordot(B, s,
                                      axes=('vR',
                                            'vL'))  #theta[vL,p,vR]=s[vL,vR]*self.psi[p,vL,vR]

            # Apply expm (-dt H) for 1-site
            chiB, chiA, d = theta.to_ndarray().shape
            Lp = self.environment.get_LP(j)
            Rp = self.environment.get_RP(j)
            W1 = self.environment.H.get_W(j)
            theta = self.update_theta_h1(Lp, Rp, theta, W1, -1j * 0.5 * self.dt)
            # SVD and update environment
            U, s, V = self.theta_svd_right_left(theta)
            self.psi.set_B(j, U, form='B')
            self.psi.set_SL(j, np.diag(s.to_ndarray()))
            self._del_correct(j)
            if j > 0:
                # Apply expm (-dt H) for 0-site

                B = self.psi.get_B(j - 1, form='A')
                B_jm1 = npc.tensordot(V, B, axes=['vL', 'vR'])
                self.psi.set_B(j - 1, B_jm1, form='A')
                Lp = npc.tensordot(Lp, V, axes=['vR', 'vL'])
                Lp = npc.tensordot(Lp, V.conj(), axes=['vR*', 'vL*'])
                H = H0_mixed(Lp, self.environment.get_RP(j - 1))

                s = self.update_s_h0(s, H, 1j * 0.5 * self.dt)
                s = s / np.linalg.norm(s.to_ndarray())
            else:
                # overwrites previous Bj
                self.psi.set_B(j, npc.tensordot(V, self.psi.get_B(j, form='A'),
                                                axes=['vR', 'vL']), form='A')

    def sweep_right_left_two(self):
        """Performs the sweep left->right of the second order TDVP scheme with two sites update.

        Evolve from 0.5*dt
        """
        theta_old = self.psi.get_theta(self.L - 1, 1)
        for j in range(self.L - 2, -1, -1):
            theta = npc.tensordot(theta_old, self.psi.get_B(j, form='A'), ('vL', 'vR'))
            theta.ireplace_label('p0', 'p1')
            theta.ireplace_label('p', 'p0')
            #theta=self.psi.get_theta(j,2)
            Lp = self.environment.get_LP(j)
            Rp = self.environment.get_RP(j + 1)
            W1 = self.environment.H.get_W(j)
            W2 = self.environment.H.get_W(j + 1)
            theta = self.update_theta_h2(Lp, Rp, theta, W1, W2, -1j * 0.5 * self.dt)
            theta = theta.combine_legs([['vL', 'p0'], ['vR', 'p1']], qconj=[+1, -1])
            # SVD and update environment
            U, s, V, err, renorm = svd_theta(theta, self.trunc_params)
            s = s / npc.norm(s)
            U = U.split_legs('(vL.p0)')
            U.ireplace_label('p0', 'p')
            V = V.split_legs('(vR.p1)')
            V.ireplace_label('p1', 'p')
            self.psi.set_B(j, U, form='A')
            self._del_correct(j)
            self.psi.set_SR(j, s)
            self.psi.set_B(j + 1, V, form='B')
            self._del_correct(j + 1)
            if j > 0:
                # Apply expm (-dt H) for 1-site
                theta = self.psi.get_theta(j, 1)
                theta.ireplace_label('p0', 'p')
                Lp = self.environment.get_LP(j)
                Rp = self.environment.get_RP(j)
                theta = self.update_theta_h1(Lp, Rp, theta, W1, 1j * 0.5 * self.dt)
                theta_old = theta
                theta.ireplace_label('p', 'p0')

    def update_theta_h1(self, Lp, Rp, theta, W, dt):
        """Update with the one site Hamiltonian.

        Parameters
        ----------
        Lp : :class:`~tenpy.linalg.np_conserved.Array`
            tensor representing the left environment
        Rp :  :class:`~tenpy.linalg.np_conserved.Array`
            tensor representing the right environment
        theta :  :class:`~tenpy.linalg.np_conserved.Array`
            the theta tensor which needs to be updated
        W : :class:`~tenpy.linalg.np_conserved.Array`
            MPO which is applied to the 'p' leg of theta
        """
        H = H1_mixed(Lp, Rp, W)
        theta = theta.combine_legs(['vL', 'p', 'vR'])
        #Initialize Lanczos
        lanczos_h1 = LanczosEvolution(H=H, psi0=theta, options=self.lanczos_options)
        theta, N_h1 = lanczos_h1.run(dt)
        theta = theta.split_legs(['(vL.p.vR)'])
        return theta

    def update_theta_h2(self, Lp, Rp, theta, W0, W1, dt):
        """Update with the two sites Hamiltonian.

        Parameters
        ----------
        Lp : :class:`tenpy.linalg.np_conserved.Array`
            tensor representing the left environment
        Rp : :class:`tenpy.linalg.np_conserved.Array`
            tensor representing the right environment
        theta : :class:`tenpy.linalg.np_conserved.Array`
            the theta tensor which needs to be updated
        W : :class:`tenpy.linalg.np_conserved.Array`
            MPO which is applied to the 'p0' leg of theta
        W1 : :class:`tenpy.linalg.np_conserved.Array`
            MPO which is applied to the 'p1' leg of theta
        """
        H = H2_mixed(Lp, Rp, W0, W1)
        theta = theta.combine_legs(['vL', 'p0', 'p1', 'vR'])
        #Initialize Lanczos
        lanczos_h1 = LanczosEvolution(H=H, psi0=theta, options=self.lanczos_options)
        theta, N_h1 = lanczos_h1.run(dt)
        theta = theta.split_legs(['(vL.p0.p1.vR)'])
        return theta

    def theta_svd_left_right(self, theta):
        """Performs the SVD from left to right.

        Parameters
        ----------
        theta: :class:`tenpy.linalg.np_conserved.Array`
            the theta tensor on which the SVD is applied
        """
        theta = theta.combine_legs(['vL', 'p'], new_axes=0)
        U, s, V = npc.svd(theta, full_matrices=0, inner_labels=('vR', 'vL'))
        U = U.split_legs(['(vL.p)'])
        s_ndarray = np.diag(s)
        vR_U = U.get_leg('vR')
        vL_V = V.get_leg('vL')
        s = npc.Array.from_ndarray(s_ndarray, [vR_U.conj(), vL_V.conj()],
                                   dtype=None,
                                   qtotal=None,
                                   cutoff=None)
        s.iset_leg_labels(['vL', 'vR'])
        return U, s, V

    def theta_svd_right_left(self, theta):
        """Performs the SVD from right to left.

        Parameters
        ----------
        theta : :class:`tenpy.linalg.np_conserved.Array`,
            The theta tensor on which the SVD is applied
        """
        theta = theta.combine_legs(['p', 'vR'], new_axes=1)
        V, s, U = npc.svd(theta, full_matrices=0, inner_labels=('vR', 'vL'))
        U = U.split_legs(['(p.vR)'])
        s_ndarray = np.diag(s)
        vL_U = U.get_leg('vL')
        vR_V = V.get_leg('vR')
        s = npc.Array.from_ndarray(s_ndarray, [vR_V.conj(), vL_U.conj()],
                                   dtype=None,
                                   qtotal=None,
                                   cutoff=None)
        s.iset_leg_labels(['vL', 'vR'])
        return U, s, V

    def update_s_h0(self, s, H, dt):
        """Update with the zero site Hamiltonian (update of the singular value)

        Parameters
        ----------
        s : :class:`tenpy.linalg.np_conserved.Array`
            representing the singular value matrix which is updated
        H : H0_mixed
            zero site Hamiltonian that we need to apply on the singular value matrix
        dt : complex number
            time step of the evolution
        """
        #Initialize Lanczos
        lanczos_h0 = LanczosEvolution(H=H,
                                      psi0=s.combine_legs(['vL', 'vR']),
                                      options=self.lanczos_options)
        s_new, N_h0 = lanczos_h0.run(dt)
        s_new = s_new.split_legs(['(vL.vR)'])
        return s_new


class Engine(OldTDVPEngine):
    """Deprecated old name of the :class:`OldTDVPEngine`.

    .. deprecated : v0.8.0
        Renamed the `Engine` to `TDVPEngine` to have unique algorithm class names.
    """
    def __init__(self, psi, model, options, **kwargs):
        msg = "Renamed `Engine` class to `TDVPEngine`."
        warnings.warn(msg, category=FutureWarning, stacklevel=2)
        TDVPEngine.__init__(self, psi, model, options, **kwargs)


class H0_mixed:
    """Class defining the zero site Hamiltonian for Lanczos.

    .. deprecated : v0.10.0
        This class is only used by the deprecated :class:`OldTDVPEngine`.

    Parameters
    ----------
    Lp : :class:`tenpy.linalg.np_conserved.Array`
        left part of the environment
    Rp : :class:`tenpy.linalg.np_conserved.Array`
        right part of the environment

    Attributes
    ----------
    Lp : :class:`tenpy.linalg.np_conserved.Array`
        left part of the environment
    Rp : :class:`tenpy.linalg.np_conserved.Array`
        right part of the environment
    """
    def __init__(self, Lp, Rp):
        self.Lp = Lp
        self.Rp = Rp

    def matvec(self, x):
        x = x.split_legs(['(vL.vR)'])
        x = npc.tensordot(self.Lp, x, axes=('vR', 'vL'))
        x = npc.tensordot(x, self.Rp, axes=(['vR', 'wR'], ['vL', 'wL']))
        #TODO:next line not needed. Since the transpose does not do anything, should not cost anything. Keep for safety ?
        x = x.transpose(['vR*', 'vL*'])
        x = x.iset_leg_labels(['vL', 'vR'])
        x = x.combine_legs(['vL', 'vR'])
        return (x)


class H1_mixed:
    """Class defining the one site Hamiltonian for Lanczos.

    .. deprecated : v0.10.0
        This class is only used by the deprecated :class:`OldTDVPEngine`.

    Parameters
    ----------
    Lp : :class:`tenpy.linalg.np_conserved.Array`
        left part of the environment
    Rp : :class:`tenpy.linalg.np_conserved.Array`
        right part of the environment
    M : :class:`tenpy.linalg.np_conserved.Array`
        MPO which is applied to the 'p' leg of theta

    Attributes
    ----------
    Lp : :class:`tenpy.linalg.np_conserved.Array`
        left part of the environment
    Rp : :class:`tenpy.linalg.np_conserved.Array`
        right part of the environment
    W : :class:`tenpy.linalg.np_conserved.Array`
        MPO which is applied to the 'p0' leg of theta
    """
    def __init__(self, Lp, Rp, W):
        self.Lp = Lp  # a,ap,m
        self.Rp = Rp  # b,bp,n
        self.W = W  # m,n,i,ip

    def matvec(self, theta):
        theta = theta.split_legs(['(vL.p.vR)'])
        Lp = self.Lp
        Rp = self.Rp
        x = npc.tensordot(Lp, theta, axes=('vR', 'vL'))
        x = npc.tensordot(x, self.W, axes=(['p', 'wR'], ['p*', 'wL']))
        x = npc.tensordot(x, Rp, axes=(['vR', 'wR'], ['vL', 'wL']))
        #TODO:next line not needed. Since the transpose does not do anything, should not cost anything. Keep for safety ?
        x = x.transpose(['vR*', 'p', 'vL*'])
        x = x.iset_leg_labels(['vL', 'p', 'vR'])
        h = x.combine_legs(['vL', 'p', 'vR'])
        return h


class H2_mixed:
    """Class defining the two sites Hamiltonian for Lanczos.

    .. deprecated : v0.10.0
        This class is only used by the deprecated :class:`OldTDVPEngine`.

    Parameters
    ----------
    Lp : :class:`tenpy.linalg.np_conserved.Array`
        left part of the environment
    Rp : :class:`tenpy.linalg.np_conserved.Array`
        right part of the environment
    W : :class:`tenpy.linalg.np_conserved.Array`
        MPO which is applied to the 'p0' leg of theta

    Attributes
    ----------
    Lp : :class:`tenpy.linalg.np_conserved.Array`
        left part of the environment
    Rp : :class:`tenpy.linalg.np_conserved.Array`
        right part of the environment
    W0 : :class:`tenpy.linalg.np_conserved.Array`
        MPO which is applied to the 'p0' leg of theta
    W1 : :class:`tenpy.linalg.np_conserved.Array`
        MPO which is applied to the 'p1' leg of theta
    """
    def __init__(self, Lp, Rp, W0, W1):
        self.Lp = Lp  # a,ap,m
        self.Rp = Rp  # b,bp,n
        self.H_MPO0 = W0  # m,n,i,ip
        self.H_MPO1 = W1

    def matvec(self, theta):
        theta = theta.split_legs(['(vL.p0.p1.vR)'])
        Lp = self.Lp
        Rp = self.Rp
        x = npc.tensordot(Lp, theta, axes=('vR', 'vL'))
        x = npc.tensordot(x, self.H_MPO0, axes=(['p0', 'wR'], ['p*', 'wL']))
        x.ireplace_label('p', 'p0')
        x = npc.tensordot(x, self.H_MPO1, axes=(['p1', 'wR'], ['p*', 'wL']))
        x.ireplace_label('p', 'p1')
        x = npc.tensordot(x, Rp, axes=(['vR', 'wR'], ['vL', 'wL']))
        x.ireplace_label('vL*', 'vR')
        x.ireplace_label('vR*', 'vL')
        h = x.combine_legs(['vL', 'p0', 'p1', 'vR'])
        return h
