"""Time Dependant Variational Principle (TDVP) with MPS (finite version only).

The TDVP MPS algorithm was first proposed by :cite:`haegeman2011`. However the stability of the
algorithm was later improved in :cite:`haegeman2016`, that we are following in this implementation.
The general idea of the algorithm is to project the quantum time evolution in the manyfold of MPS
with a given bond dimension. Compared to e.g. TEBD, the algorithm has several advantages:
e.g. it conserves the unitarity of the time evolution and the energy (for the single-site version),
and it is suitable for time evolution of Hamiltonian with arbitrary long range in the form of MPOs.
We have implemented:

1. The one-site formulation following the TDVP principle in :class:`SingleSiteTDVPEngine`,
   which **does not** allow for growth of the bond dimension.

2. The two-site algorithm in the :class:`TwoSiteTDVPEngine`, which does allow the bond
   dimension to grow - but requires truncation as in the TEBD case, and is no longer strictly TDVP,
   i.e. it does *not* strictly preserve the energy.

Much of the code is very similar to DMRG, and also based on the
:class:`~tenpy.algorithms.mps_common.Sweep` class.

.. versionchanged :: 0.10.0
    The interface changed compared to version 0.9.0:
    Just :class:`TDVPEngine` will result in a error.
    Use :class:`SingleSiteTDVPEngine` or :class:`TwoSiteTDVPEngine` instead.


.. todo ::
    extend code to infinite MPS

.. todo ::
    allow for increasing bond dimension in SingleSiteTDVPEngine, similar to DMRG Mixer
"""
# Copyright (C) TeNPy Developers, Apache license

from ..linalg.krylov_based import LanczosEvolution
from ..linalg.truncation import svd_theta, TruncationError
from .mps_common import Sweep, ZeroSiteH, OneSiteH, TwoSiteH
from .algorithm import TimeEvolutionAlgorithm, TimeDependentHAlgorithm
from ..linalg import np_conserved as npc
from ..tools.misc import consistency_check
from ..tools.params import asConfig
import logging
import warnings

logger = logging.getLogger(__name__)

__all__ = ['TDVPEngine', 'SingleSiteTDVPEngine', 'TwoSiteTDVPEngine',
           'TimeDependentSingleSiteTDVP', 'TimeDependentTwoSiteTDVP']


class TDVPEngine(TimeEvolutionAlgorithm, Sweep):
    """Time dependent variational principle algorithm for MPS.

    This class contains all methods that are generic between
    :class:`SingleSiteTDVPEngine` and :class:`TwoSiteTDVPEngine`.
    Use the latter two classes for actual TDVP runs.

    .. versionchanged :: 1.1
        Previously had separate `lanczos_options`, which have been renamed to `lanczos_params`
        for consistency with the Sweep class.

    Parameters
    ----------
    psi, model, options, **kwargs:
        Same as for :class:`~tenpy.algorithms.algorithm.Algorithm`.

    Options
    -------
    .. cfg:config :: TDVPEngine
        :include: TimeEvolutionAlgorithm, Sweep

        max_dt : float | None
            Threshold for raising errors on too large time steps. Default ``1.0``.
            See :meth:`~tenpy.tools.misc.consistency_check`.
            For large time steps, the projection to the MPS manifold that is the main building block
            of TDVP, can not be a good approximation anymore. We raise in that case.
            Can be downgraded to a warning by setting this option to ``None``.

    """
    EffectiveH = None

    def __init__(self, psi, model, options, **kwargs):
        if self.__class__.__name__ == 'TDVPEngine':
            msg = ("TDVP interface changed. \n"
                   "The new TDVPEngine has subclasses SingleSiteTDVPEngine"
                   " and TwoSiteTDVPEngine that you can use.\n"
                   )
            raise NameError(msg)
        if psi.bc != 'finite':
            raise NotImplementedError("Only finite TDVP is implemented")
        assert psi.bc == model.lat.bc_MPS
        options = asConfig(options, self.__class__.__name__)
        options.deprecated_alias("lanczos_options", "lanczos_params",
                                 "See also https://github.com/tenpy/tenpy/issues/459")
        super().__init__(psi, model, options, **kwargs)

    # run() from TimeEvolutionAlgorithm

    @property
    def lanczos_options(self):
        """Deprecated alias of :attr:`lanczos_params`."""
        warnings.warn("Accessing deprecated alias TDVPEngine.lanczos_options instead of lanczos_params",
                      FutureWarning, stacklevel=2)
        return self.lanczos_params

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
        consistency_check(dt, self.options, 'max_dt', 1.,
                          'dt > ``max_dt`` is unreasonably large for TDVP.',
                          compare=lambda dt, max_dt: abs(dt) <= max_dt)
        self.dt = dt
        trunc_err = TruncationError()
        for _ in range(N_steps):
            self.sweep()
            for eps in self.trunc_err_list:
                trunc_err += TruncationError(eps, 1 - 2 * eps)
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
    .. cfg:config :: TwoSiteTDVPEngine
        :include: TDVPEngine

    """
    EffectiveH = TwoSiteH

    def __init__(self, psi, model, options, **kwargs):
        super().__init__(psi, model, options, **kwargs)

    def get_sweep_schedule(self):
        """Slightly different sweep schedule than DMRG"""
        L = self.psi.L
        if self.finite:
            i0s = list(range(0, L - 2)) + list(range(L - 2, -1, -1))
            move_right = [True] * (L - 2) + [False] * (L - 2) + [None]
            update_LP_RP = [[True, False]] * (L - 2) + [[False, True]] * (L - 2) + [[False, False]]
        else:
            raise NotImplementedError("Only finite TDVP is implemented")
        return zip(i0s, move_right, update_LP_RP)

    def update_local(self, theta, **kwargs):
        i0 = self.i0
        L = self.psi.L

        dt = -0.5j * self.dt
        if i0 == L - 2:
            dt = 2. * dt  # instead of updating the last pair of sites twice, we double the time
        # update two-site wavefunction
        theta, N = LanczosEvolution(self.eff_H, theta, self.lanczos_params).run(dt)
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
        elif (self.move_right is False):
            self.one_site_update(i0, 0.5j * self.dt)
        # for the last update of the sweep, where move_right is None, there is no one_site_update

        return update_data

    def update_env(self, **update_data):
        """Do nothing; super().update_env() is called explicitly in :meth:`update_local`."""
        pass

    def one_site_update(self, i, dt):
        H1 = OneSiteH(self.env, i, combine=False)
        theta = self.psi.get_theta(i, n=1, cutoff=self.S_inv_cutoff)
        theta = H1.combine_theta(theta)
        theta, _ = LanczosEvolution(H1, theta, self.lanczos_params).run(dt)
        self.psi.set_B(i, theta.replace_label('p0', 'p'), form='Th')


class SingleSiteTDVPEngine(TDVPEngine):
    """Engine for the single-site TDVP algorithm.

    Parameters
    ----------
    psi, model, options, **kwargs:
        Same as for :class:`~tenpy.algorithms.algorithm.Algorithm`.

    Options
    -------
    .. cfg:config :: SingleSiteTDVPEngine
        :include: TDVPEngine

    """
    EffectiveH = OneSiteH

    def get_sweep_schedule(self):
        """slightly different sweep schedule than DMRG"""
        L = self.psi.L
        if self.finite:
            i0s = list(range(0, L - 1)) + list(range(L - 1, -1, -1))
            move_right = [True] * (L - 1) + [False] * (L - 1) + [None]
            update_LP_RP = [[True, False]] * (L - 1) + [[False, True]] * (L - 1) + [[False, False]]
        else:
            raise NotImplementedError("Only finite TDVP is implemented")
        return zip(i0s, move_right, update_LP_RP)

    def update_local(self, theta, **kwargs):
        i0 = self.i0
        L = self.psi.L

        dt = -0.5j * self.dt
        if i0 == L - 1:
            dt = 2. * dt  # instead of updating the last site twice, we double the time

        # update one-site wavefunction
        theta, N = LanczosEvolution(self.eff_H, theta, self.lanczos_params).run(dt)
        if self.move_right:
            self.right_moving_update(i0, theta)
        else:
            # note: left_moving_update() also covers the "non-moving" case move_right=None
            # of the last update in a sweep
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
            # (Believe me - I had that coded up and spent days looking for the bug...)

    def update_env(self, **update_data):
        """Do nothing; super().update_env() is called explicitly in :meth:`update_local`."""
        pass

    def zero_site_update(self, i, theta, dt):
        """Zero-site update on the left of site `i`."""
        H0 = ZeroSiteH(self.env, i)
        theta, _ = LanczosEvolution(H0, theta, self.lanczos_params).run(dt)
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
        # and reinitialize environment accordingly
        self.init_env(self.model)


class TimeDependentTwoSiteTDVP(TimeDependentHAlgorithm,TwoSiteTDVPEngine):
    """Variant of :class:`TwoSiteTDVPEngine` that can handle time-dependent Hamiltonians.

    See details in :class:`~tenpy.algorithms.algorithm.TimeDependentHAlgorithm` as well.
    """

    def reinit_model(self):
        TimeDependentSingleSiteTDVP.reinit_model(self)
