"""Density Matrix Renormalization Group (DMRG).

Although it was originally not formulated with tensor networks,
the DMRG algorithm (invented by Steven White in 1992 :cite:`white1992`) opened the whole field
with its enormous success in finding ground states in 1D.

We implement DMRG in the modern formulation of matrix product states :cite:`schollwoeck2011`,
both for finite systems (``'finite'`` or ``'segment'`` boundary conditions)
and in the thermodynamic limit (``'infinite'`` b.c.).

The function :func:`run` - well - runs one DMRG simulation.
Internally, it generates an instance of an :class:`Sweep`.
This class implements the common functionality like defining a `sweep`,
but leaves the details of the contractions to be performed to the derived classes.

Currently, there are two derived classes implementing the contractions: :class:`SingleSiteDMRGEngine`
and :class:`TwoSiteDMRGEngine`. They differ (as their name implies) in the number of sites which
are optimized simultaneously.
They should both give the same results (up to rounding errors). However, if started from a product
state, :class:`SingleSiteDMRGEngine` depends critically on the use of a
:class:`~tenpy.algorithms.mps_common.Mixer`, while :class:`TwoSiteDMRGEngine` is in principle more
computationally expensive to run and has occasionally displayed some convergence issues.
Which one is preferred in the end is not obvious a priori and might depend on the used model.
Just try both of them.

A :class:`~tenpy.algorithms.mps_common.Mixer` should be used initially to avoid that the algorithm gets stuck in local energy
minima, and then slowly turned off in the end. For :class:`SingleSiteDMRGEngine`, using a mixer is
crucial, as the one-site algorithm cannot increase the MPS bond dimension by itself.

A generic protocol for approaching a physics question using DMRG is given in
:doc:`/intro/dmrg-protocol`.
"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
import time
import warnings
import logging
logger = logging.getLogger(__name__)

from ..linalg import np_conserved as npc
from ..linalg.krylov_based import lanczos_arpack, LanczosGroundState
from ..linalg.truncation import svd_theta
from ..tools.params import asConfig
from ..tools.math import entropy
from ..tools.process import memory_usage
from .mps_common import IterativeSweeps, OneSiteH, TwoSiteH
from . import mps_common

__all__ = [
    'run',
    'DMRGEngine',
    'SingleSiteDMRGEngine',
    'TwoSiteDMRGEngine',
    'chi_list',
    'full_diag_effH',
]


def run(psi, model, options, **kwargs):
    r"""Run the DMRG algorithm to find the ground state of the given model.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        Initial guess for the ground state, which is to be optimized in-place.
    model : :class:`~tenpy.models.MPOModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    options : dict
        Further optional parameters as described in :cfg:config:`DMRG`.
    **kwargs :
        Further keyword arguments for the algorithm classes :class:`TwoSiteDMRGEngine` or
        :class:`SingleSiteDMRGEngine`.

    Returns
    -------
    info : dict
        A dictionary with keys ``'E', 'shelve', 'bond_statistics', 'sweep_statistics'``

    Options
    -------
    .. cfg:config :: DMRG
        :include: SingleSiteDMRGEngine, TwoSiteDMRGEngine

        active_sites : 1 | 2
            The number of active sites to be used by DMRG.
            If set to 1, :class:`SingleSiteDMRGEngine` is used.
            If set to 2, DMRG is handled by :class:`TwoSiteDMRGEngine`.

    """
    # initialize the engine
    options = asConfig(options, 'DMRG')
    active_sites = options.get('active_sites', 2, int)
    if active_sites == 1:
        engine = SingleSiteDMRGEngine(psi, model, options, **kwargs)
    elif active_sites == 2:
        engine = TwoSiteDMRGEngine(psi, model, options, **kwargs)
    else:
        raise ValueError("For DMRG, can only use 1 or 2 active sites, not {}".format(active_sites))
    E, _ = engine.run()
    return {
        'E': E,
        'shelve': engine.shelve,
        'bond_statistics': engine.update_stats,
        'sweep_statistics': engine.sweep_stats
    }


class DMRGEngine(IterativeSweeps):
    """DMRG base class with common methods for the TwoSiteDMRG and SingleSiteDMRG.

    This engine is implemented as a subclass of :class:`~tenpy.algorithms.mps_common.Sweep`.
    It contains all methods that are generic between
    :class:`SingleSiteDMRGEngine` and :class:`TwoSiteDMRGEngine`.
    Use the latter two classes for actual DMRG runs.

    A generic protocol for approaching a physics question using DMRG is given in
    :doc:`/intro/dmrg-protocol`.

    Options
    -------
    .. cfg:config :: DMRGEngine
        :include: IterativeSweeps

    Attributes
    ----------
    update_stats : dict
        A dictionary with detailed statistics of the convergence at local update-level.
        For each key in the following table, the dictionary contains a list where one value is
        added each time :meth:`DMRGEngine.update_bond` is called.

        =========== ===================================================================
        key         description
        =========== ===================================================================
        i0          An update was performed on sites ``i0, i0+1``.
        ----------- -------------------------------------------------------------------
        age         The number of physical sites involved in the simulation.
        ----------- -------------------------------------------------------------------
        E_total     The total energy before truncation.
        ----------- -------------------------------------------------------------------
        N_lanczos   Dimension of the Krylov space used in the lanczos diagonalization.
        ----------- -------------------------------------------------------------------
        time        Wallclock time evolved since :attr:`time0` (in seconds).
        ----------- -------------------------------------------------------------------
        ov_change   ``1. - abs(<theta_guess|theta_diag>)``, where ``|theta_guess>`` is
                    the initial guess for the wave function and ``|theta_diag>`` is the
                    *untruncated* wave function returned by :meth:`diag`.
        =========== ===================================================================

    sweep_stats : dict
        A dictionary with detailed statistics at the sweep level.
        For each key in the following table, the dictionary contains a list where one value is
        added each time :meth:`Engine.sweep` is called (with ``optimize=True``).

        ============= ===================================================================
        key           description
        ============= ===================================================================
        sweep         Number of sweeps (excluding environment sweeps) performed so far.
        ------------- -------------------------------------------------------------------
        N_updates     Number of updates (including environment sweeps) performed so far.
        ------------- -------------------------------------------------------------------
        E             The energy *before* truncation (as calculated by Lanczos).
        ------------- -------------------------------------------------------------------
        Delta_E       The change in `E` (above) since the last iteration.
        ------------- -------------------------------------------------------------------
        S             Mean entanglement entropy (over bonds).
        ------------- -------------------------------------------------------------------
        Delta_S       The change in `S` (above) since the last iteration.
        ------------- -------------------------------------------------------------------
        max_S         Max entanglement entropy (over bonds).
        ------------- -------------------------------------------------------------------
        time          Wallclock time evolved since :attr:`time0` (in seconds).
        ------------- -------------------------------------------------------------------
        max_trunc_err The maximum truncation error in the last sweep
        ------------- -------------------------------------------------------------------
        max_E_trunc   Maximum change or Energy due to truncation in the last sweep.
        ------------- -------------------------------------------------------------------
        max_chi       Maximum bond dimension used.
        ------------- -------------------------------------------------------------------
        norm_err      Error of canonical form ``np.linalg.norm(psi.norm_test())``.
        ============= ===================================================================

    _entropy_approx : list of {None, 1D array}
        While the mixer is on, the `S` stored in the MPS is a non-diagonal 2D array.
        To check convergence, we use the approximate singular values based on which we truncated
        instead to calculate the entanglement entropy and store it inside this list.
    """
    EffectiveH = None

    def __init__(self, psi, model, options, **kwargs):
        options = asConfig(options, self.__class__.__name__)
        self.diag_method = options.get('diag_method', 'default', str)
        self._entropy_approx = [None] * psi.L  # always left of a given site
        super().__init__(psi, model, options, **kwargs)
        self.N_sweeps_check = self.options.get('N_sweeps_check', 1 if self.psi.finite else 10, int)
        default_min_sweeps = int(1.5 * self.N_sweeps_check)
        if self.chi_list is not None:
            default_min_sweeps = max(max(self.chi_list.keys()), default_min_sweeps)
        self.options.setdefault('min_sweeps', default_min_sweeps)
        mixer_params = self.options.subconfig('mixer_params')
        mixer_params.setdefault('amplitude', 1.e-5)
        disable_finite = 15
        disable_infinite = 50
        decay_finite = 2.
        decay_infinite = decay_finite ** (disable_finite / disable_infinite)
        mixer_params.setdefault('decay', decay_finite if self.finite else decay_infinite)
        mixer_params.setdefault('disable_after', disable_finite if self.finite else disable_infinite)

    def pre_run_initialize(self):
        super().pre_run_initialize()
        E = np.nan
        return E, self.psi

    def run_iteration(self):
        """Perform a single iteration, consisting of ``N_sweeps_check`` sweeps.

        Options
        -------
        .. cfg:configoptions :: DMRGEngine

            E_tol_to_trunc : float
                It's reasonable to choose the Lanczos convergence criteria
                ``'E_tol'`` not many magnitudes lower than the current
                truncation error. Therefore, if `E_tol_to_trunc` is not
                ``None``, we update `E_tol` of `lanczos_params` to
                ``max_E_trunc*E_tol_to_trunc``,
                restricted to the interval [`E_tol_min`, `E_tol_max`],
                where ``max_E_trunc`` is the maximal energy difference due to
                truncation right after each Lanczos optimization during the
                sweeps.
            E_tol_max : float
                See `E_tol_to_trunc`
            E_tol_min : float
                See `E_tol_to_trunc`
            N_sweeps_check : int
                Number of sweeps to perform between checking convergence
                criteria and giving a status update.
            P_tol_to_trunc : float
                It's reasonable to choose the Lanczos convergence criteria
                ``'P_tol'`` not many magnitudes lower than the current
                truncation error. Therefore, if `P_tol_to_trunc` is not
                ``None``, we update `P_tol` of `lanczos_params` to
                ``max_trunc_err*P_tol_to_trunc``,
                restricted to the interval [`P_tol_min`, `P_tol_max`],
                where ``max_trunc_err`` is the maximal truncation error
                (discarded weight of the Schmidt values) due to truncation
                right after each Lanczos optimization during the sweeps.
            P_tol_max : float
                See `P_tol_to_trunc`
            P_tol_min : float
                See `P_tol_to_trunc`
            update_env : int
                Number of sweeps without bond optimization to update the
                environment for infinite boundary conditions,
                performed every `N_sweeps_check` sweeps.

        Returns
        -------
        E : float
            The energy of the current ground state approximation.
        psi : :class:`~tenpy.networks.mps.MPS`
            The current ground state approximation, i.e. just a reference to :attr:`psi`.
        """
        options = self.options
        # parameters for lanczos
        p_tol_to_trunc = options.get('P_tol_to_trunc', 0.05, 'real')
        if p_tol_to_trunc is not None:
            svd_min = self.trunc_params.silent_get('svd_min', 0.)
            svd_min = 0. if svd_min is None else svd_min
            trunc_cut = self.trunc_params.silent_get('trunc_cut', 0.)
            trunc_cut = 0. if trunc_cut is None else trunc_cut
            p_tol_min = max(1.e-30, svd_min**2 * p_tol_to_trunc, trunc_cut**2 * p_tol_to_trunc)
            p_tol_min = options.get('P_tol_min', p_tol_min, 'real')
            p_tol_max = options.get('P_tol_max', 1.e-4, 'real')
        e_tol_to_trunc = options.get('E_tol_to_trunc', None, 'real')
        if e_tol_to_trunc is not None:
            e_tol_min = options.get('E_tol_min', 5.e-16, 'real')
            e_tol_max = options.get('E_tol_max', 1.e-4, 'real')

        # energy and entropy before the iteration:
        if len(self.sweep_stats['E']) < 1:  # first iteration
            E_old = np.nan
            S_old = np.mean(self.psi.entanglement_entropy())
        else:
            E_old = self.sweep_stats['E'][-1]
            S_old = self.sweep_stats['S'][-1]

        # perform sweeps
        logger.info('Running sweep with optimization')
        for i in range(self.N_sweeps_check - 1):
            self.sweep(meas_E_trunc=False)
        max_trunc_err = self.sweep(meas_E_trunc=True)
        max_E_trunc = np.max(self.E_trunc_list)

        # update lanczos_params depending on truncation error(s)
        if p_tol_to_trunc is not None and max_trunc_err > p_tol_min:
            P_tol = max(p_tol_min, min(p_tol_max, max_trunc_err * p_tol_to_trunc))
            self.lanczos_params['P_tol'] = P_tol
            self.lanczos_params.touch('P_tol')  # don't warn about unused P_tol, since
            # the optimization might not even use the normal lanczos function.
            logger.debug("set lanczos_params['P_tol'] = %.2e", P_tol)
        if e_tol_to_trunc is not None and max_E_trunc > e_tol_min:
            E_tol = max(e_tol_min, min(e_tol_max, max_E_trunc * e_tol_to_trunc))
            self.lanczos_params['E_tol'] = E_tol
            self.lanczos_params.touch('E_tol')
            logger.debug("set lanczos_params['E_tol'] = %.2e", E_tol)

        # update environment
        if not self.finite:
            update_env = options.get('update_env', self.N_sweeps_check // 2, int)
            self.environment_sweeps(update_env)

        # update statistics
        entropy_bonds = self._entropy_approx
        if self.finite:
            entropy_bonds = entropy_bonds[1:]
        max_S = max(entropy_bonds)
        S = np.mean(entropy_bonds)
        if not self.finite:  # iDMRG: need energy density
            Es = self.update_stats['E_total']
            age = self.update_stats['age']
            delta = min(1 + 2 * self.env.L, len(age))
            growth = (age[-1] - age[-delta])
            E = (Es[-1] - Es[-delta]) / growth
        else:
            E = self.update_stats['E_total'][-1]
        norm_err = np.linalg.norm(self.psi.norm_test())

        self.sweep_stats['sweep'].append(self.sweeps)
        self.sweep_stats['N_updates'].append(len(self.update_stats['i0']))
        self.sweep_stats['E'].append(E)
        self.sweep_stats['Delta_E'].append((E - E_old) / self.N_sweeps_check)
        self.sweep_stats['S'].append(S)
        self.sweep_stats['Delta_S'].append((S - S_old) / self.N_sweeps_check)
        self.sweep_stats['max_S'].append(max_S)
        self.sweep_stats['time'].append(time.time() - self.time0)
        self.sweep_stats['max_trunc_err'].append(max_trunc_err)
        self.sweep_stats['max_E_trunc'].append(max_E_trunc)
        self.sweep_stats['max_chi'].append(np.max(self.psi.chi))
        self.sweep_stats['norm_err'].append(norm_err)

        return E, self.psi

    def status_update(self, iteration_start_time: float):
        logger.info(
            "checkpoint after sweep %(sweeps)d\n"
            "energy=%(E).16f, max S=%(max_S).16f, age=%(age)d, norm_err=%(norm_err).1e\n"
            "Current memory usage %(mem).1fMB, wall time: %(wall_time).1fs\n"
            "Delta E = %(dE).4e, Delta S = %(dS).4e (per sweep)\n"
            "max trunc_err = %(trunc_err).4e, max E_trunc = %(E_trunc).4e\n"
            "chi: %(chi)s\n"
            "%(sep)s", {
                'sweeps': self.sweeps,
                'E': self.sweep_stats['E'][-1],
                'max_S': self.sweep_stats['max_S'][-1],
                'age': self.update_stats['age'][-1],
                'norm_err': self.sweep_stats['norm_err'][-1],
                'mem': memory_usage(),
                'wall_time': time.time() - iteration_start_time,
                'dE': self.sweep_stats['Delta_E'][-1],
                'dS': self.sweep_stats['Delta_S'][-1],
                'trunc_err': self.sweep_stats['max_trunc_err'][-1],
                'E_trunc': self.sweep_stats['max_E_trunc'][-1],
                'chi': self.psi.chi if self.psi.L < 40 else max(self.psi.chi),
                'sep': "=" * 80,
            })

    def is_converged(self):
        """Determines if the algorithm is converged.

        Does not cover any other reasons to abort, such as reaching a time limit.
        Such checks are covered by :meth:`stopping_condition`.

        Options
        -------
        .. cfg:configoptions :: DMRGEngine

            max_E_err : float
                Convergence if the change of the energy in each step
                satisfies ``|Delta E / max(E, 1)| < max_E_err``. Note that
                this might be satisfied even if ``Delta E > 0``,
                i.e., if the energy increases (due to truncation).
            max_S_err : float
                Convergence if the relative change of the entropy in each step
                satisfies ``|Delta S|/S < max_S_err``
        """
        max_E_err = self.options.get('max_E_err', 1.e-8, 'real')
        max_S_err = self.options.get('max_S_err', 1.e-5, 'real')
        E = self.sweep_stats['E'][-1]
        Delta_E = self.sweep_stats['Delta_E'][-1]
        Delta_S = self.sweep_stats['Delta_S'][-1]
        return abs(Delta_E / max(E, 1.)) < max_E_err and abs(Delta_S) < max_S_err

    def post_run_cleanup(self):
        """Perform any final steps or clean up after the main loop has terminated.

        Options
        -------
        .. cfg:configoptions :: DMRGEngine

            norm_tol : float
                After the DMRG run, update the environment with at most
                `norm_tol_iter` sweeps until
                ``np.linalg.norm(psi.norm_err()) < norm_tol``.
            norm_tol_iter : float
                Perform at most `norm_tol_iter`*`update_env` sweeps to
                converge the norm error below `norm_tol`.
            norm_tol_final : float
                After performing `norm_tol_iter`*`update_env` sweeps, if
                ``np.linalg.norm(psi.norm_err()) < norm_tol_final``, call
                :meth:`~tenpy.networks.mps.canonical_form` to canonicalize
                instead. This tolerance should be stricter than `norm_tol`
                to ensure canonical form even if DMRG cannot fully converge.

        """
        super().post_run_cleanup()
        self._canonicalize(True)
        logger.info(f'{self.__class__.__name__} finished after {self.sweeps} sweeps, '
                    f'max chi={max(self.psi.chi)}')
        if (len(self.ortho_to_envs) > 0) and (self.sweep_stats['E'][-1] > -1e-8):
            msg = (f'{self.__class__.__name__} with orthogonal_to, i.e. searching for excited '
                   f'states, terminated with an energy consistent with zero. '
                   f'Orthogonality can not be guaranteed. Consider adding a negative constant to '
                   f'the Hamiltonian such that the target state has negative energy. '
                   f'See https://github.com/tenpy/tenpy/issues/329 for more information.')
            # stacklevel: (1) this
            #             (2) DMRGEngine.run()
            #             (3) IterativeSweeps.run()
            #             (4) user context
            warnings.warn(msg, stacklevel=4)

    def run(self):
        """Run the DMRG simulation to find the ground state.

        Returns
        -------
        E : float
            The energy of the resulting ground state MPS.
        psi : :class:`~tenpy.networks.mps.MPS`
            The MPS representing the ground state after the simulation,
            i.e. just a reference to :attr:`psi`.
        """
        return super().run()

    def _canonicalize(self, warn=False):
        #Update environment until norm_tol is reached. If norm_tol_final
        #is not reached, call canonical_form.
        if self.mixer is not None:
            return
        norm_err = np.linalg.norm(self.psi.norm_test())
        norm_tol = self.options.get('norm_tol', 1.e-5, 'real')
        norm_tol_final = self.options.get('norm_tol_final', 1.e-10, 'real')
        if not self.finite:
            update_env = self.options['update_env']
            norm_tol_iter = self.options.get('norm_tol_iter', 5, int)
        if norm_tol is None or (norm_err < norm_tol and norm_err < norm_tol_final):
            return
        if warn and norm_err > norm_tol:
            logger.warning(
                "final DMRG state not in canonical form up to "
                "norm_tol=%.2e: norm_err=%.2e", norm_tol, norm_err)
        if norm_err > norm_tol and not self.finite:
            for _ in range(norm_tol_iter):
                self.environment_sweeps(update_env)
                norm_err = np.linalg.norm(self.psi.norm_test())
                if norm_err <= norm_tol:
                    break
            else:
                logger.warning(
                    "norm_err=%.2e still too high after environment_sweeps", norm_err)
        if norm_err > norm_tol_final:
            self._resume_psi = self.psi.copy()
            if warn and not self.finite:
                logger.warning(
                "final DMRG state not in canonical form up to "
                "norm_tol_final=%.2e: norm_err=%.2e, "
                "calling psi.canonical_form()", norm_tol_final, norm_err)
            self.psi.canonical_form()

    def reset_stats(self, resume_data=None):
        """Reset the statistics, useful if you want to start a new sweep run."""
        super().reset_stats(resume_data)
        self.update_stats = {
            'i0': [],
            'age': [],
            'E_total': [],
            'N_lanczos': [],
            'time': [],
            'err': [],
            'E_trunc': [],
            'ov_change': []
        }
        self.sweep_stats = {
            'sweep': [],
            'N_updates': [],
            'E': [],
            'Delta_E': [],
            'S': [],
            'Delta_S': [],
            'max_S': [],
            'time': [],
            'max_trunc_err': [],
            'max_E_trunc': [],
            'max_chi': [],
            'norm_err': []
        }

    def sweep(self, optimize=True, meas_E_trunc=False):
        """One 'sweep' of the algorithm.

        Thin wrapper around :meth:`tenpy.algorithms.mps_common.Sweep.sweep` with one additional
        parameter `meas_E_trunc` specifying whether to measure truncation energies.
        """
        self._meas_E_trunc = meas_E_trunc
        return super().sweep(optimize)

    def update_local(self, theta, optimize=True):
        """Perform site-update on the site ``i0``.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Initial guess for the ground state of the effective Hamiltonian.
        optimize : bool
            Whether we actually optimize to find the ground state of the effective Hamiltonian.
            (If False, just update the environments).

        Returns
        -------
        update_data : dict
            Data computed during the local update, as described in the following:

            E0 : float
                Total energy, obtained *before* truncation (if ``optimize=True``),
                or *after* truncation (if ``optimize=False``) (but never ``None``).
            N : int
                Dimension of the Krylov space used for optimization in the lanczos algorithm.
                0 if ``optimize=False``.
            age : int
                Current size of the DMRG simulation: number of physical sites involved
                into the contraction.
            U, VH: :class:`~tenpy.linalg.np_conserved.Array`
                `U` and `VH` returned by :meth:`mixed_svd`.
            ov_change: float
                Change in the wave function ``1. - abs(<theta_guess|theta>)``
                induced by :meth:`diag`, *not* including the truncation!
        """
        i0 = self.i0
        n_opt = self.n_optimize
        age = self.env.get_LP_age(i0) + n_opt + self.env.get_RP_age(i0 + n_opt - 1)
        if optimize:
            E0, theta, N, ov_change = self.diag(theta)
        else:
            E0, N, ov_change = None, 0, 0.
        theta = self.prepare_svd(theta)
        U, S, VH, err, S_approx = self.mixed_svd(theta)
        self._entropy_approx[(i0 + n_opt - 1) % self.psi.L] = entropy(S_approx**2)
        self.set_B(U, S, VH)
        update_data = {
            'E0': E0,
            'err': err,
            'N': N,
            'age': age,
            'U': U,
            'VH': VH,
            'ov_change': ov_change
        }
        return update_data

    def post_update_local(self, E0, age, N, ov_change, err, **update_data):
        """Perform post-update actions.

        Compute truncation energy and collect statistics.

        Parameters
        ----------
        **update_data : dict
            What was returned by :meth:`update_local`.
        """
        E0 = E0
        i0 = self.i0
        E_trunc = None
        if self._meas_E_trunc or E0 is None:
            i = i0 if self.n_optimize == 2 or self.move_right else i0 - 1
            E_trunc = self.env.full_contraction(i).real  # uses updated LP/RP (if calculated)
            if E0 is None:
                E0 = E_trunc
            E_trunc = E_trunc - E0

        # collect statistics
        self.update_stats['i0'].append(i0)
        self.update_stats['age'].append(age)
        self.update_stats['E_total'].append(E0)
        self.update_stats['E_trunc'].append(E_trunc)
        self.update_stats['N_lanczos'].append(N)
        self.update_stats['ov_change'].append(ov_change)
        self.update_stats['err'].append(err)
        self.update_stats['time'].append(time.time() - self.time0)
        self.trunc_err_list.append(err.eps)
        self.E_trunc_list.append(E_trunc)

        if self.psi.bc == 'segment':
            self.update_segment_boundaries()

    def update_segment_boundaries(self):
        """Update the singular values at the boundaries of the segment.

        This method is called at the end of :meth:`post_update_local` for 'segment' boundary MPS.
        It just updates the singular values on the very left/right end of the MPS segment.
        """
        psi = self.psi
        if self.i0 == 0 and self.move_right:
            # need to update bond to the left of site j=0
            j = 0
            A = psi.get_B(j, form='A')
            th = psi.get_B(j, form='Th')
            U, S, V = npc.svd(th.combine_legs(psi._p_label + ['vR'], qconj=-1),
                              cutoff=0,
                              qtotal_LR=[None, th.qtotal],
                              inner_labels=['vR', 'vL'])
            S = S / np.linalg.norm(S)
            psi.set_SL(j, S)
            A_new = npc.tensordot(U.conj().replace_label('vR*', 'vL'), A, ['vL*', 'vL'])
            psi.set_B(j, A_new, form='A')

            old_UL, old_VR = psi.segment_boundaries
            new_UL = npc.tensordot(old_UL, U, axes=['vR', 'vL'])
            psi.segment_boundaries = (new_UL, old_VR)

            for env in self._all_envs:
                update_ket = env.ket is psi
                update_bra = env.bra is psi
                env._update_gauge_LP(j, U, update_bra, update_ket)
            # No need to clear the environments on the other bonds!

        elif self.i0 == psi.L - self.EffectiveH.length and not self.move_right:
            # need to update bond on the right of site j=L-1
            j = psi.L - 1
            B = psi.get_B(j, form='B')
            th = psi.get_B(j, form='Th')
            U, S, V = npc.svd(th.combine_legs(['vL'] + psi._p_label, qconj=+1),
                              cutoff=0,
                              qtotal_LR=[th.qtotal, None],
                              inner_labels=['vR', 'vL'])
            S = S / np.linalg.norm(S)
            psi.set_SR(j, S)
            B_new = npc.tensordot(B, V.conj().replace_label('vL*', 'vR'), ['vR', 'vR*'])
            psi.set_B(j, B_new, form='B')

            old_UL, old_VR = psi.segment_boundaries
            new_VR = npc.tensordot(V, old_VR, axes=['vR', 'vL'])
            psi.segment_boundaries = (old_UL, new_VR)

            for env in self._all_envs:
                update_ket = env.ket is psi
                update_bra = env.bra is psi
                env._update_gauge_RP(j, V, update_bra, update_ket)
            # No need to clear the environments on the other bonds!

    def diag(self, theta_guess):
        """Diagonalize the effective Hamiltonian represented by self.

        .. cfg:configoptions :: DMRGEngine

            max_N_for_ED : int
                Maximum matrix dimension of the effective hamiltonian
                up to which the ``'default'`` `diag_method` uses ED instead of
                Lanczos.
            diag_method : str
                One of the following strings:

                'default'
                      Same as ``'lanczos'`` for large bond dimensions, but if the
                      total dimension of the effective Hamiltonian does not exceed
                      the DMRG parameter ``'max_N_for_ED'`` it uses ``'ED_block'``.
                'lanczos'
                      :func:`~tenpy.linalg.lanczos.lanczos`
                      Default, the Lanczos implementation in TeNPy.
                'arpack'
                      :func:`~tenpy.linalg.lanczos.lanczos_arpack`
                      Based on :func:`scipy.linalg.sparse.eigsh`.
                      Slower than 'lanczos', since it needs to convert the npc arrays
                      to numpy arrays during *each* matvec, and possibly does many
                      more iterations.
                'ED_block'
                      :func:`full_diag_effH`
                      Contract the effective Hamiltonian to a (large!) matrix and
                      diagonalize the block in the charge sector of the initial state.
                      Preserves the charge sector of the explicitly conserved charges.
                      However, if you don't preserve a charge explicitly, it can break
                      it.
                      For example if you use a ``SpinChain({'conserve': 'parity'})``,
                      it could change the total "Sz", but not the parity of 'Sz'.
                'ED_all'
                      :func:`full_diag_effH`
                      Contract the effective Hamiltonian to a (large!) matrix and
                      diagonalize it completely.
                      Allows to change the charge sector *even for explicitly
                      conserved charges*.
                      For example if you use a ``SpinChain({'conserve': 'Sz'})``,
                      it **can** change the total "Sz".

        Parameters
        ----------
        theta_guess : :class:`~tenpy.linalg.np_conserved.Array`
            Initial guess for the ground state of the effective Hamiltonian.

        Returns
        -------
        E0 : float
            Energy of the found ground state.
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Ground state of the effective Hamiltonian.
        N : int
            Number of Lanczos iterations used. ``-1`` if unknown.
        ov_change : float
            Change in the wave function ``1. - abs(<theta_guess|theta_diag>)``
        """
        N = -1  # (unknown)

        if self.diag_method == 'default':
            # use ED for small matrix dimensions, but lanczos by default
            max_N = self.options.get('max_N_for_ED', 400, int)
            if self.eff_H.N < max_N:
                E, theta = full_diag_effH(self.eff_H, theta_guess, keep_sector=True)
            else:
                E, theta, N = LanczosGroundState(self.eff_H, theta_guess, self.lanczos_params).run()
        elif self.diag_method == 'lanczos':
            E, theta, N = LanczosGroundState(self.eff_H, theta_guess, self.lanczos_params).run()
        elif self.diag_method == 'arpack':
            E, theta = lanczos_arpack(self.eff_H, theta_guess, self.lanczos_params)
        elif self.diag_method == 'ED_block':
            E, theta = full_diag_effH(self.eff_H, theta_guess, keep_sector=True)
        elif self.diag_method == 'ED_all':
            E, theta = full_diag_effH(self.eff_H, theta_guess, keep_sector=False)
        else:
            raise ValueError("Unknown diagonalization method: " + repr(self.diag_method))
        ov_change = 1. - abs(npc.inner(theta_guess, theta, 'labels', do_conj=True))
        return E, theta, N, ov_change

    def plot_update_stats(self, axes, xaxis='time', yaxis='E', y_exact=None, **kwargs):
        """Plot :attr:`update_stats` to display the convergence during the sweeps.

        Parameters
        ----------
        axes : :class:`matplotlib.axes.Axes`
            The axes to plot into. Defaults to :func:`matplotlib.pyplot.gca()`
        xaxis : ``'N_updates' | 'sweep'`` | keys of :attr:`update_stats`
            Key of :attr:`update_stats` to be used for the x-axis of the plots.
            ``'N_updates'`` is just enumerating the number of bond updates,
            and ``'sweep'`` corresponds to the sweep number (including environment sweeps).
        yaxis : ``'E'`` | keys of :attr:`update_stats`
            Key of :attr:`update_stats` to be used for the y-axis of the plots.
            For 'E', use the energy (per site for infinite systems).
        y_exact : float
            Exact value for the quantity on the y-axis for comparison.
            If given, plot ``abs((y-y_exact)/y_exact)`` on a log-scale yaxis.
        **kwargs :
            Further keyword arguments given to ``axes.plot(...)``.
        """
        if axes is None:
            import matplotlib.pyplot as plt
            axes = plt.gca()
        stats = self.update_stats
        L = self.psi.L
        kwargs.setdefault('marker', 'x')
        kwargs.setdefault('linestyle', '-')

        E = np.array(stats['E_total'])
        schedule = list(self.get_sweep_schedule())
        N = len(schedule)  # bond updates per sweep
        if xaxis is None or xaxis == 'N_updates' or xaxis == 'index':
            xaxis = 'N_updates'
            x = np.arange(len(E))
        elif xaxis == 'sweep':
            x = np.arange(1, len(E) + 1) / N
        else:
            x = np.array(stats[xaxis])
        if yaxis == 'E':
            if not self.psi.finite:
                # use energy per site instead of total energy
                age = np.array(stats['age'])
                d_age = age[N:] - age[:-N]
                d_E = E[N:] - E[:-N]
                y = d_E / d_age
                x = x[N:]
            else:
                y = E
        else:
            y = np.array(stats[yaxis])
        if y_exact is not None:
            y = np.abs(y - y_exact) / np.abs(y_exact)
            axes.set_yscale('log')
        axes.plot(x, y, **kwargs)
        axes.set_xlabel(xaxis)
        axes.set_ylabel(yaxis)

    def plot_sweep_stats(self, axes=None, xaxis='time', yaxis='E', y_exact=None, **kwargs):
        """Plot :attr:`sweep_stats` to display the convergence with the sweeps.

        Parameters
        ----------
        axes : :class:`matplotlib.axes.Axes`
            The axes to plot into. Defaults to :func:`matplotlib.pyplot.gca()`
        xaxis, yaxis : key of :attr:`sweep_stats`
            Key of :attr:`sweep_stats` to be used for the x-axis and y-axis of the plots.
        y_exact : float
            Exact value for the quantity on the y-axis for comparison.
            If given, plot ``abs((y-y_exact)/y_exact)`` on a log-scale yaxis.
        **kwargs :
            Further keyword arguments given to ``axes.plot(...)``.
        """
        if axes is None:
            import matplotlib.pyplot as plt
            axes = plt.gca()
        stats = self.sweep_stats
        L = self.psi.L
        kwargs.setdefault('marker', 'x')
        kwargs.setdefault('linestyle', '-')

        x = np.array(stats[xaxis])
        y = np.array(stats[yaxis])
        if y_exact is not None:
            y = np.abs(y - y_exact) / np.abs(y_exact)
            axes.set_yscale('log')
        axes.plot(x, y, **kwargs)
        axes.set_xlabel(xaxis)
        axes.set_ylabel(yaxis)


class TwoSiteDMRGEngine(DMRGEngine):
    """Engine for the two-site DMRG algorithm.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        Initial guess for the ground state, which is to be optimized in-place.
    model : :class:`~tenpy.models.MPOModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    options : dict
        Further optional parameters.

    Options
    -------
    .. cfg:config :: TwoSiteDMRGEngine
        :include: DMRGEngine

    """
    EffectiveH = TwoSiteH
    DefaultMixer = mps_common.DensityMatrixMixer
    use_mixer_by_default = False

    def prepare_svd(self, theta):
        """Transform theta into matrix for svd."""
        if self.combine:
            return theta  # Theta is already combined.
        else:
            return theta.combine_legs([['vL', 'p0'], ['p1', 'vR']],
                                      new_axes=[0, 1],
                                      qconj=[+1, -1])

    def mixed_svd(self, theta):
        """Get (truncated) `B` from the new theta (as returned by diag).

        The goal is to split theta and truncate it::

            |   -- theta --   ==>    -- U -- S --  VH -
            |      |   |                |          |

        Without a mixer, this is done by a simple svd and truncation of Schmidt values.

        With a mixer, the state is perturbed before the SVD. The details of the perturbation are
        defined by the :class:`~tenpy.algorithms.mps_common.Mixer` class.

        Note that the returned `S` is a general (not diagonal) matrix, with labels ``'vL', 'vR'``.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            The optimized wave function, prepared for svd.

        Returns
        -------
        U : :class:`~tenpy.linalg.np_conserved.Array`
            Left-canonical part of `theta`. Labels ``'(vL.p)', 'vR'``.
        S : 1D ndarray | 2D :class:`~tenpy.linalg.np_conserved.Array`
            Without mixer just the singular values of the array; with mixer it might be a general
            matrix with labels ``'vL', 'vR'``; see comment above.
        VH : :class:`~tenpy.linalg.np_conserved.Array`
            Right-canonical part of `theta`. Labels ``'vL', '(p.vR)'``.
        err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The truncation error introduced.
        S_approx : ndarray
            Just the `S` if a 1D ndarray, or an approximation of the correct S (which was used for
            truncation) in case `S` is 2D Array.
        """
        i0 = self.i0
        update_LP, update_RP = self.update_LP_RP
        mixer = self.mixer
        if mixer is None:
            qtotal_i0 = self.env.bra.get_B(i0, form=None).qtotal
            U, S, VH, err, _ = svd_theta(
                theta, self.trunc_params, qtotal_LR=[qtotal_i0, None], inner_labels=['vR', 'vL']
            )
            S_a = S
        else:
            qtotal_LR = [self.psi.get_B(i0, form=None).qtotal,
                         self.psi.get_B(i0 + 1, form=None).qtotal]
            U, S, VH, err, S_a = mixer.mix_and_decompose_2site(
                engine=self, theta=theta, i0=self.i0, mix_left=update_LP, mix_right=update_RP,
                qtotal_LR=qtotal_LR
            )
        U.ireplace_label('(vL.p0)', '(vL.p)')
        VH.ireplace_label('(p1.vR)', '(p.vR)')
        return U, S, VH, err, S_a

    def set_B(self, U, S, VH):
        """Update the MPS with the ``U, S, VH`` returned by `self.mixed_svd`.

        Parameters
        ----------
        U, VH : :class:`~tenpy.linalg.np_conserved.Array`
            Left and Right-canonical matrices as returned by the SVD.
        S : 1D array | 2D :class:`~tenpy.linalg.np_conserved.Array`
            The middle part returned by the SVD, ``theta = U S VH``.
            Without a mixer just the singular values, with enabled `mixer` a 2D array.
        """
        B0 = U.split_legs(['(vL.p)'])
        B1 = VH.split_legs(['(p.vR)'])
        i0 = self.i0
        self.psi.set_B(i0, B0, form='A')  # left-canonical
        self.psi.set_B(i0 + 1, B1, form='B')  # right-canonical
        self.psi.set_SR(i0, S)
        # environments are cleaned/updated in :meth:`update_env`


class SingleSiteDMRGEngine(DMRGEngine):
    """Engine for the single-site DMRG algorithm.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        Initial guess for the ground state, which is to be optimized in-place.
    model : :class:`~tenpy.models.MPOModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    options : dict
        Further optional parameters.

    Options
    -------
    .. cfg:config :: SingleSiteDMRGEngine
        :include: DMRGEngine

    """
    EffectiveH = OneSiteH
    DefaultMixer = mps_common.SubspaceExpansion
    use_mixer_by_default = True

    def prepare_svd(self, theta):
        """Transform theta into matrix for svd.

        In contrast with the 2-site engine, the matrix here depends on the direction we move, as we
        need `'p'` to point away from the direction we are going in.
        """
        if self.combine:
            if self.move_right:
                theta.itranspose(['(vL.p0)', 'vR'])  # ensure the order.
            else:
                theta.itranspose(['vL', '(p0.vR)'])  # ensure the order.
        else:
            if self.move_right:
                theta = theta.combine_legs(['vL', 'p0'], qconj=+1, new_axes=0)
            else:
                theta = theta.combine_legs(['p0', 'vR'], qconj=-1, new_axes=1)
        return theta

    def mixed_svd(self, theta):
        """Get (truncated) `B` from the new theta (as returned by diag).

        The goal is to split theta and truncate it. For a move to the right::

            |             -- theta -- next_B --   ==>    -- U -- S -- VH --
            |                  |        |                   |         |

        For a move to the left::

            |   -- next_A -- theta --   ==>    -- U -- S -- VH --
            |        |         |                  |         |

        Note that `theta` lives on the same site :attr:`i0` in both cases,
        but the sites of `next_A` and `next_B` depend on whether we move right or left.
        The returned `U` and `VH` have the same labels independent of that.

        Without a mixer, this is done by a simple svd and truncation of Schmidt values of theta
        followed by the absorption of `VH` into `next_B` (`U` into `next_A`).

        With a mixer, the state/density matrix is perturbed before the SVD.
        The details of the perturbation are defined by the :class:`Mixer` class.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            The optimized wave function, prepared for svd with :meth:`prepare_svd`,
            i.e., with combined legs.

        Returns
        -------
        U : :class:`~tenpy.linalg.np_conserved.Array`
            Left-canonical part of `theta`. Labels ``'(vL.p)', 'vR'``
        S : 1D ndarray | 2D :class:`~tenpy.linalg.np_conserved.Array`
            Without mixer just the singular values of the array; with mixer it might be a general
            matrix with labels ``'vL', 'vR'``; see comment above.
        VH : :class:`~tenpy.linalg.np_conserved.Array`
            Right-canonical part of `theta`. Labels ``'vL', '(p.vR)'``.
        err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The truncation error introduced.
        S_approx : ndarray
            Just the `S` if a 1D ndarray, or an approximation of the correct S (which was used for
            truncation) in case `S` is 2D Array.
        """
        mixer = self.mixer
        move_right = self.move_right
        update_LP, update_RP = self.update_LP_RP
        if self.move_right:
            next_B = self.psi.get_B(self.i0 + 1, form='B')
            next_B = next_B.combine_legs(['p', 'vR'], qconj=-1, new_axes=1)
            if update_RP:
                # make sure that `next_B` is in right-canonical form
                assert self.psi.form[(self.i0 + 1) % self.psi.L] == (0., 1.)
        else:
            next_A = self.psi.get_B(self.i0 - 1, form='A')
            next_A = next_A.combine_legs(['vL', 'p'], qconj=1, new_axes=0)
            if update_LP:
                # make sure that `next_A` is in left-canonical form
                assert self.psi.form[(self.i0 - 1) % self.psi.L] == (1., 0.)

        if mixer is None:
            qtotal = [theta.qtotal, None] if move_right else [None, theta.qtotal]
            U, S, VH, err, _ = svd_theta(theta,
                                         self.trunc_params,
                                         qtotal_LR=qtotal,
                                         inner_labels=['vR', 'vL'])
            S_a = S
            # absorb VH/U into next_B/next_A for right/left move
            if move_right:
                # VH is at most truncation, so VH-next_B is still right-canonical,
                # (unless next_B wasn't, but then we don't need to update_RP)
                VH = npc.tensordot(VH, next_B, ['vR', 'vL'])
                U.ireplace_label('(vL.p0)', '(vL.p)')
            else:
                # U is at most truncation, so next_A-U is still left-canonical,
                # (unless next_A wasn't, but then we don't need to update_RP)
                U = npc.tensordot(next_A, U, ['vR', 'vL'])
                VH.ireplace_label('(p0.vR)', '(p.vR)')
        elif mixer.can_decompose_1site:
            U, S, VH, err = mixer.mix_and_decompose_1site(
                engine=self, theta=theta, i0=self.i0, move_right=move_right
            )
            S_a = S
            # absorb VH/U into S
            if move_right:
                # note: if update_RP, the `next_B` is a right-canonical B from the MPS.
                # Hence we *did* a subspace expansion on it, during the update when we put it
                # into the MPS.
                if isinstance(S, npc.Array):
                    S = npc.tensordot(S, VH, ['vR', 'vL'])
                else:
                    S = VH.iscale_axis(S, 'vL')
                VH = next_B
                U.ireplace_label('(vL.p0)', '(vL.p)')
            else:
                if isinstance(S, npc.Array):
                    S = npc.tensordot(U, S, ['vR', 'vL'])
                else:
                    S = U.iscale_axis(S, 'vR')
                U = next_A
                VH.ireplace_label('(p0.vR)', '(p.vR)')
        else:
            # just use two-site theta
            if self.move_right:
                next_B.ireplace_label('(p.vR)', '(p1.vR)')
                theta = npc.tensordot(theta, next_B, axes=['vR', 'vL'])
                i0 = self.i0
            else:
                next_A.ireplace_label('(vL.p)', '(vL.p0)')
                theta.ireplace_label('(p0.vR)', '(p1.vR)')
                theta = npc.tensordot(next_A, theta, axes=['vR', 'vL'])
                i0 = self.i0 - 1
            qtotal_LR = [self.psi.get_B(i0, form=None).qtotal,
                         self.psi.get_B(i0 + 1, form=None).qtotal]
            U, S, VH, err, S_a = mixer.mixed_svd_2site(
                engine=self, theta=theta, i0=i0, mix_left=update_LP, mix_right=update_RP,
                qtotal_LR=qtotal_LR
            )
            U.ireplace_label('(vL.p0)', '(vL.p)')
            VH.ireplace_label('(p1.vR)', '(p.vR)')
        return U, S, VH, err, S_a

    def set_B(self, U, S, VH):
        """Update the MPS with the ``U, S, VH`` returned by `self.mixed_svd`.

        Parameters
        ----------
        U, VH : :class:`~tenpy.linalg.np_conserved.Array`
            Left and Right-canonical matrices as returned by the SVD.
        S : 1D array | 2D :class:`~tenpy.linalg.np_conserved.Array`
            The middle part returned by the SVD, ``theta = U S VH``.
            Without a mixer just the singular values, with enabled `mixer` a 2D array.
        """
        i_L, i_R = self._update_env_inds()  # left and right updated sites
        A0 = U.split_legs(['(vL.p)'])
        B1 = VH.split_legs(['(p.vR)'])
        self.psi.set_B(i_L, A0, form='A')  # left-canonical
        self.psi.set_B(i_R, B1, form='B')  # right-canonical
        self.psi.set_SR(i_L, S)
        # environments are cleaned/updated in :meth:`update_env`

    def mixer_activate(self):
        super().mixer_activate()
        if not self.mixer.can_decompose_1site:
            msg = (f'Using {self.mixer.__class__.__name__} with single-site DMRG is inefficient. '
                   f'The resulting algorithm has two-site costs!')
            warnings.warn(msg)


def chi_list(chi_max, dchi=20, nsweeps=20):
    """Compute a 'ramping-up' chi_list.

    The resulting chi_list allows to increases `chi` by `dchi` every `nsweeps` sweeps up to a given
    maximal `chi_max`.

    Parameters
    ----------
    chi_max : int
        Final value for the bond dimension.
    dchi :int
        Step size how to increase chi
    nsweeps : int
        Step size for sweeps

    Returns
    -------
    chi_list : dict
        To be used as `chi_list` parameter for DMRG, see :func:`run`.
        Keys increase by `nsweeps`, values by `dchi`, until a maximum of `chi_max` is reached.
    """
    chi_max = int(chi_max)
    nsweeps = int(nsweeps)
    if chi_max < dchi:
        return {0: chi_max}
    chi_list = {}
    for i in range(chi_max // dchi):
        chi = int(dchi * (i + 1))
        chi_list[nsweeps * i] = chi
    if chi < chi_max:
        chi_list[nsweeps * (i + 1)] = chi_max
    return chi_list


def full_diag_effH(effH, theta_guess, keep_sector=True):
    """Perform an exact diagonalization of `effH`.

    This function offers an alternative to :func:`~tenpy.linalg.lanczos.lanczos`.

    Parameters
    ----------
    effH : :class:`~tenpy.algorithms.mps_common.EffectiveH`
        The effective Hamiltonian.
    theta_guess : :class:`~tenpy.linalg.np_conserved.Array`
        Current guess to select the charge sector. Labels as specified by ``effH.acts_on``.
    """
    theta_guess = theta_guess.combine_legs(effH.acts_on, qconj=+1)
    fullH = effH.to_matrix()
    if keep_sector:
        # diagonalize only the block of the charge sector in which `theta_guess` is.
        leg = theta_guess.legs[0]
        qi = leg.get_qindex_of_charges(theta_guess.qtotal)
        block = fullH.get_block(np.array([qi, qi], np.intp))
        if block is None:
            warnings.warn("H is zero in the given block, nothing to diagonalize."
                          "We just return the initial state again.")
            E0 = 0
            theta = theta_guess
        else:
            E, V = np.linalg.eigh(block)
            E0 = E[0]
            theta = theta_guess.zeros_like()
            theta.dtype = np.promote_types(fullH.dtype, theta_guess.dtype)
            theta_block = theta.get_block(np.array([qi], np.intp), insert=True)
            theta_block[:] = V[:, 0]  # copy data into theta
    else:  # allow to change charge sector!
        E, V = npc.eigh(fullH)
        i0 = np.argmin(E)
        E0 = E[i0]
        theta = V.take_slice(i0, 1)
    theta = theta.split_legs([0]).iset_leg_labels(effH.acts_on)
    return E0, theta
