"""Variational Uniform Matrix Product State (VUMPS)

The VUMPS algorithm was developed to find ground states directly in the thermodynamic
limit in a more principled fashion than iDMRG, which essentially is a finite algorithm
extrapolated to the thermodynamic limit. VUMPS is a tangent space MPS method, where
we look for the optimal ground state in the manifold of fixed bond dimensionMPS :cite:`vanderstraeten2019`.

The VUMPS algorithm was introduced in 2017 :cite:`zauner-stauber2018`, where it was shown
that VUMPS outperforms both iTEBD and iDMRG as ground state search algorithms for both
1D and quasi-1D models. The VUMPS algorithm uses the network :class:`~tenpy.networks.uniform_mps.UniformMPS` to represent
the current state of the uniform MPS during optimization. On each site, left canonical AL,
right canonical AR, and single-site orthogonality center AC is stored; on each bond a center
matrix C is stored. During the algorithm, the canonical form equality AL_i C_{i+1} = C_i AR_i = AC_i
is not necessarily respected, yet in the ground state, we expect this to be restored. The
difference in norm of the first two is the ``'max_split_error'`` that is reported at each checkpoint.

There are two derived classes that implement the vumps algorithm, :class:`SingleSiteVUMPSEngine` and
:class:`TwoSiteVUMPSEngine`. The first implements the algorithm found in the original paper,
where the uMPS is updated by 2 zero-site eigenvalue problems and 1 one-site eigenvalue problem.
This algorithm works at FIXED bond dimension, so the starting uMPS must be prepared with desired
bond dimension. Currently we do this with ``'MPS.from_desired_bond_dimension'`` which is NOT
compatible with charge conservation, as it is implemented with random unitaries to artificially
grow $chi$. The single-site algorithm allows for optimization of a translationally invariant uMPS
with a single-site unit cell, which is currently not possible in our iDMRG implementation.

Charge conservation and dynamic control of bond dimension is enabled by a novel two-site algorithm, in
which we solve a two-site eigenvalue problem and 2 one-site eigenvalue problems. This algorithm involves
an SVD, which allows us to dynamically grow the bond dimension. Thus, we can start the algorithm in a
product state with charge conservation and enlarge the bond dimension based on max_chi or SVD cutoff.
Best practices for multi-site unit cell uMPS would be to start with the 2-site algorithm and switch to
the single-site algorithm, which is the more principled algorithm.

"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

from ..linalg import np_conserved as npc
from ..networks.mpo import MPOEnvironment, MPOTransferMatrix
from ..networks.mps import MPS
from ..networks.uniform_mps import UniformMPS
from ..linalg.sparse import SumNpcLinearOperator
from ..algorithms.mps_common import DensityMatrixMixer, SubspaceExpansion
from ..linalg.krylov_based import LanczosGroundState
from ..tools.math import entropy
from ..tools.process import memory_usage
from .mps_common import IterativeSweeps, ZeroSiteH, OneSiteH, TwoSiteH
from .truncation import svd_theta
from .plane_wave_excitation import append_right_env, append_left_env, construct_orthogonal

__all__ = ['VUMPSEngine', 'SingleSiteVUMPSEngine', 'TwoSiteVUMPSEngine']


class VUMPSEngine(IterativeSweeps):
    """ VUMPS base class with common methods for the TwoSiteVUMPS and SingleSiteVUMPS.

    This engine is implemented as a subclass of :class:`~tenpy.algorithms.mps_common.Sweep`.
    It contains all methods that are generic between :class:`SingleSiteVUMPSEngine` and
    :class:`TwoSiteVUMPSEngine`.
    Use the latter two classes for actual VUMPS runs.

    Options
    -------
    .. cfg:config :: VUMPSEngine
        :include: IterativeSweeps

    Attributes
    ----------
    EffectiveH : class type
        Class for the effective Hamiltonian, i.e., a subclass of
        :class:`~tenpy.algorithms.mps_common.EffectiveH`. Has a `length` class attribute which
        specifies the number of sites updated at once (e.g., whether we do single-site vs. two-site
        VUMPS).
    chi_list : dict | ``None``
        See :cfg:option:`DMRGEngine.chi_list`
    eff_H : :class:`~tenpy.algorithms.mps_common.EffectiveH`
        Effective single-site or two-site Hamiltonian.
    shelve : bool
        If a simulation runs out of time (`time.time() - start_time > max_seconds`), the run will
        terminate with `shelve = True`.
    sweeps : int
        The number of sweeps already performed. (Useful for re-start).
    time0 : float
        Time marker for the start of the run.
    update_stats : dict
        A dictionary with detailed statistics of the convergence at local update-level.
        For each key in the following table, the dictionary contains a list where one value is
        added each time :meth:`VUMPSEngine.update_bond` is called.

        =========== ===================================================================
        key         description
        =========== ===================================================================
        i0          An update was performed on sites ``i0, i0+1``.
        ----------- -------------------------------------------------------------------
        e_L         Energy from left transfer matrix.
        ----------- -------------------------------------------------------------------
        e_R         Energy from right transfer matrix.
        ----------- -------------------------------------------------------------------
        e_C1        Energy from the left center matrix.
        ----------- -------------------------------------------------------------------
        e_C2        Energy from the right center matrix.
        ----------- -------------------------------------------------------------------
        e_theta     Energy from the single-site or two-site wave function.
        ----------- -------------------------------------------------------------------
        N_lanczos   Dimension of the Krylov space used in the lanczos diagonalization.
        ----------- -------------------------------------------------------------------
        split_err_L Error between AC_i and AL_i C_{i+1}
        ----------- -------------------------------------------------------------------
        split_err_R Error between AC_i and C_i AR_i
        ----------- -------------------------------------------------------------------
        time        Wallclock time evolved since :attr:`time0` (in seconds).
        =========== ===================================================================

    sweep_stats : dict
        A dictionary with detailed statistics at the sweep level.
        For each key in the following table, the dictionary contains a list where one value is
        added each time :meth:`VUMPSEngine.sweep` is called (with ``optimize=True``).

        ============= ===================================================================
        key           description
        ============= ===================================================================
        sweep         Number of sweeps (excluding environment sweeps) performed so far.
        ------------- -------------------------------------------------------------------
        N_updates     Number of updates (including environment sweeps) performed so far.
        ------------- -------------------------------------------------------------------
        E             The energy obtained from the contracted environments.
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
        max_split_err The maximum split error in the last sweep.
        ------------- -------------------------------------------------------------------
        max_N_lanczos Maximum number of used Lanczos vectors in last sweep.
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
        #options = asConfig(options, self.__class__.__name__)
        if not isinstance(psi, UniformMPS):
            assert isinstance(psi, MPS)
            psi = UniformMPS.from_MPS(psi)  # psi is an MPS, so convert it to a uMPS
        super().__init__(psi, model, options, **kwargs)
        self.guess_init_env_data = self.env.get_initialization_data()
        self.env.clear()
        self._entropy_approx = [None] * psi.L  # always left of a given site
        assert psi.L % model.H_MPO.L == 0
        self.tangent_projector_test(self.env.get_initialization_data())
        self.psi.left_U, self.psi.right_U = None, None
        self.psi.valid_umps = False

        self.N_sweeps_check = self.options.get('N_sweeps_check', 1, int)
        default_min_sweeps = int(1.5 * self.N_sweeps_check)
        if self.chi_list is not None:
            default_min_sweeps = max(max(self.chi_list.keys()), default_min_sweeps)
        self.options.setdefault('min_sweeps', default_min_sweeps)
        mixer_options = self.options.subconfig('mixer_params')
        mixer_options.setdefault('amplitude', 1.e-5)
        mixer_options.setdefault('decay', 2)
        mixer_options.setdefault('disable_after', 5)

    @property
    def S_inv_cutoff(self):
        # high cutoff for regular inverse of S, higher cutoff if we need to (pseudo-) invert
        # a matrix (S can be 2D while the mixer is on)
        return 1.e-8 if not self.psi.diagonal_gauge else 1.e-15

    def run_iteration(self):
        """Perform a single iteration, consisting of ``N_sweeps_check`` sweeps.

        Options
        -------
        .. cfg:configoptions :: VUMPSEngine
        
            diagonal_gauge_frequency : int
                Number of sweeps how often we restore the UniformMPS to the diagonal gauge
            cutoff : float
                During DMRG with a mixer, `S` may be a matrix for which we need the inverse.
                This is calculated as the Penrose pseudo-inverse, which uses a cutoff for the
                singular values.

        Returns
        -------
        E : float
            The energy of the current ground state approximation.
        psi : :class:`~tenpy.networks.uniform_mps.UniformMPS`
            The current ground state approximation, i.e. just a reference to :attr:`psi`.
        """
        options = self.options
        cutoff = options.get('cutoff', 0., 'real')
        diagonal_gauge_frequency = options.get('diagonal_gauge_frequency', 0, int)

        # energy and entropy before the iteration:
        if len(self.sweep_stats['E']) < 1:  # first iteration
            E_old = np.nan
            S_old = np.mean(self.psi.entanglement_entropy())
        else:
            E_old = self.sweep_stats['E'][-1]
            S_old = self.sweep_stats['S'][-1]

        # VUMPS specific convergence criteria
        diagonal_gauge_frequency = options.get('diagonal_gauge_frequency', 0, int)

        # perform sweeps
        logger.info('Running sweep with optimization')
        for i in range(self.N_sweeps_check):
            self.sweep()
        self.psi.diagonal_gauge = False
        if diagonal_gauge_frequency > 0 and self.sweeps % diagonal_gauge_frequency == 0:
            self.psi.to_diagonal_gauge(cutoff=cutoff)

        # update statistics
        entropy_bonds = self._entropy_approx
        max_S = max(entropy_bonds)
        S = np.mean(entropy_bonds)
        E = np.mean(self.update_stats['e_L'][-self.psi.L:] +
                    self.update_stats['e_R'][-self.psi.L:])
        norm_err = np.linalg.norm(self.psi.norm_test())
        max_split_error = np.max(self.update_stats['split_err_L'][-self.psi.L:] +
                                 self.update_stats['split_err_R'][-self.psi.L:])
        max_N_lanczos = [
            np.max([self.update_stats['N_lanczos'][-i - 1][j] for i in range(self.psi.L)])
            for j in range(3)
        ]

        self.sweep_stats['sweep'].append(self.sweeps)
        self.sweep_stats['N_updates'].append(len(self.update_stats['i0']))
        self.sweep_stats['E'].append(E)
        self.sweep_stats['Delta_E'].append((E - E_old) / self.N_sweeps_check)
        self.sweep_stats['S'].append(S)
        self.sweep_stats['Delta_S'].append((S - S_old) / self.N_sweeps_check)
        self.sweep_stats['max_S'].append(max_S)
        self.sweep_stats['time'].append(time.time() - self.time0)
        self.sweep_stats['max_chi'].append(np.max(self.psi.chi))
        self.sweep_stats['norm_err'].append(norm_err)
        self.sweep_stats['max_split_err'].append(max_split_error)
        self.sweep_stats['max_N_lanczos'].append(max_N_lanczos)

        return E, self.psi

        # self.psi.test_validity()
        # logger.info(f"VUMPS finished after {self.sweeps} sweeps, max chi={max(self.psi.chi)}")

        # # psi.norm_test() is sometimes > 1.e-10 for paramagnetic TFI. More VUMPS (>10) fixes this even though the energy is already saturated for 10 sweeps.
        # self.guess_init_env_data, Es, _ = MPOTransferMatrix.find_init_LP_RP(self.model.H_MPO, self.psi, calc_E=True, guess_init_env_data=self.guess_init_env_data)
        # self.tangent_projector_test(self.guess_init_env_data)
        # return (Es[0] + Es[1])/2, self.psi.to_MPS(check_overlap=check_overlap)

    def status_update(self, iteration_start_time: float):
        logger.info(
            "checkpoint after sweep %(sweeps)d\n"
            "energy=%(E).16f, max S=%(max_S).16f, norm_err=%(norm_err).1e\n"
            "Current memory usage %(mem).1fMB, wall time: %(wall_time).1fs\n"
            "Delta E = %(dE).4e, Delta S = %(dS).4e (per sweep)\n"
            "max split_err = %(split_err).4e\n"
            "chi: %(chi)s\n"
            "%(sep)s", {
                'sweeps': self.sweeps,
                'E': self.sweep_stats['E'][-1],
                'max_S': self.sweep_stats['max_S'][-1],
                'norm_err': self.sweep_stats['norm_err'][-1],
                'mem': memory_usage(),
                'wall_time': time.time() - iteration_start_time,
                'dE': self.sweep_stats['Delta_E'][-1],
                'dS': self.sweep_stats['Delta_S'][-1],
                'split_err': self.sweep_stats['max_split_err'][-1],
                'chi': self.psi.chi if self.psi.L < 40 else max(self.psi.chi),
                'sep': "=" * 80,
            })

    def is_converged(self):
        """Determines if the algorithm is converged.

        Does not cover any other reasons to abort, such as reaching a time limit.
        Such checks are covered by :meth:`stopping_condition`.

        Options
        -------
        .. cfg:configoptions :: VUMPSEngine
        
            max_E_err : float
                Convergence if the change of the energy in each step
                satisfies ``|Delta E / max(E, 1)| < max_E_err``. Note that
                this might be satisfied even if ``Delta E > 0``,
                i.e., if the energy increases (due to truncation).
            max_S_err : float
                Convergence if the relative change of the entropy in each step
                satisfies ``|Delta S|/S < max_S_err``
            max_split_err : float
                Convergence if the norm error between AC=AL-C and AC=C_AR is
                smaller than max_split_err.
        """
        max_E_err = self.options.get('max_E_err', 1.e-8, 'real')
        max_S_err = self.options.get('max_S_err', 1.e-5, 'real')
        max_split_error = self.options.get('max_split_err', 1.e-8, 'real')
        E = self.sweep_stats['E'][-1]
        Delta_E = self.sweep_stats['Delta_E'][-1]
        Delta_S = self.sweep_stats['Delta_S'][-1]
        split_error = self.sweep_stats['max_split_err'][-1]

        return abs(Delta_E / max(
            E, 1.)) < max_E_err and abs(Delta_S) < max_S_err and split_error < max_split_error

    def post_run_cleanup(self):
        """
        Perform any final steps or clean up after the main loop has terminated.
        Try to convert uniform MPS back to iMPS.

        Options
        -------
        .. cfg:configoptions :: VUMPSEngine
        
            check_overlap : bool
                Since AL C = C AR is not identically true, the MPS defined by AL and AR are not exactly the same.
                We can compute the overlap of the two to check.
            norm_tol : float
                Check if final state is in canonical form.
        
        """
        super().post_run_cleanup()
        check_overlap = self.options.get('check_overlap', True, bool)
        norm_tol = self.options.get('norm_tol', 1.e-10, 'real')

        self.psi.test_validity()
        logger.info(f'{self.__class__.__name__} finished after {self.sweeps} sweeps, '
                    f'max chi={max(self.psi.chi)}')

        norm_err = np.linalg.norm(self.psi.norm_test())
        if norm_err > norm_tol:
            logger.warning(
                "final VUMPS state not in canonical form up to "
                "norm_tol=%.2e: norm_err=%.2e", norm_tol, norm_err)
            E = self.sweep_stats['E'][-1]
        else:
            self.guess_init_env_data, Es, _ = MPOTransferMatrix.find_init_LP_RP(
                self.model.H_MPO,
                self.psi,
                calc_E=True,
                guess_init_env_data=self.guess_init_env_data)
            self.tangent_projector_test(self.guess_init_env_data)
            E = (Es[0] + Es[1]) / 2

        return E, self.psi.to_MPS(check_overlap=check_overlap)

    def mixer_cleanup(self):
        """For uniform MPS there is no need to clean up after the mixer.
        """
        pass

    def run(self):
        """Run the VUMPS simulation to find the ground state.

        Returns
        -------
        E : float
            The energy of the resulting ground state MPS.
        psi : :class:`~tenpy.networks.mps.MPS`
            The MPS representing the ground state after the simulation,
            i.e. just a reference to :attr:`psi`.
        """
        self.shelve = False
        result = self.pre_run_initialize()
        is_first_sweep = True
        while True:
            iteration_start_time = time.time()
            if self.stopping_criterion(iteration_start_time=iteration_start_time):
                break
            if not is_first_sweep:
                self.checkpoint.emit(self)
            result = self.run_iteration()
            self.status_update(iteration_start_time=iteration_start_time)
            is_first_sweep = False
        return self.post_run_cleanup()

    def environment_sweeps(self, N_sweeps):
        """
        In VUMPS we don't want to do this as we regenerate the environment each time we do an update.
        """
        pass

    def reset_stats(self, resume_data=None):
        """Reset the statistics, useful if you want to start a new sweep run.

        .. cfg:configoptions :: VUMPSEngine

            chi_list : dict | None
                A dictionary to gradually increase the `chi_max` parameter of
                `trunc_params`. The key defines starting from which sweep
                `chi_max` is set to the value, e.g. ``{0: 50, 20: 100}`` uses
                ``chi_max=50`` for the first 20 sweeps and ``chi_max=100``
                afterwards. Overwrites `trunc_params['chi_list']``.
                By default (``None``) this feature is disabled.
            sweep_0 : int
                The number of sweeps already performed. (Useful for re-start).
        """

        super().reset_stats(resume_data)
        self.update_stats = {
            'i0': [],
            'e_L': [],
            'e_R': [],
            'e_C1': [],
            'e_C2': [],
            'e_theta': [],
            'N_lanczos': [],
            'split_err_L': [],
            'split_err_R': [],
            'time': [],
        }

        self.sweep_stats = {
            'sweep': [],
            'N_updates': [],
            'E': [],
            'Delta_E': [],
            'S': [],
            'max_S': [],
            'Delta_S': [],
            'time': [0],
            'max_chi': [],
            'norm_err': [],
            'max_split_err': [],
            'max_N_lanczos': [],
        }

    def get_sweep_schedule(self):
        """Sweep from site 0 to L-1"""
        L = self.psi.L

        i0s = list(range(0, L))
        move_right = [
            True
        ] * L  # Should we also sweep left? not necessary but may increase convergence
        update_LP_RP = [[False, False]
                        ] * L  # Never update the envs since we replace them each time
        return zip(i0s, move_right, update_LP_RP)

    def prepare_update_local(self):
        """
        For each update, we need to rebuild the environments from scratch using the most recent tensors
        """
        i0 = self.i0
        H = self.model.H_MPO
        psi = self.psi

        self.update_env(**{})  # Call this here to update the env guess due to diagonal changes.
        boundary_env_data, Es, _ = MPOTransferMatrix.find_init_LP_RP(
            H, self.psi, calc_E=True,
            guess_init_env_data=self.guess_init_env_data)  # E is already the energy density.
        self.env = MPOEnvironment(psi, H, psi, **boundary_env_data)
        self.transfer_matrix_energy = Es

        self.make_eff_H()
        theta = self.psi.get_theta(i0, n=self.n_optimize,
                                   cutoff=self.S_inv_cutoff)  #n_optimize will be 1
        assert self.eff_H.combine == False
        theta = self.eff_H.combine_theta(theta)  #combine should be false.
        C1, C2 = self.psi.get_C(i0), self.psi.get_C(i0 + self.n_optimize)

        return (theta, C1, C2)

    def make_eff_H(self):
        """
        Create new instance of `self.EffectiveH` at `self.i0`.
        Also create zero-site Hamiltonians left of `self.i0` and right of `self.i0+self.n_optimize`.
        """
        self.eff_H0_1 = ZeroSiteH(self.env, self.i0)  # This saves more envs than optimal.
        self.eff_H0_2 = ZeroSiteH(self.env,
                                  self.i0 + self.n_optimize)  # This saves more envs than optimal.
        self.eff_H = self.EffectiveH(self.env, self.i0, self.combine, self.move_right)

        if hasattr(self.env, 'H') and self.env.H.explicit_plus_hc:
            self.eff_H = SumNpcLinearOperator(self.eff_H, self.eff_H.adjoint())
        if hasattr(self.env, 'H') and self.env.H.explicit_plus_hc:
            self.eff_H0_1 = SumNpcLinearOperator(self.eff_H0_1, self.eff_H0_1.adjoint())
        if hasattr(self.env, 'H') and self.env.H.explicit_plus_hc:
            self.eff_H0_2 = SumNpcLinearOperator(self.eff_H0_2, self.eff_H0_2.adjoint())

    def _wrap_ortho_eff_H(self):
        raise NotImplementedError("Do we want this for VUMPS?")

    def post_update_local(self, e_L, e_R, eps_L, eps_R, e_C1, e_C2, e_theta, N0_L, N0_R, N1,
                          **update_data):
        """Perform post-update actions.

        Collect statistics.

        Parameters
        ----------
        **update_data : dict
            What was returned by :meth:`update_local`.
        """
        self.update_stats['i0'].append(self.i0)
        self.update_stats['e_L'].append(e_L)
        self.update_stats['e_R'].append(e_R)
        self.update_stats['e_C1'].append(e_C1)
        self.update_stats['e_C2'].append(e_C2)
        self.update_stats['e_theta'].append(e_theta)
        self.update_stats['N_lanczos'].append([N0_L, N0_R, N1])
        self.update_stats['split_err_L'].append(eps_L)
        self.update_stats['split_err_R'].append(eps_R)
        self.update_stats['time'].append(time.time() - self.time0)

    def free_no_longer_needed_envs(self):
        for env in self._all_envs:
            env.clear()  # TODO: Can we do better? Is this doing anything at all?

    def resume_run(self):
        raise NotImplementedError("TODO")

    def tangent_projector_test(self, env_data):
        """
        The ground state projector P_GS
        """
        LW = env_data['init_LP']
        RW = env_data['init_RP']

        VLs = [construct_orthogonal(self.psi.get_B(i, form='AL')) for i in range(self.psi.L)]
        VRs = [
            construct_orthogonal(self.psi.get_B(i, form='AR'), left=False)
            for i in range(self.psi.L)
        ]
        ALs = self.psi._AL
        ARs = self.psi._AR
        ACs = self.psi._AC
        Ws = self.model.H_MPO._W * int(self.psi.L / self.model.H_MPO.L)
        strange_left = []
        strange_right = []
        for i in range(self.psi.L):
            temp_L = append_left_env(ALs[:i], ALs[:i], LW, Ws=Ws[:i])
            temp_R = append_right_env(ARs[i + 1:], ARs[i + 1:], RW, Ws=Ws[i + 1:])

            temp_VL = append_left_env([VLs[i]], [ACs[i]], temp_L, Ws=[Ws[i]])
            temp_VL = npc.tensordot(temp_VL, temp_R, axes=(['wR', 'vR*'], ['wL', 'vL*']))

            temp_VR = append_right_env([VRs[i]], [ACs[i]], temp_R, Ws=[Ws[i]])
            temp_VR = npc.tensordot(temp_L, temp_VR, axes=(['wR', 'vR*'], ['wL', 'vL*']))

            strange_left.append(npc.norm(temp_VL))
            strange_right.append(npc.norm(temp_VR))
        logger.info(f'Strange cancellation left: {strange_left}, right: {strange_right}.')

        return strange_left, strange_right


class SingleSiteVUMPSEngine(VUMPSEngine):
    """Engine for the single-site VUMPS algorithm.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        Initial guess for the ground state, which is to be optimized in-place.
    model : :class:`~tenpy.models.model.MPOModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    options : dict
        Further optional parameters.

    Options
    -------
    .. cfg:config :: SingleSiteDMRGEngine
        :include: DMRGEngine
    """
    EffectiveH = OneSiteH

    def __init__(self, psi, model, options, **kwargs):
        super().__init__(psi, model, options, **kwargs)
        if self.mixer is not None:
            raise NotImplementedError("No mixer for SingleSiteVUMPS implemented")

    def update_env(self, **update_data):
        # Get guesses for the next LP and RP
        self.guess_init_env_data = self.env.get_initialization_data()

        # Use unitary gauges from diagonalizing C matrices to update envs
        # Since we update to diagonal gauge after a complete sweep (if we choose to do so),
        # this function is called from sweep.prepare_update.
        if self.psi.left_U is not None:
            init_LP = self.guess_init_env_data['init_LP']
            init_LP = npc.tensordot(self.psi.left_U.conj(), init_LP, axes=(['vL*'], ['vR*']))
            init_LP = npc.tensordot(init_LP, self.psi.left_U, axes=(['vR'], ['vL']))
            self.guess_init_env_data['init_LP'] = init_LP
        if self.psi.right_U is not None:
            init_RP = self.guess_init_env_data['init_RP']
            init_RP = npc.tensordot(self.psi.right_U, init_RP, axes=(['vR'], ['vL']))
            init_RP = npc.tensordot(init_RP, self.psi.right_U.conj(), axes=(['vL*'], ['vR*']))
            self.guess_init_env_data['init_RP'] = init_RP
        # Reset unitary gauges
        self.psi.left_U, self.psi.right_U = None, None

    def update_local(self, theta, **kwargs):
        """Perform single-site update on the site ``i0``.

        Parameters
        ----------
        theta : 3-tuple of :class:`~tenpy.linalg.np_conserved.Array`
            Initial guesses for the ground state of the effective Hamiltonian and zero-site Hamiltonians.

        Returns
        -------
        update_data : dict
            Data computed during the local update.
        """
        psi = self.psi
        i0 = self.i0
        H0_1, H0_2, H1 = self.eff_H0_1, self.eff_H0_2, self.eff_H
        AC, C1, C2 = theta
        lanczos_options = self.options.subconfig('lanczos_options')

        E0_1, theta0_1, N0_1 = LanczosGroundState(H0_1, C1, lanczos_options).run()

        if self.psi.L > 1:
            E0_2, theta0_2, N0_2 = LanczosGroundState(H0_2, C2, lanczos_options).run()
        E1, theta1, N1 = LanczosGroundState(H1, AC, lanczos_options).run()

        if self.psi.L == 1:
            E0_2, theta0_2, N0_2 = E0_1, theta0_1, N0_1

        theta1.ireplace_label('p0', 'p')
        psi.set_C(i0, theta0_1)
        psi.set_C(i0 + 1, theta0_2)
        psi.set_B(i0, theta1, form='AC')
        AL, AR, eps_L, eps_R, entropy_1, entropy_2 = self.polar_max(theta1, theta0_1, theta0_2)
        psi.set_B(i0, AL, form='AL')
        psi.set_B(i0, AR, form='AR')
        self._entropy_approx[i0 % self.psi.L] = entropy_1
        self._entropy_approx[(i0 + self.n_optimize) % self.psi.L] = entropy_2

        update_data = {
            'e_L': self.transfer_matrix_energy[1],
            'e_R': self.transfer_matrix_energy[0],
            'eps_L': eps_L,
            'eps_R': eps_R,
            'e_C1': E0_1,
            'e_C2': E0_2,
            'e_theta': E1,
            'N0_L': N0_1,
            'N0_R': N0_2,
            'N1': N1
        }

        self.trunc_err_list.append(0)

        return update_data

    def polar_max(self, AC, C1, C2):
        """
        Polar decompositions: Given AC and C, find AL and AR such that AL C = AC = C AR

        Parameters
        ----------
        AC : :class:`~tenpy.linalg.np_conserved.Array`
            Center-site tensor at site ``i0``
        C1: :class:`~tenpy.linalg.np_conserved.Array`
            Center matrix left of site ``i0``
        C2: :class:`~tenpy.linalg.np_conserved.Array`
            Center matrix right of site ``i0``
        
        Returns
        -------
        AL : :class:`~tenpy.linalg.np_conserved.Array`
            Left-orthonormal tensor such that AL C2 = AC
        AR : :class:`~tenpy.linalg.np_conserved.Array`
            Right-orthonormal tensor such that C1 AR = AC
        eps_L : float
            Norm error, || AC - AL C2 ||
        eps_R : float
            Norm error, || AC - C1 AR ||
        entropy_left : float
            entanglement entropy left of site ``i0``
        entropy_right : float
            entanglement entropy right of site ``i0``
        """
        U_ACL, _, _ = npc.polar(AC.combine_legs(['vL', 'p'], qconj=[+1]), left=False)
        U_CL, _, s1 = npc.polar(C2, left=False)
        AL = npc.tensordot(U_ACL.split_legs(), U_CL.conj(),
                           axes=(['vR'], ['vR*'])).replace_label('vL*', 'vR')

        U_ACR, _, _ = npc.polar(AC.combine_legs(['p', 'vR'], qconj=[+1]), left=True)
        U_CR, _, s2 = npc.polar(C1, left=True)
        AR = npc.tensordot(U_CR.conj(), U_ACR.split_legs(),
                           axes=(['vL*'], ['vL'])).replace_label('vR*', 'vL')

        eps_L = npc.norm(AC - npc.tensordot(AL, C2, axes=['vR', 'vL']))
        eps_R = npc.norm(AC - npc.tensordot(C1, AR, axes=['vR', 'vL']))

        entropy_left = entropy(s1**2, n=1)
        entropy_right = entropy(s2**2, n=1)

        return AL, AR, eps_L, eps_R, entropy_left, entropy_right


class TwoSiteVUMPSEngine(VUMPSEngine):
    """Engine for the two-site VUMPS algorithm.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        Initial guess for the ground state, which is to be optimized in-place.
    model : :class:`~tenpy.models.model.MPOModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    options : dict
        Further optional parameters.

    Options
    -------
    .. cfg:config :: TwoSiteDMRGEngine
        :include: DMRGEngine
    """
    EffectiveH = TwoSiteH
    DefaultMixer = SubspaceExpansion
    use_mixer_by_default = False

    def __init__(self, psi, model, options, **kwargs):
        super().__init__(psi, model, options, **kwargs)
        if not self.psi.L > 1:
            raise ValueError("Two-site methods require a two-site unit cell.")
        if not self.psi.L > 2 and isinstance(self.mixer, DensityMatrixMixer):
            raise NotImplementedError(
                "DensityMatrixMixer currently only works for unit cells larger than 2")

    def update_env(self, **update_data):
        # Get guesses for the next LP and RP
        # TODO: Since bond dimension is changing, is there anyway to reuse old envs?
        self.guess_init_env_data = None

    def update_local(self, theta, **kwargs):
        """Perform two-site update on the site ``i0`` and ``i0+1``.

        Parameters
        ----------
        theta : 3-tuple of :class:`~tenpy.linalg.np_conserved.Array`
            Initial guesses for the ground state of the effective Hamiltonian and zero-site Hamiltonians.

        Returns
        -------
        update_data : dict
            Data computed during the local update.
        """
        psi = self.psi
        i0 = self.i0
        H0_1, H0_2, H2 = self.eff_H0_1, self.eff_H0_2, self.eff_H
        AC, C1, C2 = theta

        lanczos_options = self.options.subconfig('lanczos_options')
        E0_1, theta0_1, N0_1 = LanczosGroundState(H0_1, C1, lanczos_options).run()
        E0_2, theta0_2, N0_2 = LanczosGroundState(H0_2, C2, lanczos_options).run()
        E2, theta2, N2 = LanczosGroundState(H2, AC, lanczos_options).run()

        U, S, VH, err, S_approx = self.mixed_svd(
            theta2.combine_legs([['vL', 'p0'], ['p1', 'vR']], qconj=[+1, -1]))
        AL1 = U.split_legs()
        AR2 = VH.split_legs()

        AC1 = npc.tensordot(AL1, S, axes=['vR', 'vL'])
        AC2 = npc.tensordot(S, AR2, axes=['vR', 'vL'])

        psi.set_C(i0, theta0_1)
        psi.set_C(i0 + 2, theta0_2)
        psi.set_C(i0 + 1, S)
        psi.set_B(i0, AL1, form='AL')
        psi.set_B(i0 + 1, AR2, form='AR')
        psi.set_B(i0, AC1, form='AC')
        psi.set_B(i0 + 1, AC2, form='AC')

        AL2, AR1, eps_L, eps_R, entropy_1, entropy_2 = self.polar_max(AC1, AC2, theta0_1, theta0_2)
        psi.set_B(i0, AR1, form='AR')
        psi.set_B(i0 + 1, AL2, form='AL')

        self._entropy_approx[i0 % self.psi.L] = entropy_1
        self._entropy_approx[(i0 + 1) % self.psi.L] = entropy(S_approx**2, n=1)
        self._entropy_approx[(i0 + 2) % self.psi.L] = entropy_2
        update_data = {
            'e_L': self.transfer_matrix_energy[1],
            'e_R': self.transfer_matrix_energy[0],
            'eps_L': eps_L,
            'eps_R': eps_R,
            'e_C1': E0_1,
            'e_C2': E0_2,
            'e_theta': E2,
            'N0_L': N0_1,
            'N0_R': N0_2,
            'N1': N2
        }

        self.trunc_err_list.append(err.eps)

        return update_data

    def polar_max(self, AC1, AC2, C1, C3):
        """
        Polar decompositions on two sites:
        Given AC1 and C1, find AR1 such that AC1 = C1 AR1
        and from AC2 and C3, find AL2 such that AC2 = AC2 C3

        Parameters
        ----------
        AC1 : :class:`~tenpy.linalg.np_conserved.Array`
            Center-site tensor at site ``i0``
        AC2 : :class:`~tenpy.linalg.np_conserved.Array`
            Center-site tensor at site ``i0+1``
        C1: :class:`~tenpy.linalg.np_conserved.Array`
            Center matrix left of site ``i0``
        C3: :class:`~tenpy.linalg.np_conserved.Array`
            Center matrix right of site ``i0+1``
        
        Returns
        -------
        AL2 : :class:`~tenpy.linalg.np_conserved.Array`
            Left-orthonormal tensor such that AL2 C3 = AC2
        AR1 : :class:`~tenpy.linalg.np_conserved.Array`
            Right-orthonormal tensor such that C1 AR1 = AC1
        eps_L : float
            Norm error, || AC1 - AL2 C3 ||
        eps_R : float
            Norm error, || AC2 - C1 AR1 ||
        entropy_left : float
            entanglement entropy left of site ``i0``
        entropy_right : float
            entanglement entropy right of site ``i0+1``
        """

        U_ACL, _, _ = npc.polar(AC2.combine_legs(['vL', 'p'], qconj=[+1]), left=False)
        U_CL, _, s1 = npc.polar(C3, left=False)
        AL2 = npc.tensordot(U_ACL.split_legs(), U_CL.conj(),
                            axes=(['vR'], ['vR*'])).replace_label('vL*', 'vR')

        U_ACR, _, _ = npc.polar(AC1.combine_legs(['p', 'vR'], qconj=[+1]), left=True)
        U_CR, _, s2 = npc.polar(C1, left=True)
        AR1 = npc.tensordot(U_CR.conj(), U_ACR.split_legs(),
                            axes=(['vL*'], ['vL'])).replace_label('vR*', 'vL')

        eps_L = npc.norm(AC2 - npc.tensordot(AL2, C3, axes=['vR', 'vL']))
        eps_R = npc.norm(AC1 - npc.tensordot(C1, AR1, axes=['vR', 'vL']))

        entropy_left = entropy(s1**2, n=1)
        entropy_right = entropy(s2**2, n=1)

        return AL2, AR1, eps_L, eps_R, entropy_left, entropy_right

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
        S : 2D :class:`~tenpy.linalg.np_conserved.Array`
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
        mixer = self.mixer
        if mixer is None:
            # simple case: real svd, defined elsewhere.
            qtotal_i0 = self.env.bra.get_B(i0, form=None).qtotal
            U, S, VH, err, _ = svd_theta(theta,
                                         self.trunc_params,
                                         qtotal_LR=[qtotal_i0, None],
                                         inner_labels=['vR', 'vL'])
            S_a = S
            S = npc.diag(S, U.split_legs().get_leg('vR').conj(), labels=['vL', 'vR'])
        else:
            qtotal_LR = [
                self.psi.get_B(i0, form=None).qtotal,
                self.psi.get_B(i0 + 1, form=None).qtotal
            ]
            U, S, VH, err, S_a = mixer.mix_and_decompose_2site(engine=self,
                                                               theta=theta,
                                                               i0=self.i0,
                                                               mix_left=False,
                                                               mix_right=True,
                                                               qtotal_LR=qtotal_LR)
            if not isinstance(S, npc.Array):
                S = npc.diag(S, U.split_legs().get_leg('vR').conj(), labels=['vL', 'vR'])
        U.ireplace_label('(vL.p0)', '(vL.p)')
        VH.ireplace_label('(p1.vR)', '(p.vR)')
        return U, S, VH, err, S_a
