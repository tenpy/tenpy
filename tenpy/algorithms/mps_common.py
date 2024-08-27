"""'Sweep' algorithm and effective Hamiltonians.

Many MPS-based algorithms use a 'sweep' structure, wherein local updates are
performed on the MPS tensors sequentially, first from left to right, then from
right to left. This procedure is common to DMRG, TDVP, MPO-based time evolution,
etc.

Another common feature of these algorithms is the use of an effective local
Hamiltonian to perform the local updates. The most prominent example of this is
probably DMRG, where the local MPS object is optimized with respect to the rest
of the MPS-MPO-MPS network, the latter forming the effective Hamiltonian.

The :class:`Sweep` class attempts to generalize as many aspects of 'sweeping'
algorithms as possible. :class:`EffectiveH` and its subclasses implement the
effective Hamiltonians mentioned above. Currently, effective Hamiltonians for
1-site and 2-site optimization are implemented.

The :class:`VariationalCompression` and :class:`VariationalApplyMPO`
implemented here also directly use the :class:`Sweep` class.
"""
# Copyright (C) TeNPy Developers, GNU GPLv3

from ..linalg import np_conserved as npc
from .algorithm import Algorithm
from ..linalg.sparse import NpcLinearOperator, SumNpcLinearOperator, OrthogonalNpcLinearOperator
from ..networks.mpo import MPOEnvironment
from ..networks.mps import MPSEnvironment
from ..linalg.truncation import truncate, svd_theta, decompose_theta_qr_based, TruncationError
from ..linalg import np_conserved as npc
from ..tools.params import asConfig
from ..tools.misc import find_subclass, consistency_check
from ..tools.process import memory_usage
import numpy as np
import time
import warnings
import copy
import itertools
import logging

logger = logging.getLogger(__name__)

__all__ = [
    'Sweep',
    'IterativeSweeps',
    'EffectiveH',
    'OneSiteH',
    'TwoSiteH',
    'ZeroSiteH',
    'DummyTwoSiteH',
    'Mixer',
    'DensityMatrixMixer',
    'SubspaceExpansion',
    'VariationalCompression',
    'VariationalApplyMPO',
    'QRBasedVariationalApplyMPO',
]


class Sweep(Algorithm):
    r"""Prototype class for a 'sweeping' algorithm.

    This is a base class, intended to cover common procedures in all algorithms that 'sweep'
    left-right over the MPS (for infinite MPS: over the MPS unit cell).
    Examples for such algorithms are DMRG, TDVP, and variational compression.

    Parameters
    ----------
    psi, model, options, **kwargs:
        Other parameters as described in :class:`~tenpy.algorithms.algorithm.Algorithm`.
    orthogonal_to : None | list of :class:`~tenpy.networks.mps.MPS` | list of dict
        States to orthogonalize against, see :meth:`init_env`.

    Options
    -------
    .. cfg:config :: Sweep
        :include: Algorithm

        combine : bool
            Whether to combine legs into pipes. This combines the virtual and
            physical leg for the left site (when moving right) or right side
            (when moving left) into pipes. This reduces the overhead of
            calculating charge combinations in the contractions, but one
            :meth:`matvec` is formally more expensive,
            :math:`O(2 d^3 \chi^3 D)`.
        lanczos_params : dict
            Lanczos parameters as described in :cfg:config:`KrylovBased`.

    Attributes
    ----------
    EffectiveH : class
        Class attribute; a subclass of :class:`~tenpy.algorithms.mps_common.EffectiveH`.
        It's length attribute determines how many sites are optimized/updated at once,
        see also :attr:`n_optimize`.
    E_trunc_list : list
        List of truncation energies throughout a sweep.
    env : :class:`~tenpy.networks.mpo.MPOEnvironment`
        Environment for contraction ``<psi|H|psi>``.
    finite : bool
        Whether the MPS boundary conditions are finite (True) or infinite (False)
    i0 : int
        Only set during sweep.
        Left-most of the `EffectiveH.length` sites to be updated in :meth:`update_local`.
    move_right : bool | None
        Only set during sweep.
        Whether the next `i0` of the sweep will be right (`True`), left (`False`) or at the same
        position (`None`) as the current one.
    update_LP_RP : (bool, bool)
        Only set during a sweep, see :meth:`get_sweep_schedule`.
        Indicates whether it is necessary to update the `LP` and `RP` in :meth:`update_env`.
    ortho_to_envs : list of :class:`~tenpy.networks.mps.MPSEnvironment`
        List of environments ``<psi|psi_ortho>``, where `psi_ortho` is an MPS to orthogonalize
        against.
    shelve : bool
        If a simulation runs out of time (`time.time() - start_time > max_seconds`), the run will
        terminate with `shelve = True`.
    sweeps : int
        The number of sweeps already performed.
    S_inv_cutoff : float
        Cutoff for singular values when taking inverses of them is required.
    time0 : float
        Time marker for the start of the run.
    eff_H : :class:`~tenpy.algorithms.mps_common.EffectiveH`
        Effective single-site or two-site Hamiltonian.
    trunc_err_list : list
        List of truncation errors from the last sweep.
    chi_list : dict | ``None``
        A dictionary to gradually increase the `chi_max` parameter of `trunc_params`.
        See :cfg:option:`Sweep.chi_list`
    mixer : :class:`Mixer` | ``None``
        If ``None``, no mixer is used (anymore), otherwise the mixer instance.
    """
    DefaultMixer = None
    use_mixer_by_default = False  # The default for the "mixer" config option

    def __init__(self, psi, model, options, *, orthogonal_to=None, **kwargs):
        if not hasattr(self, "EffectiveH"):
            raise NotImplementedError("Subclass needs to set EffectiveH")
        super().__init__(psi, model, options, **kwargs)
        options = self.options

        self.combine = options.get('combine', False, bool)
        self.finite = self.psi.finite
        self.lanczos_params = options.subconfig('lanczos_params')
        self.mixer = None  # set to an actual mixer (if at all) in :meth:`mixer_activate``

        self.env = None
        self.ortho_to_envs = []
        self.init_env(model, resume_data=self.resume_data, orthogonal_to=orthogonal_to)
        self.i0 = 0
        self.move_right = True
        self.update_LP_RP = (True, False)

    @property
    def _all_envs(self):
        return [self.env] + self.ortho_to_envs

    @property
    def S_inv_cutoff(self):
        # high cutoff for regular inverse of S, higher cutoff if we need to (pseudo-) invert
        # a matrix (S can be 2D while the mixer is on)
        return 1.e-8 if any(isinstance(S, npc.Array) for S in self.psi._S) else 1.e-15

    def get_resume_data(self, sequential_simulations=False):
        data = super().get_resume_data(sequential_simulations)
        data['init_env_data'] = self.env.get_initialization_data()
        if not sequential_simulations:
            data['sweeps'] = self.sweeps
            if len(self.ortho_to_envs) > 0:
                if self.psi.bc == 'finite':
                    data['orthogonal_to'] = [e.ket for e in self.ortho_to_envs]
                else:
                    # need the environments as well
                    data['orthogonal_to'] = [e.get_initialization_data(include_ket=True)
                                            for e in self.ortho_to_envs]
        return data

    @property
    def n_optimize(self):
        """The number of sites to be optimized at once.

        Indirectly set by the class attribute :attr:`EffectiveH` and it's `length`.
        For example, :class:`~tenpy.algorithms.dmrg.TwoSiteDMRGEngine` uses the
        :class:`~tenpy.algorithms.mps_common.TwoSiteH` and hence has ``n_optimize=2``,
        while the :class:`~tenpy.algorithms.dmrg.SingleSiteDMRGEngine` has ``n_optimize=1``.
        """
        return self.EffectiveH.length

    def init_env(self, model=None, resume_data=None, orthogonal_to=None):
        """(Re-)initialize the environment.

        This function is useful to (re-)start a Sweep with a slightly different
        model or different (engine) parameters.
        Note that we assume that we still have the same `psi`.
        Calls :meth:`reset_stats`.

        Parameters
        ----------
        model : :class:`~tenpy.models.MPOModel`
            The model representing the Hamiltonian for which we want to find the ground state.
            If ``None``, keep the model used before.
        resume_data : None | dict
            Given when resuming a simulation, as returned by :meth:`get_resume_data`.
            Can contain another dict under the key `init_env_data`; the contents of
            `init_env_data` get passed as keyword arguments to the environment initialization.
        orthogonal_to : None | list of :class:`~tenpy.networks.mps.MPS` | list of dict
            List of other matrix product states to orthogonalize against.
            Instead of just the state, you can specify a dict with the state as `ket`
            and further keyword arguments for initializing the
            :class:`~tenpy.networks.mps.MPSEnvironment`; the :attr:`psi` to be optimized is
            used as `bra`.
            Works only for finite or segment MPS; for infinite MPS it must be `None`.
            This can be used to find (a few) excited states as follows.
            First, run DMRG to find the ground state,
            and then run DMRG again while orthogonalizing against the ground state,
            which yields the first excited state (in the same symmetry sector), and so on.
            Note that ``resume_data['orthogonal_to']`` takes precedence over the argument.

        Options
        -------

        .. cfg:configoptions :: Sweep

            start_env : int
                Number of sweeps to be performed without optimization to update the environment.

        Raises
        ------
        ValueError
            If the engine is re-initialized with a new model, which legs are incompatible with
            those of hte old model.
        """
        H = model.H_MPO if model is not None else self.env.H
        # extract `init_env_data` from options or previous env
        if resume_data is None:
            resume_data = {}
        init_env_data = {}
        if self.env is not None and self.psi.bc != 'finite':
            # reuse previous environments.
            # if legs are incompatible, MPOEnvironment.init_first_LP_last_RP will regenerate
            init_env_data = self.env.get_initialization_data()
        init_env_data = resume_data.get('init_env_data', init_env_data)
        if not self.psi.finite and init_env_data and \
                self.options.get('chi_list', None) is not None:
            warnings.warn("Re-using environment with `chi_list` set! Do you want this?")

        # actually initialize the environment
        self._init_mpo_env(H, init_env_data)
        self._init_ortho_to_envs(orthogonal_to, resume_data)

        self.reset_stats(resume_data)

        # initial sweeps of the environment (without mixer)
        if not self.finite:
            start_env = self.options.get('start_env', 1, int)
            self.environment_sweeps(start_env)

    def _init_mpo_env(self, H, init_env_data):
        if self.env is None:
            cache = self.cache.create_subcache('env')
        else:
            cache = self.env.cache  # re-initialize and reuse the cache!
            cache.clear()  # remove old entries which might no longer be valid
        self.env = MPOEnvironment(self.psi, H, self.psi, cache=cache, **init_env_data)

    def _init_ortho_to_envs(self, orthogonal_to, resume_data):
        # (re)initialize ortho_to_envs
        if 'orthogonal_to' in resume_data:
            orthogonal_to = resume_data['orthogonal_to']  # precedence for resume_data!

        if orthogonal_to:
            if not self.finite:
                raise ValueError("Can't orthogonalize for infinite MPS: overlap not well defined.")
            logger.info("got %d states to orthogonalize against", len(orthogonal_to))
            self.ortho_to_envs = []
            for i, ortho in enumerate(orthogonal_to):
                if isinstance(ortho, dict):
                    self.ortho_to_envs.append(MPSEnvironment(self.psi, **ortho))
                else:
                    self.ortho_to_envs.append(MPSEnvironment(self.psi, ortho))
        # done

    def reset_stats(self, resume_data=None):
        """Reset the statistics. Useful if you want to start a new Sweep run.

        This method is expected to be overwritten by subclass, and should then define
        self.update_stats and self.sweep_stats dicts consistent with the statistics generated by
        the algorithm particular to that subclass.

        Parameters
        ----------
        resume_data : dict
            Given when resuming a simulation, as returned by :meth:`get_resume_data`.
            Here, we read out the `sweeps`.

        Options
        -------

        .. cfg:configoptions :: Sweep

            chi_list : None | dict(int -> int)
                By default (``None``) this feature is disabled.
                A dict allows to gradually increase the `chi_max`.
                An entry `at_sweep: chi` states that starting from sweep `at_sweep`,
                the value `chi` is to be used for ``trunc_params['chi_max']``.
                For example ``chi_list={0: 50, 20: 100}`` uses ``chi_max=50`` for the first
                20 sweeps and ``chi_max=100`` afterwards.
                A value of `None` is initialized to the current value of
                ``trunc_params['chi_max']`` at algorithm initialization.
        """
        self.sweeps = 0
        if resume_data is not None and 'sweeps' in resume_data:
            self.sweeps = resume_data['sweeps']
        self.shelve = False
        self.chi_list = self.options.get('chi_list', None)
        if self.chi_list is not None:
            for k, v in self.chi_list.items():
                if v is None:
                    self.chi_list[k] = chi_max = self.trunc_params['chi_max']
                    logger.info("Setting chi_list[%d]=%d", k, chi_max)
            done = [k for k in self.chi_list.keys() if k < self.sweeps]
            if len(done) > 0:
                chi_max = self.chi_list[max(done)]
                self.trunc_params['chi_max'] = chi_max
                logger.info("Setting chi_max=%d", chi_max)
        self.time0 = time.time()

    def environment_sweeps(self, N_sweeps):
        """Perform `N_sweeps` sweeps without optimization to update the environment.

        Parameters
        ----------
        N_sweeps : int
            Number of sweeps to run without optimization
        """
        if N_sweeps <= 0:
            return
        logger.info("start environment_sweep")
        for k in range(N_sweeps):
            self.sweep(optimize=False)

    def sweep(self, optimize=True):
        """One 'sweep' of a sweeper algorithm.

        Iterate over the bond which is optimized, to the right and
        then back to the left to the starting point.

        Parameters
        ----------
        optimize : bool, optional
            Whether we actually optimize the state, e.g. to find the ground state of the effective
            Hamiltonian in case of a DMRG. (If False, just update the environments).

        Options
        -------
        .. cfg:configoptions :: Sweep

            chi_list_reactivates_mixer : bool
                If True, the mixer is reset/reactivated each time the bond dimension growths
                due to :cfg:option:`Sweep.chi_list`.

        Returns
        -------
        max_trunc_err : float
            Maximal truncation error introduced.
        """
        self._resume_psi = None  # if we had a separate _resume_psi previously, it's now invalid!
        self.E_trunc_list = []
        self.trunc_err_list = []
        schedule = self.get_sweep_schedule()

        if optimize and self.chi_list is not None:
            new_chi_max = self.chi_list.get(self.sweeps, None)
            if new_chi_max is not None:
                logger.info("Setting chi_max=%d", new_chi_max)
                self.trunc_params['chi_max'] = new_chi_max
                if self.options.get('chi_list_reactivates_mixer', True, bool):
                    self.mixer_activate()

        # the actual sweep
        for i0, move_right, update_LP_RP in schedule:
            self.i0 = i0
            self.move_right = move_right
            self.update_LP_RP = update_LP_RP
            self._cache_optimize()
            logger.debug("in sweep: i0 =%d", i0)
            # --------- the main work --------------
            theta = self.prepare_update_local()
            update_data = self.update_local(theta, optimize=optimize)
            self.update_env(**update_data)
            self.post_update_local(**update_data)
            self.free_no_longer_needed_envs()

        if optimize:  # count optimization sweeps
            self.sweeps += 1
            # update mixer
            if self.mixer is not None:
                mixer = self.mixer.update_amplitude(self.sweeps)
                if mixer is None:
                    self.mixer_deactivate()
                else:
                    self.mixer = mixer

        return np.max(self.trunc_err_list)

    def get_sweep_schedule(self):
        """Define the schedule of the sweep.

        One 'sweep' is a full sequence from the leftmost site to the right and back.

        Returns
        -------
        schedule : iterable of (int, bool, (bool, bool))
            Schedule for the sweep. Each entry is ``(i0, move_right, (update_LP, update_RP))``,
            where `i0` is the leftmost of the ``self.EffectiveH.length`` sites to be updated in
            :meth:`update_local`, `move_right` indicates whether the next `i0` in the schedule is
            right (`True`), left (`False`) or equal (`None`) of the current one, and `update_LP`,
            `update_RP` indicate whether it is necessary to update the `LP` and `RP` of the
            environments.
        """
        # warning: set only those `LP` and `RP` to True, which can/will be used later again
        # otherwise, the assumptions in :meth:`free_no_longer_needed_envs` will not hold,
        # and you need to update that method as well!
        L = self.psi.L
        n = self.EffectiveH.length
        if self.finite:
            assert L > n
            i0s = list(range(0, L - n)) + list(range(L - n, 0, -1))
            move_right = [True] * (L - n) + [False] * (L - n)
            update_LP_RP = [[True, False]] * (L - n) + [[False, True]] * (L - n)
        elif n == 2:
            assert L >= 2
            i0s = list(range(0, L)) + list(range(L, 0, -1))
            move_right = [True] * L + [False] * L
            update_LP_RP = [[True, True]] * 2 + [[True, False]] * (L-2) + \
                           [[True, True]] * 2 + [[False, True]] * (L-2)
        elif n == 1:
            i0s = list(range(0, L)) + list(range(L, 0, -1))
            move_right = [True] * L + [False] * L
            update_LP_RP = [[True, True]] + [[True, False]] * (L-1) + \
                           [[True, True]] + [[False, True]] * (L-1)
        else:
            assert False, "n_optimize is neither 1 nor 2!?"
        return zip(i0s, move_right, update_LP_RP)

    def _cache_optimize(self):
        """call ``env.cache_optimize`` to preload next env tensors and avoid unnecessary reads."""
        i0 = self.i0
        move_right = self.move_right
        if self.n_optimize == 2:
            kwargs = {
                'short_term_LP': [i0, i0 + 1],
                'short_term_RP': [i0, i0 + 1],
            }
            if move_right:
                kwargs['preload_RP'] = i0 + 2
            elif move_right is None:
                pass  # not moving. nothing to preload
            else:
                kwargs['preload_LP'] = i0 - 1
        elif self.n_optimize == 1:
            if move_right:
                kwargs = {
                    'short_term_LP': [i0, i0 + 1],
                    'short_term_RP': [i0],
                    'preload_RP': i0 + 1,
                }
            elif move_right is None:
                kwargs = {
                    'short_term_LP': [i0],
                    'short_term_RP': [i0],
                }
            else:
                kwargs = {
                    'short_term_LP': [i0],
                    'short_term_RP': [i0 - 1, i0],
                    'preload_LP': i0 - 1,
                }
        else:
            raise ValueError(f"unexpected `n_optimize` = {self.n_optimize!r}")
        for env in self._all_envs:
            env.cache_optimize(**kwargs)

    def prepare_update_local(self):
        """Prepare `self` for calling :meth:`update_local`.

        Returns
        -------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Current best guess for the ground state, which is to be optimized.
            Labels are ``'vL', 'p0', 'p1', 'vR'``, or combined versions of it (if `self.combine`).
            For single-site DMRG, the ``'p1'`` label is missing.
        """
        self.make_eff_H()  # self.eff_H represents tensors LP, W0, RP
        # make theta
        theta = self.psi.get_theta(self.i0, n=self.n_optimize, cutoff=self.S_inv_cutoff)
        theta = self.eff_H.combine_theta(theta)
        return theta

    def make_eff_H(self):
        """Create new instance of `self.EffectiveH` at `self.i0` and set it to `self.eff_H`."""
        self.eff_H = self.EffectiveH(self.env, self.i0, self.combine, self.move_right)
        # note: this order of wrapping is most effective.
        if hasattr(self.env, 'H') and self.env.H.explicit_plus_hc:
            self.eff_H = SumNpcLinearOperator(self.eff_H, self.eff_H.adjoint())
        if len(self.ortho_to_envs) > 0:
            self._wrap_ortho_eff_H()

    def _wrap_ortho_eff_H(self):
        assert len(self.ortho_to_envs) > 0
        ortho_vecs = []
        i0 = self.i0
        for o_env in self.ortho_to_envs:
            # environments are of form <psi|ortho>
            theta = o_env.ket.get_theta(i0, n=self.eff_H.length)
            LP = o_env.get_LP(i0, store=True)
            RP = o_env.get_RP(i0 + self.eff_H.length - 1, store=True)
            theta = npc.tensordot(LP, theta, axes=('vR', 'vL'))
            theta = npc.tensordot(theta, RP, axes=('vR', 'vL'))
            theta.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
            if self.eff_H.combine:
                theta = self.eff_H.combine_theta(theta)
            theta.itranspose(self.eff_H.acts_on)
            ortho_vecs.append(theta)
        self.eff_H = OrthogonalNpcLinearOperator(self.eff_H, ortho_vecs)

    def update_local(self, theta, **kwargs):
        """Perform algorithm-specific local update.

        For two-site algorithms with :attr:`n_optimize` = 2, this always optimizes the
        sites :attr:`i0` and `i0` + 1.
        For single-site algorithms, the effective H only acts on site `i0`, but afterwards it
        also updates the bond to the *right* if :attr:`move_right` is True,
        or the bond to the left if :attr:`move_right` is False.
        Since the svd for truncation gives tensors to be multiplied into the tensors on both sides
        of the bond, tensors of two sites are updated even for single-site algorithms:
        when right-moving, site `i0` + 1 is also updated; site `i0` - 1 when left-moving.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Local single- or two-site wave function, as returned by :meth:`prepare_update_local`.

        Returns
        -------
        update_data : dict
            Data to be processed by :meth:`update_env` and :meth:`post_update_local`,
            e.g. containing the truncation error as `err`.
            If :attr:`combine` is set, it should also contain the `U` and `VH` from the SVD.
        """
        raise NotImplementedError("needs to be overridden by subclass")

    def update_env(self, **update_data):
        """Update the left and right environments after an update of the state.

        Parameters
        ----------
        **update_data :
            Whatever is returned by :meth:`update_local`.
        """
        i_L, i_R = self._update_env_inds()  # left and right updated sites
        all_envs = self._all_envs
        for env in all_envs:
            # clean up the updated center bond
            env.del_LP(i_R)
            env.del_RP(i_L)
        # possibly recalculated updated center bonds
        update_LP, update_RP = self.update_LP_RP
        if update_LP:
            self.eff_H.update_LP(self.env, i_R, update_data['U'])  # possibly optimized
            for env in self.ortho_to_envs:
                env.get_LP(i_R, store=True)
        if update_RP:
            self.eff_H.update_RP(self.env, i_L, update_data['VH'])  # possibly optimized
            for env in self.ortho_to_envs:
                env.get_RP(i_L, store=True)

    def _update_env_inds(self):
        n = self.n_optimize  # = 1 or 2
        move_right = self.move_right
        if n == 2 or move_right:
            i_L = self.i0
            i_R = self.i0 + 1
        else:  # n == 1 and left moving
            # TODO is this also correct if move_right is None?
            i_L = self.i0 - 1
            i_R = self.i0
        return i_L, i_R

    def post_update_local(self, err, **update_data):
        """Algorithm-specific actions to be taken after local update.

        An example would be to collect statistics.
        """
        self.trunc_err_list.append(err.eps)

    def free_no_longer_needed_envs(self):
        """Remove no longer needed environments after an update.

        This allows to minimize the number of environments to be kept.
        For large MPO bond dimensions, these environments are by far the biggest part in memory,
        so this is a valuable optimization to reduce memory requirements.
        """
        i_L, i_R = self._update_env_inds()  # left and right updated site
        # envs between `i_L` and `i_R` where already deleted and updated in `update_env`
        i0 = self.i0
        n = self.n_optimize
        update_LP, update_RP = self.update_LP_RP
        all_envs = self._all_envs
        if n == 2:
            if update_RP:
                # growing right environment: will update (i_L-1, i_L) in future
                # so current LP[i_L] is useless
                for env in all_envs:
                    env.del_LP(i_L)
            if update_LP:
                # growing left environment: will update (i_R, i_R + 1) in future
                # so current RP[i_R] is useless
                for env in all_envs:
                    env.del_RP(i_R)
        elif n == 1:
            if self.move_right and update_RP:
                # will update site i_L coming from the left in the future
                # so current LP[i_L] is useless
                for env in all_envs:
                    env.del_LP(i_L)
            elif (self.move_right is False) and update_LP:
                # will update site i_R coming from the right in the future
                # so current RP[i_R] is useless
                for env in all_envs:
                    env.del_RP(i_R)
        else:
            assert False, "n_optimize != 1, 2"
        self.eff_H = None  # free references to environments held by eff_H
        # done

    def mixer_activate(self):
        """Set `self.mixer` to the class specified by `options['mixer']`.

        .. cfg:configoptions :: Sweep

            mixer : str | class | bool | None
                Specifies which :class:`Mixer` to use, if any.
                A string stands for one of the mixers defined in this module.
                A class is assumed to have the same interface as :class:`Mixer` and is used
                to instantiate the :attr:`mixer`.
                ``None`` uses no mixer.
                ``True`` uses the mixer specified by the :attr:`DefaultMixer` class attribute.
                The default depends on the subclass of :class:`Sweep`.
            mixer_params : dict
                Mixer parameters as described in :cfg:config:`Mixer`.

        See Also
        --------
        mixer_deactivate
        """
        Mixer_class = self.options.get('mixer', self.use_mixer_by_default)
        if not Mixer_class:
            return  # no mixer -> nothing to do
        if Mixer_class is True:
            Mixer_class = self.DefaultMixer
        if isinstance(Mixer_class, str):
            Mixer_class = find_subclass(Mixer, Mixer_class)
        mixer_params = self.options.subconfig('mixer_params')
        self.mixer = Mixer_class(mixer_params, self.sweeps)
        logger.info(f'activate {Mixer_class.__name__} with initial amplitude {self.mixer.amplitude}')

    def mixer_deactivate(self):
        """Deactivate the mixer.

        Set ``self.mixer=None`` and revert any other effects of :meth:`mixer_activate`.
        """
        logger.info(f'deactivate {self.mixer.__class__.__name__} with final amplitude ' \
                    f'{self.mixer.amplitude}')
        self.mixer = None

    def mixer_cleanup(self):
        """Cleanup the effects of a mixer.

        A :meth:`sweep` with an enabled :class:`~tenpy.algorithms.mps_common.Mixer` leaves the MPS
        `psi` with 2D arrays in `S`. This method recovers the original form by performing SVDs
        of the `S` and updating the MPS tensors accordingly.
        """
        # Do SVDs ::  S[i] = U[i] * new_S[i] * V[i]
        # Keep state consistent by absorbing into Gammas:
        #   new_G[i] = V[i] * G[i] * U[i + 1]
        # For Th form tensors this means
        #   new_Th[i] = new_S[i] * new_G[i] * new_S[i + 1]
        #             = hc(U[i]) * S[i] * G[i] * S[i + 1] * hc(V[i])
        #             = hc(U[i]) * Th[i] * hc(V[i])
        # For A and B form tensors, we get a mix of the above, i.e.
        #   new_A[i] = hc(U[i]) * A[i] * U[i + 1]
        #   new_B[i] = V[i] * B[i] * hc(V[i + 1])
        # LP environments transform like A tensors on the vR(*) leg(s)
        # RP environments transform like B tensors on the vL(*) leg(s)

        if self.psi.finite:
            assert self.psi.get_SL(0).ndim == 1
            assert self.psi.get_SR(self.psi.L - 1).ndim == 1
            first = 1
        else:
            first = 0

        for i in range(first, self.psi.L):  # converting S to the left of site i
            S = self.psi.get_SL(i)
            if S.ndim == 1:
                # nothing to do
                continue
            U, S, V = npc.svd(S, full_matrices=False, inner_labels=['vR', 'vL'])
            _, form_L = self.psi.form[self.psi._to_valid_index(i - 1)]
            form_R, _ = self.psi.form[i]
            B_L = self.psi.get_B(i - 1, form=None)
            B_R = self.psi.get_B(i, form=None)
            # Update psi._B to the left and right
            if form_L == 0.:  # A or Gamma to the left
                B_L = npc.tensordot(B_L, U, ['vR', 'vL'])
            elif form_L == 1.:  # B or C to the left
                X_L = V.conj().replace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                B_L = npc.tensordot(B_L, X_L, ['vR', 'vL'])
            else:
                msg = (f'Array S are only supported in A, B, Th or G form. '
                       f'Got form {self.psi.form[self.psi._to_valid_index(i - 1)]} on site {i - 1}.')
                raise RuntimeError(msg)
            if form_R == 0.:  # B or Gamma to the right
                B_R = npc.tensordot(V, B_R, ['vR', 'vL'])
            elif form_R == 1.:  # A or C to the left
                X_R = U.conj().replace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                B_R = npc.tensordot(X_R, B_R, ['vR', 'vL'])
            else:
                msg = (f'Array S are only supported in A, B, Th or G form. '
                       f'Got form {self.psi.form[i]} on site {i}.')
                raise RuntimeError(msg)
            self.psi.set_B(i - 1, B_L, form=self.psi.form[i - 1])
            self.psi.set_SL(i, S)
            self.psi.set_B(i, B_R, form=self.psi.form[i])

            # Update environment LP and RP
            assert self.env.bra is self.psi
            update_env_ket_leg = (self.env.ket is self.psi)
            if self.env.has_LP(i):
                LP = self.env.get_LP(i)
                LP = npc.tensordot(LP, U.conj(), ['vR*', 'vL*'])
                if update_env_ket_leg:
                    LP = npc.tensordot(LP, U, ['vR', 'vL'])
                LP.itranspose(['vR*', 'wR', 'vR'])
                self.env.set_LP(i, LP, age=self.env.get_LP_age(i))
            if self.env.has_RP(i - 1):
                RP = self.env.get_RP(i - 1)
                RP = npc.tensordot(V.conj(), RP, ['vR*', 'vL*'])
                if update_env_ket_leg:
                    RP = npc.tensordot(V, RP, ['vR', 'vL'])
                RP.itranspose(['vL', 'wL', 'vL*'])
                self.env.set_RP(i - 1, RP, age=self.env.get_RP_age(i - 1))


class IterativeSweeps(Sweep):
    r"""Prototype class for algorithms that iterate the same sweep until convergence.

    Examples for such algorithms are DMRG and variational compression.
    This is a base class, implementing :meth:`run` in terms of other methods.
    Subclasses should implement :meth:`run_iteration` and :meth:`is_converged`.
    It might be useful to overwrite :meth:`pre_run_initialize`, :meth:`status_update`,
    :meth:`stopping_criterion` or :meth:`post_run_cleanup`.

    Options
    -------
    .. cfg:config :: IterativeSweeps
        :include: Sweep

        max_trunc_err : float
            Threshold for raising errors on too large truncation errors. Default ``0.0001``.
            See :meth:`~tenpy.tools.misc.consistency_check`.
            If the any truncation error :attr:`~tenpy.algorithms.truncation.TruncationError.eps`
            on the final sweep exceeds this value, we raise.
            Can be downgraded to a warning by setting this option to ``None``.

    """

    def run(self):
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
        self.post_run_cleanup()
        consistency_check(np.max(self.trunc_err_list), self.options, 'max_trunc_err', 1e-4,
                          'Maximum truncation error (``max_trunc_err``) exceeded.')
        return result

    def pre_run_initialize(self):
        """Perform preparations before :meth:`run_iteration` is iterated.

        Returns
        -------
        result
            The object to be returned by :meth:`run` in case of immediate convergence, i.e.
            if no iterations are performed.
        """
        self.mixer_activate()
        return None

    def run_iteration(self):
        """Perform a single iteration.

        Returns
        -------
        result
            The object to be returned by :meth:`run` if the main loop terminates after this
            iteration
        """
        raise NotImplementedError("Subclasses should implement this.")

    def status_update(self, iteration_start_time: float):
        """Emits a status message to the logging system after an iteration.

        Parameters
        ----------
        iteration_start_time: float
            The ``time.time()`` at the start of the last iteration
        """
        # only print the bare bones information that is guaranteed to be available
        # subclasses should overwrite this
        logger.info(
            "checkpoint after sweep %(sweeps)d\n"
            "Current memory usage %(mem).1fMB, wall time: %(wall_time).1fs\n"
            "chi: %(chi)s\n"
            "%(sep)s", {
                'sweeps': self.sweeps,
                'mem': memory_usage(),
                'wall_time': time.time() - iteration_start_time,
                'chi': self.psi.chi if self.psi.L < 40 else max(self.psi.chi),
                'sep': "=" * 80,
            }
        )

    def stopping_criterion(self, iteration_start_time: float) -> bool:
        """Determines if the main loop should be terminated.

        Parameters
        ----------
        iteration_start_time : float
            The ``time.time()`` at the start of the last iteration

        Options
        -------
        .. cfg:configoptions :: IterativeSweeps

            min_sweeps : int
                Minimum number of sweeps to perform.
            max_sweeps : int
                Maximum number of sweeps to perform.
            max_hours : float
                If the DMRG took longer (measured in wall-clock time),
                'shelve' the simulation, i.e. stop and return with the flag
                ``shelve=True``.

        Returns
        -------
        should_break : bool
            If ``True``, the main loop in :meth:`run` is broken.
        """
        min_sweeps = self.options.get('min_sweeps', 1, int)
        max_sweeps = self.options.get('max_sweeps', 1000, int)
        max_seconds = 3600 * self.options.get('max_hours', 24 * 365, 'real')

        if self.sweeps > max_sweeps:
            if self.is_converged():
                logger.info(f'{self.__class__.__name__}: Converged.')
            else:
                logger.info(f'{self.__class__.__name__}: Maximum number of sweeps reached')
            return True
        if self.sweeps > min_sweeps and self.is_converged():
            if self.mixer is None:
                return True
            else:
                logger.info(f"{self.__class__.__name__}: Convergence criterion reached with "
                            "enabled mixer. Disable mixer and continue.")
                self.mixer_deactivate()
                return False
        if iteration_start_time - self.time0 > max_seconds:
            self.shelve = True
            logger.warning(f'{self.__class__.__name__}: maximum time limit reached. '
                           f'Shelve simulation.')
            return True
        return False

    def is_converged(self) -> bool:
        """Determines if the algorithm is converged.

        Does not cover any other reasons to abort, such as reaching a time limit.
        Such checks are covered by :meth:`stopping_criterion`.
        """
        raise NotImplementedError("Subclasses should implement this.")

    def post_run_cleanup(self):
        """Perform any final steps or clean up after the main loop has terminated."""
        self.mixer_cleanup()


class EffectiveH(NpcLinearOperator):
    """Prototype class for local effective Hamiltonians used in sweep algorithms.

    As an example, the local effective Hamiltonian for a two-site (DMRG) algorithm
    looks like::

            |        .---       ---.
            |        |    |   |    |
            |       LP----H0--H1---RP
            |        |    |   |    |
            |        .---       ---.

    where ``H0`` and ``H1`` are MPO tensors.

    Parameters
    ----------
    env : :class:`~tenpy.networks.mpo.MPOEnvironment`
        Environment for contraction ``<psi|H|psi>``.
    i0 : int
        Index of left-most site it acts on.
    combine : bool, optional
        Whether to combine legs into pipes as far as possible. This reduces the overhead of
        calculating charge combinations in the contractions.
    move_right : bool | None, optional
        Whether the sweeping algorithm that calls for an `EffectiveH` is moving to the right,
        to the left or not moving.

    Attributes
    ----------
    length : int
        Number of (MPS) sites the effective hamiltonian covers. NB: Class attribute.
    i0 : int
        Index of left-most site it acts on.
    dtype : np.dtype
        The data type of the involved arrays.
    N : int
        Contracting `self` with :meth:`as_matrix` will result in an `N`x`N` matrix .
    acts_on : list of str
        Labels of the state on which `self` acts. NB: class attribute.
        Overwritten by normal attribute, if `combine`.
    combine : bool
        Whether to combine legs into pipes as far as possible. This reduces the overhead of
        calculating charge combinations in the contractions.
    """
    length = None
    acts_on = None

    def __init__(self, env, i0, combine=False, move_right=True):
        raise NotImplementedError("This function should be implemented in derived classes")

    def combine_theta(self, theta):
        """Combine the legs of `theta`, such that it fits to how we combined the legs of `self`.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Wave function to apply the effective Hamiltonian to, with uncombined legs.

        Returns
        -------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Wave function with labels as given by `self.acts_on`.
        """
        raise NotImplementedError("This function should be implemented in derived classes")

    def update_LP(self, env, i, U=None):
        """Equivalent to ``env.get_LP(i, store=True)``; optimized for `combine`.

        Parameters
        ----------
        env : :class:`~tenpy.networks.mpo.MPOEnvironment`
            The same environment as given during class initialization.
        i : int
            We update the part left of site `i`.
            Can optimize if `i` == :attr:`i0` and :attr:`combine` is True.
        U : None | :class:`~tenpy.linalg.np_conserved.Array`
            The tensor on the left-most site `self` acts on, with combined legs after SVD.
            Only used if trying to optimize.
        """
        # non-optimized case
        env.get_LP(i, store=True)

    def update_RP(self, env, i, VH=None):
        """Equivalent to ``env.get_RP(i, store=True)``; optimized for `combine`.

        Parameters
        ----------
        env : :class:`~tenpy.networks.mpo.MPOEnvironment`
            The same environment as given during class initialization.
        i : int
            We update the part right of site `i`.
            Can optimize if `i` == :attr:`i0` + 2 - :attr:`length` and :attr:`combine` is True.
        U : None | :class:`~tenpy.linalg.np_conserved.Array`
            The tensor on the right-most site `self` acts on, with combined legs after SVD.
            Only used if trying to optimize.
        """
        # non-optimized case
        env.get_RP(i, store=True)


class OneSiteH(EffectiveH):
    r"""Class defining the one-site effective Hamiltonian for Lanczos.

    The effective one-site Hamiltonian looks like this::

            |        .---    ---.
            |        |    |     |
            |       LP----W0----RP
            |        |    |     |
            |        .---    ---.

    If `combine` is True, we define either `LHeff` as contraction of `LP` with `W` (in the case
    `move_right` is True) or `RHeff` as contraction of `RP` and `W`.

    Parameters
    ----------
    env : :class:`~tenpy.networks.mpo.MPOEnvironment`
        Environment for contraction ``<psi|H|psi>``.
    i0 : int
        Index of the active site if length=1, or of the left-most active site if length>1.
    combine : bool
        Whether to combine legs into pipes. This combines the virtual and
        physical leg for the left site (when moving right) or right side (when moving left)
        into pipes. This reduces the overhead of calculating charge combinations in the
        contractions, but one :meth:`matvec` is formally more expensive, :math:`O(2 d^3 \chi^3 D)`.
        Is originally from the wo-site method; unclear if it works well for 1 site.
    move_right : bool | None
        Whether the sweeping algorithm that calls for an `EffectiveH` is moving to the right,
        to the left or not moving.

    Attributes
    ----------
    length : int
        Number of (MPS) sites the effective hamiltonian covers.
    acts_on : list of str
        Labels of the state on which `self` acts. NB: class attribute.
        Overwritten by normal attribute, if `combine`.
    combine, move_right : bool
        See above.
    LHeff, RHeff : :class:`~tenpy.linalg.np_conserved.Array`
        Only set if :attr:`combine`, and only one of them depending on :attr:`move_right`.
        If `move_right` was True, `LHeff` is set with labels ``'(vR*.p0)', 'wR', '(vR.p0*)'``
        for bra, MPO, ket; otherwise `RHeff` is set with labels ``'(p0*.vL)', 'wL', '(p0, vL*)'``
    LP, W0, RP : :class:`~tenpy.linalg.np_conserved.Array`
        Tensors making up the network of `self`.
    """
    length = 1
    acts_on = ['vL', 'p0', 'vR']

    def __init__(self, env, i0, combine=False, move_right=True):
        self.i0 = i0
        self.LP = env.get_LP(i0)
        self.RP = env.get_RP(i0)
        self.W0 = env.H.get_W(i0).replace_labels(['p', 'p*'], ['p0', 'p0*'])
        self.dtype = env.H.dtype
        self.combine = combine
        self.move_right = move_right
        self.N = (self.LP.get_leg('vR').ind_len * self.W0.get_leg('p0').ind_len *
                  self.RP.get_leg('vL').ind_len)
        if combine:
            self.combine_Heff(env)

    @classmethod
    def from_LP_W0_RP(cls, LP, W0, RP, i0=0, combine=False, move_right=True):
        self = cls.__new__(cls)
        if combine:
            raise NotImplementedError("Shouldn't need this for vumps")
        self.i0 = i0
        self.LP = LP.itranspose(['vR*', 'wR', 'vR'])
        self.RP = RP.itranspose(['wL', 'vL', 'vL*'])
        self.W0 = W0.replace_labels(['p', 'p*'], ['p0', 'p0*'])
        self.dtype = LP.dtype
        self.combine = combine
        self.move_right = move_right
        self.N = (self.LP.get_leg('vR').ind_len * self.W0.get_leg('p0').ind_len *
                  self.RP.get_leg('vL').ind_len)
        return self

    def matvec(self, theta):
        """Apply the effective Hamiltonian to `theta`.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Labels: ``vL, p0, vR`` if combine=False, ``(vL.p0), vR`` or ``vL, (p0.vR)`` if True
            (depending on the direction of movement)

        Returns
        -------
        theta :class:`~tenpy.linalg.np_conserved.Array`
            Product of `theta` and the effective Hamiltonian.
        """
        labels = theta.get_leg_labels()
        if self.combine:
            if self.move_right:
                theta = npc.tensordot(self.LHeff, theta, axes=['(vR.p0*)', '(vL.p0)'])
                # '(vR*.p0)', 'wR', 'vR'
                theta = npc.tensordot(theta, self.RP, axes=[['wR', 'vR'], ['wL', 'vL']])
                theta.ireplace_labels(['(vR*.p0)', 'vL*'], ['(vL.p0)', 'vR'])
            else:
                theta = npc.tensordot(theta, self.RHeff, axes=['(p0.vR)', '(p0*.vL)'])
                # 'vL', 'wL', '(p0.vL*)'
                theta = npc.tensordot(self.LP, theta, axes=[['vR', 'wR'], ['vL', 'wL']])
                theta.ireplace_labels(['vR*', '(p0.vL*)'], ['vL', '(p0.vR)'])
        else:
            theta = npc.tensordot(self.LP, theta, axes=['vR', 'vL'])
            theta = npc.tensordot(self.W0, theta, axes=[['wL', 'p0*'], ['wR', 'p0']])
            theta = npc.tensordot(theta, self.RP, axes=[['wR', 'vR'], ['wL', 'vL']])
            theta.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
        theta.itranspose(labels)  # if necessary, transpose
        return theta

    def combine_Heff(self, env):
        """Combine LP and RP with W to form LHeff and RHeff, depending on the direction.

        In a move to the right, we need LHeff. In a move to the left, we need RHeff. Both contain
        the same W.

        Parameters
        ----------
        env : :class:`~tenpy.networks.mpo.MPOEnvironment`
            Environment for contraction ``<psi|H|psi>``.
        """
        if self.move_right:
            self.LHeff = env._contract_LHeff(self.i0, 'p0')
            self.pipeL = self.LHeff.get_leg('(vR*.p0)')
            self.acts_on = ['(vL.p0)', 'vR']
        else:
            self.RHeff = env._contract_RHeff(self.i0, 'p0')
            self.pipeR = self.RHeff.get_leg('(p0.vL*)')
            self.acts_on = ['vL', '(p0.vR)']

    def combine_theta(self, theta):
        """Combine the legs of `theta`, such that it fits to how we combined the legs of `self`.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Wave function with labels ``'vL', 'p0', 'vR'``

        Returns
        -------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Wave function with labels ``'(vL.p0)', 'vR'``
        """
        if self.combine:
            if self.move_right:
                theta = theta.combine_legs(['vL', 'p0'], pipes=self.pipeL)
            else:
                theta = theta.combine_legs(['p0', 'vR'], pipes=self.pipeR)
        return theta.itranspose(self.acts_on)

    def to_matrix(self):
        """Contract `self` to a matrix."""
        if self.combine:
            if self.move_right:
                contr = npc.tensordot(self.LHeff, self.RP, axes=['wR', 'wL'])
                contr = contr.combine_legs([['(vR*.p0)', 'vL*'], ['(vR.p0*)', 'vL']],
                                           qconj=[+1, -1])
            else:
                contr = npc.tensordot(self.LP, self.RHeff, axes=['wR', 'wL'])
                contr = contr.combine_legs([['vR*', '(p0.vL*)'], ['vR', '(p0*.vL)']],
                                           qconj=[+1, -1])
        else:
            contr = npc.tensordot(self.LP, self.W0, axes=['wR', 'wL'])
            contr = npc.tensordot(contr, self.RP, axes=['wR', 'wL'])
            contr = contr.combine_legs([['vR*', 'p0', 'vL*'], ['vR', 'p0*', 'vL']], qconj=[+1, -1])
        return contr

    def adjoint(self):
        """Return the hermitian conjugate of `self`."""
        adj = copy.copy(self)
        adj.LP = self.LP.conj().ireplace_label('wR*', 'wR')
        adj.RP = self.RP.conj().ireplace_label('wL*', 'wL')
        adj.W0 = self.W0.conj().ireplace_labels(['wL*', 'wR*'], ['wL', 'wR'])
        if self.combine:
            adj.LHeff = self.LHeff.conj().ireplace_label('wR*', 'wR')
            adj.RHeff = self.RHeff.conj().ireplace_label('wL*', 'wL')
        tensors = ['LP', 'RP', 'W0']
        if self.combine:
            tensors.extend(['LHeff', 'RHeff'])
        for key in tensors:
            getattr(adj, key).itranspose(getattr(self, key).get_leg_labels())
        return adj

    def update_LP(self, env, i, U=None):
        if self.combine and self.move_right:
            assert i == self.i0 + 1  # TODO: hit this in single-site?!?
            LP = npc.tensordot(self.LHeff, U, axes=['(vR.p0*)', '(vL.p)'])
            LP = npc.tensordot(U.conj(), LP, axes=['(vL*.p*)', '(vR*.p0)'])
            env.set_LP(i, LP, age=env.get_LP_age(i - 1) + 1)
        else:
            env.get_LP(i, store=True)

    def update_RP(self, env, i, VH=None):
        if self.combine and (self.move_right is False):
            assert i == self.i0 - 1
            RP = npc.tensordot(VH, self.RHeff, axes=['(p.vR)', '(p0*.vL)'])
            RP = npc.tensordot(RP, VH.conj(), axes=['(p0.vL*)', '(p*.vR*)'])
            env.set_RP(i, RP, age=env.get_RP_age(i + 1) + 1)
        else:
            env.get_RP(i, store=True)


class TwoSiteH(EffectiveH):
    r"""Class defining the two-site effective Hamiltonian for Lanczos.

    The effective two-site Hamiltonian looks like this::

            |        .---       ---.
            |        |    |   |    |
            |       LP----W0--W1---RP
            |        |    |   |    |
            |        .---       ---.

    If `combine` is True, we define `LHeff` and `RHeff`, which are the contractions of `LP` with
    `W0`, and `RP` with `W1`, respectively.

    Parameters
    ----------
    env : :class:`~tenpy.networks.mpo.MPOEnvironment`
        Environment for contraction ``<psi|H|psi>``.
    i0 : int
        Left-most site of the MPS it acts on.
    combine : bool
        Whether to combine legs into pipes. This combines the virtual and
        physical leg for the left site (when moving right) or right side (when moving left)
        into pipes. This reduces the overhead of calculating charge combinations in the
        contractions, but one :meth:`matvec` is formally more expensive, :math:`O(2 d^3 \chi^3 D)`.
    move_right : bool | None
        Whether the the sweep is moving right or left for the next update (or doesn't move).
        Ignored for the :class:`TwoSiteH`.

    Attributes
    ----------
    i0 : int
        Left-most site of the MPS it acts on.
    combine : bool
        Whether to combine legs into pipes. This combines the virtual and
        physical leg for the left site and right site into pipes. This reduces
        the overhead of calculating charge combinations in the contractions,
        but one :meth:`matvec` is formally more expensive, :math:`O(2 d^3 \chi^3 D)`.
    length : int
        Number of (MPS) sites the effective hamiltonian covers.
    acts_on : list of str
        Labels of the state on which `self` acts. NB: class attribute.
        Overwritten by normal attribute, if `combine`.
    LHeff : :class:`~tenpy.linalg.np_conserved.Array`
        Left part of the effective Hamiltonian.
        Labels ``'(vR*.p0)', 'wR', '(vR.p0*)'`` for bra, MPO, ket.
    RHeff : :class:`~tenpy.linalg.np_conserved.Array`
        Right part of the effective Hamiltonian.
        Labels ``'(p1*.vL)', 'wL', '(p1.vL*)'`` for ket, MPO, bra.
    LP, W0, W1, RP : :class:`~tenpy.linalg.np_conserved.Array`
        Tensors making up the network of `self`.
    """
    length = 2
    acts_on = ['vL', 'p0', 'p1', 'vR']

    def __init__(self, env, i0, combine=False, move_right=True):
        self.i0 = i0
        self.LP = env.get_LP(i0)
        self.RP = env.get_RP(i0 + 1)
        self.W0 = env.H.get_W(i0).replace_labels(['p', 'p*'], ['p0', 'p0*'])
        # 'wL', 'wR', 'p0', 'p0*'
        self.W1 = env.H.get_W(i0 + 1).replace_labels(['p', 'p*'], ['p1', 'p1*'])
        # 'wL', 'wR', 'p1', 'p1*'
        self.dtype = env.H.dtype
        self.combine = combine
        self.N = (self.LP.get_leg('vR').ind_len * self.W0.get_leg('p0').ind_len *
                  self.W1.get_leg('p1').ind_len * self.RP.get_leg('vL').ind_len)
        if combine:
            self.combine_Heff(env)

    def matvec(self, theta):
        """Apply the effective Hamiltonian to `theta`.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Labels: ``vL, p0, p1, vR`` if combine=False, ``(vL.p0), (p1.vR)`` if True

        Returns
        -------
        theta :class:`~tenpy.linalg.np_conserved.Array`
            Product of `theta` and the effective Hamiltonian.
        """
        labels = theta.get_leg_labels()
        if self.combine:
            theta = npc.tensordot(self.LHeff, theta, axes=['(vR.p0*)', '(vL.p0)'])
            theta = npc.tensordot(theta, self.RHeff, axes=[['wR', '(p1.vR)'], ['wL', '(p1*.vL)']])
            theta.ireplace_labels(['(vR*.p0)', '(p1.vL*)'], ['(vL.p0)', '(p1.vR)'])
        else:
            theta = npc.tensordot(self.LP, theta, axes=['vR', 'vL'])
            theta = npc.tensordot(self.W0, theta, axes=[['wL', 'p0*'], ['wR', 'p0']])
            theta = npc.tensordot(theta, self.W1, axes=[['wR', 'p1'], ['wL', 'p1*']])
            theta = npc.tensordot(theta, self.RP, axes=[['wR', 'vR'], ['wL', 'vL']])
            theta.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
        theta.itranspose(labels)  # if necessary, transpose
        # This is where we would truncate. Separate mode from combine?
        return theta

    def combine_Heff(self, env, left=True, right=True):
        """Combine LP and RP with W to form LHeff and RHeff.

        Combine LP with W0 and RP with W1 to get the effective parts of the Hamiltonian with piped
        legs.

        Parameters
        ----------
        env : :class:`~tenpy.networks.mpo.MPOEnvironment`
            Environment for contraction ``<psi|H|psi>``.
        left, right : bool
            The mixer might need only one of LHeff/RHeff after the Lanczos optimization even for
            `combine=True`.
            These flags allow to calculate them specifically.
        """
        if left:
            self.LHeff = env._contract_LHeff(self.i0, 'p0')
            self.pipeL = self.LHeff.get_leg('(vR*.p0)')
        if right:
            self.RHeff = env._contract_RHeff(self.i0 + 1, 'p1')
            self.pipeR = self.RHeff.get_leg('(p1.vL*)')
        self.acts_on = ['(vL.p0)', '(p1.vR)']  # overwrites class attribute!

    def combine_theta(self, theta):
        """Combine the legs of `theta`, such that it fits to how we combined the legs of `self`.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Wave function with labels ``'vL', 'p0', 'p1', 'vR'``

        Returns
        -------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Wave function with labels ``'vL', 'p0', 'p1', 'vR'``
        """
        if self.combine:
            theta = theta.combine_legs([['vL', 'p0'], ['p1', 'vR']],
                                       pipes=[self.pipeL, self.pipeR])
        return theta.itranspose(self.acts_on)

    def to_matrix(self):
        """Contract `self` to a matrix."""
        if self.combine:
            contr = npc.tensordot(self.LHeff, self.RHeff, axes=['wR', 'wL'])
            contr = contr.combine_legs([['(vR*.p0)', '(p1.vL*)'], ['(vR.p0*)', '(p1*.vL)']],
                                       qconj=[+1, -1])
        else:
            contr = npc.tensordot(self.LP, self.W0, axes=['wR', 'wL'])
            contr = npc.tensordot(contr, self.W1, axes=['wR', 'wL'])
            contr = npc.tensordot(contr, self.RP, axes=['wR', 'wL'])
            contr = contr.combine_legs([['vR*', 'p0', 'p1', 'vL*'], ['vR', 'p0*', 'p1*', 'vL']],
                                       qconj=[+1, -1])
        return contr

    def adjoint(self):
        """Return the hermitian conjugate of `self`."""
        adj = copy.copy(self)
        adj.LP = self.LP.conj().ireplace_label('wR*', 'wR')
        adj.RP = self.RP.conj().ireplace_label('wL*', 'wL')
        adj.W0 = self.W0.conj().ireplace_labels(['wL*', 'wR*'], ['wL', 'wR'])
        adj.W1 = self.W1.conj().ireplace_labels(['wL*', 'wR*'], ['wL', 'wR'])
        if self.combine:
            adj.LHeff = self.LHeff.conj().ireplace_label('wR*', 'wR')
            adj.RHeff = self.RHeff.conj().ireplace_label('wL*', 'wL')
        tensors = ['LP', 'RP', 'W0', 'W1']
        if self.combine:
            tensors.extend(['LHeff', 'RHeff'])
        for key in tensors:
            getattr(adj, key).itranspose(getattr(self, key).get_leg_labels())
        return adj

    def update_LP(self, env, i, U=None):
        if self.combine:
            assert i == self.i0 + 1
            LP = npc.tensordot(self.LHeff, U, axes=['(vR.p0*)', '(vL.p)'])
            LP = npc.tensordot(U.conj(), LP, axes=['(vL*.p*)', '(vR*.p0)'])
            env.set_LP(i, LP, age=env.get_LP_age(i - 1) + 1)
        else:
            env.get_LP(i, store=True)

    def update_RP(self, env, i, VH=None):
        if self.combine:
            assert i == self.i0
            RP = npc.tensordot(VH, self.RHeff, axes=['(p.vR)', '(p1*.vL)'])
            RP = npc.tensordot(RP, VH.conj(), axes=['(p1.vL*)', '(p*.vR*)'])
            env.set_RP(i, RP, age=env.get_RP_age(i + 1) + 1)
        else:
            env.get_RP(i, store=True)


class ZeroSiteH(EffectiveH):
    r"""Class defining the zero-site effective Hamiltonian for Lanczos.

    The effective zero-site Hamiltonian looks like this::

            |        .---    ---.
            |        |          |
            |       LP----------RP
            |        |          |
            |        .---    ---.


    Note that this class has less functionality than the :class:`OneSiteH` and :class:`TwoSiteH`.

    Parameters
    ----------
    env : :class:`~tenpy.networks.mpo.MPOEnvironment`
        Environment for contraction ``<psi|H|psi>``.
    i0 : int
        Site index such that `LP` is everything strictly left of `i0`.

    Attributes
    ----------
    length : int
        Number of (MPS) sites the effective hamiltonian covers.
    acts_on : list of str
        Labels of the state on which `self` acts. NB: class attribute.
        Overwritten by normal attribute, if `combine`.
    LHeff, RHeff : :class:`~tenpy.linalg.np_conserved.Array`
        Only set if :attr:`combine`, and only one of them depending on :attr:`move_right`.
        If `move_right` was True, `LHeff` is set with labels ``'(vR*.p0)', 'wR', '(vR.p0*)'``
        for bra, MPO, ket; otherwise `RHeff` is set with labels ``'(p0*.vL)', 'wL', '(p0, vL*)'``
    LP, W0, RP : :class:`~tenpy.linalg.np_conserved.Array`
        Tensors making up the network of `self`.
    """
    length = 0
    acts_on = ['vL', 'vR']

    def __init__(self, env, i0):
        self.i0 = i0
        self.LP = env.get_LP(i0)
        self.RP = env.get_RP(i0 - 1)
        self.dtype = env.H.dtype
        self.N = self.LP.get_leg('vR').ind_len * self.RP.get_leg('vL').ind_len

    @classmethod
    def from_LP_RP(cls, LP, RP, i0=0):
        self = cls.__new__(cls)
        self.i0 = i0
        self.LP = LP.itranspose(['vR*', 'wR', 'vR'])
        self.RP = RP.itranspose(['wL', 'vL', 'vL*'])
        self.dtype = LP.dtype
        self.N = LP.get_leg('vR').ind_len * RP.get_leg('vL').ind_len
        return self

    def matvec(self, theta):
        """Apply the effective Hamiltonian to `theta`.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Labels: ``vL, vR``.

        Returns
        -------
        theta :class:`~tenpy.linalg.np_conserved.Array`
            Product of `theta` and the effective Hamiltonian.
        """
        labels = theta.get_leg_labels()
        theta = npc.tensordot(self.LP, theta, axes=['vR', 'vL'])
        theta = npc.tensordot(theta, self.RP, axes=[['wR', 'vR'], ['wL', 'vL']])
        theta.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
        theta.itranspose(labels)  # if necessary, transpose
        return theta

    def to_matrix(self):
        """Contract `self` to a matrix."""
        contr = npc.tensordot(self.LP, self.RP, axes=['wR', 'wL'])
        contr = contr.combine_legs([['vR*', 'vL*'], ['vR', 'vL']], qconj=[+1, -1])
        return contr

    def adjoint(self):
        """Return the hermitian conjugate of `self`."""
        adj = copy.copy(self)
        adj.LP = self.LP.conj().ireplace_label('wR*', 'wR').itranspose(self.LP.get_leg_labels())
        adj.RP = self.RP.conj().ireplace_label('wL*', 'wL').itranspose(self.RP.get_leg_labels())
        return adj


class DummyTwoSiteH(EffectiveH):
    """A dummy replacement for :meth:`TwoSiteH` with similar methods but no actual MPO.

    This allows to base the :class:`VariationalCompression` on the :class:`Sweep` class.
    """
    length = 2

    def __init__(self, *args, **kwargs):
        pass

    def combine_theta(self, theta):
        return theta


class Mixer:
    """Base class for a general Mixer.

    Mixers are optional algorithmic steps during sweeping MPS algorithms that increase the
    variational power and / or improve convergence.
    They expand the virtual Hilbert space associated with a single bond of the MPS, while keeping
    the physical state itself unchanged (up to possible truncation).
    This allows subsequent updates adjacent to that bond to explore a larger variational space.
    In particular, applying the mixer to a bond ``i0, i0 + 1`` of an MPS with tensors `M_i` --
    a.k.a. expanding that bond -- gives us two new tensors `M_new_i0` and `M_new_i1` such that::

        |   -- M_i0 --- M_i1 --   ~=   -- M_new_i0 === M_new_i1 ---
        |       |        |                   |            |

    I.e. such that the physical state represented by the MPS is unchanged.
    We use double lines (``===``) to emphasize that the central virtual index on the right side
    has a larger dimension than its counterpart on the left.

    While the above picture gives possibly the best intuitive understanding of what mixing does to
    an MPS, we only implement the combination of two conceptually separate algorithmic steps;
    Namely the SVD of a `theta` wave function (either on 1 or on 2 sites) that is usually done in
    sweeping algorithms anyway, together with the expansion of a bond, as shown above. This
    simplifies the implementation and different Mixer subclasses may choose to perform these steps
    in different orders or even inseparably.

    Different mixers typically implement _either_ :meth:`mixed_svd_2site` _or_
    :meth:`mix_and_decompose_1site` but not both. This in turn means that all mixers offer
    :meth:`mix_and_decompose_2site`, which serves as the access point e.g. for two-site DMRG.

    Parameters
    ----------
    options : dict
        Optional parameters as described in the following table. See :cfg:config:`Mixer`.
    sweep_activated : int
        The first sweep where the mixer was activated; `disable_after` is relative to that.

    Options
    -------
    .. cfg:config :: Mixer

        amplitude : float | None
            Initial :attr:`amplitude` of the mixer.
        decay : float | None
            To slowly turn off the mixer, we divide `amplitude` by `decay` after each sweep.
            (Should be >= 1.). Or ``None``, in which case the amplitude does not decay.
        disable_after : int | None
            We disable the mixer completely after this number of sweeps.
            ``None`` means to never disable the mixer.

    Attributes
    ----------
    can_decompose_1site : bool
        Class attribute indicating if :meth:`mix_and_decompose_1site` is implemented.
    amplitude : float | None
        Current amplitude of the mixer. Meaning is specific to the concrete Mixer subclass.
        A value of ``None`` indicates that the given mixer has no tuneable amplitude.
    decay : float | None
        If both `amplitude` and `decay` are not None, the `amplitude` is divided by `decay` after
        each sweep.
    disable_after : int | None
        We disable the mixer completely after this number of sweeps.
        ``None`` means to never disable the mixer.

    """
    can_decompose_1site = False
    _default_amplitude = 1.e-5
    _default_decay = 2.
    _default_disable_after = 15

    def __init__(self, options, sweep_activated=0):
        self.options = options = asConfig(options, 'Mixer')
        self.amplitude = options.get('amplitude', self._default_amplitude, 'real')
        self.decay = decay = options.get('decay', self._default_decay, 'real')
        assert decay is None or decay >= 1.
        self.disable_after = disable_after = options.get('disable_after', self._default_disable_after, int)
        assert disable_after is None or disable_after > 0
        self.sweep_activated = sweep_activated

    def update_amplitude(self, sweeps):
        """Update the amplitude, possibly disable the mixer.

        Parameters
        ----------
        sweeps : int
            The total number of performed sweeps, to check if we need to disable the mixer.

        Returns
        -------
        mixer : :class:`Mixer` | None
            Returns `self` if we should continue mixing, or ``None``, if the mixer
            should be disabled.
        """
        if self.disable_after is None:
            should_disable = False
        else:
            should_disable = (sweeps >= self.sweep_activated + self.disable_after)
        if self.amplitude is not None and self.decay is not None:
            # otherwise the mixer has no amplitude or it should not decay
            self.amplitude /= self.decay
            if self.amplitude <= np.finfo('float').eps:
                should_disable = True
        if should_disable:
            logger.info(f'Disable mixer after {sweeps} sweeps, final amplitude {self.amplitude}.')
            return None
        return self

    def mixed_svd_2site(self, engine: Sweep, theta: npc.Array, i0: int, mix_left: bool,
                        mix_right: bool, qtotal_LR=[None, None]):
        """Mix and SVD-like decompose a two-site wavefunction.

        The goal is to split theta as follows::

            |   -- theta --   ==>   -- U === S --- VH --
            |      |   |               |           |

        The LHS is equal to the RHS up to truncation and rescaling (we normalize to ``norm(S)==1``).
        The double lines (``===``) indicate the mixed/expanded bond, here e.g.
        for ``mix_left=True, mix_right=False``.
        `U` and `VH` are isometries like in an SVD, but `S` may be a general bond-matrix and in
        particular not necessarily diagonal or even square.
        Either one (or both) of the bonds next to `S` can be expanded / mixed.
        The isometry on a non-mixed side (e.g. `U` if ``mix_left=False``) could have been obtained
        from an SVD of `theta`.

        Parameters
        ----------
        engine :  :class:`Sweep`
            The engine that is using this mixer.
        theta : 2D :class:`~tenpy.linalg.np_conserved.Array`
            Two-site wavefunction prepared for SVD. Labels ``'(vL.p0)', '(p1.vR)'``.
        i0 : int
            The site index of the left site, i.e. such that `theta` lives on sites ``i0, i0 + 1``.
        mix_left : bool
            If the virtual index left of `S` should be expanded.
        mix_right : bool
            If the virtual index right of `S` should be expanded.
        qtotal_LR : [{charges}, {charges}] | None
            The desired `qtotal` for `U` and `VH`, respectively.
            If ``None``, the `qtotal` are arbitrary.

        Returns
        -------
        U : :class:`~tenpy.linalg.np_conserved.Array`
            Left isometry as defined above. Labels ``'(vL.p)', 'vR'``.
        S : 1D ndarray | 2D :class:`~tenpy.linalg.np_conserved.Array`
            Singular values (1D ndarray) or general bond matrix (2D Array, labels ``'vL', 'vR'``).
        VH : :class:`~tenpy.linalg.np_conserved.Array`
            Right isometry as defined above. Labels ``'vL', '(p.vR)'``.
        err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The truncation error introduced.
        S_approx : ndarray
            Approximation of the singular values of `theta`. Exact if available.

        See Also
        --------
        mix_and_decompose_2site
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not implement mixed_svd_2site')

    def mix_and_decompose_1site(self, engine: Sweep, theta: npc.Array, i0: int, move_right: bool):
        """Decompose single-site wavefunction and expand/mix an adjacent bond.

        For a right move, we decompose::

            |   -- theta --   ==>   -- U === S === VH --
            |        |                 |

        For a left move::

            |   -- theta --   ==>   -- U === S === VH --
            |        |                             |

        The LHS is equal to the RHS up to truncation and rescaling (we normalize to ``norm(S)==1``).
        The double lines (``===``) indicate the mixed/expanded bonds.
        Only the tensor with a physical leg (e.g. `U` for a right move) is an isometry and is
        equivalent to the corresponding output of :meth:`mixed_svd_2site`.
        It carries the `qtotal` of `theta`.
        The other (e.g. `VH` for a right move) is in general not isometric.
        `S` are the usual singular values.

        The mixer can be injected in a sweeping algorithm by replacing the usual SVD of `theta`
        that shifts the canonical form  with this method.

        Parameters
        ----------
        engine :  :class:`Sweep`
            The engine that is using this mixer.
        theta : 2D :class:`~tenpy.linalg.np_conserved.Array`
            Single-site wavefunction prepared for SVD. Labels either ``'(vL.p0)', 'vR'`` for a
            right move, or ``'vL', '(p0.vR)'`` for a left move.
        i0 : int
            The site that ``theta`` lives on. The bond to be expanded is ``i0, i0 + 1`` for a right
            move or ``i0 - 1, i0`` for a left move.
        move_right : bool | None
            Whether we move to the right (``True``), left (``False``), or dont move (``None``).

        Returns
        -------
        U : :class:`~tenpy.linalg.np_conserved.Array`
            Left part as defined above. Isometric for a right move. Labels ``'(vL.p)', 'vR'``
            for a right move or ``'vL', '(p.vR)'`` for a left move.
        S : 1D ndarray
            Singular values on the new bond.
        VH : :class:`~tenpy.linalg.np_conserved.Array`
            Right part as defined above. Isometric for a left move. Labels ``'vL', '(p.vR)'``
            for a right move or ``'(vL.p)', 'vR'`` for a left move.
        err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The truncation error introduced.
        """
        msg = f'{self.__class__.__name__} does not implement mix_and_decompose_1site'
        raise NotImplementedError(msg)

    def mix_and_decompose_2site(self, engine: Sweep, theta: npc.Array, i0: int, mix_left: bool,
                                mix_right: bool, qtotal_LR=None):
        """Decompose two-site wavefunction and expand/mix enclosed bond(s).

        This is a weaker version of :meth:`mixed_svd_2site`. The decomposition is also::

            |   -- theta --   ==>   -- U === S --- VH --
            |      |   |               |           |

        But only the tensors on mixed sites (e.g. only `U` for the case depicted above, i.e.
        ``mix_left=True, mix_right=False``) are guaranteed to be isometric, while any non-mixed
        tensor (`VH` in this example) is in general not isometric.
        Other than that, parameters and returns are the same as for :meth:`mixed_svd_2site`.

        The reason to relax the isometry condition is that the decomposition described above
        can be done using :meth:`mix_and_decompose_1site` if :meth:`mixed_svd_2site` is not
        implemented.
        """
        try:
            return self.mixed_svd_2site(engine, theta, i0, mix_left, mix_right, qtotal_LR)
        except NotImplementedError:
            pass  # fall back to using mix_and_decompose_1site

        if mix_left and mix_right:
            # mix left site by treating p1 as part of vR leg
            theta_L = theta.replace_label('(p1.vR)', 'vR')
            U, _, _, err_L = self.mix_and_decompose_1site(engine, theta_L, i0, move_right=True)
            if qtotal_LR is not None:
                U = U.gauge_total_charge(1, qtotal_LR[0])
            # mix right site by treating p0 as part of vL leg
            theta_R = theta.replace_labels(['(vL.p0)', '(p1.vR)'], ['vL', '(p0.vR)'])
            _, S_approx, VH, err_R = self.mix_and_decompose_1site(engine, theta_R, i0 + 1, move_right=False)
            if qtotal_LR is not None:
                VH = VH.gauge_total_charge(0, qtotal_LR[1])
            VH.ireplace_label('(p0.vR)', '(p1.vR)')
            # calculate S = U^H theta V
            theta = npc.tensordot(U.conj(), theta, axes=['(vL*.p0*)', '(vL.p0)'])
            theta = npc.tensordot(theta, VH.conj(), axes=['(p1.vR)', '(p1*.vR*)'])
            theta.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
            theta /= np.linalg.norm(npc.svd(theta, compute_uv=False))
            S = theta
            err = err_L + err_R
        elif mix_left:
            theta_L = theta.replace_label('(p1.vR)', 'vR')
            U, S, VH, err = self.mix_and_decompose_1site(engine, theta_L, i0, move_right=True)
            # note: VH is not isometric
            VH.ireplace_label('vR', '(p1.vR)')
            S_approx = S
        elif mix_right:
            theta_R = theta.replace_labels(['(vL.p0)', '(p1.vR)'], ['vL', '(p0.vR)'])
            U, S, VH, err = self.mix_and_decompose_1site(engine, theta_R, i0 + 1, move_right=False)
            # note: U is not isometric
            U.ireplace_label('vL', '(vL.p0)')
            VH.ireplace_label('(p0.vR)', '(p1.vR)')
            S_approx = S
        else:
            raise ValueError('Expected mix_left=True and/or mix_right=True.')
        return U, S, VH, err, S_approx


def _mix_LR(H, i0, amplitude):
    """Helper function to compute the ``mix_L, mix_R`` matrices.

    These are used in :class:`DensityMatrixMixer` and :class:`SubspaceExpansion` and defined
    in their respective docstrings.

    Parameters
    ----------
    H : :class:`~tenpy.networks.mpo.MPO`
        The MPO used for mixing, e.g. the Hamiltonian
    i0 : int
        The site index left of the bond to be mixed.
    amplitude : float
        The diagonal entry for mix_L and mix_R.

    Returns
    -------
    mix_L, mix_R : 1D ndarray
        Diagonal bond matrices as 1D numpy arrays.
    IdL, IdR : int
        MPO indices representing "only identities to the left (right)".
    explicit_plus_hc : bool
        :attr:`~tenpy.networks.mpo.MPO.explicit_plus_hc` attribute of the MPO.
    """
    chi_MPO = H.get_W(i0).get_leg('wR').ind_len
    IdL, IdR = H.get_IdL(i0 + 1), H.get_IdR(i0)
    mix_L = np.full((chi_MPO, ), amplitude)
    mix_R = np.full((chi_MPO, ), amplitude)
    one = 1. if not H.explicit_plus_hc else 0.5
    if IdL is not None:
        mix_L[IdL] = one
        mix_R[IdL] = 0.
    if IdR is not None:
        mix_L[IdR] = 0.
        mix_R[IdR] = one
    return mix_L, mix_R, IdL, IdR, H.explicit_plus_hc


def _get_LHeff(env, i, eff_H):
    # return LHeff with p0 labels on site `i`
    if i == eff_H.i0 and hasattr(eff_H, 'LHeff'):
        return eff_H.LHeff
    # else:
    return env._contract_LHeff(i)


def _get_RHeff(env, i, eff_H):
    # return RHeff with 'p1' labels on site `i`
    if i == eff_H.i0 + eff_H.length - 1 and hasattr(eff_H, 'RHeff'):
        if eff_H.length == 1:
            return eff_H.RHeff.replace_labels(['(p0.vL*)', '(p0*.vL)'], ['(p1.vL*)', '(p1*.vL)'])
        return eff_H.RHeff
    # else:
    return env._contract_RHeff(i)


class DensityMatrixMixer(Mixer):
    r"""Mixer based on reduced density matrices.

    This mixer constructs density matrices as described in the original paper :cite:`white2005`.

    It implements :meth:`mixed_svd_2site`, i.e. it replaces at the svd ``theta = U S VH`` with
    ``U-> A[i0]`` and `VH -> B[i0+1]`` being the new tensors in the MPS.
    It is thus best suited for two-site updates, e.g. two-site DMRG.
    Using it with one-site updates is possible, but introduces two-site costs and is inadvisable.

    Given `theta`, one way to obtain `U` in the non-mixed case is to calculate and diagonalize the
    reduced density matrices ``rho_L = tr_R |theta><theta| = U S^2 U^H``, and similarly diagonalize
    `rho_R` for `VH`. With the mixer, we perturb the `rho_L` (and/or `rho_R`) with terms from the
    relevant MPO -- e.g. from the Hamiltonian in a DMRG groundstate search -- before diagonalizing,
    see notes below.

    See Also
    --------
    SubspaceExpansion :
        This mixer does mathematically the same, but circumvents the explicit contraction
        of the `rho_L` and `rho_R`.

    Notes
    -----
    The perturbation of `rho_L` is

    .. math ::

        rho_L = tr_R(|\theta><\theta|)
        \rightarrow  tr_R(|\theta><\theta|) + a \sum_l h_l tr_R(|\theta><\theta|) h_l^\dagger

    where `a` is the (small) perturbation :attr:`amplitude` and `h_l` are the left parts of
    the Hamiltonian going across the center bond (i0, i0+1).
    This perturbs eigenvalues of `rho_L` on the order of that amplitude.
    Note, however, that the eigenvalues of the perturbed `rho_L` are no longer related to the
    singular values of `theta`. Since we recover `theta` (at least up to truncation), the singular
    values are unchanged.

    Pictorially, the left density matrix `rho_L` is given by::

        |     mix_left=False           mix_left=True
        |
        |    .---theta---.            .---theta----.
        |    |   |   |   |            |   |    \   |
        |            |   |           LP---W0-.  \  |
        |    |   |   |   |            |   |   \  | |
        |    .---theta*--.                  mixL | |
        |                             |   |   /  | |
        |                            LP*--W0*-  /  |
        |                             |   |    /   |
        |                             .---theta*---.

    Here, the `mixL` is a diagonal matrix with mostly the :attr:`amplitude` on the diagonal, except
    for the `IdL` and `IdR` indices of the MPO, where the entries are 1. and 0., respectively.

    The right density matrix `rho_R` is mirrored accordingly.
    """

    def __init__(self, options, sweep_activated=0):
        super().__init__(options, sweep_activated)
        assert self.amplitude <= 1.

    def mixed_svd_2site(self, engine: Sweep, theta: npc.Array, i0: int, mix_left: bool,
                        mix_right: bool, qtotal_LR=[None, None]):
        rho_L, rho_R = self.mix_rho(engine, theta, i0, mix_left, mix_right)
        return self.svd_from_rho(engine, rho_L, rho_R, theta, qtotal_LR)

    def mix_rho(self, engine: Sweep, theta: npc.Array, i0: int, mix_left: bool, mix_right: bool):
        """Calculate the (possibly mixed) reduced density matrices.

        Parameters
        ----------
        engine : :class:`Sweep`
            The engine that is using this mixer.
        theta : 2D :class:`~tenpy.linalg.np_conserved.Array`
            Two-site wavefunction prepared for SVD. Labels ``'(vL.p0)', '(p1.vR)'``.
        i0 : int
            The site index of the left site, i.e. such that `theta` lives on sites ``i0, i0 + 1``.
        mix_left : bool
            If the virtual index left of `S` should be expanded by perturbing `rho_L`.
        mix_right : bool
            If the virtual index right of `S` should be expanded by perturbing `rho_R`.

        Returns
        -------
        rho_L : :class:`~tenpy.linalg.np_conserved.Array`
            Reduced density matrix on the left site or a perturbation thereof.
            Hermitian square array with labels ``'(vL.p0)', '(vL*.p0*)'``.
        rho_R : :class:`~tenpy.linalg.np_conserved.Array`
            Reduced density matrix on the right site or a perturbation thereof.
            Hermitian square array with labels ``'(p1.vR)', '(p1*.vR*)'``.
        """
        mix_L, mix_R, IdL, IdR, explicit_plus_hc = _mix_LR(engine.env.H, i0, self.amplitude)

        if mix_left:
            LHeff = _get_LHeff(env=engine.env, i=i0, eff_H=engine.eff_H)
            rho_L = npc.tensordot(LHeff, theta, axes=['(vR.p0*)', '(vL.p0)'])
            rho_L.ireplace_label('(vR*.p0)', '(vL.p0)')
            rho_c = rho_L.conj()
            rho_L.iscale_axis(mix_L, 'wR')
            rho_L = npc.tensordot(rho_L, rho_c, axes=[['wR', '(p1.vR)'], ['wR*', '(p1*.vR*)']])
            if explicit_plus_hc:
                rho_L = rho_L + rho_L.conj().itranspose()
            if IdL is None:  # can't set mix_L[IdL] = 1.
                rho_L = rho_L + npc.tensordot(theta, theta.conj(), axes=['(p1.vR)', '(p1*.vR*)'])
        else:
            rho_L = npc.tensordot(theta, theta.conj(), axes=['(p1.vR)', '(p1*.vR*)'])

        if mix_right:
            RHeff = _get_RHeff(env=engine.env, i=i0 + 1, eff_H=engine.eff_H)
            rho_R = npc.tensordot(theta, RHeff, axes=['(p1.vR)', '(p1*.vL)'])
            rho_R.ireplace_label('(p1.vL*)', '(p1.vR)')
            rho_c = rho_R.conj()
            rho_R.iscale_axis(mix_R, 'wL')
            rho_R = npc.tensordot(rho_c, rho_R, axes=[['wL*', '(vL*.p0*)'], ['wL', '(vL.p0)']])
            if explicit_plus_hc:
                rho_R = rho_R + rho_R.conj().itranspose()
            if IdR is None:
                rho_R = rho_R + npc.tensordot(theta.conj(), theta, axes=['(vL*.p0*)', '(vL.p0)'])
        else:
            rho_R = npc.tensordot(theta.conj(), theta, axes=['(vL*.p0*)', '(vL.p0)'])
        return rho_L, rho_R

    def svd_from_rho(self, engine: Sweep, rho_L: npc.Array, rho_R: npc.Array, theta: npc.Array,
                     qtotal_LR):
        r"""Diagonalize ``rho_L, rho_R`` to rewrite `theta` as ``U S V`` with isometric U/V.

        If `rho_L` and `rho_R` were the actual density matrices of `theta`, this function
        just performs an SVD by diagonalizing `rho_L` with U and `rho_R` with `VH` and then
        rewriting `theta == U (U^\dagger theta VH^\dagger VH) = U S V``.
        Since the actual `rho_L` and `rho_R` passed as arguments are perturbed by `mix_rho`,
        we get a similar decomposition but `S` is a general (non-diagonal) bond matrix.

        Returns
        -------
        U, S, VH, err, S_approx:
            As defined in :meth:`mixed_svd_2site`.
        """
        rho_L.itranspose(['(vL.p0)', '(vL*.p0*)'])  # just to be sure of the order
        rho_R.itranspose(['(p1.vR)', '(p1*.vR*)'])  # just to be sure of the order
        # consider the SVD `theta = U S V^H` (with real, diagonal S>0)
        # rho_L ~=  theta theta^H = U S V^H V S U^H = U S S U^H  (for mixer -> 0)
        # Thus, rho_L U = U S S, i.e. columns of U are the eigenvectors of rho_L,
        # eigenvalues are S^2.
        val_L, U = npc.eigh(rho_L)
        U.iset_leg_labels(['(vL.p0)', 'vR'])
        val_L[val_L < 0.] = 0.  # for stability reasons
        val_L /= np.sum(val_L)
        S_a = np.sqrt(val_L)
        keep_L, _, err_L = truncate(S_a, engine.trunc_params)
        U.iproject(keep_L, axes='vR')  # in place
        if qtotal_LR is not None:
            U = U.gauge_total_charge(1, qtotal_LR[0])
        # rho_R ~=  theta^T theta^* = V^* S U^T U* S V^T = V^* S S V^T  (for mixer -> 0)
        # Thus, rho_R V^* = V^* S S, i.e. columns of V^* are eigenvectors of rho_R
        val_R, Vc = npc.eigh(rho_R)
        Vc.iset_leg_labels(['(p1.vR)', 'vL'])
        VH = Vc.itranspose(['vL', '(p1.vR)'])
        val_R[val_R < 0.] = 0.  # for stability reasons
        val_R /= np.sum(val_R)
        keep_R, _, err_R = truncate(np.sqrt(val_R), engine.trunc_params)
        VH.iproject(keep_R, axes='vL')
        if qtotal_LR is not None:
            VH = VH.gauge_total_charge(0, qtotal_LR[1])

        # calculate S = U^H theta V
        theta = npc.tensordot(U.conj(), theta, axes=['(vL*.p0*)', '(vL.p0)'])  # axes 0, 0
        theta = npc.tensordot(theta, VH.conj(), axes=['(p1.vR)', '(p1*.vR*)'])  # axes 1, 1
        theta.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
        # normalize `S` (as in svd_theta) to avoid blowing up numbers
        theta /= theta.norm()  # norm(singular values) = norm(whole array)
        S_a = S_a[keep_L]
        return U, theta, VH, err_L + err_R, S_a


class SubspaceExpansion(Mixer):
    """Mixer of a direct subspace expansion.

    Performs a subspace expansion following :cite:`hubig2015`.
    It operates on single-site wave functions `theta` and is thus suitable for both
    single-site DMRG and two-site DMRG.

    It is actually not necessary to fill the `next_B` with zeros as described in Hubig's paper;
    rather we directly project the `wR` leg of `VH` onto the `IdL` index, which corresponds to
    taking the original `theta` (up to truncation).

    See Also
    --------
    DensityMatrixMixer

    Notes
    -----
    Pictorially for a subspace expansion of the left `U` while moving right::

        |  --theta---            .-theta---                                  --U---S---VH---
        |     |                  |   |                                         |       |
        |             =dot=>    LP---H0--mix_L--     =SVD=>                    |       .---[IdL]
        |                        |   |          (vL.p0),(wR.vR)

    For a left-move::

        |  --theta---            --theta--.                         ---U---S---VH--
        |     |                      |    |                            |       |
        |            =dot=>  --mix_R-H0---RP         =SVD=>      [IdR]-.       |
        |                            |    |     (vL.wL),(p0.vR)


    Note that only the `U` during the right move (or `VH` during left-move) is guaranteed to be
    an isometry as expected in the canonical form; `VH` during the right-move contains a
    "subspace expansion" and does not fulfill the canonical ``VH.dot(VH.conj().T) == eye``.
    Moreover, the `U` constructed from a two-site `theta` viewing the ``'(p1.vR)`` leg as just `vR`
    in the right-move is (mathematically) equivalent to the `U` returned by the
    :class:`DensityMatrixMixer` (up to degenerate singular values).

    In other words, the :meth:`mix_and_decompose_2site` methods of :class:`SubspaceExpansion` and
    :class:`DensityMatrixMixer` should produce equivalent results; they only differ in the way
    they calculate `U` and `V` internally.
    """
    can_decompose_1site = True

    def __init__(self, options, sweep_activated=0):
        super().__init__(options, sweep_activated)
        assert self.amplitude <= 1.

    def mix_and_decompose_1site(self, engine: Sweep, theta: npc.Array, i0: int, move_right: bool):
        bond = i0 if move_right else i0 - 1
        # the mix_L / mix_R bond matrix is the sqrt of the analogous matrix in DensityMatrixMixer
        # so by taking the sqrt here, we make amplitude mean the same thing for both mixers
        amplitude = np.sqrt(self.amplitude)
        mix_L, mix_R, IdL, IdR, explicit_plus_hc = _mix_LR(engine.env.H, bond, amplitude)

        if move_right:
            LHeff = _get_LHeff(env=engine.env, i=i0, eff_H=engine.eff_H)
            LHeff = LHeff.transpose(['(vR*.p0)', 'wR', '(vR.p0*)'])
            if not explicit_plus_hc and IdL is not None:
                theta_expand = npc.tensordot(LHeff.iscale_axis(mix_L, 'wR'), theta,
                                             ['(vR.p0*)', '(vL.p0)'])
                theta_expand.ireplace_label('(vR*.p0)', '(vL.p0)')
            else:
                # need to stack different parts of the wR leg
                wR = LHeff.get_leg('wR')
                stack = [theta.add_trivial_leg(1, 'wR', wR.qconj)]  # explicitly add the identity
                proj = np.ones(wR.ind_len - (IdL is not None) - (IdR is not None), bool)
                if IdL is not None:
                    proj[IdL] = False
                if IdR is not None:
                    proj[IdR] = False
                LHeff.iproject(proj, 'wR')
                LHeff = LHeff * np.sqrt(self.amplitude)
                stack.append(npc.tensordot(LHeff, theta, ['(vR.p0*)', '(vL.p0)']))
                if explicit_plus_hc:
                    # apply (LHeff^dagger theta) = conj(dot(LHeff.T, theta.conj()))
                    th = npc.tensordot(LHeff, theta.conj(), ['(vR*.p0)', '(vL*.p0*)'])
                    stack.append(th.itranspose(['(vR.p0*)', 'wR', 'vR*']).iconj())
                theta_expand = npc.concatenate(stack, axis='wR')
                IdL = 0  # of the new, concatenated leg.
            theta_expand = theta_expand.combine_legs(['wR', 'vR'], qconj=-1)
            U, S, VH, err, _ = svd_theta(theta_expand, engine.trunc_params,
                                         qtotal_LR=[theta.qtotal, None], inner_labels=['vR', 'vL'])
            VH = VH.split_legs('(wR.vR)')
            VH = VH.take_slice(IdL, 'wR')  # project back such that U-S-VH is original theta
        else:  # move left
            RHeff = _get_RHeff(env=engine.env, i=i0, eff_H=engine.eff_H)
            # RHeff is on site i0, but has p1 label
            RHeff = RHeff.transpose(['(p1*.vL)', 'wL', '(p1.vL*)'])
            if not explicit_plus_hc and IdR is not None:
                theta_expand = npc.tensordot(theta, RHeff.iscale_axis(mix_R, 'wL'),
                                             ['(p0.vR)', '(p1*.vL)'])
                theta_expand.ireplace_label('(p1.vL*)', '(p0.vR)')
            else:
                # need to stack different parts of the wR leg
                wL = RHeff.get_leg('wL')
                stack = [theta.add_trivial_leg(1, 'wL', wL.qconj)]  # explicitly add the identity
                proj = np.ones(wL.ind_len - (IdL is not None) - (IdR is not None), bool)
                if IdL is not None:
                    proj[IdL] = False
                if IdR is not None:
                    proj[IdR] = False
                RHeff.iproject(proj, 'wR')
                stack.append(npc.tensordot(theta, RHeff, ['(p0.vR)', '(p1*.vL)']))
                if explicit_plus_hc:
                    # apply (RHeff^dagger theta) = conj(dot(RHeff.T, theta.conj()))
                    th = npc.tensordot(theta.conj(), RHeff, ['(p0*.vR*)', '(p1.vL*)'])
                    stack.append(th.itranspose(['vL*', 'wL', '(p1*.vL*)']).iconj())
                theta_expand = npc.concatenate(stack, axis='wR')
                IdR = 0  # of the new, concatenated leg.
            theta_expand = theta_expand.combine_legs(['vL', 'wL'], qconj=+1)
            U, S, VH, err, _ = svd_theta(theta_expand, engine.trunc_params,
                                         qtotal_LR=[None, theta.qtotal], inner_labels=['vR', 'vL'])
            U = U.split_legs('(vL.wL)')
            U = U.take_slice(IdR, 'wL')  # project back such that U-S-VH is original theta

        return U, S, VH, err


class VariationalCompression(IterativeSweeps):
    """Variational compression of an MPS (in place).

    To compress an MPS `psi`, use ``VariationalCompression(psi, options).run()``.

    The algorithm is the same as described in :class:`VariationalApplyMPO`,
    except that we don't have an MPO in the networks - one can think of the MPO being trivial.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        The state to be compressed.
    options : dict
        See :cfg:config:`VariationalCompression`.
    resume_data : None | dict
        By default (``None``) ignored. If a `dict`, it should contain the data returned by
        :meth:`get_resume_data` when intending to continue/resume an interrupted run,
        in particular `'init_env_data'`.

    Options
    -------
    .. cfg:config :: VariationalCompression
        :include: IterativeSweeps

        trunc_params : dict
            Truncation parameters as described in :cfg:config:`truncation`.
        tol_theta_diff: float | None
            Stop after less than `max_sweeps` sweeps if the 1-site wave function changed by less
            than this value, ``1.-|<theta_old|theta_new>| < tol_theta_diff``, where
            theta_old/new are two-site wave functions during the sweep to the left.
            ``None`` disables this convergence check, always performing `max_sweeps` sweeps.
        start_env_sites : int
            Number of sites to contract for the initial LP/RP environment in case of infinite MPS.

    Attributes
    ----------
    renormalize : list
        Used to keep track of renormalization in the last sweep for `psi.norm`.
    """
    EffectiveH = DummyTwoSiteH

    def __init__(self, psi, options, resume_data=None):
        super().__init__(psi, None, options, resume_data=resume_data)
        self.renormalize = []
        self._theta_diff = []
        self.options.setdefault('max_sweeps', 2)

    def pre_run_initialize(self):
        super().pre_run_initialize()
        max_sweeps = self._max_sweeps = self.options.get("max_sweeps", 2, int)
        min_sweeps = self._min_sweeps = self.options.get("min_sweeps", 1, int)
        tol_diff = self._tol_theta_diff = self.options.get("tol_theta_diff", 1.e-8, 'real')
        if min_sweeps == max_sweeps and tol_diff is not None:
            warnings.warn("VariationalCompression with min_sweeps=max_sweeps: "
                          "we recommend to set tol_theta_diff=None to avoid overhead")
        return TruncationError()

    def run_iteration(self):
        self.renormalize = []
        self._theta_diff = []
        max_trunc_err = self.sweep()
        return TruncationError(max_trunc_err, 1. - 2. * max_trunc_err)

    def is_converged(self):
        if self.sweeps >= self._min_sweeps and self._tol_theta_diff is not None:
            max_diff = max(self._theta_diff[-(self.psi.L - self.n_optimize):])
            if max_diff < self._tol_theta_diff:
                logger.debug(f'VariationalCompression converged after {self.sweeps} sweeps '
                             f'with theta_diff={max_diff}')
                return True
        return False

    def post_run_cleanup(self):
        super().post_run_cleanup()
        if self.psi.finite:
            self.psi.norm *= max(self.renormalize)

    def run(self):
        """Run the compression.

        The state :attr:`psi` is compressed in place.

        .. warning ::
            Call this function directly after initializing the class, without modifying `psi`
            inbetween. A copy of :attr:`psi` is made during :meth:`init_env`!

        Returns
        -------
        max_trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The maximal truncation error of a two-site wave function.
        """
        return super().run()

    def init_env(self, model=None, resume_data=None, orthogonal_to=None):
        """Initialize the environment.

        Parameters
        ----------
        model, orthogonal_to :
            Ignored, only there for compatibility with the :class:`Sweep` class.
        resume_data : dict
            May contain `init_env_data`.
        """
        if resume_data is None:
            resume_data = {}
        init_env_data = resume_data.get("init_env_data", {})
        old_psi = self.psi.copy()
        start_env_sites = self.options.get('start_env_sites', 2, int)
        if start_env_sites is not None and not self.psi.finite:
            init_env_data['start_env_sites'] = start_env_sites
        if self.env is None:
            cache = self.cache.create_subcache('env')
        else:
            cache = self.env.cache
            cache.clear()
        self.env = MPSEnvironment(self.psi, old_psi, cache=cache, **init_env_data)
        self._init_ortho_to_envs(orthogonal_to, resume_data)
        self.reset_stats()

    def get_sweep_schedule(self):
        """Define the schedule of the sweep.

        Compared to :meth:`~tenpy.algorithms.mps_common.Sweep.get_sweep_schedule`, we add one
        extra update at the end with i0=0 (which is the same as the first update of the sweep).
        This is done to ensure proper convergence after each sweep, even if that implies that
        the site 0 is then updated twice per sweep.
        """
        extra = (0, True, [False, False])
        return itertools.chain(super().get_sweep_schedule(), [extra])

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
        """Given a new two-site wave function `theta`, split it and save it in :attr:`psi`."""
        i0 = self.i0
        new_psi = self.psi
        old_A0 = new_psi.get_B(i0, form='A')
        U, S, VH, err, renormalize = svd_theta(theta,
                                               self.trunc_params,
                                               qtotal_LR=[old_A0.qtotal, None],
                                               inner_labels=['vR', 'vL'])
        U.ireplace_label('(vL.p0)', '(vL.p)')
        VH.ireplace_label('(p1.vR)', '(p.vR)')
        A0 = U.split_legs(['(vL.p)'])
        B1 = VH.split_legs(['(p.vR)'])
        self.renormalize.append(renormalize)
        # first compare to old best guess to check convergence of the sweeps
        if self._tol_theta_diff is not None and self.update_LP_RP[0] == False:
            theta_old = new_psi.get_theta(i0)
            theta_new_trunc = npc.tensordot(A0.scale_axis(S, 'vR'), B1, ['vR', 'vL'])
            theta_new_trunc.iset_leg_labels(['vL', 'p0', 'p1', 'vR'])
            ov = npc.inner(theta_new_trunc, theta_old, do_conj=True, axes='labels')
            theta_diff = 1. - abs(ov)
            self._theta_diff.append(theta_diff)
        # now set the new tensors to the MPS
        new_psi.set_B(i0, A0, form='A')  # left-canonical
        new_psi.set_B(i0 + 1, B1, form='B')  # right-canonical
        new_psi.set_SR(i0, S)
        return {'U': U, 'VH': VH, 'err': err}


class VariationalApplyMPO(VariationalCompression):
    """Variational compression for applying an MPO to an MPS (in place).

    To apply an MPO `U_MPO` to an MPS `psi`, use
    ``VariationalApplyMPO(psi, U_MPO, options).run()``.

    The goal is to find a new MPS `phi` (with `N` tensors) which is optimally close
    to ``U_MPO|psi>``, i.e. it is normalized and maximizes ``| <phi|U_MPO|psi> |^2``.
    The network for this (with `M` tensors for `psi`) is given by::


        |     .-------M[0]----M[1]----M[2]---- ...  ----.
        |     |       |       |       |                 |
        |     LP[0]---W[0]----W[1]----W[2]---- ...  --- RP[-1]
        |     |       |       |       |                 |
        |     .-------N[0]*---N[1]*---N[2]*--- ...  ----.

    Here `LP` and `RP` are the environments with partial contractions,
    see also :class:`~tenpy.networks.mpo.MPOEnvironment`.
    This algorithms sweeps through the sites, updating 2 `N` tensors in each :meth:`update_local`,
    say on sites `i0` and `i1` = `i0` +1. We need to maximize::

        |     .-------M[i0]---M[i1]---.
        |     |       |       |       |
        |     LP[i0]--W[i0]---W[i1]---RP[i1]
        |     |       |       |       |
        |     .-------N[i0]*--N[i1]*--.

    The optimal solution is given by::

        |                                     .-------M[i0]---M[i1]---.
        |   ---N[i0]---N[i1]---               |       |       |       |
        |      |       |          = SVD of    LP[i0]--W[i0]---W[i1]---RP[i1]
        |                                     |       |       |       |
        |                                     .-----                --.


    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        The state to which
    U_MPO : :class:`~tenpy.networks.mpo.MPO`
        MPO to be applied to the state.
    options : dict
        See :cfg:config:`VariationalCompression`.
    **kwargs :
        Further keyword arguments as described in the :class:`Sweep` class.

    Options
    -------
    .. cfg:config :: VariationalApplyMPO
        :include: VariationalCompression


    Attributes
    ----------
    renormalize : list
        Used to keep track of renormalization in the last sweep for `psi.norm`.
    """
    EffectiveH = TwoSiteH

    def __init__(self, psi, U_MPO, options, **kwargs):
        Sweep.__init__(self, psi, U_MPO, options, **kwargs)
        self.renormalize = [None] * (psi.L - int(psi.finite))

    def init_env(self, U_MPO, resume_data=None, orthogonal_to=None):
        """Initialize the environment.

        Parameters
        ----------
        U_MPO : :class:`~tenpy.networks.mpo.MPO`
            The MPO to be applied to the sate.
        resume_data : dict
            May contain `init_env_data`.
        orthogonal_to :
            Ignored.
        """
        if resume_data is None:
            resume_data = {}
        init_env_data = resume_data.get("init_env_data", {})
        old_psi = self.psi.copy()
        start_env_sites = 0 if self.psi.finite else self.psi.L
        start_env_sites = self.options.get("start_env_sites", start_env_sites, int)
        if start_env_sites is not None:
            init_env_data['start_env_sites'] = start_env_sites
        # note: we need explicit `start_env_sites` since `bra` != `ket`, so we can't converge
        # with MPOTransferMatrix.find_init_LP_RP
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


class QRBasedVariationalApplyMPO(VariationalApplyMPO):
    r"""Variational MPO application, using QR-based decompositions instead of SVD.

    The QR-based decomposition, introduced in :arxiv:`2212.09782` is used for TEBD, as implemented
    in :class:`~tenpy.algorithms.tebd.QRBasedTEBDEngine`. This engine is a version of
    :class:`VariationalApplyMPO` that uses the same QR-based decomposition instead of SVD in
    the truncation step after the variational update.

    Options
    -------
    .. cfg:config :: QRBasedVariationalApplyMPO
        :include: VariationalApplyMPO

        cbe_expand : float
            Expansion rate. The QR-based decomposition is carried out at an expanded bond dimension
            ``eta = (1 + cbe_expand) * chi``, where ``chi`` is the bond dimension before the time step.
            Default is `0.1`.
        cbe_expand_0 : float
            Expansion rate at low ``chi``.
            If given, the expansion rate decreases linearly from ``cbe_expand_0`` at ``chi == 1``
            to ``cbe_expand`` at ``chi == trunc_params['chi_max']``, then remains constant.
            If not given, the expansion rate is ``cbe_expand`` at all ``chi``.
        cbe_min_block_increase : int
            Minimum bond dimension increase for each block. Default is `1`.
        use_eig_based_svd : bool
            Whether the SVD of the bond matrix :math:`\Xi` should be carried out numerically via
            the eigensystem. This is faster on GPUs, but less accurate.
            It makes no sense to do this on CPU. It is currently not supported for update_imag.
            Default is `False`.
        compute_err : bool
            Whether the truncation error should be computed exactly.
            Compared to SVD-based TEBD, computing the truncation error is significantly more expensive.
            If `True` (default), the full error is computed.
            Otherwise, the truncation error is set to NaN.
    """

    def _expansion_rate(self, i):
        """get expansion rate for updating bond i"""
        expand = self.options.get('cbe_expand', 0.1, 'real')
        expand_0 = self.options.get('cbe_expand_0', None, 'real')

        if expand_0 is None or expand_0 == expand:
            return expand

        chi_max = self.trunc_params.get('chi_max', None, int)
        if chi_max is None:
            raise ValueError('Need to specify trunc_params["chi_max"] in order to use cbe_expand_0.')

        chi = min(self.psi.get_SL(i).shape)
        return max(expand_0 - chi / chi_max * (expand_0 - expand), expand)

    def update_new_psi(self, theta: npc.Array):
        """Given a new two-site wave function `theta`, split it and save it in :attr:`psi`."""
        i0 = self.i0
        new_psi = self.psi

        if self.move_right:
            old_T_L = new_psi.get_B(i0, 'Th')
            old_T_R = new_psi.get_B(i0+1, 'B')
            old_bond_leg = old_T_R.get_leg('vL')
            # for old_T_L `'B'` form fine as well, but i0 in `'Th'` form if ``use_eig_based_svd=True``
        else:
            old_T_L = new_psi.get_B(i0, 'A')
            old_T_R = new_psi.get_B(i0+1, 'Th')
            old_bond_leg = old_T_L.get_leg('vR')
            # for old_T_R `'A'` form fine as well, but i0+1 in `'Th'` form if ``use_eig_based_svd=True``
        expand = self._expansion_rate(i0)
        use_eig_based_svd = self.options.get('use_eig_based_svd', False, bool)

        T_Lc, S, T_Rc, form, err, renormalize = decompose_theta_qr_based(
            old_qtotal_L=old_T_L.qtotal, old_qtotal_R=old_T_R.qtotal, old_bond_leg=old_bond_leg,
            theta=theta, move_right=self.move_right,
            expand=expand, min_block_increase = self.options.get('cbe_min_block_increase', 1, int),
            use_eig_based_svd=use_eig_based_svd,
            trunc_params=self.trunc_params,
            compute_err=self.options.get('compute_err', True, bool),
            return_both_T=True
        )

        if self.move_right:
            assert form[0] == 'A'
            U = T_Lc
        else:
            assert form[1] == 'B'
            VH = T_Rc

        T_L = T_Lc.split_legs(['(vL.p)'])
        T_R = T_Rc.split_legs(['(p.vR)'])
        U, VH = None, None

        self.renormalize.append(renormalize)

        # compare to old best guess to check convergence of the sweeps
        if self._tol_theta_diff is not None and self.update_LP_RP[0] == False:
            theta_old = new_psi.get_theta(i0)
            if use_eig_based_svd:
                theta_new_trunc = npc.tensordot(T_L, T_R, ['vR', 'vL'])
            else:
                theta_new_trunc = npc.tensordot(T_L.scale_axis(S, 'vR'), T_R, ['vR', 'vL'])
            theta_new_trunc.iset_leg_labels(['vL', 'p0', 'p1', 'vR'])
            ov = npc.inner(theta_new_trunc, theta_old, do_conj=True, axes='labels')
            theta_diff = 1. - abs(ov)
            self._theta_diff.append(theta_diff)

        # set the new tensors to the MPS
        new_psi.set_B(i0, T_L, form=form[0])
        new_psi.set_B(i0+1, T_R, form=form[1])
        new_psi.set_SR(i0, S)
        return {'U': U, 'VH': VH, 'err': err}
