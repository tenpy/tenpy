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
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

from .algorithm import Algorithm
from ..linalg.sparse import NpcLinearOperator, SumNpcLinearOperator, OrthogonalNpcLinearOperator
from ..networks.mpo import MPOEnvironment
from ..networks.mps import MPSEnvironment, MPS
from .truncation import svd_theta, TruncationError
from ..linalg import np_conserved as npc
import numpy as np
import time
import warnings
import copy
import itertools
import logging

logger = logging.getLogger(__name__)

__all__ = [
    'Sweep',
    'EffectiveH',
    'OneSiteH',
    'TwoSiteH',
    'ZeroSiteH',
    'DummyTwoSiteH',
    'VariationalCompression',
    'VariationalApplyMPO',
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
            Lanczos parameters as described in :cfg:config:`Lanczos`.

    Attributes
    ----------
    EffectiveH : class
        Class attribute; a sublcass of :class:`~tenpy.algorithms.mps_common.EffectiveH`.
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
    move_right : bool
        Only set during sweep.
        Whether the next `i0` of the sweep will be right or left of the current one.
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
    trunc_err_list : list
        List of truncation errors from the last sweep.
    chi_list : dict | ``None``
        A dictionary to gradually increase the `chi_max` parameter of `trunc_params`.
        See :cfg:option:`Sweep.chi_list`
    """
    def __init__(self, psi, model, options, *, orthogonal_to=None, **kwargs):
        if not hasattr(self, "EffectiveH"):
            raise NotImplementedError("Subclass needs to set EffectiveH")
        super().__init__(psi, model, options, **kwargs)
        options = self.options

        self.combine = options.get('combine', False)
        self.finite = self.psi.finite
        self.S_inv_cutoff = 1.e-15
        self.lanczos_params = options.subconfig('lanczos_params')

        self.env = None
        self.ortho_to_envs = []
        self.init_env(model, resume_data=self.resume_data, orthogonal_to=orthogonal_to)
        self.i0 = 0
        self.move_right = True
        self.update_LP_RP = (True, False)

    @property
    def engine_params(self):
        warnings.warn("renamed self.engine_params -> self.options", FutureWarning, stacklevel=2)
        return self.options

    @property
    def _all_envs(self):
        return [self.env] + self.ortho_to_envs

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
            `init_env_data` get passed as keyword arguments to the environment initializaiton.
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
        .. deprecated :: 0.6.0
            Options `LP`, `LP_age`, `RP` and `RP_age` are now collected in a dictionary
            `init_env_data` with different keys `init_LP`, `init_RP`, `age_LP`, `age_RP`

        .. deprecated :: 0.8.0
            Instead of passing the `init_env_data` as a option, it should be passed
            as dict entry of `resume_data`.

        .. cfg:configoptions :: Sweep

            init_env_data : dict
                Dictionary as returned by ``self.env.get_initialization_data()`` from
                :meth:`~tenpy.networks.mpo.MPOEnvironment.get_initialization_data`.
                Deprecated, use the `resume_data` function/class argument instead.
            orthogonal_to : list of :class:`~tenpy.networks.mps.MPS`
                Deprecated in favor of the `orthogonal_to` function argument (forwarded from the
                class argument) with the same effect.
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
        if 'init_env_data' in self.options:
            warnings.warn("put init_env_data in resume_data instead of options!", FutureWarning)
            resume_data.setdefault('init_env_data', self.options['init_env_data'])
        init_env_data = {}
        if self.env is not None and self.psi.bc != 'finite':
            # reuse previous environments.
            # if legs are incompatible, MPOEnvironment.init_first_LP_last_RP will regenerate
            init_env_data = self.env.get_initialization_data()
        init_env_data = resume_data.get('init_env_data', init_env_data)
        if not self.psi.finite and init_env_data and \
                self.options.get('chi_list', None) is not None:
            warnings.warn("Re-using environment with `chi_list` set! Do you want this?")
        replaced = [('LP', 'init_LP'), ('LP_age', 'age_LP'), ('RP', 'init_RP'),
                    ('RP_age', 'age_RP')]
        if any([key_old in self.options for key_old, _ in replaced]):
            warnings.warn("Deprecated options LP/RP/LP_age/RP_age: collected in `init_env_data`",
                          FutureWarning)
            for key_old, key_new in replaced:
                if key_old in self.options:
                    init_env_data[key_new] = self.options[key_old]

        # actually initialize the environment
        if self.env is None:
            cache = self.cache.create_subcache('env')
        else:
            cache = self.env.cache  # re-initialize and reuse the cache!
            cache.clear()  # remove old entries which might no longer be valid
        self.env = MPOEnvironment(self.psi, H, self.psi, cache=cache, **init_env_data)
        self._init_ortho_to_envs(orthogonal_to, resume_data)

        self.reset_stats(resume_data)

        # initial sweeps of the environment (without mixer)
        if not self.finite:
            start_env = self.options.get('start_env', 1)
            self.environment_sweeps(start_env)

    def _init_ortho_to_envs(self, orthogonal_to, resume_data):
        # (re)initialize ortho_to_envs
        if 'orthogonal_to' in self.options:
            warnings.warn(
                "Deprecated `orthogonal_to` in dmrg options: instead give "
                "`orthogonal_to` as keyword to the Algorithm class.", FutureWarning)
            assert orthogonal_to is None
            orthogonal_to = self.options['orthogonal_to']
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
        .. deprecated : 0.9
            sweep_0 : int
                Number of sweeps that have already been performed.
                Pass as ``resume_data['sweeps']`` instead.

        .. cfg:configoptions :: Sweep

            chi_list : None | dict(int -> int)
                By default (``None``) this feature is disabled.
                A dict allows to gradually increase the `chi_max`.
                An entry `at_sweep: chi` states that starting from sweep `at_sweep`,
                the value `chi` is to be used for ``trunc_params['chi_max']``.
                For example ``chi_list={0: 50, 20: 100}`` uses ``chi_max=50`` for the first
                20 sweeps and ``chi_max=100`` afterwards.
        """
        self.sweeps = 0
        if 'sweep_0' in self.options:
            warnings.warn("Deprecated sweep_0 option: set as resume_data['sweep'] instead.",
                          FutureWarning)
            self.sweeps = self.options['sweep_0']
        if resume_data is not None and 'sweeps' in resume_data:
            self.sweeps = resume_data['sweeps']
        self.shelve = False
        self.chi_list = self.options.get('chi_list', None)
        if self.chi_list is not None:
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

        Iteratate over the bond which is optimized, to the right and
        then back to the left to the starting point.
        If optimize=False, don't actually diagonalize the effective hamiltonian,
        but only update the environment.

        Parameters
        ----------
        optimize : bool, optional
            Whether we actually optimize to find the ground state of the effective Hamiltonian.
            (If False, just update the environments).

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
            rigth (`True`) of the current one, and `update_LP`, `update_RP` indicate
            whether it is necessary to update the `LP` and `RP` of the environments.
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
        """call ``env.cache_optimize`` to preload next env tensors and avoid unncessary reads."""
        i0 = self.i0
        move_right = self.move_right
        if self.n_optimize == 2:
            kwargs = {
                'short_term_LP': [i0, i0 + 1],
                'short_term_RP': [i0, i0 + 1],
            }
            if move_right:
                kwargs['preload_RP'] = i0 + 2
            else:
                kwargs['preload_LP'] = i0 - 1
        elif self.n_optimize == 1:
            if move_right:
                kwargs = {
                    'short_term_LP': [i0, i0 + 1],
                    'short_term_RP': [i0],
                    'preload_RP': i0 + 1,
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
        so this is a valuable optimiztion to reduce memory requirements.
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
            elif not self.move_right and update_LP:
                # will update site i_R coming from the right in the future
                # so current RP[i_R] is useless
                for env in all_envs:
                    env.del_RP(i_R)
        else:
            assert False, "n_optimize != 1, 2"
        self.eff_H = None  # free references to environments held by eff_H
        # done


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
    move_right : bool, optional
        Whether the sweeping algorithm that calls for an `EffectiveH` is moving to the right.

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
    move_right : bool
        Whether the sweeping algorithm that calls for an `EffectiveH` is moving to the right.
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
    move_right : bool
        Whether the the sweep is moving right or left for the next update.

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
        if self.combine and not self.move_right:
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
    move_right : bool
        Whether the the sweep is moving right or left for the next update.
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


    Note that this class has less funcitonality than the :class:`OneSiteH` and :class:`TwoSiteH`.

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
    combine, move_right : bool
        See above.
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


class VariationalCompression(Sweep):
    """Variational compression of an MPS (in place).

    To compress an MPS `psi`, use ``VariationalCompression(psi, options).run()``.

    The algorithm is the same as described in :class:`VariationalApplyMPO`,
    except that we don't have an MPO in the networks - one can think of the MPO being trivial.

    .. deprecated :: 0.9.1
        Renamed the optoin `N_sweeps` to `max_sweeps`.

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
        :include: Sweep

        trunc_params : dict
            Truncation parameters as described in :cfg:config:`truncation`.
        min_sweeps, max_sweeps : int
            Minimum and maximum number of sweeps to perform for the compression.
        tol_theta_diff: float | None
            Stop after less than `max_sweeps` sweeps if the 1-site wave function changed by less
            than this value, ``1.-|<theta_old|theta_new>| < tol_theta_diff``, where
            theta_old/new are two-site wave functions during the sweep to the left.
            ``None`` disables this convergence check, always performing `max_sweeps` sweeps.
        start_env_sites : int
            Number of sites to contract for the inital LP/RP environment in case of infinite MPS.

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
        self.options.deprecated_alias("N_sweeps", "max_sweeps",
                                      "Also check out the other new convergence parameters "
                                      "min_N_sweeps and tol_theta_diff!")
        max_sweeps = self.options.get("max_sweeps", 2)
        min_sweeps = self.options.get("min_sweeps", 1)
        tol_diff = self._tol_theta_diff = self.options.get("tol_theta_diff", 1.e-8)
        if min_sweeps == max_sweeps and tol_diff is not None:
            warnings.warn("VariationalCompression with min_sweeps=max_sweeps: "
                          "we recommend to set tol_theta_diff=None to avoid overhead")

        for i in range(max_sweeps):
            self.renormalize = []
            self._theta_diff = []
            max_trunc_err = self.sweep()
            if i + 1 >= min_sweeps and tol_diff is not None:
                max_diff = max(self._theta_diff[-(self.psi.L - self.n_optimize):])
                if max_diff < tol_diff:
                    logger.debug("break VariationalCompression after %d sweeps "
                                "with theta_diff=%.2e", i + 1, max_diff)
                    break
        if self.psi.finite:
            self.psi.norm *= max(self.renormalize)
        return TruncationError(max_trunc_err, 1. - 2. * max_trunc_err)

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
        start_env_sites = self.options.get('start_env_sites', 2)
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
    The network for this (with `M` tensors for `psi`) is given by


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
        start_env_sites = self.options.get("start_env_sites", start_env_sites)
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
