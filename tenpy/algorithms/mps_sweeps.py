"""'Sweep' algorithm and effective Hamiltonians.

Many MPS-based algorithms use a 'sweep' structure, wherein local updates are
performed on the MPS tensors sequentially, first from left to right, then from
right to left. This procedure is common to DMRG, TDVP, sequential time evolution,
etc.

Another common feature of these algorithms is the use of an effective local
Hamiltonian to perform the local updates. The most prominent example of this is
probably DMRG, where the local MPS object is optimized with respect to the rest
of the MPS-MPO-MPS network, the latter forming the effective Hamiltonian.

The :class:`Sweep` class attempts to generalize as many aspects of 'sweeping'
algorithms as possible. :class:`EffectiveH` and its subclasses implement the
effective Hamiltonians mentioned above. Currently, effective Hamiltonians for
1-site and 2-site optimization are implemented.

.. todo ::
    Rebuild TDVP engine as subclasses of sweep
    Do testing
"""
# Copyright 2018-2020 TeNPy Developers, GNU GPLv3

import numpy as np
import time
import warnings
import copy

from ..linalg import np_conserved as npc
from ..linalg.lanczos import gram_schmidt
from ..networks.mps import MPSEnvironment
from ..networks.mpo import MPOEnvironment
from ..linalg.sparse import NpcLinearOperator, SumNpcLinearOperator, OrthogonalNpcLinearOperator
from ..tools.params import asConfig

__all__ = ['Sweep', 'EffectiveH', 'OneSiteH', 'TwoSiteH']


class Sweep:
    r"""Prototype class for a 'sweeping' algorithm.

    This is a superclass, intended to cover common procedures in all algorithms that 'sweep'. This
    includes DMRG, TDVP, TEBD, etc. Only DMRG is currently implemented in this way.


    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        Initial guess for the ground state, which is to be optimized in-place.
    model : :class:`~tenpy.models.MPOModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    options : dict
        Further optional configuration parameters.

    Options
    -------
    .. cfg:config :: Sweep

        combine : bool
            Whether to combine legs into pipes. This combines the virtual and
            physical leg for the left site (when moving right) or right side
            (when moving left) into pipes. This reduces the overhead of
            calculating charge combinations in the contractions, but one
            :meth:`matvec` is formally more expensive,
            :math:`O(2 d^3 \chi^3 D)`.
        lanczos_params : :class:`Config`
            Lanczos parameters as described in
            :func:`~tenpy.linalg.lanczos.lanczos`
        trunc_params : dict
            Truncation parameters as described in
            :func:`~tenpy.algorithms.truncation.truncate`
        verbose : bool | int
            Level of verbosity (i.e. how much status information to print); higher=more output.

    Attributes
    ----------
    options: :class:`~tenpy.tools.params.Config`
        Optional parameters.
    E_trunc_list : list
        List of truncation energies throughout a sweep.
    env : :class:`~tenpy.networks.mpo.MPOEnvironment`
        Environment for contraction ``<psi|H|psi>``.
    finite : bool
        Whether the MPS boundary conditions are finite (True) or infinite (False)
    i0 : int
        Only set during sweep.
        Left-most of the `EffectiveH.length` sites to be updated in :meth:`update_local`.
    mixer : :class:`Mixer` | ``None``
        If ``None``, no mixer is used (anymore), otherwise the mixer instance.
    move_right : bool
        Only set during sweep.
        Whether the next `i0` of the sweep will be right or left of the current one.
    ortho_to_envs : list of :class:`~tenpy.networks.mps.MPSEnvironment`
        List of environments ``<psi|psi_ortho>``, where `psi_ortho` is an MPS to orthogonalize
        against.
    shelve : bool
        If a simulation runs out of time (`time.time() - start_time > max_seconds`), the run will
        terminate with `shelve = True`.
    sweeps : int
        The number of sweeps already performed. (Useful for re-start).
    time0 : float
        Time marker for the start of the run.
    trunc_err_list : list
        List of truncation errors.
    update_LP_RP : (bool, bool)
        Only set during a sweep.
        Whether it is necessary to update the `LP` and `RP`.
        The latter are chosen such that the environment is growing for infinite systems, but
        we only keep the minimal number of environment tensors in memory (inside :attr:`env`).
    """
    def __init__(self, psi, model, options):
        if not hasattr(self, "EffectiveH"):
            raise NotImplementedError("Subclass needs to set EffectiveH")
        self.options = options = asConfig(options, "Sweep")
        self.psi = psi
        self.verbose = options.verbose

        self.combine = options.get('combine', False)
        self.finite = self.psi.finite
        self.mixer = None  # means 'ignore mixer'; the mixer is activated in in :meth:`run`.

        self.lanczos_params = options.subconfig('lanczos_params')
        self.trunc_params = options.subconfig('trunc_params')

        self.env = None
        self.ortho_to_envs = []
        self.init_env(model)
        self.i0 = 0
        self.move_right = True
        self.update_LP_RP = (True, False)

    @property
    def engine_params(self):
        warnings.warn("renamed self.engine_params -> self.options", FutureWarning, stacklevel=2)
        return self.options

    def init_env(self, model=None):
        """(Re-)initialize the environment.

        This function is useful to (re-)start a Sweep with a slightly different
        model or different (engine) parameters. Note that we assume that we
        still have the same `psi`.
        Calls :meth:`reset_stats`.

        Parameters
        ----------
        model : :class:`~tenpy.models.MPOModel`
            The model representing the Hamiltonian for which we want to find the ground state.
            If ``None``, keep the model used before.

        Options
        -------
        .. deprecated :: 0.6.0
            Options `LP`, `LP_age`, `RP` and `RP_age` are now collected in a dictionary
            `init_env_data` with different keys `init_LP`, `init_RP`, `age_LP`, `age_RP`

        .. cfg:configoptions :: Sweep

            chi_list : dict | ``None``
                A dictionary to gradually increase the `chi_max` parameter of `trunc_params`.
                The key defines starting from which sweep `chi_max` is set to the value,
                e.g. ``{0: 50, 20: 100}`` uses ``chi_max=50`` for the first 20 sweeps and
                ``chi_max=100`` afterwards. Overwrites ``trunc_params['chi_list']``.
                By default (``None``) this feature is disabled.
            init_env_data : dict
                Dictionary as returned by ``self.env.get_initialization_data()`` from
                :meth:`~tenpy.networks.mps.MPOEnvironment.get_initialization_data`.
            orthogonal_to : list of :class:`~tenpy.networks.mps.MPSEnvironment`
                List of other matrix product states to orthogonalize against.
                Works only for finite systems.
                This parameter can be used to find (a few) excited states as
                follows. First, run DMRG to find the ground state and then
                run DMRG again while orthogonalizing against the ground state,
                which yields the first excited state (in the same symmetry
                sector), and so on.
            start_env : int
                Number of sweeps to be performed without optimization to update
                the environment.

        Raises
        ------
        ValueError
            If the engine is re-initialized with a new model, which legs are incompatible with
            those of hte old model.
        """
        H = model.H_MPO if model is not None else self.env.H
        if self.env is None or self.psi.bc == 'finite':
            init_env_data = self.options.get("init_env_data", {})
        else:  # re-initialize
            compatible = True
            if model is not None:
                try:
                    H.get_W(0).get_leg('wL').test_equal(self.env.H.get_W(0).get_leg('wL'))
                except ValueError:
                    compatible = False
                    warnings.warn("The leg of the new model is incompatible with the previous one."
                                  "Rebuild environment from scratch.")
            if compatible:
                init_env_data = self.env.get_initialization_data()
            else:
                init_env_data = self.options.get("init_env_data", {})
            if self.options.get('chi_list', None) is not None:
                warnings.warn("Re-using environment with `chi_list` set! Do you want this?")
        replaced = [('LP', 'init_LP'), ('LP_age', 'age_LP'), ('RP', 'init_RP'),
                    ('RP_age', 'age_RP')]
        if any([key_old in self.options for key_old, _ in replaced]):
            warnings.warn("Deprecated options LP/RP/LP_age/RP_age: collected in `init_env_data`",
                          FutureWarning)
            for key_old, key_new in replaced:
                if key_old in self.options:
                    init_env_data[key_new] = self.options[key_old]

        self.env = MPOEnvironment(self.psi, H, self.psi, **init_env_data)

        # (re)initialize ortho_to_envs
        orthogonal_to = self.options.get('orthogonal_to', [])
        if len(orthogonal_to) > 0:
            if not self.finite:
                raise ValueError("Can't orthogonalize for infinite MPS: overlap not well defined.")
            self.ortho_to_envs = [MPSEnvironment(self.psi, ortho) for ortho in orthogonal_to]

        self.reset_stats()

        # initial sweeps of the environment (without mixer)
        if not self.finite:
            print("Initial sweeps...")
            # print(self.options['start_env'])
            start_env = self.options.get('start_env', 1)
            self.environment_sweeps(start_env)

    def reset_stats(self):
        """Reset the statistics. Useful if you want to start a new Sweep run.

        This method is expected to be overwritten by subclass, and should then define
        self.update_stats and self.sweep_stats dicts consistent with the statistics generated by
        the algorithm particular to that subclass.

        .. cfg:configoptions :: Sweep

            sweep_0 : int
                Number of sweeps that have already been performed.

        """
        warnings.warn(
            "reset_stats() is not overwritten by the engine. No statistics will be collected!")
        self.sweeps = self.options.get('sweep_0', 0)
        self.shelve = False
        self.chi_list = self.options.get('chi_list', None)
        if self.chi_list is not None:
            chi_max = self.chi_list[max([k for k in self.chi_list.keys() if k <= self.sweeps])]
            self.trunc_params['chi_max'] = chi_max
            if self.verbose >= 1:
                print("Setting chi_max =", chi_max)
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
        if self.verbose >= 1:
            print("Updating environment")
        for k in range(N_sweeps):
            self.sweep(optimize=False)
            if self.verbose >= 1:
                print('.', end='', flush=True)
        if self.verbose >= 1:
            print("", flush=True)  # end line

    def sweep(self, optimize=True, meas_E_trunc=False):
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
        meas_E_trunc : bool, optional
            Whether to measure truncation energies.

        Returns
        -------
        max_trunc_err : float
            Maximal truncation error introduced.
        max_E_trunc : ``None`` | float
            ``None`` if meas_E_trunc is False, else the maximal change of the energy due to the
            truncation.
        """
        self.E_trunc_list = []
        self.trunc_err_list = []
        schedule = self.get_sweep_schedule()

        # the actual sweep
        for i0, move_right, update_LP_RP in schedule:
            self.i0 = i0
            self.move_right = move_right
            self.update_LP_RP = update_LP_RP
            update_LP, update_RP = update_LP_RP
            if self.verbose >= 10:
                print("in sweep: i0 =", i0)
            # --------- the main work --------------
            theta = self.prepare_update()
            update_data = self.update_local(theta, optimize=optimize)
            if update_LP:
                self.update_LP(update_data['U'])  # (requires updated B)
                for o_env in self.ortho_to_envs:
                    o_env.get_LP(i0 + 1, store=True)
            if update_RP:
                self.update_RP(update_data['VH'])
                for o_env in self.ortho_to_envs:
                    o_env.get_RP(i0, store=True)
            self.post_update_local(update_data, meas_E_trunc)

        if optimize:  # count optimization sweeps
            self.sweeps += 1
            if self.chi_list is not None:
                new_chi_max = self.chi_list.get(self.sweeps, None)
                if new_chi_max is not None:
                    self.trunc_params['chi_max'] = new_chi_max
                    if self.verbose >= 1:
                        print("Setting chi_max =", new_chi_max)
            # update mixer
            if self.mixer is not None:
                self.mixer = self.mixer.update_amplitude(self.sweeps)
        if meas_E_trunc:
            return np.max(self.trunc_err_list), np.max(self.E_trunc_list)
        else:
            return np.max(self.trunc_err_list), None

    def get_sweep_schedule(self):
        """Define the schedule of the sweep.

        One 'sweep' is a full sequence from the leftmost site to the right and
        back. Only those `LP` and `RP` that can be used later should be updated.

        Returns
        -------
        schedule : iterable of (int, bool, (bool, bool))
            Schedule for the sweep. Each entry is ``(i0, move_right, (update_LP, update_RP))``,
            where `i0` is the leftmost of the ``self.EffectiveH.length`` sites to be updated in
            :meth:`update_local`, `move_right` indicates whether the next `i0` in the schedule is
            rigth (`True`) of the current one, and `update_LP`, `update_RP` indicate
            whether it is necessary to update the `LP` and `RP`.
            The latter are chosen such that the environment is growing for infinite systems, but
            we only keep the minimal number of environment tensors in memory.
        """
        L = self.psi.L
        if self.finite:
            n = self.EffectiveH.length
            assert L >= n
            i0s = list(range(0, L - n)) + list(range(L - n, 0, -1))
            move_right = [True] * (L - n) + [False] * (L - n)
            update_LP_RP = [[True, False]] * (L - n) + [[False, True]] * (L - n)
        else:
            assert L >= 2
            i0s = list(range(0, L)) + list(range(L, 0, -1))
            move_right = [True] * L + [False] * L
            update_LP_RP = [[True, True]] * 2 + [[True, False]] * (L-2) + \
                           [[True, True]] * 2 + [[False, True]] * (L-2)
        return zip(i0s, move_right, update_LP_RP)

    def mixer_cleanup(self):
        """Cleanup the effects of a mixer.

        A :meth:`sweep` with an enabled :class:`Mixer` leaves the MPS `psi` with 2D arrays in `S`.
        To recover the originial form, this function simply performs one sweep with disabled mixer.
        """
        if any([self.psi.get_SL(i).ndim > 1 for i in range(self.psi.L)]):
            mixer = self.mixer
            self.mixer = None  # disable the mixer
            self.sweep(optimize=False)  # (discard return value)
            self.mixer = mixer  # recover the original mixer

    def mixer_activate(self):
        """Set `self.mixer` to the class specified by `options['mixer']`.

        It is expected that different algorithms have differen ways of implementing mixers (with
        different defaults). Thus, this is algorithm-specific.
        """
        raise NotImplementedError("needs to be overwritten by subclass")

    def prepare_update(self):
        """Prepare everything algorithm-specific to perform a local update."""
        raise NotImplementedError("needs to be overwritten by subclass")

    def update_local(self, theta, **kwargs):
        """Perform algorithm-specific local update."""
        raise NotImplementedError("needs to be overwritten by subclass")

    def post_update_local(self, **kwargs):
        """Algorithm-specific actions to be taken after local update.

        An example would be to collect statistics.
        """
        raise NotImplementedError("needs to be overwritten by subclass")

    def make_eff_H(self):
        """Create new instance of `self.EffectiveH` at `self.i0` and set it to `self.eff_H`."""
        self.eff_H = self.EffectiveH(self.env, self.i0, self.combine, self.move_right)
        # note: this order of wrapping is most effective.
        if self.env.H.explicit_plus_hc:
            self.eff_H = SumNpcLinearOperator(self.eff_H, self.eff_H.adjoint())
        if len(self.ortho_to_envs) > 0:
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
        Index of the active site if length=1, or of the left-most active site if length>1.
    combine : bool, optional
        Whether to combine legs into pipes as far as possible. This reduces the overhead of
        calculating charge combinations in the contractions.
    move_right : bool, optional
        Whether the sweeping algorithm that calls for an `EffectiveH` is moving to the right.

    Attributes
    ----------
    length : int
        Number of (MPS) sites the effective hamiltonian covers. NB: Class attribute.
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
        self.LP = env.get_LP(i0)
        self.RP = env.get_RP(i0)
        self.W0 = env.H.get_W(i0).replace_labels(['p', 'p*'], ['p0', 'p0*'])
        self.dtype = env.H.dtype
        self.combine = combine
        self.move_right = move_right
        self.N = (self.LP.get_leg('vR').ind_len * self.W0.get_leg('p0').ind_len *
                  self.RP.get_leg('vL').ind_len)
        if combine:
            self.combine_Heff()

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

    def combine_Heff(self):
        """Combine LP and RP with W to form LHeff and RHeff, depending on the direction.

        In a move to the right, we need LHeff. In a move to the left, we need RHeff. Both contain
        the same W.
        """
        # Always compute both L/R, because we might need them. Could change later.
        LHeff = npc.tensordot(self.LP, self.W0, axes=['wR', 'wL'])
        self.pipeL = pipeL = LHeff.make_pipe(['vR*', 'p0'], qconj=+1)
        self.LHeff = LHeff.combine_legs([['vR*', 'p0'], ['vR', 'p0*']],
                                        pipes=[pipeL, pipeL.conj()],
                                        new_axes=[0, 2])
        RHeff = npc.tensordot(self.W0, self.RP, axes=['wR', 'wL'])
        self.pipeR = pipeR = RHeff.make_pipe(['p0', 'vL*'], qconj=-1)
        self.RHeff = RHeff.combine_legs([['p0', 'vL*'], ['p0*', 'vL']],
                                        pipes=[pipeR, pipeR.conj()],
                                        new_axes=[-1, 0])
        if self.move_right:
            self.acts_on = ['(vL.p0)', 'vR']
        else:
            self.acts_on = ['vL', '(p0.vR)']

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
        for key in ['LP', 'RP', 'W0', 'W1']:
            getattr(adj, key).itranspose(getattr(self, key).get_leg_labels())
        if self.combine:
            for key in ['LHeff', 'RHeff']:
                getattr(adj, key).itranspose(getattr(self, key).get_leg_labels())
        return adj


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
        Index of the active site if length=1, or of the left-most active site if length>1.
    combine : bool
        Whether to combine legs into pipes. This combines the virtual and
        physical leg for the left site (when moving right) or right side (when moving left)
        into pipes. This reduces the overhead of calculating charge combinations in the
        contractions, but one :meth:`matvec` is formally more expensive, :math:`O(2 d^3 \chi^3 D)`.
    move_right : bool
        Whether the the sweep is moving right or left for the next update.

    Attributes
    ----------
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
            self.combine_Heff()

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

    def combine_Heff(self):
        """Combine LP and RP with W to form LHeff and RHeff.

        Combine LP with W0 and RP with W1 to get the effective parts of the Hamiltonian with piped
        legs.
        """
        LHeff = npc.tensordot(self.LP, self.W0, axes=['wR', 'wL'])
        self.pipeL = pipeL = LHeff.make_pipe(['vR*', 'p0'], qconj=+1)
        self.LHeff = LHeff.combine_legs([['vR*', 'p0'], ['vR', 'p0*']],
                                        pipes=[pipeL, pipeL.conj()],
                                        new_axes=[0, 2])
        RHeff = npc.tensordot(self.RP, self.W1, axes=['wL', 'wR'])
        self.pipeR = pipeR = RHeff.make_pipe(['p1', 'vL*'], qconj=-1)
        self.RHeff = RHeff.combine_legs([['p1', 'vL*'], ['p1*', 'vL']],
                                        pipes=[pipeR, pipeR.conj()],
                                        new_axes=[2, 1])
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
            adj.LHeff = self.LHeff.conj().ireplace_labels('wR*', 'wR')
            adj.RHeff = self.RHeff.conj().ireplace_labels('wL*', 'wL')
        for key in ['LP', 'RP', 'W0', 'W1']:
            getattr(adj, key).itranspose(getattr(self, key).get_leg_labels())
        if self.combine:
            for key in ['LHeff', 'RHeff']:
                getattr(adj, key).itranspose(getattr(self, key).get_leg_labels())
        return adj
