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

The :class:`VariationalCompression` and :class:`VariationalApplyMPO`
implemented here also directly use the :class:`Sweep` class.

"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import time
import warnings
import copy

from ..linalg import np_conserved as npc
from .truncation import svd_theta, TruncationError
from ..networks.mps import MPSEnvironment
from ..networks.mpo import MPOEnvironment
from ..linalg.sparse import NpcLinearOperator, SumNpcLinearOperator, OrthogonalNpcLinearOperator
from .algorithm import Algorithm

__all__ = [
    'Sweep', 'EffectiveH', 'OneSiteH', 'TwoSiteH', 'VariationalCompression', 'VariationalApplyMPO'
]


class Sweep(Algorithm):
    r"""Prototype class for a 'sweeping' algorithm.

    This is a superclass, intended to cover common procedures in all algorithms that 'sweep'. This
    includes DMRG, TDVP, etc.

    .. todo ::
        TDVP is currently not implemented with the sweep class.

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
        super().__init__(psi, model, options)
        options = self.options

        self.combine = options.get('combine', False)
        self.finite = self.psi.finite

        self.lanczos_params = options.subconfig('lanczos_params')

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

    @property
    def n_optimize(self):
        """the number of sites to be optimized over at once.

        Indirectly set by the class attribute :attr:`EffectiveH` and it's `length`.
        For example, :class:`~tenpy.algorithms.dmrg.TwoSiteDMRGEngine` uses the
        :class:`~tenpy.algorithms.mps_common.TwoSiteH` and hence has ``n_optimize=2``,
        while the :class:`~tenpy.algorithms.dmrg.SingleSiteDMRGEngine` has ``n_optimize=1``.
        """
        return self.EffectiveH.length

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

            init_env_data : dict
                Dictionary as returned by ``self.env.get_initialization_data()`` from
                :meth:`~tenpy.networks.mpo.MPOEnvironment.get_initialization_data`.
            orthogonal_to : list of :class:`~tenpy.networks.mps.MPS`
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
            if self.verbose >= 1:
                print("Initial sweeps...")
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
            self.post_update_local(update_data)

        if optimize:  # count optimization sweeps
            self.sweeps += 1
            if self.chi_list is not None:
                new_chi_max = self.chi_list.get(self.sweeps, None)
                if new_chi_max is not None:
                    self.trunc_params['chi_max'] = new_chi_max
                    if self.verbose >= 1:
                        print("Setting chi_max =", new_chi_max)
        return np.max(self.trunc_err_list)

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

    def prepare_update(self):
        """Prepare everything algorithm-specific to perform a local update."""
        pass  # should usually be overridden by subclassed

    def update_local(self, theta, **kwargs):
        """Perform algorithm-specific local update."""
        raise NotImplementedError("needs to be overridden by subclass")

    def post_update_local(self, update_data):
        """Algorithm-specific actions to be taken after local update.

        An example would be to collect statistics.
        """
        self.trunc_err_list.append(update_data['err'].eps)

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

    def update_LP(self, _):
        self.env.get_LP(self.i0 + 1, store=True)

    def update_RP(self, _):
        self.env.get_RP(self.i0, store=True)


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
        :include: Sweep

        trunc_params : dict
            Truncation parameters as described in :cfg:config:`truncation`.
        N_sweeps : int
            Number of sweeps to perform.

    Attributes
    ----------
    renormalize : list
        Used to keep track of renormalization in the last sweep for `psi.norm`.
    """
    EffectiveH = TwoSiteH

    def __init__(self, psi, options):
        super().__init__(psi, None, options)
        self.renormalize = []

    def run(self):
        """Run the compression.

        The state :attr:`psi` is compressed in place.

        Returns
        -------
        max_trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The maximal truncation error of a two-site wave function.
        """
        N_sweeps = self.options.get("N_sweeps", 2)

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
        if (not self.psi.finite and 'init_LP' not in init_env_data
                and 'init_RP' not in init_env_data):
            start_env_sites = self.options.get("start_env_sites", 0)
            self._init_env_from_start_env_sites(start_env_sites)
        self.reset_stats()

    def _init_env_from_start_env_sites(self, start_env_sites):
        """initialize LP[0] and RP[L-1] to already include `start_env_sites` sites."""
        if start_env_sites is None or start_env_sites == 0 or start_env_sites == np.inf:
            return
        env = self.env
        L = env.L
        LP = env.init_LP(-start_env_sites)
        for i in range(-start_env_sites, 0):
            LP = env._contract_LP(i, LP)
        env.set_LP(0, LP, age=start_env_sites)
        RP = env.init_RP(L - 1 + start_env_sites)
        for i in range(L - 1 + start_env_sites, L - 1, -1):
            RP = env._contract_RP(i, RP)
        env.set_RP(L - 1, RP, age=start_env_sites)

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
        Sweep.__init__(self, psi, U_MPO, options)
        self.renormalize = [None] * (psi.L - int(psi.finite))

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
        if not self.psi.finite:
            start_env_sites = self.options.get("start_env_sites", 1)
            self._init_env_from_start_env_sites(start_env_sites)
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
