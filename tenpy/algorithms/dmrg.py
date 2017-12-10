"""Density Matrix Renormalization Group (DMRG).

Although it was originally not formulated with tensor networks,
the DMRG algorithm (invented by Steven White in 1992 [White1992]_) opened the whole field
with its enormous success in finding ground states in 1D.

We implement DMRG in the modern formulation of matrix product states [Schollwoeck2011]_,
both for finite systems (``'finite'`` or ``'segment'`` boundary conditions)
and in the thermodynamic limit (``'periodic'`` b.c.).

The function :func:`run` - well - runs one DMRG simulation.
Internally, it generates an instance of an :class:`Engine`.
It implements the common functionality like defining a `sweep`,
but leaves the details of the contractions to be performed to the derived classes.

Currently, there are two derived classes implementing the contractions.
They should both give the same results (up to rounding errors).
Which one is in the end faster is not obvious a priory and might depend on the used model.
Just try both of them.

Currently, there is only one :class:`Mixer` implemented.
The mixer should be used initially to avoid that the algorithm gets stuck in local energy minima,
and then slowly turned off in the end.

.. todo ::
    abort on too large NormErr -> need MPS.normerr()
    Need function to plot the statistics in the end
    Write UserGuide/Example!!!
    Allow to keep MPS orthogonal to other states, for finding excited states
"""


import numpy as np
import time
import itertools
import warnings

from ..linalg import np_conserved as npc
from ..networks.mpo import MPOEnvironment
from ..linalg.lanczos import lanczos
from .truncation import truncate, svd_theta
from ..tools.params import get_parameter, unused_parameters
from ..tools.process import memory_usage

__all__ = ['run', 'Engine', 'EngineCombine', 'EngineFracture', 'Mixer']


def run(psi, model, DMRG_params):
    r"""Run the DMRG algorithm to find the ground state of `M`.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        Initial guess for the ground state, which is to be optimized in-place.
    model : :class:`~tenpy.models.MPOModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    DMRG_params : dict
        Further optional parameters as described in the following table.
        Use ``verbose>0`` to print the used parameters during runtime.

        ============== ========= ===============================================================
        key            type      description
        ============== ========= ===============================================================
        LP             npc.Array Initial left-most `LP` and right-most `RP` ('left/right part')
        RP                       of the environment. By default (``None``) generate trivial,
                                 see :class:`~tenpy.networks.mpo.MPOEnvironment` for details.
        -------------- --------- ---------------------------------------------------------------
        LP_age         int       The 'age' (i.e. number of physical sites invovled into the
        RP_age                   contraction) of the left-most `LP` and right-most `RP`
                                 of the environment.
        -------------- --------- ---------------------------------------------------------------
        mixer          str |     Chooses the :class:`Mixer` to be used.
                       class     A string stands for one of the mixers defined in this module,
                                 a class is used as custom mixer.
                                 Default (``None``) uses no mixer, ``True`` uses :class:`Mixer`.
        -------------- --------- ---------------------------------------------------------------
        mixer_params   dict      Non-default initialization arguments of the mixer.
                                 Options may be custom to the specified mixer, so they're
                                 documented in the class doc-string of the mixer.
        -------------- --------- ---------------------------------------------------------------
        engine         str |     Chooses the (derived class of) :class:`Engine` to be used.
                       class     A string stands for one of the engines defined in this module,
                                 a class (not an instance!) can be used as custom engine.
        -------------- --------- ---------------------------------------------------------------
        trunc_params   dict      Truncation parameters as described in
                                 :func:`~tenpy.algorithms.truncation.truncate`
        -------------- --------- ---------------------------------------------------------------
        lanczos_params dict      Lanczos parameters as described in
                                 :func:`~tenpy.linalg.lanczos.lanczos`
        -------------- --------- ---------------------------------------------------------------
        N_sweeps_check int       Number of sweeps to perform between checking convergence
                                 criteria and giving a status update.
        -------------- --------- ---------------------------------------------------------------
        sweep_0        int       The number of sweeps already performed. (Useful for re-start).
        -------------- --------- ---------------------------------------------------------------
        start_env      int       Number of initial sweeps without bond optimizaiton to
                                 initialize the environment.
        -------------- --------- ---------------------------------------------------------------
        update_env     int       Number of sweeps without bond optimizaiton to update the
                                 environment for infinite bc,
                                 performed every `N_sweeps_check` sweeps.
        -------------- --------- ---------------------------------------------------------------
        norm_tol       float     After the DMRG run, update the environment by 'update_env'
                                 sweeps until ``np.linalg.norm(psi.norm_err()) < norm_tol``.
        -------------- --------- ---------------------------------------------------------------
        norm_tol_iter  float     Perform at most `update_env` * `norm_tol_iter` iteration to
                                 converge the norm error below `norm_tol`.
        -------------- --------- ---------------------------------------------------------------
        max_sweeps     int       Maximum number of sweeps to be performed.
        -------------- --------- ---------------------------------------------------------------
        min_sweeps     int       Minimum number of sweeps to be performed.
                                 Defaults to 1.5*N_sweeps_check.
        -------------- --------- ---------------------------------------------------------------
        max_E_err      int       Convergence if the change of the energy in each step
                                 satisfies ``-\Delta E / |E| < max_E_err``. Note that this is
                                 also satisfied if Delta E > 0, i.e. if the energy increases.
        -------------- --------- ---------------------------------------------------------------
        max_S_err      int       Convergence if the relative change of the entropy in each step
                                 satisfies ``|\Delta S|/S < max_S_err``
        -------------- --------- ---------------------------------------------------------------
        max_hours      float     If the DMRG took longer (measured in wall-clock time),
                                 'shelve' the simulation, i.e. stop and return with the flag
                                 ``shelve=True``.
        -------------- --------- ---------------------------------------------------------------
        P_tol_to_trunc float     It's reasonable to choose the Lanczos convergence criteria
        P_tol_max                ``'P_tol'`` not many magnitudes lower than the current
        P_tol_min                truncation error. Therefore, if `P_tol_to_trunc` is not
                                 ``None``, we update `P_tol` of `lanczos_params` to
                                 ``max_trunc_err*P_tol_to_trunc``,
                                 restricted to the interval [`P_tol_min`,`P_tol_max`].
        -------------- --------- ---------------------------------------------------------------
        E_tol_to_trunc float     It's reasonable to choose the Lanczos convergence criteria
        E_tol_max                ``'E_tol'`` not many magnitudes lower than the current
        E_tol_min                truncation error. Therefore, if `E_tol_to_trunc` is not
                                 ``None``, we update `E_tol` of `lanczos_params` to
                                 ``max_E_trunc*E_tol_to_trunc``,
                                 restricted to the interval [`E_tol_min`,`E_tol_max`].
        ============== ========= ===============================================================

    Returns
    -------
    info : dict
        A dictionary with keys ``'E', 'shelve', 'bond_statistics', 'sweep_statistics'``
    """
    # initialize the engine
    Engine_class = get_parameter(DMRG_params, 'engine', 'EngineCombine', 'DMRG')
    if isinstance(Engine_class, str):
        Engine_class = globals()[Engine_class]
    engine = Engine_class(psi, model, DMRG_params)
    verbose = engine.verbose

    # prepare parameters
    chi_max_default = engine.trunc_params.get('chi_max', max(50, np.max(psi.chi)))
    chi_list = get_parameter(DMRG_params, 'chi_list', {0: chi_max_default}, 'DMRG')
    chi_max = chi_list[max([k for k in list(chi_list.keys()) if k <= engine.sweeps])]
    engine.trunc_params['chi_max'] = chi_max
    if verbose >= 1:
        print("Setting chi_max =", chi_max)
    p_tol_to_trunc = get_parameter(DMRG_params, 'P_tol_to_trunc', None, 'DMRG')
    if p_tol_to_trunc is not None:
        p_tol_min = get_parameter(DMRG_params, 'P_tol_min', None, 'DMRG')
        p_tol_max = get_parameter(DMRG_params, 'P_tol_max', None, 'DMRG')
    e_tol_to_trunc = get_parameter(DMRG_params, 'E_tol_to_trunc', None, 'DMRG')
    if e_tol_to_trunc is not None:
        e_tol_min = get_parameter(DMRG_params, 'E_tol_min', None, 'DMRG')
        e_tol_max = get_parameter(DMRG_params, 'E_tol_max', None, 'DMRG')

    # get parameters for DMRG convergence criteria
    N_sweeps_check = get_parameter(DMRG_params, 'N_sweeps_check', 10, 'DMRG')
    min_sweeps = get_parameter(DMRG_params, 'min_sweeps', 1.5 * N_sweeps_check, 'DMRG')
    max_sweeps = get_parameter(DMRG_params, 'max_sweeps', 1000, 'DMRG')
    max_E_err = get_parameter(DMRG_params, 'max_E_err', 0.1, 'DMRG')
    max_S_err = get_parameter(DMRG_params, 'max_S_err', 0.1, 'DMRG')
    max_seconds = 3600 * get_parameter(DMRG_params, 'max_hours', 24 * 365, 'DMRG')
    start_time = time.time()

    # initial sweeps of the environment
    start_env = get_parameter(DMRG_params, 'start_env', 0, 'DMRG')
    # update environement sweeps
    default_update_env = min((N_sweeps_check // 2 + 1), 10)
    if psi.finite:
        default_update_env = 0
    update_env = get_parameter(DMRG_params, 'update_env', default_update_env, 'DMRG')
    norm_tol = get_parameter(DMRG_params, 'norm_tol', 1.e-3, 'DMRG')
    norm_tol_iter = get_parameter(DMRG_params, 'norm_tol_iter', 10, 'DMRG')
    if psi.finite:
        if start_env > 0 or update_env > 0:
            warnings.warn("Ignore `start_env` and `update_env` for finite MPS: nothing to do.")
        start_env = update_env = 0
        norm_tol = None
    engine.environment_sweeps(start_env)

    # initialize statistics
    shelve = False
    E_old, S_old = np.nan, np.nan  # initial dummy values
    E, Delta_E, Delta_S = 1., 1., 1.

    sweep_statistics = {
        'sweep': [],
        'E': [],
        'S': [],
        'max_trunc_err': [],
        'max_E_trunc': [],
        'max_chi': [],
        'norm_err': []
    }

    while True:
        # check abortion criteria
        if engine.sweeps >= max_sweeps:
            break
        if engine.sweeps > min_sweeps and -Delta_E / abs(E) < max_E_err and abs(
                Delta_S) < max_S_err:
            if engine.mixer is None:
                break
            else:
                if verbose >= 1:
                    print("Convergence criterium reached with enabled mixer.\n" +  \
                        "disable mixer and continue")
                    engine.mixer = None
        if time.time() - start_time > max_seconds:
            shelve = True
            warnings.warn("DMRG: maximum time limit reached. Shelve simulation.")
            break
        # the time-consuming part: the actual sweeps
        for i in range(N_sweeps_check):
            # --------- the main work --------------
            max_trunc_err, max_E_trunc = engine.sweep(meas_E_trunc=(i + 1 == N_sweeps_check))
            # --------------------------------------
            if engine.sweeps in chi_list:
                engine.trunc_params['chi_max'] = chi_list[engine.sweeps]
                if verbose >= 1:
                    print("Setting chi_max =", chi_list[engine.sweeps])
        # update lancos_params depending on truncation error(s)
        if p_tol_to_trunc is not None and max_trunc_err > p_tol_min:
            engine.lanczos_params['P_tol'] = max(p_tol_min,
                                                 min(p_tol_max, max_trunc_err * p_tol_to_trunc))
        if e_tol_to_trunc is not None and max_E_trunc > e_tol_min:
            engine.lanczos_params['E_tol'] = max(e_tol_min,
                                                 min(e_tol_max, max_E_trunc * e_tol_to_trunc))
        # update environment
        engine.environment_sweeps(update_env)
        try:
            S = np.average(psi.entanglement_entropy())
            Delta_S = (S - S_old) / N_sweeps_check
        except ValueError:
            S = np.nan
            Delta_S = 0.
        S_old = S
        if not psi.finite:  # iDMRG: need energy density
            Es = engine.statistics['E_total']
            age = engine.statistics['age']
            if N_sweeps_check > 1:
                growth = (age[-1] - age[-1 - 2 * engine.env.L])
                E = (Es[-1] - Es[-1 - 2 * engine.env.L]) / growth
            else:
                E = (Es[-1] - Es[0]) / (age[-1] - age[0])
        else:
            E = engine.statistics['E_total'][-1]
        Delta_E = E - E_old
        E_old = E
        norm_err = np.linalg.norm(psi.norm_test())
        sweep_statistics['sweep'].append(engine.sweeps)
        sweep_statistics['E'].append(E)
        sweep_statistics['S'].append(S)
        sweep_statistics['max_trunc_err'].append(max_trunc_err)
        sweep_statistics['max_E_trunc'].append(max_E_trunc)
        sweep_statistics['max_chi'].append(np.max(psi.chi))
        sweep_statistics['norm_err'].append(norm_err)

        if verbose >= 1:
            # print a status update
            print("=" * 80)
            msg = "sweep {sweep:d}, age = {age:d}\n"
            msg += "Energy = {E:.16f}, norm_err = {norm_err:.1e}\n"
            msg += "Current memory usage {mem:.1f} MB, time elapsed: {time:.1f} s\n"
            msg += "Delta E = {DE:.4e}, Delta S = {DS:.4e} (per sweep)\n"
            msg += "max_trunc_err = {trerr:.4e}, max_E_trunc = {Eerr:.4e}\n"
            msg += "MPS bond dimensions: {chi!s}"
            print(msg.format(
                sweep=engine.sweeps,
                time=time.time() - start_time,
                mem=memory_usage(),
                chi=psi.chi,
                age=engine.statistics['age'][-1],
                E=E,
                DE=Delta_E,
                DS=Delta_S,
                trerr=max_trunc_err,
                Eerr=max_E_trunc,
                norm_err=norm_err))
    # clean up from mixer
    engine.mixer_cleanup(optimize=False)
    # update environment until norm_tol is reached
    if norm_tol is not None and norm_err > norm_tol:
        warnings.warn("final DMRG state not in canonical form: too much truncation!")
        if psi.finite:
            psi.canonical_form()
        else:
            for _ in range(norm_tol_iter):
                engine.environment_sweeps(update_env)
                norm_err = np.linalg.norm(psi.norm_test())
                if norm_err <= norm_tol:
                    break

    if verbose >= 1:
        print("=" * 80)
        msg = "DMRG finished after {sweep:d} sweeps.\n"
        msg += "Age (=total size) = {age:d}, maximum chi = {chimax}"
        print(msg.format(
            sweep=engine.sweeps, age=engine.statistics['age'][-1], chimax=np.max(psi.chi)))
        print("=" * 80)
    unused_parameters(DMRG_params['lanczos_params'], "DMRG")
    unused_parameters(DMRG_params['trunc_params'], "DMRG")
    unused_parameters(DMRG_params, "DMRG")
    return {
        'E': E,
        'shelve': shelve,
        'bond_statistics': engine.statistics,
        'sweep_statistics': sweep_statistics
    }


class Engine(object):
    """Prototype for an DMRG 'Engine'.

    This class is the working horse of :func:`DMRG`. It implements the :meth:`sweep` and large
    parts of the (two-site) optimization.
    During the diagonalization (i.e. after calling :meth:`prepare_diag`), the class represents
    the effective two-site Hamiltonian, which looks like this::

        |        .---            ----.
        |        |     |      |      |
        |        LP----W[i0]--W[i1]--RP
        |        |     |      |      |
        |        .---            ----.

    `LP` and `RP` are left and right parts of the :class:`~tenpy.networks.mpo.MPOEnvironment`,
    `W[i0]` and `W[i1]` are the MPO matrices of the Hamiltonian at the two sites ``i0, i1=i0+1``.
    How this network is then actually contracted in detail is left to derived classes.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        Initial guess for the ground state, which is to be optimized in-place.
    model : :class:`~tenpy.models.MPOModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    DMRG_params : dict
        Further optional parameters. See :func:`run` for more details.

    Attributes
    ----------
    verbose : int
        Level of verbosity (i.e. how much status information to print); higher=more output.
    sweeps : int
        The number of performed sweeps (with ``optimize=True``, i.e. not counting the
        environment updates).
    statistics : dict
        A dictionary with detailed statistics of the convergence.
        For each key in the following table, the dictionary contains a list where one value is
        added each :meth:`Engine.update_bond`.

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
        =========== ===================================================================

    env : :class:`~tenpy.networks.mpo.MPOEnvironment`
        Environment for contraction ``<psi|H|psi>``.
    mixer : :class:`Mixer` | ``None``
        If ``None``, no mixer is used, otherwise the mixer instance.
    lanczos_params : dict
        Parameters for :func:`~tenpy.linalg.lanczos.lanczos`.
    trunc_params : dict
        Parameters for :func:`~tenpy.algorithms.truncation.truncate`.
    """

    def __init__(self, psi, model, DMRG_params):
        self.verbose = get_parameter(DMRG_params, 'verbose', 1, 'DMRG')
        self.sweeps = get_parameter(DMRG_params, 'sweep_0', 0, 'DMRG')
        self.statistics = {'i0': [], 'age': [], 'E_total': [], 'N_lanczos': []}

        # set up environment
        LP = get_parameter(DMRG_params, 'LP', None, 'DMRG')
        RP = get_parameter(DMRG_params, 'RP', None, 'DMRG')
        LP_age = get_parameter(DMRG_params, 'LP_age', 0, 'DMRG')
        RP_age = get_parameter(DMRG_params, 'RP_age', 0, 'DMRG')
        self.env = MPOEnvironment(psi, model.H_MPO, psi, LP, RP, LP_age, RP_age)
        # (checks compatibility of psi with model)

        # generate mixer instance, if a mixer is to be used.
        self.mixer = None  # means 'ignore mixer'
        Mixer_class = get_parameter(DMRG_params, 'mixer', None, 'DMRG')
        if Mixer_class is not None:
            if Mixer_class is True:
                Mixer_class = Mixer
            if isinstance(Mixer_class, str):
                Mixer_class = globals()[Mixer_class]
            mixer_params = get_parameter(DMRG_params, 'mixer_params', {}, 'DMRG')
            mixer_params.setdefault('verbose', self.verbose / 10)  # reduced verbosity
            self.mixer = Mixer_class(mixer_params)

        self.lanczos_params = get_parameter(DMRG_params, 'lanczos_params', {}, 'DMRG')
        self.lanczos_params.setdefault('verbose', self.verbose / 10)  # reduced verbosity

        self.trunc_params = get_parameter(DMRG_params, 'trunc_params', {}, 'DMRG')
        self.trunc_params.setdefault('verbose', self.verbose / 10)  # reduced verbosity

    def environment_sweeps(self, N_sweeps):
        """Perform `N_sweeps` sweeps without bond optimization to update the environment."""
        if N_sweeps <= 0:
            return
        if self.verbose >= 1:
            print("Updating environment")
        for k in range(N_sweeps):
            self.sweep(optimize=False)
            if self.verbose >= 1:
                print('.', end=' ')
        if self.verbose >= 1:
            print("")  # end line

    def sweep(self, optimize=True, meas_E_trunc=False):
        """One 'sweep' of the DMRG algorithm.

        Iteratate over the bond which is optimized, to the right and
        then back to the left to the starting point.
        If optimize=False, don't actually diagonalize the effective hamiltonian,
        but only update the environment.

        Parameters
        ----------
        optimize : bool
            Wheter we actually optimize to find the ground state of the effective Hamiltonian.
            (If False, just update the environments).
        meas_E_trunc : bool
            Wheter to measure the energy after truncation.

        Returns
        -------
        max_trunc_err : float
            Maximal truncation error introduced.
        max_E_trunc : ``None`` | float
            ``None`` if meas_E_trunc is False, else the maximal change of the energy due to the
            truncation.
        """
        E_trunc_list = []
        trunc_err_list = []
        # get schedule
        L = self.env.L
        if self.env.finite:
            schedule_i0 = list(range(0, L - 1)) + list(range(L - 3, 0, -1))
            update_env = [[True, False]] * (L - 2) + [[False, True]] * (L - 2)
        else:
            assert (L >= 2)
            schedule_i0 = list(range(0, L)) + list(range(L, 0, -1))
            update_env = [[True, True]] * 2 + [[True, False]] * (L-2) + \
                         [[True, True]] * 2 + [[False, True]] * (L-2)

        # the actual sweep
        for i0, upd_env in zip(schedule_i0, update_env):
            if self.verbose >= 10:
                print("in sweep: i0 =", i0)
            # --------- the main work --------------
            E_total, E_trunc, trunc_err, N_lanczos, age = self.update_bond(
                i0, upd_env[0], upd_env[1], optimize=optimize, meas_E_trunc=meas_E_trunc)
            # collect statistics
            self.statistics['i0'].append(i0)
            self.statistics['age'].append(age)
            self.statistics['E_total'].append(E_total)
            self.statistics['N_lanczos'].append(N_lanczos)
            E_trunc_list.append(E_trunc)
            trunc_err_list.append(trunc_err.eps)

        if optimize:  # count optimization sweeps
            self.sweeps += 1

        # update mixer
        if self.mixer is not None:
            self.mixer = self.mixer.update_amplitude(self.sweeps)
        if meas_E_trunc:
            return np.max(trunc_err_list), np.max(E_trunc_list)
        else:
            return np.max(trunc_err_list), None

    def update_bond(self, i0, update_LP, update_RP, optimize=True, meas_E_trunc=False):
        """Perform bond-update on the sites ``(i0, i0+1)``.

        Parameters
        ----------
        i0 : int
            Site left to the bond which should be optimized.
        update_LP : bool
            Whether to calculate the next ``env.LP[i0+1]``.
        update_LP : bool
            Whether to calculate the next ``env.RP[i0]``.
        optimize : bool
            Wheter we actually optimize to find the ground state of the effective Hamiltonian.
            (If False, just update the environments).
        meas_E_trunc : bool
            Wheter to measure the energy after truncation.

        Returns
        -------
        E_total : float
            Total energy, obtained *before* truncation (if ``optimize=True``),
            or *after* truncation (if ``optimize=False``) (but never ``None``).
        E_trunc : float | ``None``
            The energy difference of the total energy after minus before truncation,
            ``E_truncated - E_total``. ``None`` if ``meas_E_trunc=False``.
        err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The truncation error introduced after bond optimization.
        N_lanczos : int
            Dimension of the Krylov space used for optimization in the lanczos algorithm.
            0 if ``optimize=False``.
        age : int
            Current size of the DMRG simulation: number of physical sites involved
            into the contraction.
        """
        theta = self.prepare_diag(i0, update_LP, update_RP)
        age = self.env.get_LP_age(i0) + 2 + self.env.get_RP_age(i0 + 1)
        if optimize:
            E0, theta, N = self.diag(theta)
        else:
            E0, N = None, 0
        theta = self.prepare_svd(theta)
        U, S, VH, err = self.mixed_svd(theta, i0, update_LP, update_RP)
        self.set_B(i0, U, S, VH)
        if update_LP:
            self.update_LP(i0, U)  # (requires updated B)
        if update_RP:
            self.update_RP(i0, VH)
        E_trunc = None
        if meas_E_trunc or E0 is None:
            E_trunc = self.env.full_contraction(i0).real
            if E0 is None:
                E0 = E_trunc
            E_trunc = E_trunc - E0
        return E0, E_trunc, err, N, age

    def prepare_diag(self, i0):
        """Prepare `self` to represent the effective Hamiltonian on sites ``(i0, i0+1)``.

        Parameters
        ----------
        i0 : int
            We want to optimize on sites ``(i0, i0+1)``.

        Returns
        -------
        theta_guess : :class:`~tenpy.linalg.np_conserved.Array`
            Current best guess for the ground state, which is to be optimized.
        """
        raise NotImplementedError("This function should be implemented in derived classes")

    def diag(self, theta_guess):
        """Diagonalize the effective Hamiltonian represented by self.

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
            Number of Lanczos iterations used.
        """
        E, theta, N = lanczos(self, theta_guess, self.lanczos_params)
        return E, theta, N

    def matvec(self, theta):
        r"""Apply the effective Hamiltonian to `theta`.

        This function turns :class:`Engine` to a linear operator, which can be
        used for :func:`~tenpy.linalg.lanczos.lanczos`. Pictorially::

            |        .----theta---.
            |        |    |   |   |
            |       LP----H0--H1--RP
            |        |    |   |   |
            |        .---       --.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Wave function to apply the effective Hamiltonian to.

        Returns
        -------
        H_theta : :class:`~tenpy.linalg.np_conserved.Array`
            Result of applying the effective Hamiltonian to `theta`, :math:`H |\theta>`.
        """
        raise NotImplementedError("This function should be implemented in derived classes")

    def prepare_svd(self, theta):
        """Transform theta into a matrix for svd.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Ground state of the effective Hamiltonian as returned by `diag`.

        Returns
        -------
        theta_matrix : :class:`~tenpy.linalg.np_conserved.Array`
            Same as `theta`, but with legs combined into a 2D array for svd partition.
        """
        raise NotImplementedError("This function should be implemented in derived classes")

    def mixed_svd(self, theta, i0, update_LP, update_RP):
        """Get (truncated) `B` from the new theta (as returned by diag).

        The goal ist to split theta and truncate it::

            |   -- theta --   ==>    -- U -- S --  VH -
            |      |   |                |          |

        Whithout a mixer, this is done by a simple svd and truncation of Schmidt values.

        Whith a mixer, we calculate the left and right reduced density using the mixer
        (which might include applications of `H`).
        These density matrices are diagonalized and truncated such that we effectively perform
        a svd for the case ``mixer.amplitude=0``.
        Note that the returned `S` is a general (not diagonal) matrix, with labels ``'vL', 'vR'``.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            The optimized wave function, prepared for svd.
        i0 : int
            Site index; `theta` lives on ``i0, i0+1``.
        update_LP : bool
            Whether to calculate the next ``env.LP[i0+1]``.
        update_RP : bool
            Whether to calculate the next ``env.RP[i0]``.

        Returns
        -------
        U : :class:`~tenpy.linalg.np_conserved.Array`
            Left-canonical part of `theta`. Labels ``'(vL.p0)', 'vR'``.
        S : 1D ndarray | 2D :class:`~tenpy.linalg.np_conserved.Array`
            Without mixer just the singluar values of the array; with mixer it might be a general
            matrix; see comment above.
        VH : :class:`~tenpy.linalg.np_conserved.Array`
            Right-canonical part of `theta`. Labels ``'vL', '(vR.p1)'``.
        err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The truncation error introduced.
        """
        # get qtotal_LR from i0
        qtotal_i0 = self.env.ket.get_B(i0, form=None).qtotal
        if self.mixer is None:
            # simple case: real svd, devined elsewhere.
            U, S, VH, err, _ = svd_theta(
                theta, self.trunc_params, qtotal_LR=[qtotal_i0, None], inner_labels=['vR', 'vL'])
            return U, S, VH, err
        rho_L = self.mix_rho_L(theta, i0, update_LP)
        # don't mix left parts, when we're going to the right
        rho_L.itranspose(['(vL.p0)', '(vL*.p0*)'])  # just to be sure of the order
        rho_R = self.mix_rho_R(theta, i0, update_RP)
        rho_R.itranspose(['(vR.p1)', '(vR*.p1*)'])  # just to be sure of the order

        # consider the SVD `theta = U S V^H` (with real, diagonal S>0)
        # rho_L ~=  theta theta^H = U S V^H V S U^H = U S S U^H  (for mixer -> 0)
        # Thus, rho_L U = U S S, i.e. columns of U are the eigenvectors of rho_L,
        # eigenvalues are S^2.
        val_L, U = npc.eigh(rho_L)
        U.legs[1] = U.legs[1].to_LegCharge()  # explicit conversion: avoid warning in `iproject`
        U.iset_leg_labels(['(vL.p0)', 'vR'])
        val_L[val_L < 0.] = 0.  # for stability reasons
        val_L /= np.sum(val_L)
        keep_L, _, errL = truncate(np.sqrt(val_L), self.trunc_params)
        U.iproject(keep_L, axes='vR')  # in place
        # rho_R ~=  theta^T theta^* = V^* S U^T U* S V^T = V^* S S V^T  (for mixer -> 0)
        # Thus, rho_L V^* = V^* S S, i.e. columns of V^* are eigenvectors of rho_L
        val_R, Vc = npc.eigh(rho_R)
        Vc.legs[1] = Vc.legs[1].to_LegCharge()
        Vc.iset_leg_labels(['(vR.p1)', 'vL'])
        VH = Vc.itranspose(['vL', '(vR.p1)'])
        val_R[val_R < 0.] = 0.  # for stability reasons
        val_R /= np.sum(val_R)
        keep_R, _, err_R = truncate(np.sqrt(val_R), self.trunc_params)
        VH.iproject(keep_R, axes='vL')

        # calculate S = U^H theta V
        theta = npc.tensordot(U.conj(), theta, axes=['(vL*.p0*)', '(vL.p0)'])  # axes 0, 0
        theta = npc.tensordot(theta, VH.conj(), axes=['(vR.p1)', '(vR*.p1*)'])  # axes 1, 1
        theta.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])  # for left/right
        # normalize `S` (as in svd_theta) to avoid blowing up numbers
        theta /= np.linalg.norm(npc.svd(theta, compute_uv=False))
        return U, theta, VH, errL + err_R

    def mix_rho_L(self, theta, i0, mix_enabled):
        """Calculated mixed reduced density matrix for left site.

        Pictorially::

            |     mix_enabled=False           mix_enabled=True
            |
            |    .---theta---.            .---theta-------.
            |    |   |   |   |            |   |   |       |
            |            |   |           LP---H0--H1--.   |
            |    |   |   |   |            |   |   |   |   |
            |    .---theta*--.                    |   xR  |
            |                             |   |   |   |   |
            |                            LP*--H0*-H1*-.   |
            |                             |   |   |       |
            |                             .---theta*------.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Ground state of the effective Hamiltonian, prepared for svd.
        i0 : int
            Site index; `theta` lives on ``i0, i0+1``.
        mix_enabled : bool
            Wheter we actually use mix_R or not.

        Returns
        -------
        rho_L : :class:`~tenpy.linalg.np_conserved.Array`
            A (hermitian) square array with labels ``'(vL.p0)', '(vL*.p0*)'``,
            Mainly the reduced density matrix of the left part, but with some additional mixing.
        """
        raise NotImplementedError("This function should be implemented in derived classes")

    def mix_rho_R(self, theta, i0, mix_enabled):
        """Calculated mixed reduced density matrix for left site.

        Pictorially::

            |     mix_enabled=False           mix_enabled=True
            |
            |    .---theta---.           .------theta---.
            |    |   |   |   |           |      |   |   |
            |    |   |                   |   .--H0--H1--RP
            |    |   |   |   |           |   |  |   |   |
            |    .---theta*--.           |  wL  |
            |                            |   |  |   |   |
            |                            |   .--H0*-H1*-RP*
            |                            |      |   |   |
            |                            .------theta*--.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Ground state of the effective Hamiltonian, prepared for svd.
        i0 : int
            Site index; `theta` lives on ``i0, i0+1``.
        mix_enabled : bool
            Wheter to actually use the mixer or not.

        Returns
        -------
        rho_R : :class:`~tenpy.linalg.np_conserved.Array`
            A (hermitian) square array with labels ``'(vR.p1)', '(vR*.p1*)'``.
            Mainly the reduced density matrix of the right part, but with some additional mixing.
        """
        raise NotImplementedError("This function should be implemented in derived classes")

    def mixer_cleanup(self, *args, **kwargs):
        """Cleanup the effects of the mixer.

        A :meth:`sweep` with an enabled :class:`Mixer` leaves the MPS `psi` with 2D arrays in `S`.
        To recover the originial form, this function simply performs a sweep with disabled mixer.
        """
        if self.mixer is not None:
            mixer = self.mixer
            self.mixer = None  # disable the mixer
            self.sweep(*args, **kwargs)  # (discard return value)
            self.mixer = mixer  # recover the original mixer

    def set_B(self, i0, U, S, VH):
        """Update the MPS with the ``U, S, VH`` returned by `self.mixed_svd`.

        Parameters
        ----------
        i0 : int
            We update the MPS `B` at sites ``i0, i0+1``.
        U, VH : :class:`~tenpy.linalg.np_conserved.Array`
            Left and Right-canonical matrices as returned by the SVD.
        S : 1D array | 2D :class:`~tenpy.linalg.np_conserved.Array`
            The middle part returned by the SVD, ``theta = U S VH``.
            Without a mixer just the singular values, with enabled `mixer` a 2D array.
        """
        B0 = U.split_legs(['(vL.p0)']).replace_label('p0', 'p')
        B1 = VH.split_legs(['(vR.p1)']).replace_label('p1', 'p')
        self.env.ket.set_B(i0, B0, form='A')  # left-canonical
        self.env.ket.set_B(i0 + 1, B1, form='B')  # right-canonical
        self.env.del_LP(i0 + 1)  # the old stored environments are now invalid
        self.env.del_RP(i0)
        self.env.ket.set_SR(i0, S)

    def update_LP(self, i0, U):
        """Update left part of the environment.

        Parameters
        ----------
        i0 : int
            Site index. We calculate ``self.env.get_LP(i0+1)``.
        U : :class:`~tenpy.linalg.np_conserved.Array`
            The U as returned by SVD with combined legs, labels ``'(vL.p0)', 'vR'``.
        """
        raise NotImplementedError("This function should be implemented in derived classes")

    def update_RP(self, i0, VH):
        """Update right part of the environment.

        Parameters
        ----------
        i0 : int
            Site index. We calculate ``self.env.get_RP(i0)``.
        VH : :class:`~tenpy.linalg.np_conserved.Array`
            The VH as returned by SVD with combined legs, labels ``'vL', '(vR.p1)'``.
        """
        raise NotImplementedError("This function should be implemented in derived classes")


class EngineCombine(Engine):
    """Engine which combines legs into pipes as far as possible.

    This engine combines the virtual and physical leg for the left site and right site into pipes.
    This reduces the overhead of calculating charge combinations in the contractions,
    but one :meth:`matvec` is more expensive, :math:`O(2 d^3 \chi^3 D)`.

    Attributes
    ----------
    LHeff: :class:`~tenpy.linalg.np_conserved.Array`
        Left part of the effective Hamiltonian.
        Labels ``'(vR*.p0)', 'wR', '(vR.p0*)'`` for bra, MPO, ket.
    RHeff: :class:`~tenpy.linalg.np_conserved.Array`
        Right part of the effective Hamiltonian.
        Labels ``'(vL.p1*)', 'wL', '(vL*.p1)'`` for ket, MPO, bra.
    """

    def prepare_diag(self, i0, update_LP, update_RP):
        """Prepare `self` to represent the effective Hamiltonian on sites ``(i0, i0+1)``.

        Parameters
        ----------
        i0 : int
            We want to optimize on sites ``(i0, i0+1)``.

        Returns
        -------
        theta_guess : :class:`~tenpy.linalg.np_conserved.Array`
            Current best guess for the ground state, which is to be optimized.
            Labels ``'(vL.p0)', '(vR.p1)'``.
        """
        env = self.env
        LP = env.get_LP(i0, store=True)  # labels 'vR*', 'wR', 'vR'
        H1 = env.H.get_W(i0).replace_labels(['p', 'p*'], ['p0', 'p0*'])  # 'wL', 'wR', 'p0', 'p0*'
        RP = env.get_RP(i0 + 1, store=True)  # labels 'vL*', 'wL', 'vL'
        H2 = env.H.get_W(i0 + 1).replace_labels(['p', 'p*'],
                                                ['p1', 'p1*'])  # ('wL', 'wR', 'p1', 'p1*')
        # calculate LHeff
        LHeff = npc.tensordot(LP, H1, axes=['wR', 'wL'])
        pipeL = LHeff.make_pipe(['vR*', 'p0'])
        self.LHeff = LHeff.combine_legs(
            [['vR*', 'p0'], ['vR', 'p0*']], pipes=[pipeL, pipeL.conj()], new_axes=[0, -1])
        # calculate RHeff
        RHeff = npc.tensordot(RP, H2, axes=['wL', 'wR'])
        pipeR = RHeff.make_pipe(['vL*', 'p1'])
        self.RHeff = RHeff.combine_legs(
            [['vL*', 'p1'], ['vL', 'p1*']], pipes=[pipeR, pipeR.conj()], new_axes=[-1, 0])
        # make theta
        cutoff = 1.e-16 if self.mixer is None else 1.e-8
        theta = env.ket.get_theta(i0, n=2, cutoff=cutoff)  # labels 'vL', 'vR', 'p0', 'p1'
        theta = theta.combine_legs([['vL', 'p0'], ['vR', 'p1']], pipes=[pipeL, pipeR])
        return theta

    def matvec(self, theta):
        r"""Apply the effective Hamiltonian to `theta`.


        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Wave function to apply the effective Hamiltonian to.

        Returns
        -------
        H_theta : :class:`~tenpy.linalg.np_conserved.Array`
            The effective Hamiltonian applied to the wave function, :math:`H |\theta>`.
        """
        labels = theta.get_leg_labels()
        theta = npc.tensordot(self.LHeff, theta, axes=['(vR.p0*)', '(vL.p0)'])
        theta = npc.tensordot(theta, self.RHeff, axes=[['(vR.p1)', 'wR'], ['(vL.p1*)', 'wL']])
        theta.ireplace_labels(['(vR*.p0)', '(vL*.p1)'], ['(vL.p0)', '(vR.p1)'])
        theta.itranspose(labels)  # if necessary, transpose
        return theta

    def prepare_svd(self, theta):
        """Transform theta into matrix for svd."""
        return theta  # For this engine nothing to do.

    def mix_rho_L(self, theta, i0, mix_enabled):
        """Calculated mixed reduced density matrix for left site.

        Pictorially::

            |     mix_enabled=False           mix_enabled=True
            |
            |    .---theta---.            .---theta-------.
            |    |   |   |   |            |   |   |       |
            |            |   |           LP---H0--H1--.   |
            |    |   |   |   |            |   |   |   |   |
            |    .---theta*--.                    |   xR  |
            |                             |   |   |   |   |
            |                            LP*--H0*-H1*-.   |
            |                             |   |   |       |
            |                             .---theta*------.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Ground state of the effective Hamiltonian, prepared for svd.
        i0 : int
            Site index; `theta` lives on ``i0, i0+1``.
        mix_enabled : bool
            Wheter we actually use mix_R or not.

        Returns
        -------
        rho_L : :class:`~tenpy.linalg.np_conserved.Array`
            A (hermitian) square array with labels ``'(vL.p0)', '(vL*.p0*)'``,
            Mainly the reduced density matrix of the left part, but with some additional mixing.
        """
        if not mix_enabled:
            return npc.tensordot(theta, theta.conj(), axes=['(vR.p1)', '(vR*.p1*)'])
        H = self.env.H
        H1 = H.get_W(i0 + 1).replace_labels(['p', 'p*'], ['p1', 'p1*'])
        mixer_xR, add_separate_Id = self.mixer.get_xR(
            H1.get_leg('wR'), H.get_IdL(i0 + 2), H.get_IdR(i0 + 1))
        rho = npc.tensordot(self.LHeff, theta.split_legs('(vR.p1)'), axes=['(vR.p0*)', '(vL.p0)'])
        rho = npc.tensordot(rho, H1, axes=[['p1', 'wR'], ['p1*', 'wL']])
        rho_c = rho.conj()
        rho = npc.tensordot(rho, mixer_xR, axes=['wR', 'wL'])
        rho = npc.tensordot(rho, rho_c, axes=(['p1', 'wL*', 'vR'], ['p1*', 'wR*', 'vR*']))
        rho = rho.ireplace_labels(['(vR*.p0)', '(vR.p0*)'], ['(vL.p0)', '(vL*.p0*)'])
        if add_separate_Id:
            rho = rho + npc.tensordot(theta, theta.conj(), axes=['(vR.p1)', '(vR*.p1*)'])
        return rho

    def mix_rho_R(self, theta, i0, mix_enabled):
        """Calculated mixed reduced density matrix for left site.

        Pictorially::

            |     mix_enabled=False           mix_enabled=True
            |
            |    .---theta---.           .------theta---.
            |    |   |   |   |           |      |   |   |
            |    |   |                   |   .--H0--H1--RP
            |    |   |   |   |           |   |  |   |   |
            |    .---theta*--.           |  wL  |
            |                            |   |  |   |   |
            |                            |   .--H0*-H1*-RP*
            |                            |      |   |   |
            |                            .------theta*--.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Ground state of the effective Hamiltonian, prepared for svd.
        i0 : int
            Site index; `theta` lives on ``i0, i0+1``.
        mix_enabled : bool
            Wheter to actually use the mixer or not.

        Returns
        -------
        rho_R : :class:`~tenpy.linalg.np_conserved.Array`
            A (hermitian) square array with labels ``'(vR.p1)', '(vR*.p1*)'``.
            Mainly the reduced density matrix of the right part, but with some additional mixing.
        """
        if not mix_enabled:
            return npc.tensordot(theta, theta.conj(), axes=['(vL.p0)', '(vL*.p0*)'])
        H = self.env.H
        H0 = H.get_W(i0).replace_labels(['p', 'p*'], ['p0', 'p0*'])
        mixer_xL, add_separate_Id = self.mixer.get_xL(
            H0.get_leg('wL'), H.get_IdL(i0), H.get_IdR(i0 - 1))
        rho = npc.tensordot(self.RHeff, theta.split_legs('(vL.p0)'), axes=['(vL.p1*)', '(vR.p1)'])
        rho = npc.tensordot(rho, H0, axes=[['p0', 'wL'], ['p0*', 'wR']])
        rho_c = rho.conj()
        rho = npc.tensordot(rho, mixer_xL, axes=['wL', 'wR'])
        rho = npc.tensordot(rho, rho_c, axes=(['p0', 'wR*', 'vL'], ['p0*', 'wL*', 'vL*']))
        rho.ireplace_labels(['(vL*.p1)', '(vL.p1*)'], ['(vR.p1)', '(vR*.p1*)'])
        if add_separate_Id:
            rho = rho + npc.tensordot(theta, theta.conj(), axes=['(vL.p0)', '(vL*.p0*)'])
        return rho

    def update_LP(self, i0, U):
        """Update left part of the environment.

        Parameters
        ----------
        i0 : int
            Site index. We calculate ``self.env.get_LP(i0+1)``.
        U : :class:`~tenpy.linalg.np_conserved.Array`
            The U as returned by SVD with combined legs, labels ``'(vL.p0)', 'vR'``.
        """
        # make use of self.LHeff
        LP = npc.tensordot(self.LHeff, U, axes=['(vR.p0*)', '(vL.p0)'])
        LP = npc.tensordot(U.conj(), LP, axes=['(vL*.p0*)', '(vR*.p0)'])
        self.env.set_LP(i0 + 1, LP, age=self.env.get_LP_age(i0) + 1)

    def update_RP(self, i0, VH):
        """Update right part of the environment.

        Parameters
        ----------
        i0 : int
            Site index. We calculate ``self.env.get_RP(i0)``.
        VH : :class:`~tenpy.linalg.np_conserved.Array`
            The U as returned by SVD, with combined legs, labels ``'vL', '(vR.p1)'``.
        """
        # make use of self.RHeff
        RP = npc.tensordot(self.RHeff, VH, axes=['(vL.p1*)', '(vR.p1)'])
        RP = npc.tensordot(VH.conj(), RP, axes=['(vR*.p1*)', '(vL*.p1)'])
        self.env.set_RP(i0, RP, age=self.env.get_RP_age(i0 + 1) + 1)


class EngineFracture(Engine):
    """Engine which keeps the legs separate.

    Due to a different contraction order in :meth:`matvec`, this engine might be faster than
    :class:`EngineCombine`, at least for large physical dimensions and if the MPO is sparse.
    One :meth:`matvec` is :math:`O(2 \chi^3 d^2 D + 2 \chi^2 d^3 W^2 )`.

    Attributes
    ----------
    LP: :class:`~tenpy.linalg.np_conserved.Array`
        Left part of the effective Hamiltonian. Labels ``'vR*', 'wR', 'vR'``.
    RP: :class:`~tenpy.linalg.np_conserved.Array`
        Right part of the effective Hamiltonian. Labels ``'vL*', 'wL', 'vL'``.
    H0, H1: :class:`~tenpy.linalg.np_conserved.Array`
        MPO on the two sites to be optimized.
        Labels ``'wL, 'wR', 'p0', 'p0*'`` and ``'wL, 'wR', 'p1', 'p1*'``.
    """

    def prepare_diag(self, i0, update_LP, update_RP):
        """Prepare `self` to represent the effective Hamiltonian on sites ``(i0, i0+1)``.

        Parameters
        ----------
        i0 : int
            We want to optimize on sites ``(i0, i0+1)``.

        Returns
        -------
        theta_guess : :class:`~tenpy.linalg.np_conserved.Array`
            Current best guess for the ground state, which is to be optimized.
            Labels ``'vL', 'p0', 'vR', 'p1'``.
        """
        env = self.env
        self.LP = env.get_LP(i0, store=True)  # labels 'vR*', 'wR', 'vR'
        self.H0 = env.H.get_W(i0).replace_labels(['p', 'p*'],
                                                 ['p0', 'p0*'])  # 'wL', 'wR', 'p0', 'p0*'
        self.RP = env.get_RP(i0 + 1, store=True)  # labels 'vL*', 'wL', 'vL'
        self.H1 = env.H.get_W(i0 + 1).replace_labels(['p', 'p*'],
                                                     ['p1', 'p1*'])  # 'wL', 'wR', 'p1', 'p1*'
        # make theta
        cutoff = 1.e-16 if self.mixer is None else 1.e-8
        theta = env.ket.get_theta(i0, n=2, cutoff=cutoff)  # labels 'vL', 'vR', 'p0', 'p1'
        return theta.itranspose(['vL', 'p0', 'vR', 'p1'])

    def matvec(self, theta):
        r"""Apply the effective Hamiltonian to `theta`.


        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Wave function to apply the effective Hamiltonian to.

        Returns
        -------
        H_theta : :class:`~tenpy.linalg.np_conserved.Array`
            Wave function to apply the effective Hamiltonian to,  :math:`H |\theta>`
        """
        labels = theta.get_leg_labels()
        theta = npc.tensordot(self.LP, theta, axes=['vR', 'vL'])
        theta = npc.tensordot(theta, self.H0, axes=[['wR', 'p0'], ['wL', 'p0*']])
        theta = npc.tensordot(theta, self.H1, axes=[['wR', 'p1'], ['wL', 'p1*']])
        theta = npc.tensordot(theta, self.RP, axes=[['wR', 'vR'], ['wL', 'vL']])
        theta.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
        theta.itranspose(labels)  # if necessary, transpose
        return theta

    def prepare_svd(self, theta):
        """Transform theta into matrix for svd."""
        return theta.combine_legs([['vL', 'p0'], ['vR', 'p1']], new_axes=[0, 1])

    def mix_rho_L(self, theta, i0, mix_enabled):
        """Calculated mixed reduced density matrix for left site.

        Pictorially::

            |     mix_enabled=False           mix_enabled=True
            |
            |    .---theta---.            .---theta-------.
            |    |   |   |   |            |   |   |       |
            |            |   |           LP---H0--H1--.   |
            |    |   |   |   |            |   |   |   |   |
            |    .---theta*--.                    |   xR  |
            |                             |   |   |   |   |
            |                            LP*--H0*-H1*-.   |
            |                             |   |   |       |
            |                             .---theta*------.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Ground state of the effective Hamiltonian, prepared for svd.
        i0 : int
            Site index; `theta` lives on ``i0, i0+1``.
        mix_enabled : bool
            Wheter we actually use mix_R or not.

        Returns
        -------
        rho_L : :class:`~tenpy.linalg.np_conserved.Array`
            A (hermitian) square array with labels ``'(vL.p0)', '(vL*.p0*)'``,
            Mainly the reduced density matrix of the left part, but with some additional mixing.
        """
        if not mix_enabled:
            return npc.tensordot(theta, theta.conj(), axes=['(vR.p1)', '(vR*.p1*)'])
        H = self.env.H
        mixer_xR, add_separate_Id = self.mixer.get_xR(
            self.H1.get_leg('wR'), H.get_IdL(i0 + 2), H.get_IdR(i0 + 1))
        rho = npc.tensordot(self.LP, theta.split_legs(['(vL.p0)', '(vR.p1)']), axes=['vR', 'vL'])
        rho = npc.tensordot(rho, self.H0, axes=[['wR', 'p0'], ['wL', 'p0*']])
        H1m = npc.tensordot(self.H1, mixer_xR, axes=['wR', 'wL'])
        H1m = npc.tensordot(H1m, self.H1.conj(), axes=[['p1', 'wL*'], ['p1*', 'wR*']])
        rho = rho.ireplace_label('vR*', 'vL').combine_legs(['vL', 'p0'])
        rho_c = rho.conj()
        rho = npc.tensordot(rho, H1m, axes=[['p1', 'wR'], ['p1*', 'wL']])
        rho = npc.tensordot(rho, rho_c, axes=(['p1', 'wL*', 'vR'], ['p1*', 'wR*', 'vR*']))
        if add_separate_Id:
            rho = rho + npc.tensordot(theta, theta.conj(), axes=['(vR.p1)', '(vR*.p1*)'])
        return rho

    def mix_rho_R(self, theta, i0, mix_enabled):
        """Calculated mixed reduced density matrix for left site.

        Pictorially::

            |     mix_enabled=False           mix_enabled=True
            |
            |    .---theta---.           .------theta---.
            |    |   |   |   |           |      |   |   |
            |    |   |                   |   .--H0--H1--RP
            |    |   |   |   |           |   |  |   |   |
            |    .---theta*--.           |  wL  |
            |                            |   |  |   |   |
            |                            |   .--H0*-H1*-RP*
            |                            |      |   |   |
            |                            .------theta*--.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Ground state of the effective Hamiltonian, prepared for svd.
        i0 : int
            Site index; `theta` lives on ``i0, i0+1``.
        mix_enabled : bool
            Wheter to actually use the mixer or not.

        Returns
        -------
        rho_R : :class:`~tenpy.linalg.np_conserved.Array`
            A (hermitian) square array with labels ``'(vR.p1)', '(vR*.p1*)'``.
            Mainly the reduced density matrix of the right part, but with some additional mixing.
        """
        if not mix_enabled:
            return npc.tensordot(theta, theta.conj(), axes=[['(vL.p0)'], ['(vL*.p0*)']])
        H = self.env.H
        mixer_xL, add_separate_Id = self.mixer.get_xL(
            self.H0.get_leg('wL'), H.get_IdL(i0), H.get_IdR(i0 - 1))
        rho = npc.tensordot(theta.split_legs(['(vL.p0)', '(vR.p1)']), self.RP, axes=['vR', 'vL'])
        rho = npc.tensordot(rho, self.H1, axes=[['wL', 'p1'], ['wR', 'p1*']])
        H0m = npc.tensordot(mixer_xL, self.H0, axes=['wR', 'wL'])
        H0m = npc.tensordot(H0m, self.H0.conj(), axes=[['wR*', 'p0'], ['wL*', 'p0*']])
        rho = rho.ireplace_label('vL*', 'vR').combine_legs(['vR', 'p1'])
        rho_c = rho.conj()
        rho = npc.tensordot(H0m, rho, axes=[['p0*', 'wR'], ['p0', 'wL']])
        rho = npc.tensordot(rho, rho_c, axes=(['p0', 'wR*', 'vL'], ['p0*', 'wL*', 'vL*']))
        if add_separate_Id:
            rho = rho + npc.tensordot(theta, theta.conj(), axes=['(vL.p0)', '(vL*.p0*)'])
        return rho

    def update_LP(self, i0, U):
        """Update left part of the environment.

        Parameters
        ----------
        i0 : int
            Site index. We calculate ``self.env.get_LP(i0+1)``.
        U : :class:`~tenpy.linalg.np_conserved.Array`
            The U as returned by SVD with combined legs, labels ``'(vL.p0)', 'vR'``.
        """
        self.env.get_LP(i0 + 1, store=True)  # as implemented directly in the environment

    def update_RP(self, i0, VH):
        """Update right part of the environment.

        Parameters
        ----------
        i0 : int
            Site index. We calculate ``self.env.get_RP(i0)``.
        VH : :class:`~tenpy.linalg.np_conserved.Array`
            The U as returned by SVD, with combined legs, labels ``'vL', '(vR.p1)'``.
        """
        self.env.get_RP(i0, store=True)  # as implemented directly in the environment


class Mixer(object):
    """Mixer class.

    .. todo ::
        documentation/reference

    Parameters
    ----------
    env : :class:`~tenpy.networks.mpo.MPOEnvironment`
        Environment for contraction ``<psi|H|psi>`` for later
    mixer_params : dict
        Optional parameters as described in the following table.
        Use ``verbose>0`` to print the used parameters during runtime.

        ============== ========= ===============================================================
        key            type      description
        ============== ========= ===============================================================
        amplitude      float     Initial strength of the mixer. (Should be chosen < 1.)
        -------------- --------- ---------------------------------------------------------------
        decay          float     To slowly turn off the mixer, we divide `amplitude` by `decay`
                                 after each sweep.
        -------------- --------- ---------------------------------------------------------------
        disable_after  int       We disable the mixer completely after this number of sweeps.
        ============== ========= ===============================================================

    Attributes
    ----------
    amplitude : float
        Current amplitude for mixing.
    decay : float
        Factor by which `amplitude` is divided after each sweep.
    disable_after : int
        The number of sweeps after which the mixer should be disabled.
    verbose : int
        Level of output vebosity.
    """

    def __init__(self, mixer_params):
        self.amplitude = get_parameter(mixer_params, 'amplitude', 1.e-2, 'Mixer')
        self.decay = get_parameter(mixer_params, 'decay', 2., 'Mixer')
        self.disable_after = get_parameter(mixer_params, 'disable_after', 15, 'Mixer')
        self.verbose = mixer_params.get('verbose', 0)

    def update_amplitude(self, sweeps):
        """Update the amplitude, possibly disable the mixer.

        Parameters
        ----------
        sweeps : int
            The number of performed sweeps, to check if we need to disable the mixer.

        Returns
        -------
        mixer : :class:`Mixer` | None
            Returns `self` if we should continue mixing, or ``None``, if the mixer
            should be disabled.
        """
        self.amplitude /= self.decay
        if sweeps >= self.disable_after and self.amplitude >= np.finfo('float').eps:
            if self.verbose >= 0.1:  # increased verbosity: the same level as DMRG
                print("disable mixer")
            return None  # disable mixer
        return self

    def get_xR(self, wR_leg, Id_L, Id_R):
        """Generate the coupling of the MPO legs for the reduced density matrix.

        Parameters
        ----------
        wR_leg : :class:`~tenpy.linalg.charges.LegCharge`
            LegCharge to be connected to.
        IdL : int | ``None``
            Index within the leg for which the MPO has only identities to the left.
        IdR : int | ``None``
            Index within the leg for which the MPO has only identities to the right.

        Returns
        -------
        mixed_xR : :class:`~tenpy.linalg.np_conserved.Array`
            Connection of the MPOs on the right for the reduced density matrix `rhoL`.
            Labels ``('wL', 'wL*')``.
        add_separate_Id : bool
            If Id_L is ``None``, we can't include the identity into `mixed_xR`,
            so it has to be added directly in :meth:`Engine.mix_rho_L`.
        """
        x = self.amplitude * np.ones(wR_leg.ind_len, dtype=np.float)
        separate_Id = Id_L is None
        if not separate_Id:
            x[Id_L] = 1.
        if Id_R is not None:
            x[Id_R] = 0.
        x = npc.diag(x, wR_leg)
        x.iset_leg_labels(['wL*', 'wL'])
        return x, separate_Id

    def get_xL(self, wL_leg, Id_L, Id_R):
        """Generate the coupling of the MPO legs for the reduced density matrix.

        Parameters
        ----------
        wL_leg : :class:`~tenpy.linalg.charges.LegCharge`
            LegCharge to be connected to.
        Id_L : int | ``None``
            Index within the leg for which the MPO has only identities to the left.
        Id_R : int | ``None``
            Index within the leg for which the MPO has only identities to the right.

        Returns
        -------
        mixed_xL : :class:`~tenpy.linalg.np_conserved.Array`
            Connection of the MPOs on the left for the reduced density matrix `rhoR`.
            Labels ``('wR', 'wR*')``.
        add_separate_Id : bool
            If Id_R is ``None``, we can't include the identity into `mixed_xL`,
            so it has to be added directly in :meth:`Engine.mix_rho_R`.
        """
        x = self.amplitude * np.ones(wL_leg.ind_len, dtype=np.float)
        separate_Id = Id_R is None
        if not separate_Id:
            x[Id_R] = 1.
        if Id_L is not None:
            x[Id_L] = 0.
        x = npc.diag(x, wL_leg)
        x.iset_leg_labels(['wR*', 'wR'])
        return x, separate_Id
