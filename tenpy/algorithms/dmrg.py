"""Density Matrix Renormalization Group (DMRG).

Although it was originally not formulated with tensor networks,
the DMRG algorithm (invented by Steven White in 1992 [1]_, [2]_) opened the whole field
with its enormous success in finding ground states in 1D.

We implement DMRG in the modern formulation of matrix product states [3]_,
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

References
----------
.. [1] S. White, Phys. Rev. Lett. 69, 2863 (1992),
       S. White, Phys. Rev. B 84, 10345 (1992)
.. [2] U. Schollwoeck, Annals of Physics 326, 96 (2011), arXiv:1008.3477
"""

from __future__ import division
import numpy as np
import time
import itertools
import warnings

from ..linalg import np_conserved as npc
from ..networks.mpo import MPOEnvironment
from ..linalg.lanczos import lanczos
from .truncation import truncate, svd_theta
from ..tools.params import get_parameter

__all__ = ['run', 'Engine', 'EngineCombine', 'EngineFracture', 'Mixer']


def run(psi, model, DMRG_params):
    """Run the DMRG algorithm to find the ground state of `M`.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        Initial guess for the ground state, which is to be optimized in-place.
    model : :class:`~tenpy.models.MPOModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    DMRG_params : dict
        Further optional parameters as described in the following table.
        Use ``verbose=1`` to print the used parameters during runtime.

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
                       class     A string stands for one of the mixers defined in module,
                                 a class is used as custom mixer.
        -------------- --------- ---------------------------------------------------------------
        mixer_params   dict      Non-default initialization arguments of the mixer.
                                 Options may be custom to the specified mixer, so they're
                                 documented in the class doc-string of the mixer.
        -------------- --------- ---------------------------------------------------------------
        engine         str |     Chooses the (derived class of) :class:`Engine` to be used.
                       class     A string stands for one of the engines defined in this module,
                                 a class (not an instance!) can be used as custom engine.
        -------------- --------- ---------------------------------------------------------------
        initial_sweep  int       The number of sweeps already performed. (Useful for re-start).
        -------------- --------- ---------------------------------------------------------------
        max_seeps      int       Maximum number of sweeps to be performed.
        -------------- --------- ---------------------------------------------------------------
        min_sweep      int       Minimum number of sweeps to be performed.
        -------------- --------- ---------------------------------------------------------------
        N_sweeps_check int       Number of sweeps to perform between checking convergence
                                 criteria and giving a status update.
        -------------- --------- ---------------------------------------------------------------
        max_hours      float     If the DMRG took longer (measured in wall-clock time),
                                 'shelve' the simulation, i.e. stop and return with the flag
                                 ``shelve=True``.
        -------------- --------- ---------------------------------------------------------------
        trunc_params   dict       Truncation parameters as described in
                                  :func:`~tenpy.algorithms.truncation.truncate`
        -------------- --------- ---------------------------------------------------------------
        lanczos_params dict       Lanczos parameters as described in
                                  :func:`~tenpy.linalg.lanczos.lanczos`
        ============== ========= ===============================================================
    """
    Engine_class = get_parameter(DMRG_params, 'engine', 'EngineCombine', 'DMRG')
    if isinstance(Engine_class, str):
        Engine_class = globals()[Engine_class]
    engine = Engine_class(psi, model, DMRG_params)

    # get parameters for DMRG convergence criteria
    min_sweeps = get_parameter(DMRG_params, 'min_sweeps', 4, 'DMRG')
    max_sweeps = get_parameter(DMRG_params, 'max_sweeps', 1000, 'DMRG')
    N_sweeps_check = get_parameter(DMRG_params, 'N_sweeps_check', 10, 'DMRG')
    max_seconds = 3600 * get_parameter(DMRG_params, 'max_hours', 24*365, 'DMRG')
    start_time = time.time()
    shelve = False

    # TODO: collect statistics, truncation error

    sweeps = get_parameter(DMRG_params, 'initial_sweep', 0, 'DMRG')
    while True:
        # check abortion criteria
        if sweeps >= max_sweeps:
            break
        if sweeps > min_sweeps:  # TODO: and E_err < E_err_tol and ...
            break
        if time.time() - start_time > max_seconds:
            shelve = True
            warnings.warn("DMRG: maximum time limit reached. Shelve simulation.")
            break
        # the time-consuming part: the actual sweeps
        for i in range(N_sweeps_check):
            # TODO: update trunc_params, ...
            # trunc_params['chi_max'] = chi_max
            engine.sweep(sweeps)  # TODO statistics
            sweeps += 1

    # cleanup
    engine.mixer_cleanup()
    # TODO: check norm condition?
    raise NotImplementedError()
    return shelve  # TODO: ...


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

    Attributes
    ----------
    verbose : int
        Level of verbosity (i.e. how much status information to print); higher=more output.
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
        # TODO: print status messages

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
            if isinstance(Mixer_class, str):
                Mixer_class = globals()[Mixer_class]
            mixer_params = get_parameter(DMRG_params, 'mixer_params', {}, 'DMRG')
            self.mixer = Mixer_class(mixer_params)

        self.lanczos_params = get_parameter(DMRG_params, 'lanczos_params', {}, 'DMRG')
        self.trunc_params = get_parameter(DMRG_params, 'trunc_params', {}, 'DMRG')

    def sweep(self, sweep_count, optimize=True):
        """One 'sweep' of the DMRG algorithm.

        Iteratate over the bond which is optimized, to the right and
        then back to the left to the starting point.
        If optimize=False, don't actually diagonalize the effective hamiltonian,
        but only update the environment.
        """
        # get schedule
        L = self.env.L
        if self.env.finite:
            schedule_i0 = range(0, L-1) + range(L-3, 0, -1)
            update_env = [[True, False]] * (L-2) + [[False, True]] * (L-2)
        else:
            assert(L >= 2)
            schedule_i0 = range(0, L) + range(L, 0, -1)
            update_env = [[True, True]] * 2 + [[True, False]] * (L-2) + \
                         [[True, True]] * 2 + [[False, True]] * (L-2)

        # the actual sweep
        for i0, upd_env in itertools.izip(schedule_i0, update_env):
            print "sweep", i0, upd_env # TODO
            self.update_bond(i0, upd_env[0], upd_env[1])

        # update mixer
        if self.mixer is not None:
            self.mixer.amplitude /= self.mixer.decay
            if sweep_count + 1 >= self.mixer.disable_after:
                self.mixer = None  # disable mixer
        # TODO: return statistis/truncation error

    def update_bond(self, i0, update_LP, update_RP, optimize=True):
        """Perform bond-update on the sites ``(i0, i0+1)``.

        Parameters
        ----------
        i0 : int
            Site left to the bond which should be optimized.
        update_LP : bool
            Whether to calculate the next ``env.LP[i0+1]``.
        update_LP : bool
            Whether to calculate the next ``env.RP[i0]``.
        """
        theta = self.prepare_diag(i0, update_LP, update_RP)
        if optimize:
            E0, theta, N = self.diag(theta)
        theta = self.prepare_svd(theta)
        U, S, VH, err = self.mixed_svd(theta, i0, update_LP, update_RP)
        self.set_B(i0, U, S, VH)  # needs to be called before update_{L,R}P
        if update_LP:
            self.update_LP(i0, U)
        if update_RP:
            self.update_RP(i0, VH)
        return err  # TODO


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
        return lanczos(self, theta_guess, self.lanczos_params)

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

        The goal ist to split theta and truncate it:

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
            The optimized wave function.
        """
        # get qtotal_LR from i0
        qtotal_i0 = self.env.ket._B[i0].qtotal
        if self.mixer is None:
            # simple case: real svd, devined elsewhere.
            return svd_theta(theta, self.trunc_params, qtotal_LR=[qtotal_i0, None],
                             inner_labels=['vR', 'vL'])
        rho_L = self.mix_rho_L(theta, i0, update_LP)
        # don't mix left parts, when we're going to the right  # TODO: why
        rho_L.itranspose(['(vL.p0)', '(vL*.p0*)'])  # just to be sure of the order
        rho_R = self.mix_rho_R(theta, i0, update_RP)
        rho_R.itranspose(['(vR.p1)', '(vR*.p1*)'])  # just to be sure of the order

        # consider the SVD `theta = U S V^H` (with real, diagonal S>0)
        # rho_L ~=  theta theta^H = U S V^H V S U^H = U S S U^H  (for mixer -> 0)
        # Thus, rho_L U = U S S, i.e. columns of U are the eigenvectors of rho_L,
        # eigenvalues are S^2.
        val_L, U = npc.eigh(rho_L)
        val_L[val_L < 0.] = 0.  # for stability reasons
        val_L /= np.sum(val_L)
        keep_L, _, errL = truncate(np.sqrt(val_L), self.trunc_params)
        U.set_leg_labels(['(vL.p0)', 'vR'])
        U.iproject(keep_L, axes='vR')  # in place
        # rho_R ~=  theta^T theta^* = V^* S U^T U* S V^T = V^* S S V^T  (for mixer -> 0)
        # Thus, rho_L V^* = V^* S S, i.e. columns of V^* are eigenvectors of rho_L
        val_R, Vc = npc.eigh(rho_R)
        VH = Vc.transpose()
        VH.set_leg_labels(['vL', '(vR.p1)'])
        val_R[val_R < 0.] = 0.  # for stability reasons
        keep_R, _, err_R = truncate(np.sqrt(val_R), self.trunc_params)
        VH.itranspose()
        VH.iproject(keep_R, axes='vL')

        # calculate S = U^H theta V
        theta = npc.tensordot(U.conj(), theta, axes=['(vL*.p0*)', '(vL.p0)'])  # axes 0, 0
        theta = npc.tensordot(theta, VH.conj(), axes=['(vR.p1)', '(vR*.p1)'])  # axes 1, 1
        return U, theta, VH, errL + err_R  # TODO: error ok like that?

    def mix_rho_L(self, theta, i0, mix_enabled):
        """Calculated mixed reduced density matrix for left site.

        Pictorially:

            |     mix_enabled=False           mix_enabled=True
            |
            |    .---theta---.            .---theta-------.
            |    |   |   |   |            |   |   |       |
            |            |   |           LP---H0--H1--.   |
            |    |   |   |   |            |   |   |   |   |
            |    .---theta*--.                    |   wR  |
            |                             |   |   |   |   |
            |                            LP*--H0*-H1*-.   |
            |                             |   |   |       |
            |                             .---theta*------.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Ground state of the effective Hamiltonian.
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

        Pictorially:

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
            Ground state of the effective Hamiltonian.
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

    def mixer_cleanup(self, *args):
        """Cleanup the effects of the mixer.

        A :meth:`sweep` with an enabled :class:`Mixer` leaves the MPS `psi` with 2D arrays `s`.
        To recover the originial form, this function simply performs a sweep with disabled mixer.
        """
        if self.mixer is not None:
            mixer = self.mixer
            self.mixer = None   # disable the mixer
            self.sweep(*args)   # TODO: return value
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
        B0 = U.split_legs([0]).replace_label('p0', 'p')
        B1 = VH.split_legs([1]).replace_label('p1', 'p')
        self.env.ket.set_B(i0, B0, form='A')  # left-canonical
        self.env.ket.set_B(i0+1, B1, form='B')  # right-canonical
        self.env.del_LP(i0+1)  # the old stored environments are now invalid
        self.env.del_RP(i0)
        self.env.ket.set_SR(i0, S)
        return B0, B1

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
            The U as returned by SVD, with combined legs, labels ``'vL', '(vR.p1)'``.
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
        Labels ``'(vL.p0)', 'wL*', '(vL*.p0*)'`` for bra, MPO, ket.
    RHeff: :class:`~tenpy.linalg.np_conserved.Array`
        Right part of the effective Hamiltonian.
        Labels ``'(vR*.p1*)', 'wR*', '(vR.p1)'`` for ket, MPO, bra.

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
        LP = env.get_LP(i0, update_LP)      # labels ('vL', 'wL*', 'vL*')
        H1 = env.H.get_W(i0).replace_labels(['p', 'p*'],
                                            ['p0', 'p0*'])  # ('wL', 'wR', 'p0', 'p0*')
        RP = env.get_RP(i0+1, update_RP)    # labels ('vR', 'wR*', 'vR*')
        H2 = env.H.get_W(i0+1).replace_labels(['p', 'p*'],
                                              ['p1', 'p1*'])  # ('wL', 'wR', 'p1', 'p1*')
        # calculate LHeff
        LHeff = npc.tensordot(LP, H1, axes=['wL*', 'wL'])
        pipeL = LHeff.make_pipe(['vL', 'p0'])
        self.LHeff = LHeff.combine_legs([['vL', 'p0'], ['vL*', 'p0*']],
                                        pipes=[pipeL, pipeL.conj()],
                                        new_axes=[0, -1])  # avoid transpositions during matvec
        # calculate RHeff
        RHeff = npc.tensordot(RP, H2, axes=['wR*', 'wR'])
        pipeR = RHeff.make_pipe(['vR', 'p1'])
        self.RHeff = RHeff.combine_legs([['vR', 'p1'], ['vR*', 'p1*']],
                                        pipes=[pipeR, pipeR.conj()],
                                        new_axes=[-1, 0])  # avoid transpositions during matvec
        # make theta
        theta = env.ket.get_theta(i0, n=2)  # labels ('vL', 'vR', 'p0', 'p1')
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
            Wave function to apply the effective Hamiltonian to,  :math:`H |\theta>`
        """
        theta = npc.tensordot(self.LHeff, theta, axes=['(vL*.p0*)', '(vL.p0)'])
        theta = npc.tensordot(theta, self.RHeff, axes=[['(vR.p1)', 'wR'] , ['(vR*.p1*)', 'wL']])
        return theta

    def prepare_svd(self, theta):
        """Transform theta into matrix for svd."""
        return theta  # For this engine nothing to do.

    def mix_rho_L(self, theta, i0, mix_enabled):
        """Calculated mixed reduced density matrix for left site.

        Pictorially:

            |     mix_enabled=False           mix_enabled=True
            |
            |    .---theta---.            .---theta-------.
            |    |   |   |   |            |   |   |       |
            |            |   |           LP---H0--H1--.   |
            |    |   |   |   |            |   |   |   |   |
            |    .---theta*--.                    |   wR  |
            |                             |   |   |   |   |
            |                            LP*--H0*-H1*-.   |
            |                             |   |   |       |
            |                             .---theta*------.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Ground state of the effective Hamiltonian.
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
        H1 = H.get_W(i0+1).replace_labels(['p', 'p*'], ['p1', 'p1*'])
        mixer_wR = self.mixer.get_w_R(H1.get_leg('wR'),
                                      H.get_IdL(i0+2), H.get_IdR(i0+1))
        # TODO: wired index convention. change in MPO.
        rho = npc.tensordot(self.LHeff, theta.split_legs('(vR.p1)'),
                            axes=['(vL*.p0*)', '(vL.p0)'])
        rho = npc.tensordot(rho, H1, axes=[['p1', 'wR'], ['p1*', 'wL']])
        rho_c = rho.conj()
        rho = npc.tensordot(rho, mixer_wR, axes=['wR', 'wR*'])
        return npc.tensordot(rho_c, rho, axes=(['p1*', 'wR*', 'vR*'], ['p1', 'wR', 'vR']))

    def mix_rho_R(self, theta, i0, mix_enabled):
        """Calculated mixed reduced density matrix for left site.

        Pictorially:

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
            Ground state of the effective Hamiltonian.
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
            return npc.tensordot(theta, theta.conj(), axes=['(vR.p1)', '(vR*.p1*)'])
        H = self.env.H
        H0 = H.get_W(i0).replace_labels(['p', 'p*'], ['p0', 'p8*'])
        mixer_wL = self.mixer.get_w_L(H0.get_leg('wL'),
                                      H.get_IdL(i0), H.get_IdR(i0-1))
        rho = npc.tensordot(self.RHeff, theta.split_legs('(vL.p0)'), axes=['(vR*.p1*)', '(vR.p1)'])
        rho = npc.tensordot(rho, H0, axes=[['p0', 'wL'], ['p0*', 'wR']])
        rho_c = rho.conj()
        rho = npc.tensordot(rho, mixer_wL, axes=['wL', 'wL*'])
        return npc.tensordot(rho, rho_c, axes=(['p0', 'wL', 'vL'], ['p0*', 'wL*', 'vL*']))

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
        LP = npc.tensordot(self.LHeff, U, axes=['(vL*.p0*)', '(vL.p0)'])
        LP = npc.tensordot(U.conj(), LP, axes=['(vL*.p0*)', '(vL.p0)'])
        LP = LP.replace_labels(['vR*', 'wR', 'vR'], ['vL', 'wL*', 'vL*'])
        self.env.set_LP(i0+1, LP, age=self.env.get_LP_age(i0)+1)

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
        RP = npc.tensordot(self.RHeff, VH, axes=['(vR*.p1*)', '(vR.p1)'])
        RP = npc.tensordot(VH.conj(), RP, axes=['(vR*.p1*)', '(vR.p1)'])
        RP = RP.replace_labels(['vL*', 'wL', 'vL'], ['vR', 'wR*', 'vR*'])
        self.env.set_RP(i0, RP, age=self.env.get_RP_age(i0+1)+1)


class EngineFracture(Engine):
    """Engine which keeps the legs separate.

    Due to a different contraction order in :meth:`matvec`, this engine might be faster than
    :class:`EngineCombine`, at least for large physical dimensions and if the MPO is sparse.
    One :meth:`matvec` is :math:`O(2 \chi^3 d^2 D + 2 \chi^2 d^3 W^2 )`.

    Attributes
    ----------
    LP: :class:`~tenpy.linalg.np_conserved.Array`
        Left part of the effective Hamiltonian. Labels ``'vL', 'wL*', 'vL*'``.
    RP: :class:`~tenpy.linalg.np_conserved.Array`
        Right part of the effective Hamiltonian. Labels ``'vR', 'wR*', 'vR*'``.
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
        self.LP = env.get_LP(i0, update_LP)      # labels ('vL', 'wL*', 'vL*')
        self.H1 = env.H.get_W(i0).replace_labels(['p', 'p*'],
                                                 ['p0', 'p0*'])  # ('wL', 'wR', 'p0', 'p0*')
        self.RP = env.get_RP(i0+1, update_RP)    # labels ('vR', 'wR*', 'vR*')
        self.H2 = env.H.get_W(i0+1).replace_labels(['p', 'p*'],
                                                   ['p1', 'p1*'])  # ('wL', 'wR', 'p1', 'p1*')
        # make theta
        theta = env.ket.get_theta(i0, n=2)  # labels ('vL', 'vR', 'p0', 'p1')
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
            Wave function to apply the effective Hamiltonian to,  :math:`H |\theta>`
        """
        theta = npc.tensordot(self.LP, theta, axes=['vL*', 'vL'])
        theta = npc.tensordot(theta, self.H0, axes=[['wL*', 'p0'], ['wL', 'p0*']])
        theta = npc.tensordot(theta, self.H1, axes=[['wR', 'p1'], ['wL', 'p1*']])
        theta = npc.tensordot(theta, self.RP, axes=[['wR', 'p1'], ['wL', 'vL']])
        return theta

    def prepare_svd(self, theta):
        """Transform theta into matrix for svd."""
        return theta.combine_legs([['vL', 'p0'], ['vR', 'p1']])

    def mix_rho_L(self, theta, i0, mix_enabled):
        """Calculated mixed reduced density matrix for left site.

        Pictorially:

            |     mix_enabled=False           mix_enabled=True
            |
            |    .---theta---.            .---theta-------.
            |    |   |   |   |            |   |   |       |
            |            |   |           LP---H0--H1--.   |
            |    |   |   |   |            |   |   |   |   |
            |    .---theta*--.                    |   wR  |
            |                             |   |   |   |   |
            |                            LP*--H0*-H1*-.   |
            |                             |   |   |       |
            |                             .---theta*------.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Ground state of the effective Hamiltonian.
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
            return npc.tensordot(theta, theta.conj(), axes=[['vR', 'p1'], ['vR*', 'p1*']])
        H = self.env.H
        mixer_wR = self.mixer.get_w_R(self.H1.get_leg('wR'), H.get_IdL(i0+2), H.get_IdR(i0+1))
        rho = npc.tensordot(self.LP, theta, axes=['vL*', 'vL'])
        rho = npc.tensordot(rho, self.H0, axes=[['wL*', 'p0'], ['wL', 'p0*']])
        H1m = npc.tensordot(self.H1, mixer_wR, axes=['wR', 'wR*'])
        H1m = npc.tensordot(H1m, self.H1.conj(), axes=[['wR', 'p1'], ['wR', 'p1*']])
        rho_c = rho.conj()
        rho = npc.tensordot(rho, H1m, axes=[['p1', 'wR'], ['p1*', 'wL']])
        return npc.tensordot(rho_c, rho, axes=(['p1*', 'wR*', 'vR*'], ['p1', 'wL*', 'vR']))

    def mix_rho_R(self, theta, i0, mix_enabled):
        """Calculated mixed reduced density matrix for left site.

        Pictorially:

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
            Ground state of the effective Hamiltonian.
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
            return npc.tensordot(theta, theta.conj(), axes=['(vR.p1)', '(vR*.p1*)'])
        H = self.env.H
        mixer_wL = self.mixer.get_w_L(self.H0.get_leg('wL'), H.get_IdL(i0), H.get_IdR(i0-1))
        rho = npc.tensordot(self.RP, theta, axes=['vR*', 'vR'])
        rho = npc.tensordot(rho, self.H1, axes=[['wR*', 'p0'], ['wL', 'p0*']])
        H0m = npc.tensordot(self.H0, mixer_wL, axes=['wL', 'wL*'])
        H0m = npc.tensordot(H0m, self.H0.conj(), axes=[['wL', 'p0'], ['wL', 'p0*']])
        rho_c = rho.conj()
        rho = npc.tensordot(rho, H0m, axes=[['p0', 'wL'], ['p0*', 'wR']])
        return npc.tensordot(rho_c, rho, axes=(['p0*', 'wL*', 'vL*'], ['p0', 'wR*', 'vL']))

    def update_LP(self, i0, U):
        """Update left part of the environment.

        Parameters
        ----------
        i0 : int
            Site index. We calculate ``self.env.get_LP(i0+1)``.
        U : :class:`~tenpy.linalg.np_conserved.Array`
            The U as returned by SVD with combined legs, labels ``'(vL.p0)', 'vR'``.
        """
        self.env.get_LP(i0+1, store=True)  # as implemented directly in the environment

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

    Parameters
    ----------
    env : :class:`~tenpy.networks.mpo.MPOEnvironment`
        Environment for contraction ``<psi|H|psi>`` for later
    mixer_params : dict
        Optional parameters as described in the following table.
        Use ``verbose=1`` to print the used parameters during runtime.

        ============== ========= ===============================================================
        key            type      description
        ============== ========= ===============================================================
        amplitude      float     Initial strength of the mixer. (Should be chosen < 1.)
        -------------- --------- ---------------------------------------------------------------
        decay          float     To slowly turn off the mixer, we divide `amplitude` by `decay`
                                 after each sweep.
        -------------- --------- ---------------------------------------------------------------
        disable_after  int       We disable the mixer completely after this number of sweeps.
        -------------- --------- ---------------------------------------------------------------


    Attributes
    ----------
    env : :class:`~tenpy.networks.mpo.MPOEnvironment`
        Provides the environment.
    amplitude : float
        Current amplitude for mixing.
    decay : float
        Factor by which `amplitude` is divided after each sweep.
    .. todo : documentation/reference
    """
    def __init__(self, env, mixer_params):
        self.env = env
        self.amplitude = get_parameter(mixer_params, 'amplitude', 1.e-2, 'Mixer')
        self.decay = get_parameter(mixer_params, 'decay', 2., 'Mixer')
        self.disable_after = get_parameter(mixer_params, 'disable_after', 50, 'Mixer')

    def get_w_R(self, wR_leg, Id_L, Id_R):
        """Generate the coupling of the MPO legs for the reduced density matrix.

        Parameters
        ----------
        wR_leg : :class:`~tenpy.linalg.charges.LegCharge`
            LegCharge to be connected to.
        IdL : int
            Index within the leg for which the MPO has only identities to the left.
        IdR : int
            Index within the leg for which the MPO has only identities to the right.

        Returns
        -------
        mixed_wR : :class:`~tenpy.linalg.np_conserved.Array`
            Connection of the MPOs on the right for the reduced density matrix `rhoL`.
            Labels ``('wR*', 'wR')``.
        """
        w = self.amplitude * np.ones(wR_leg.ind_len, dtype=np.float)
        w[Id_L] = 1.  # TODO: what if IdL, IdR is None ???
        w[Id_R] = 0.
        w = npc.diag(w, wR_leg)
        w.set_leg_labels(['wR*', 'wR'])
        return w

    def get_w_L(self, wL_leg, Id_L, Id_R):
        """Generate the coupling of the MPO legs for the reduced density matrix.

        Parameters
        ----------
        wL_leg : :class:`~tenpy.linalg.charges.LegCharge`
            LegCharge to be connected to.
        IdL : int
            Index within the leg for which the MPO has only identities to the left.
        IdR : int
            Index within the leg for which the MPO has only identities to the right.

        Returns
        -------
        mixed_wL : :class:`~tenpy.linalg.np_conserved.Array`
            Connection of the MPOs on the left for the reduced density matrix `rhoR`.
            Labels ``('wL*', 'wL')``.
        """
        w = self.amplitude * np.ones(wL_leg.ind_len, dtype=np.float)
        w[Id_L] = 0.  # TODO: what if IdL, IdR is None ???
        w[Id_R] = 1.
        w = npc.diag(w, wL_leg)
        w.set_leg_labels(['wL*', 'wL'])
        return w
