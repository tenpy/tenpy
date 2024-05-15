"""Plane wave excitations ansatz

The quasiparticle ansatz, utilizing plane wave states, offers a powerful approach for computing
low-energy excitations. For finite systems, we could employ DMRG with different charge sectors to
find these states. For infinite systems, an efficient algorithm was introduced
in :cite:`haegeman2012`. By working directly in the tangent space of a uniform
MPS :cite:`vanderstraeten2019`, we can use translational invariance to specify momentum.

The plane wave excitation finds excitation of a :class:`~tenpy.networks.uniform_mps.UniformMPS`.
By summing over all states, where one tensor is replaced with an excited tensor B, we get a state
with a fixed momentum :class:`~tenpy.networks.momentum_mps.MomentumMPS`. The tensors B are
decomposed into -B- = -V-X-, where V is the orthogonal complement of the usual A-tensors of the
uniform MPS, and X contains the variational parameters. The algorithm constructs an effective
Hamiltonian for X and finds the low-energy states using an iterative eigensolver. Since we don't
sweep as e.g. in DMRG, more Lanczos steps may be required. Increase those until the energy is
converged!

Additionally, we can specify X to be in a given charge sector.

The :class:`PlaneWaveExcitationEngine` optimizes the X for each tensor in the unit cell. The ansatz
can be extended to include excitations that span several sites. This is implemented
in :class:`MultiSitePlaneWaveExcitationEngine`. Note that with the current implementation, the
numerical costs scale exponentially with the number of exciting sites.
"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import logging

logger = logging.getLogger(__name__)

from ..linalg import np_conserved as npc
from ..linalg.charges import LegPipe
from ..networks.momentum_mps import MomentumMPS
from ..networks.mpo import MPOEnvironment, MPOTransferMatrix
from ..linalg.krylov_based import GMRES, LanczosGroundState, Arnoldi
from ..linalg.sparse import NpcLinearOperator, SumNpcLinearOperator, BoostNpcLinearOperator
from ..algorithms.algorithm import Algorithm
from ..algorithms.mps_common import ZeroSiteH

__all__ = [
    'append_right_env', 'append_left_env', 'construct_orthogonal', 'PlaneWaveExcitationEngine',
    'MultiSitePlaneWaveExcitationEngine'
]


def append_right_env(As, Bs, R, Ws=None):
    """Contract all tensors in As and Bs to the right environment R.

    If Ws given, contract as a MPO environment.

    Parameters
    ----------
    As : list of :class:`~tenpy.linalg.np_conserved.Array`
        Arrays to add to the top leg of the environment, labels are ``vL, vR, p`` (in any order).
    Bs : list of :class:`~tenpy.linalg.np_conserved.Array`
        Arrays to add to the bottom leg of the environment, labels are ``vL, vR, p`` (in any order).
    R : :class:`~tenpy.linalg.np_conserved.Array`
        Initial right environment, labels are ``vL, vR`` (in any order), if 'Ws' is `None`,
        otherwise also need a leg ``wL``
    Ws : list of :class:`~tenpy.linalg.np_conserved.Array`
        Arrays to add to the middle leg of the environment, labels are ``wL, wR, p, p*`` (in any order).

    Returns
    -------
    temp : :class:`~tenpy.linalg.np_conserved.Array`
        The new environment with the tensors above included.
    """
    temp = R.copy()
    for i in reversed(range(len(As))):
        temp = npc.tensordot(Bs[i].conj(), temp, axes=(['vR*'], ['vL*']))
        if Ws is not None:
            temp = npc.tensordot(Ws[i], temp, axes=(['wR', 'p'], ['wL', 'p*']))
        temp = npc.tensordot(As[i], temp, axes=(['vR', 'p'], ['vL', 'p*']))
    return temp


def append_left_env(As, Bs, L, Ws=None):
    """Contract all tensors in As and Bs to the left environment L.

    If Ws given, contract as a MPO environment.

    Parameters
    ----------
    As : list of :class:`~tenpy.linalg.np_conserved.Array`
        Arrays to add to the top leg of the environment, labels are ``vL, vR, p`` (in any order).
    Bs : list of :class:`~tenpy.linalg.np_conserved.Array`
        Arrays to add to the bottom leg of the environment, labels are ``vL, vR, p`` (in any order).
    L : :class:`~tenpy.linalg.np_conserved.Array`
        Initial left environment, labels are ``vL, vR`` (in any order), if 'Ws' is `None`,
        otherwise also need a leg ``wR``
    Ws : list of :class:`~tenpy.linalg.np_conserved.Array`
        Arrays to add to the middle leg of the environment, labels are ``wL, wR, p, p*`` (in any order).

    Returns
    -------
    temp : :class:`~tenpy.linalg.np_conserved.Array`
        The new environment with the tensors above included.
    """
    temp = L.copy()
    for i in range(len(As)):
        temp = npc.tensordot(temp, Bs[i].conj(), axes=(['vR*'], ['vL*']))
        if Ws is not None:
            temp = npc.tensordot(temp, Ws[i], axes=(['wR', 'p*'], ['wL', 'p']))
        temp = npc.tensordot(temp, As[i], axes=(['vR', 'p*'], ['vL', 'p']))
    return temp


def construct_orthogonal(M, left=True):
    """find (left) orthogonal complement of tensor M

    It finds Q such that::

      .--Q --
      |  |
      |  |    = 0
      |  |
      .--M --

    if left==False, find the right complement accordingly.

    Parameters
    ----------
    M : list of :class:`~tenpy.linalg.np_conserved.Array`
        Array for which we want to find the orthogonal complement, labels are ``vL, vR, p`` (in any order).
    left : bool
        Whether we want to compute the left or right complement.

    Returns
    -------
    Q : :class:`~tenpy.linalg.np_conserved.Array`
        The orthogonal complement, such that the relation above is fulfilled.
    """
    if left:
        M = M.copy().combine_legs([['vL', 'p'], ['vR']], qconj=[+1, -1])
        Q = npc.orthogonal_columns(M, 'vR')
        assert npc.norm(npc.tensordot(Q, M.conj(), axes=(['(vL.p)'], ['(vL*.p*)']))) < 1.e-12
    else:
        M = M.copy().combine_legs([['vL'], ['p', 'vR']], qconj=[+1, -1])
        Q = npc.orthogonal_columns(M.transpose(['(p.vR)', '(vL)']),
                                   'vL').itranspose(['vL', '(p.vR)'])
        assert npc.norm(npc.tensordot(Q, M.conj(), axes=(['(p.vR)'], ['(p*.vR*)']))) < 1.e-12
    return Q.split_legs()


class PlaneWaveExcitationEngine(Algorithm):
    r""" Base engine to compute quasiparticle excitations for uniform MPS.

    Parameters are the same as for :class:`~tenpy.algorithms.algorithm.Algorithm`.

    Options
    -------
    .. cfg:config :: PlaneWaveExcitationEngine
        :include: Algorithm

        lanczos_params : dict
            Lanczos parameters as described in :cfg:config:`KrylovBased`.
        lambda_C1 : float
            Energy shift from contracting the infinite environments. If `None`, compute it again.
        init_env_data : dict
            Dictionary as returned by ``self.env.get_initialization_data()`` from
            :meth:`~tenpy.networks.mpo.MPOEnvironment.get_initialization_data`.


    Attributes
    ----------
    psi : :class:`~tenpy.networks.uniform_mps.UniformMPS`
        The uniform MPS for which we compute (orthogonal) excitations.
    model : :class:`~tenpy.models.model.MPOModel`
        The model defining the Hamiltonian.
    ACs : list of :class:`~tenpy.linalg.np_conserved.Array`
        The 'center-site' tensors of psi.
    ALs : list of :class:`~tenpy.linalg.np_conserved.Array`
        The 'left-orthonormal' tensors of psi.
    AR : list of :class:`~tenpy.linalg.np_conserved.Array`
        The 'right-orthonormal' tensors of psi.
    Cs : list of :class:`~tenpy.linalg.np_conserved.Array`
        The center matrices of psi.
    Ws : list of :class:`~tenpy.linalg.np_conserved.Array`
        The matrices of the MPO.
    VLs : list of :class:`~tenpy.linalg.np_conserved.Array`
        The orthogonal complements for each AL.
    """

    def __init__(self, psi, model, options, **kwargs):
        super().__init__(psi, model, options, **kwargs)

        assert self.psi.L == self.model.H_MPO.L
        self.L = self.psi.L

        self.ALs = [self.psi.get_AL(i) for i in range(self.L)]
        self.ARs = [self.psi.get_AR(i) for i in range(self.L)]
        self.ACs = [self.psi.get_AC(i) for i in range(self.L)]
        self.Cs = [self.psi.get_C(i) for i in range(self.L)]  # C on the left
        self.H = self.model.H_MPO
        self.Ws = [self.H.get_W(i) for i in range(self.L)]
        if len(self.Ws) < len(self.ALs):
            assert len(self.ALs) % len(self.Ws)
            self.Ws = self.Ws * len(self.ALs) // len(self.Ws)

        self.IdL = self.H.get_IdL(0)
        self.IdR = self.H.get_IdR(-1)

        self.guess_init_env_data = self.options.get('init_env_data', None)

        # Construct VL, needed to parametrize - B - as - VL - X -
        #                                       |        |
        # Use prescription under Eq. 85 in Tangent Space lecture notes.
        self.VLs = [construct_orthogonal(self.ALs[i]) for i in range(self.L)]

        # Get left and right generalized eigenvalues
        self.boundary_env_data, self.energy_density, _ = MPOTransferMatrix.find_init_LP_RP(
            self.H, self.psi, calc_E=True, guess_init_env_data=self.guess_init_env_data)
        self.energy_density = np.mean(self.energy_density)
        self.LW = self.boundary_env_data['init_LP']
        self.RW = self.boundary_env_data['init_RP']

        # We create GS_env_L and GS_env_R to make topological easier.
        self.GS_env = self.GS_env_L = self.GS_env_R = MPOEnvironment(self.psi, self.H, self.psi,
                                                                     **self.boundary_env_data)
        self.lambda_C1 = options.get('lambda_C1', None, 'real')
        if self.lambda_C1 is None:
            C0_L = self.Cs[0]
            norm = npc.tensordot(C0_L, C0_L.conj(), axes=(['vL', 'vR'], ['vL*', 'vR*']))
            self.lambda_C1 = npc.tensordot(C0_L, self.RW, axes=(['vR'], ['vL']))
            self.lambda_C1 = npc.tensordot(self.LW,
                                           self.lambda_C1,
                                           axes=(['wR', 'vR'], ['wL', 'vL']))
            self.lambda_C1 = npc.tensordot(
                self.lambda_C1, C0_L.conj(), axes=(['vR*', 'vL*'], ['vL*', 'vR*'])) / norm

        self.aligned_H = self.Aligned_Effective_H(self)

        strange = []
        for i in range(self.L):
            temp_L = self.GS_env.get_LP(i)
            temp_R = self.GS_env.get_RP(i)
            temp = append_left_env([self.VLs[i]], [self.ACs[i]], temp_L, Ws=[self.Ws[i]])
            temp = npc.tensordot(temp, temp_R, axes=(['wR', 'vR*'], ['wL', 'vL*']))
            strange.append(npc.norm(temp))
        logger.info("Norm of H|psi> projected into the tangent space on each site: %r.", strange)

    def run(self, p, qtotal_change=None, orthogonal_to=[], E_boosts=[], num_ev=1):
        """ Run the plane-wave algorithm to find excited states of the given model.

        Parameters
        ----------
        p : float
            The momentum of the state; for unit cells larger than 1, we already include the
            factor of the smaller Brillouin zone: p*L.
        qtotal_change : list of int
            Charge sectors for each of the defined charges.
        orthogonal_to : list of list of :class:`~tenpy.linalg.np_conserved.Array`
            Find excitations orthogonal to previously found tensors X.
        E_boosts: list of float
            energy boosts for orthogonal states
        num_ev: int
            Number of eigenvalues and eigenvectors, that we extract from a single Arnoldi/ Lanczos run

        Returns
        -------
        Es : list of float
            Energies of the lowest-energy excitations. Number equal to `num_ev`.
        psis : list of :class:`~tenpy.networks.momentum_mps.MomentumMPS`
            MomentumMPS corresponding to the lowest-energy excitations. Number equal to `num_ev`.

        Options
        -------
        .. cfg:config :: PlaneWaveExcitationEngine

            E_boost : float
                uniform strength of the energy boosts (instead of specifying a list), default to E_boost=100

        """
        self.unaligned_H = self.Unaligned_Effective_H(self, p)
        effective_H = SumNpcLinearOperator(self.aligned_H, self.unaligned_H)
        lanczos_params = self.options.subconfig('lanczos_params')
        X_init = self.initial_guess(qtotal_change)
        if len(E_boosts) != len(orthogonal_to):
            E_boost = self.options.get('E_boost', 100, 'real')
            E_boosts = [E_boost] * len(orthogonal_to)
        if len(orthogonal_to) > 0:
            effective_H = BoostNpcLinearOperator(effective_H, E_boosts, orthogonal_to)

        if num_ev > 1:
            lanczos_params['which'] = 'SR'
            lanczos_params['num_ev'] = num_ev
            energies, Xs, N = Arnoldi(effective_H, X_init, lanczos_params).run()
            psis = []
            Es = []
            for E, X in zip(energies, Xs):
                psis.append(MomentumMPS(X, self.psi, p))
                Es.append(E - self.lambda_C1 - self.energy_density * self.L)
        else:
            energy, X, N = LanczosGroundState(effective_H, X_init, lanczos_params).run()
            Es = [energy - self.lambda_C1 - self.energy_density * self.L]
            psis = [MomentumMPS(X, self.psi, p)]

        if N == lanczos_params.get('N_max', 20, int):
            import warnings
            warnings.warn('Maximum Lanczos iterations needed; be wary of results.')

        return np.real_if_close(Es), psis, N

    def resume_run(self):
        raise NotImplementedError()

    def energy(self, p, X):
        """
        Compute the energy of excited states

        Parameters
        ----------
        p : float
            The momentum of the state; for unit cells larger than 1, we already include the
            factor of the smaller Brillouin zone: p*L.
        X : list of :class:`~tenpy.linalg.np_conserved.Array`
            Excitation tensors for each site of the unit cell.

        Returns
        -------
        energy : float
            Energy

        """
        self.unaligned_H = self.Unaligned_Effective_H(self, p)
        effective_H = SumNpcLinearOperator(self.aligned_H, self.unaligned_H)
        HX = effective_H.matvec(X)
        E = np.real(npc.inner(X, HX)).item()
        return E - self.energy_density * self.L - self.lambda_C1

    def infinite_sum_right(self, p, X):
        """
        Infinite sum to the right, see Eq. (194) in :cite:`vanderstraeten2019`

        p : float
            The momentum of the state; for unit cells larger than 1, we already include the
            factor of the smaller Brillouin zone: p*L.
        X : list of :class:`~tenpy.linalg.np_conserved.Array`
            Current excitation tensors for each site of the unit cell.

        Returns
        -------
        R_sum : :class:`~tenpy.linalg.np_conserved.Array`
            Array representing the right environment including the `B` tensors at each site.

        Options
        -------
        .. cfg:config :: PlaneWaveExcitationEngine

            sum_method : ``explicit`` | ``GMRES``
                Whether to explicitly sum the environment by applying the unit cell tensors until convergence (default) or solving the geometric series with the GMRES method.
            sum_tol : float
                Convergence criterion for the explicit summation.
            sum_iterations : int
                Maximum number of iterations for the explicit summation (default sum_iterations=100).
        """
        sum_tol = self.options.get('sum_tol', 1.e-10, 'real')
        sum_iterations = self.options.get('sum_iterations', 100, int)
        sum_method = self.options.get('sum_method', 'explicit', str)

        B = npc.tensordot(self.VLs[self.L - 1], X[self.L - 1], axes=(['vR'], ['vL']))
        RB = append_right_env([B], [self.ARs[self.L - 1]], self.RW, Ws=[self.Ws[self.L - 1]])
        for i in reversed(range(0, self.L - 1)):
            B = npc.tensordot(self.VLs[i], X[i], axes=(['vR'], ['vL']))
            RB = append_right_env([B], [self.ARs[i]], self.GS_env_R.get_RP(i), Ws=[self.Ws[i]]) + \
                 append_right_env([self.ALs[i]], [self.ARs[i]], RB, Ws=[self.Ws[i]])
        R = RB

        if np.isclose(npc.norm(R), 0):
            return R
        if sum_method == 'explicit':
            R_sum = R.copy()
            for _ in range(sum_iterations):
                R = np.exp(-1.0j * p * self.L) * append_right_env(
                    self.ALs, self.ARs, R, Ws=self.Ws)
                R_sum.iadd_prefactor_other(1., R)
                if npc.norm(R) < sum_tol:
                    break
            return R_sum
        elif 'GMRES' in sum_method:

            class helper_matvec(NpcLinearOperator):

                def __init__(self, excit, ALs, ARs, Ws, sum_method):
                    self.ALs = ALs
                    self.ARs = ARs
                    self.Ws = Ws
                    self.sum_method = sum_method
                    self.excit = excit

                def matvec(self, vec):
                    Tr = append_right_env(self.ALs, self.ARs, vec, Ws=self.Ws)
                    if 'reg' in self.sum_method:
                        raise NotImplementedError(
                            'GMRES-reg not implemented for multi-site unit cell.')
                        lr = npc.tensordot(self.excit.l_LR,
                                           vec,
                                           axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        llr = npc.tensordot(self.excit.LWCc,
                                            vec,
                                            axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        T1_r = self.excit.r_LR * (
                            (self.excit.e_LR - 1) * lr + llr) + self.excit.CRW * lr
                        Tr = Tr - T1_r
                    return vec - np.exp(-1.0j * p * self.excit.L) * Tr

            tm_op = helper_matvec(self, self.ALs, self.ARs, self.Ws, sum_method)
            GMRES_params = self.options.subconfig('GMRES_params')
            R_sum, _, _, _ = GMRES(tm_op, npc.Array.zeros_like(R) * 1.j, R, GMRES_params).run()
            return R_sum
        else:
            raise ValueError('Sum method', sum_method, 'not recognized!')

    def infinite_sum_left(self, p, X):
        """
        Infinite sum to the left, see Eq. (194) in :cite:`vanderstraeten2019`

        p : float
            The momentum of the state; for unit cells larger than 1, we already include the
            factor of the smaller Brillouin zone: p*L.
        X : list of :class:`~tenpy.linalg.np_conserved.Array`
            Current excitation tensors for each site of the unit cell.

        Returns
        -------
        L_sum : :class:`~tenpy.linalg.np_conserved.Array`
            Array representing the left environment including the `B` tensors at each site.

        Options
        -------
        .. cfg:config :: PlaneWaveExcitationEngine

            sum_method : ``explicit`` | ``GMRES``
                Whether to explicitly sum the environment by applying the unit cell tensors until convergence (default) or solving the geometric series with the GMRES method.
            sum_tol : float
                Convergence criterion for the explicit summation.
            sum_iterations : int
                Maximum number of iterations for the explicit summation (default sum_iterations=100).
        """
        sum_tol = self.options.get('sum_tol', 1.e-10, 'real')
        sum_iterations = self.options.get('sum_iterations', 100, int)
        sum_method = self.options.get('sum_method', 'explicit', str)

        B = npc.tensordot(self.VLs[0], X[0], axes=(['vR'], ['vL']))
        LB = append_left_env([B], [self.ALs[0]], self.LW, Ws=[self.Ws[0]])
        for i in range(1, self.L):
            B = npc.tensordot(self.VLs[i], X[i], axes=(['vR'], ['vL']))
            LB = append_left_env([B], [self.ALs[i]], self.GS_env_L.get_LP(i), Ws=[self.Ws[i]]) + \
                 append_left_env([self.ARs[i]], [self.ALs[i]], LB, Ws=[self.Ws[i]])
        L = LB

        if np.isclose(npc.norm(L), 0):
            return L
        if sum_method == 'explicit':
            L_sum = L.copy()
            for i in range(sum_iterations):
                L = np.exp(1.0j * p * self.L) * append_left_env(self.ARs, self.ALs, L, Ws=self.Ws)
                L_sum.iadd_prefactor_other(1., L)
                if npc.norm(L) < sum_tol:
                    break
            return L_sum
        elif 'GMRES' in sum_method:

            class helper_matvec(NpcLinearOperator):

                def __init__(self, excit, ALs, ARs, Ws, sum_method):
                    self.ALs = ALs
                    self.ARs = ARs
                    self.Ws = Ws
                    self.sum_method = sum_method
                    self.excit = excit

                def matvec(self, vec):
                    lT = append_left_env(self.ARs, self.ALs, vec, Ws=self.Ws)
                    if 'reg' in self.sum_method:
                        raise NotImplementedError(
                            'GMRES-reg not implemented for multi-site unit cell.')
                        lr = npc.tensordot(vec,
                                           self.excit.r_RL,
                                           axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        lrr = npc.tensordot(vec,
                                            self.excit.CcRW,
                                            axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        T1_l = self.excit.l_RL * (
                            (self.excit.e_RL - 1) * lr + lrr) + self.excit.LWC * lr
                        lT = lT - T1_l
                    return vec - np.exp(1.0j * p * self.excit.L) * lT

            tm_op = helper_matvec(self, self.ALs, self.ARs, self.Ws, sum_method)
            GMRES_params = self.options.subconfig('GMRES_params')
            L_sum, _, _, _ = GMRES(tm_op, npc.Array.zeros_like(L) * 1.j, L, GMRES_params).run()
            return L_sum
        else:
            raise ValueError('Sum method', sum_method, 'not recognized!')

    class Aligned_Effective_H(NpcLinearOperator):
        r"""Class defining the effective Hamiltonian for the excitation tensors `X`.

        Where the `B` tensors are in the same unit cell as the tensors we want to update.

        For a single-site unit cell the effective Hamiltonian looks like this::

                |        .--- B  ---.
                |        |    |     |
                |       LW----W0----RW
                |        |    |     |
                |        .---VL-  --.

        Parameters
        ----------
        outer : :class:`PlaneWaveExcitationEngine`
            Parent engine for the plane wave excitation ansatz.
        """

        def __init__(self, outer):
            self.ALs = outer.ALs
            self.ARs = outer.ARs
            self.VLs = outer.VLs
            self.LW = outer.LW
            self.RW = outer.RW
            self.Ws = outer.Ws
            self.outer = outer

        def matvec(self, vec):

            total_vec = [npc.Array.zeros_like(v) for v in vec]

            for i in range(self.outer.L):
                LB = npc.Array.zeros_like(self.LW)
                RB = npc.Array.zeros_like(self.RW)
                for j in range(i):
                    B = npc.tensordot(self.VLs[j], vec[j], axes=(['vR'], ['vL']))
                    if j > 0:
                        LB = append_left_env([B], [self.ALs[j]], self.outer.GS_env_L.get_LP(j), Ws=[self.Ws[j]]) + \
                             append_left_env([self.ARs[j]], [self.ALs[j]], LB, Ws=[self.Ws[j]]) # Does one extra multiplication when i = 0
                    else:
                        LB = append_left_env([B], [self.ALs[j]],
                                             self.outer.GS_env_L.get_LP(j),
                                             Ws=[self.Ws[j]])

                B = npc.tensordot(self.VLs[i], vec[i], axes=(['vR'], ['vL']))
                LB = append_left_env([self.ARs[i]], [self.VLs[i]], LB, Ws=[self.Ws[i]])
                LP1 = append_left_env([self.ALs[i]], [self.VLs[i]],
                                      self.outer.GS_env_L.get_LP(i),
                                      Ws=[self.Ws[i]])
                LP2 = append_left_env([B], [self.VLs[i]],
                                      self.outer.GS_env_L.get_LP(i),
                                      Ws=[self.Ws[i]])

                for j in reversed(range(i + 1, self.outer.L)):
                    B = npc.tensordot(self.VLs[j], vec[j], axes=(['vR'], ['vL']))
                    if j < self.outer.L - 1:
                        RB = append_right_env([B], [self.ARs[j]], self.outer.GS_env_R.get_RP(j), Ws=[self.Ws[j]]) + \
                             append_right_env([self.ALs[j]], [self.ARs[j]], RB, Ws=[self.Ws[j]])
                    else:
                        RB = append_right_env([B], [self.ARs[j]],
                                              self.outer.GS_env_R.get_RP(j),
                                              Ws=[self.Ws[j]])
                if i > 0:
                    total_vec[i] += npc.tensordot(LB,
                                                  self.outer.GS_env_R.get_RP(i),
                                                  axes=(['vR', 'wR'], ['vL', 'wL']))
                if i < self.outer.L - 1:
                    total_vec[i] += npc.tensordot(LP1, RB, axes=(['vR', 'wR'], ['vL', 'wL']))
                total_vec[i] += npc.tensordot(LP2,
                                              self.outer.GS_env_R.get_RP(i),
                                              axes=(['vR', 'wR'], ['vL', 'wL']))

            return total_vec

    class Unaligned_Effective_H(NpcLinearOperator):
        r"""Class defining the effective Hamiltonian for the excitation tensors `X`, where the `B` tensors are in left (LB) or right (RB) environment.

        For a single-site unit cell the effective Hamiltonian looks like this::

                |         .---  AR ---.            .---  AL ---.
                |     ip  |     |     |       -ip  |     |     |
                |   e     LB----W0----RW   + e     LW----W0----RB
                |         |     |     |            |     |     |
                |         .---VL-   --.            .---VL-   --.

        Parameters
        ----------
        outer : :class:`PlaneWaveExcitationEngine`
            Parent engine for the plane wave excitation ansatz.
        p : float
            The momentum of the state; for unit cells larger than 1, we already include the
            factor of the smaller Brillouin zone: p*L.
        """

        def __init__(self, outer, p):
            self.ALs = outer.ALs
            self.ARs = outer.ARs
            self.VLs = outer.VLs
            self.LW = outer.LW
            self.RW = outer.RW
            self.Ws = outer.Ws
            self.p = p
            self.outer = outer

        def matvec(self, vec):

            total = [npc.Array.zeros_like(v) for v in vec]

            inf_sum_TR = self.outer.infinite_sum_right(self.p, vec)
            cached_TR = [inf_sum_TR]
            for i in reversed(range(1, self.outer.L)):
                cached_TR.insert(
                    0, append_right_env([self.ALs[i]], [self.ARs[i]],
                                        cached_TR[0],
                                        Ws=[self.Ws[i]]))
            for i in range(self.outer.L):
                LP_VL = append_left_env([self.ALs[i]], [self.VLs[i]],
                                        self.outer.GS_env_L.get_LP(i),
                                        Ws=[self.Ws[i]])
                X_out_left = np.exp(-1.0j * self.p * self.outer.L) * npc.tensordot(
                    LP_VL, cached_TR[i], axes=(['vR', 'wR'], ['vL', 'wL']))
                X_out_left.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                total[i] += X_out_left
            cached_TR = []

            inf_sum_TL = self.outer.infinite_sum_left(self.p, vec)
            cached_TL = [inf_sum_TL]
            for i in range(0, self.outer.L - 1):
                cached_TL.append(
                    append_left_env([self.ARs[i]], [self.ALs[i]], cached_TL[-1], Ws=[self.Ws[i]]))
            for i in reversed(range(self.outer.L)):
                TL_VL = append_left_env([self.ARs[i]], [self.VLs[i]],
                                        cached_TL[i],
                                        Ws=[self.Ws[i]])
                X_out_left = np.exp(1.0j * self.p * self.outer.L) * npc.tensordot(
                    TL_VL, self.outer.GS_env_R.get_RP(i), axes=(['vR', 'wR'], ['vL', 'wL']))
                X_out_left.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                total[i] += X_out_left
            cached_TL = []

            return total

    def initial_guess(self, qtotal_change):
        """
        Initial guess for the `X` tensors within a fixed charge sector.

        Parameters
        ----------
        qtotal_change : list of int
            For each charge sector specify how `X` should change the charge.

        Returns
        -------
        X_init : list of :class:`~tenpy.linalg.np_conserved.Array`
            Initial guess for excitation tensors for each site of the unit cell.
        """
        X_init = []
        valid_charge = False
        for i in range(self.L):
            vL = self.VLs[i].get_leg('vR').conj()
            vR = self.ALs[(i + 1) % self.L].get_leg('vL').conj()
            th0 = npc.Array.from_func(np.ones, [vL, vR],
                                      dtype=self.psi.dtype,
                                      qtotal=qtotal_change,
                                      labels=['vL', 'vR'])

            if np.isclose(npc.norm(th0), 0):
                logger.warn("Initial guess for an X is zero; charges not be allowed on site %d.",
                            i)
            else:
                valid_charge = True
                LP = self.GS_env_L.get_LP(i, store=True)
                RP = self.GS_env_R.get_RP(i, store=True)
                LP = append_left_env([self.VLs[i]], [self.VLs[i]], LP, Ws=[self.Ws[i]])

                H0 = ZeroSiteH.from_LP_RP(LP, RP)
                if self.model.H_MPO.explicit_plus_hc:
                    H0 = SumNpcLinearOperator(H0, H0.adjoint())

                lanczos_params = self.options.subconfig('lanczos_params')
                _, th0, _ = LanczosGroundState(H0, th0, lanczos_params).run()

            X_init.append(th0)

        logger.info("Norms of the initial guess: %r.", [npc.norm(x) for x in X_init])
        assert valid_charge, "No X is non-zero; charge is not valid for gluing."
        return X_init


class MultiSitePlaneWaveExcitationEngine(Algorithm):
    r""" Engine to compute quasiparticle excitations across multiple sites for uniform MPS. For each site in the unit cell one multi-site excitation tensor is computed.

    Parameters are the same as for :class:`~tenpy.algorithms.algorithm.Algorithm`.

    Options
    -------
    .. cfg:config :: MultiSitePlaneWaveExcitationEngine
        :include: Algorithm

        lanczos_params : dict
            Lanczos parameters as described in :cfg:config:`KrylovBased`.
        lambda_C1 : float
            Energy shift from contracting the infinite environments. If `None`, compute it again.
        init_env_data : dict
            Dictionary as returned by ``self.env.get_initialization_data()`` from
            :meth:`~tenpy.networks.mpo.MPOEnvironment.get_initialization_data`.
        excitation_size : int
            Number of sites of the excitation, i.e. how many sites in the uniform MPS are replaced with orthogonal tensors. This can be larger than the unit cell or incommensurate.

    Attributes
    ----------
    psi : :class:`~tenpy.networks.uniform_mps.UniformMPS`
        The uniform MPS for which we compute (orthogonal) excitations.
    model : :class:`~tenpy.models.model.MPOModel`
        The model defining the Hamiltonian.
    ACs : list of :class:`~tenpy.linalg.np_conserved.Array`
        The 'center-site' tensors of psi.
    ALs : list of :class:`~tenpy.linalg.np_conserved.Array`
        The 'left-orthonormal' tensors of psi.
    AR : list of :class:`~tenpy.linalg.np_conserved.Array`
        The 'right-orthonormal' tensors of psi.
    Cs : list of :class:`~tenpy.linalg.np_conserved.Array`
        The center matrices of psi.
    Ws : list of :class:`~tenpy.linalg.np_conserved.Array`
        The matrices of the MPO.
    VLs : list of :class:`~tenpy.linalg.np_conserved.Array`
        The orthogonal complements for each AL.
    """

    def __init__(self, psi, model, options, **kwargs):
        super().__init__(psi, model, options, **kwargs)

        assert self.psi.L == self.model.H_MPO.L
        self.L = self.psi.L

        self.size = self.options.get('excitation_size', 1, int)
        assert self.size >= 1

        self.ALs = [self.psi.get_AL(i) for i in range(self.L)]
        self.ARs = [self.psi.get_AR(i) for i in range(self.L)]
        self.ACs = [self.psi.get_AC(i) for i in range(self.L)]
        self.Cs = [self.psi.get_C(i) for i in range(self.L)]  # C on the left
        self.H = self.model.H_MPO
        self.Ws = [self.H.get_W(i) for i in range(self.L)]
        if len(self.Ws) < len(self.ALs):
            assert len(self.ALs) % len(self.Ws)
            self.Ws = self.Ws * len(self.ALs) // len(self.Ws)

        self.IdL = self.H.get_IdL(0)
        self.IdR = self.H.get_IdR(-1)

        self.guess_init_env_data = self.options.get('init_env_data', None)

        # Construct VL, needed to parametrize - B - as - VL - X -
        #                                       |        |
        # Use prescription under Eq. 85 in Tangent Space lecture notes.
        self.VLs = [construct_orthogonal(self.ALs[i]) for i in range(self.L)]

        # Get left and right generalized eigenvalues
        self.boundary_env_data, self.energy_density, _ = MPOTransferMatrix.find_init_LP_RP(
            self.H, self.psi, calc_E=True, guess_init_env_data=self.guess_init_env_data)
        self.energy_density = np.mean(self.energy_density)
        self.LW = self.boundary_env_data['init_LP']
        self.RW = self.boundary_env_data['init_RP']

        # We create GS_env_L and GS_env_R to make topological easier.
        self.GS_env = self.GS_env_L = self.GS_env_R = MPOEnvironment(self.psi, self.H, self.psi,
                                                                     **self.boundary_env_data)
        self.lambda_C1 = options.get('lambda_C1', None, 'real')
        if self.lambda_C1 is None:
            C0_L = self.Cs[0]
            norm = npc.tensordot(C0_L, C0_L.conj(), axes=(['vL', 'vR'], ['vL*', 'vR*']))
            self.lambda_C1 = npc.tensordot(C0_L, self.RW, axes=(['vR'], ['vL']))
            self.lambda_C1 = npc.tensordot(self.LW,
                                           self.lambda_C1,
                                           axes=(['wR', 'vR'], ['wL', 'vL']))
            self.lambda_C1 = npc.tensordot(
                self.lambda_C1, C0_L.conj(), axes=(['vR*', 'vL*'], ['vL*', 'vR*'])) / norm

        strange = []
        for i in range(self.L):
            temp_L = self.GS_env.get_LP(i)
            temp_R = self.GS_env.get_RP(i)
            temp = append_left_env([self.VLs[i]], [self.ACs[i]], temp_L, Ws=[self.Ws[i]])
            temp = npc.tensordot(temp, temp_R, axes=(['wR', 'vR*'], ['wL', 'vL*']))
            strange.append(npc.norm(temp))
        logger.info("Norm of H|psi> projected into the tangent space on each site: %r.", strange)

    def run(self, p, qtotal_change=None, orthogonal_to=[], E_boosts=[], num_ev=1):
        """ Run the plane-wave algorithm to find excited states of the given model.

        Parameters
        ----------
        p : float
            The momentum of the state; for unit cells larger than 1, we already include the
            factor of the smaller Brillouin zone: p*L.
        qtotal_change : list of int
            Charge sectors for each of the defined charges.
        orthogonal_to : list of list of :class:`~tenpy.linalg.np_conserved.Array`
            Find excitations orthogonal to previously found tensors X.
        E_boosts: list of float
            energy boosts for orthogonal states
        num_ev: int
            Number of eigenvalues and eigenvectors, that we extract from a single Arnoldi/ Lanczos run

        Returns
        -------
        Es : list of float
            Energies of the lowest-energy excitations. Number equal to `num_ev`.
        psis : list of :class:`~tenpy.networks.momentum_mps.MomentumMPS`
            MomentumMPS corresponding to the lowest-energy excitations. Number equal to `num_ev`.

        Options
        -------
        .. cfg:config :: PlaneWaveExcitationEngine

            E_boost : float
                uniform strength of the energy boosts (instead of specifying a list), default to E_boost=100

        """
        self.aligned_H = self.Aligned_Effective_H(self, p)
        self.unaligned_H = self.Unaligned_Effective_H(self, p)
        effective_H = SumNpcLinearOperator(self.aligned_H, self.unaligned_H)
        lanczos_params = self.options.subconfig('lanczos_params')
        X_init = self.initial_guess(qtotal_change)
        if len(E_boosts) != len(orthogonal_to):
            E_boost = self.options.get('E_boost', 100, 'real')
            E_boosts = [E_boost] * len(orthogonal_to)
        if len(orthogonal_to) > 0:
            effective_H = BoostNpcLinearOperator(effective_H, E_boosts, orthogonal_to)

        # how many unit cells to include all excitations
        multiple_unit_cell = int(np.ceil((self.L - 1 + self.size) / self.L))

        if num_ev > 1:
            lanczos_params['which'] = 'SR'
            lanczos_params['num_ev'] = num_ev
            energies, Xs, N = Arnoldi(effective_H, X_init, lanczos_params).run()
            psis = []
            Es = []
            for E, X in zip(energies, Xs):
                psis.append(MomentumMPS(X, self.psi, p, self.size))
                Es.append(E - self.lambda_C1 - self.energy_density * (self.L * multiple_unit_cell))
        else:
            energy, X, N = LanczosGroundState(effective_H, X_init, lanczos_params).run()
            Es = [energy - self.lambda_C1 - self.energy_density * (self.L * multiple_unit_cell)]
            psis = [MomentumMPS(X, self.psi, p, self.size)]

        if N == lanczos_params.get('N_max', 20, int):
            import warnings
            warnings.warn('Maximum Lanczos iterations needed; be wary of results.')

        return np.real_if_close(Es), psis, N

    def resume_run(self):
        raise NotImplementedError()

    def energy(self, p, X):
        """
        Compute the energy of excited states

        Parameters
        ----------
        p : float
            The momentum of the state; for unit cells larger than 1, we already include the
            factor of the smaller Brillouin zone: p*L.
        X : list of :class:`~tenpy.linalg.np_conserved.Array`
            Excitation tensors for each site of the unit cell.

        Returns
        -------
        energy : float
            Energy

        """
        multiple_unit_cell = int(np.ceil((self.L - 1 + self.size) / self.L))
        self.aligned_H = self.Aligned_Effective_H(self, p)
        self.unaligned_H = self.Unaligned_Effective_H(self, p)
        effective_H = SumNpcLinearOperator(self.aligned_H, self.unaligned_H)
        HX = effective_H.matvec(X)
        E = np.real(npc.inner(X, HX)).item()
        return E - self.lambda_C1 - self.energy_density * (self.L * multiple_unit_cell)

    def attach_right(self, VL, X, As, R, Ws=None):
        """
        attach excitation tensors to a right environment
        """
        B = npc.tensordot(VL.replace_label('p', 'p0'), X, axes=(['vR'], ['vL']))
        RB = npc.tensordot(B, R, axes=(['vR'], ['vL']))

        for i in reversed(range(len(As))):
            p = 'p' + str(i)
            if Ws is not None:
                RB = npc.tensordot(RB, Ws[i], axes=([p, 'wL'], ['p*', 'wR']))
            RB = npc.tensordot(RB, As[i].conj(), axes=(['p', 'vL*'], ['p*', 'vR*']))
        return RB

    def _starting_right_TR(self, X):
        """
        Sum up all single-B environments from the right and fill with tensors to complete unit cell
        """
        i = 0
        RP = self.GS_env_R.get_RP(i + self.size - 1)
        RB = self.attach_right(self.VLs[i],
                               X[i], [self.ARs[j % self.L] for j in range(i, i + self.size)],
                               RP,
                               Ws=[self.Ws[j % self.L] for j in range(i, i + self.size)])
        RB = append_right_env(self.ALs[:i], self.ARs[:i], RB, Ws=self.Ws[:i])
        RW = RB
        for i in range(1, self.L):
            RP = self.GS_env_R.get_RP(i + self.size - 1)
            RB = self.attach_right(self.VLs[i],
                                   X[i], [self.ARs[j % self.L] for j in range(i, i + self.size)],
                                   RP,
                                   Ws=[self.Ws[j % self.L] for j in range(i, i + self.size)])
            RB = append_right_env(self.ALs[:i], self.ARs[:i], RB, Ws=self.Ws[:i])
            RW += RB
        return RW

    def infinite_sum_right(self, p, X):
        """
        Infinite sum to the right, see Eq. (194) in :cite:`vanderstraeten2019`

        p : float
            The momentum of the state; for unit cells larger than 1, we already include the
            factor of the smaller Brillouin zone: p*L.
        X : list of :class:`~tenpy.linalg.np_conserved.Array`
            Current excitation tensors for each site of the unit cell.

        Returns
        -------
        R_sum : :class:`~tenpy.linalg.np_conserved.Array`
            Array representing the right environment including the `B` tensors at each site.

        Options
        -------
        .. cfg:config :: PlaneWaveExcitationEngine

            sum_method : ``explicit`` | ``GMRES``
                Whether to explicitly sum the environment by applying the unit cell tensors until convergence (default) or solving the geometric series with the GMRES method.
            sum_tol : float
                Convergence criterion for the explicit summation.
            sum_iterations : int
                Maximum number of iterations for the explicit summation (default sum_iterations=100).
        """
        sum_tol = self.options.get('sum_tol', 1.e-10, 'real')
        sum_iterations = self.options.get('sum_iterations', 100, int)
        sum_method = self.options.get('sum_method', 'explicit', str)

        R = self._starting_right_TR(X)

        if np.isclose(npc.norm(R), 0):
            return R
        if sum_method == 'explicit':
            R_sum = R.copy()
            for _ in range(sum_iterations):
                R = np.exp(-1.0j * p * self.L) * append_right_env(
                    self.ALs, self.ARs, R, Ws=self.Ws)
                R_sum.iadd_prefactor_other(1., R)
                if npc.norm(R) < sum_tol:
                    break
            return R_sum
        elif 'GMRES' in sum_method:

            class helper_matvec(NpcLinearOperator):

                def __init__(self, excit, ALs, ARs, Ws, sum_method):
                    self.ALs = ALs
                    self.ARs = ARs
                    self.Ws = Ws
                    self.sum_method = sum_method
                    self.excit = excit

                def matvec(self, vec):
                    Tr = append_right_env(self.ALs, self.ARs, vec, Ws=self.Ws)
                    if 'reg' in self.sum_method:
                        raise NotImplementedError(
                            'GMRES-reg not implemented for multi-site unit cell.')
                        lr = npc.tensordot(self.excit.l_LR,
                                           vec,
                                           axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        llr = npc.tensordot(self.excit.LWCc,
                                            vec,
                                            axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        T1_r = self.excit.r_LR * (
                            (self.excit.e_LR - 1) * lr + llr) + self.excit.CRW * lr
                        Tr = Tr - T1_r
                    return vec - np.exp(-1.0j * p * self.excit.L) * Tr

            tm_op = helper_matvec(self, self.ALs, self.ARs, self.Ws, sum_method)
            GMRES_params = self.options.subconfig('GMRES_params')
            R_sum, _, _, _ = GMRES(tm_op, npc.Array.zeros_like(R) * 1.j, R, GMRES_params).run()
            return R_sum
        else:
            raise ValueError('Sum method', sum_method, 'not recognized!')

    def attach_left(self, VL, X, As, L, Ws=None):
        """
        attach excitation tensors to a left environment
        """
        B = npc.tensordot(VL.replace_label('p', 'p0'), X, axes=(['vR'], ['vL']))
        LB = npc.tensordot(L, B, axes=(['vR'], ['vL']))
        for i in range(len(As)):
            p = 'p' + str(i)
            if Ws is not None:
                LB = npc.tensordot(Ws[i], LB, axes=(['p*', 'wL'], [p, 'wR']))
            LB = npc.tensordot(As[i].conj(), LB, axes=(['p*', 'vL*'], ['p', 'vR*']))
        return LB

    def _starting_left_TL(self, X):
        """
        Sum up all single-B environments from the left and fill with tensors to complete unit cell
        """
        multiple_unit_cell = int(np.ceil(
            (self.L - 1 + self.size) / self.L))  # number of extension of unit cells
        i = 0
        LP = self.GS_env_L.get_LP(i)
        LB = self.attach_left(self.VLs[i],
                              X[i], [self.ALs[j % self.L] for j in range(i, i + self.size)],
                              LP,
                              Ws=[self.Ws[j % self.L] for j in range(i, i + self.size)])
        for j in range(i + self.size, multiple_unit_cell * self.L):
            LB = append_left_env([self.ARs[j % self.L]], [self.ALs[j % self.L]],
                                 LB,
                                 Ws=[self.Ws[j % self.L]])
        LW = LB
        for i in range(1, self.L):
            LP = self.GS_env_L.get_LP(i)
            LB = self.attach_left(self.VLs[i],
                                  X[i], [self.ALs[j % self.L] for j in range(i, i + self.size)],
                                  LP,
                                  Ws=[self.Ws[j % self.L] for j in range(i, i + self.size)])
            for j in range(i + self.size, multiple_unit_cell * self.L):
                LB = append_left_env([self.ARs[j % self.L]], [self.ALs[j % self.L]],
                                     LB,
                                     Ws=[self.Ws[j % self.L]])
            LW += LB
        return LW

    def infinite_sum_left(self, p, X):
        """
        Infinite sum to the left, see Eq. (194) in :cite:`vanderstraeten2019`

        p : float
            The momentum of the state; for unit cells larger than 1, we already include the
            factor of the smaller Brillouin zone: p*L.
        X : list of :class:`~tenpy.linalg.np_conserved.Array`
            Current excitation tensors for each site of the unit cell.

        Returns
        -------
        L_sum : :class:`~tenpy.linalg.np_conserved.Array`
            Array representing the left environment including the `B` tensors at each site.

        Options
        -------
        .. cfg:config :: PlaneWaveExcitationEngine

            sum_method : ``explicit`` | ``GMRES``
                Whether to explicitly sum the environment by applying the unit cell tensors until convergence (default) or solving the geometric series with the GMRES method.
            sum_tol : float
                Convergence criterion for the explicit summation.
            sum_iterations : int
                Maximum number of iterations for the explicit summation (default sum_iterations=100).
        """
        sum_tol = self.options.get('sum_tol', 1.e-10, 'real')
        sum_iterations = self.options.get('sum_iterations', 100, int)
        sum_method = self.options.get('sum_method', 'explicit', str)

        # shift unit cell to the left to include all excitations
        self.shift_unit_cell = None
        if self.size == 1:
            self.shift_unit_cell = 0
        elif self.L == 1:
            self.shift_unit_cell = self.size - 1
        elif self.size > self.L:
            self.shift_unit_cell = self.size // self.L
        else:
            self.shift_unit_cell = 1
        LB = np.exp(1.0j * p * self.L * self.shift_unit_cell) * self._starting_left_TL(X)

        if np.isclose(npc.norm(LB), 0):
            return LB
        if sum_method == 'explicit':
            L_sum = LB.copy()
            for i in range(sum_iterations):
                LB = np.exp(1.0j * p * self.L) * append_left_env(
                    self.ARs, self.ALs, LB, Ws=self.Ws)
                L_sum.iadd_prefactor_other(1., LB)
                if npc.norm(LB) < sum_tol:
                    break
            return L_sum
        elif 'GMRES' in sum_method:

            class helper_matvec(NpcLinearOperator):

                def __init__(self, excit, ALs, ARs, Ws, sum_method):
                    self.ALs = ALs
                    self.ARs = ARs
                    self.Ws = Ws
                    self.sum_method = sum_method
                    self.excit = excit

                def matvec(self, vec):
                    lT = append_left_env(self.ARs, self.ALs, vec, Ws=self.Ws)
                    if 'reg' in self.sum_method:
                        raise NotImplementedError(
                            'GMRES-reg not implemented for multi-site unit cell.')
                        lr = npc.tensordot(vec,
                                           self.excit.r_RL,
                                           axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        lrr = npc.tensordot(vec,
                                            self.excit.CcRW,
                                            axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        T1_l = self.excit.l_RL * (
                            (self.excit.e_RL - 1) * lr + lrr) + self.excit.LWC * lr
                        lT = lT - T1_l
                    return vec - np.exp(1.0j * p * self.excit.L) * lT

            tm_op = helper_matvec(self, self.ALs, self.ARs, self.Ws, sum_method)
            GMRES_params = self.options.subconfig('GMRES_params')
            L_sum, _, _, _ = GMRES(tm_op, npc.Array.zeros_like(LB) * 1.j, LB, GMRES_params).run()
            return L_sum
        else:
            raise ValueError('Sum method', sum_method, 'not recognized!')

    class Aligned_Effective_H(NpcLinearOperator):
        r"""Class defining the effective Hamiltonian for the multi-site excitation tensors `X`, where there is overlap between `B` tensors and the unit cell we want to update.

        Parameters
        ----------
        outer : :class:`PlaneWaveExcitationEngine`
            Parent engine for the plane wave excitation ansatz.
        p : float
            The momentum of the state; for unit cells larger than 1, we already include the
            factor of the smaller Brillouin zone: p*L.
        """

        def __init__(self, outer, p):
            self.ALs = outer.ALs
            self.ARs = outer.ARs
            self.VLs = outer.VLs
            self.LW = outer.LW
            self.RW = outer.RW
            self.Ws = outer.Ws
            self.p = p
            self.outer = outer

        def matvec(self, vec):
            size = self.outer.size
            L = self.outer.L

            total_vec = [npc.Array.zeros_like(v) for v in vec]
            multiple_unit_cell = int(np.ceil((L - 1 + size) / L))
            for i in range(L):
                # all contributions from shifting Bs to the right
                for j in range(size):
                    LW = self.outer.GS_env_L.get_LP(i)
                    RW = self.outer.GS_env_R.get_RP((i + j + size - 1) % L)
                    for _ in range(int(np.ceil((i + j + size) / L)),
                                   multiple_unit_cell):  # attach complete unit cells if necessary
                        RW = append_right_env(
                            [self.ARs[n % L] for n in range(i + j + size, i + j + size + L)],
                            [self.ARs[n % L] for n in range(i + j + size, i + j + size + L)], RW,
                            [self.Ws[n % L] for n in range(i + j + size, i + j + size + L)])

                    B = npc.tensordot(self.VLs[(i + j) % L].replace_label('p', 'p0'),
                                      vec[(i + j) % L],
                                      axes=(['vR'], ['vL']))
                    RW = npc.tensordot(B, RW, axes=(['vR'], ['vL']))
                    for n in reversed(range(j, size + j)):
                        p = 'p' + str(n - j)
                        RW = npc.tensordot(RW,
                                           self.Ws[(n + i) % L],
                                           axes=([p, 'wL'], ['p*', 'wR']))
                        if n >= size:
                            RW = npc.tensordot(RW,
                                               self.ARs[(n + i) % L].conj(),
                                               axes=(['p', 'vL*'], ['p*', 'vR*']))
                        else:
                            RW.ireplace_label('p', 'p' + str(n))

                    for k in range(j):
                        LW = npc.tensordot(LW, self.ALs[(i + k) % L], axes=(['vR'], ['vL']))
                        LW = npc.tensordot(LW,
                                           self.Ws[(i + k) % L],
                                           axes=(['wR', 'p'], ['wL', 'p*']))
                        LW.ireplace_label('p', 'p' + str(k))

                    if j == 0:
                        LW = npc.tensordot(LW, self.VLs[i].conj(), axes=(['vR*'], ['vL*']))
                        X_out_right = npc.tensordot(LW,
                                                    RW,
                                                    axes=(['vR', 'wR', 'p*'], ['vL', 'wL', 'p0']))
                    else:
                        LW = npc.tensordot(LW,
                                           self.VLs[i].conj(),
                                           axes=(['vR*', 'p0'], ['vL*', 'p*']))
                        X_out_right = npc.tensordot(LW, RW, axes=(['vR', 'wR'], ['vL', 'wL']))
                    X_out_right.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                    phase = (i + j) // L
                    X_out_right.itranspose(total_vec[i].get_leg_labels())
                    total_vec[i] += X_out_right * np.exp(-1.0j * self.p * L * phase)

                # all contributions from shifting Bs to the left
                for j in range(i - size + 1, i):
                    LW = self.outer.GS_env_L.get_LP(j % L)
                    RW = self.outer.GS_env_R.get_RP((size - 1 + i) % L)

                    B = npc.tensordot(self.VLs[j % L].replace_label('p', 'p0'),
                                      vec[j % L],
                                      axes=(['vR'], ['vL']))
                    LW = npc.tensordot(LW, B, axes=(['vR'], ['vL']))
                    for n in range(j, j + size):
                        p = 'p' + str(n - j)
                        LW = npc.tensordot(LW, self.Ws[n % L], axes=([p, 'wR'], ['p*', 'wL']))
                        if n < i:
                            LW = npc.tensordot(LW,
                                               self.ALs[n % L].conj(),
                                               axes=(['p', 'vR*'], ['p*', 'vL*']))
                        else:
                            LW.ireplace_label('p', 'p' + str(n - i))

                    for k in reversed(range(j + size, size + i)):
                        RW = npc.tensordot(self.ARs[k % L], RW, axes=(['vR'], ['vL']))
                        RW = npc.tensordot(self.Ws[k % L], RW, axes=(['wR', 'p*'], ['wL', 'p']))
                        RW.ireplace_label('p', 'p' + str(k - i))

                    LW = npc.tensordot(LW, self.VLs[i].conj(), axes=(['vR*', 'p0'], ['vL*', 'p*']))

                    X_out_left = npc.tensordot(LW, RW, axes=(['vR', 'wR'], ['vL', 'wL']))
                    X_out_left.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                    X_out_left.itranspose(total_vec[i].get_leg_labels())
                    phase = j // L
                    total_vec[i] += X_out_left * np.exp(-1.0j * self.p * L * phase)
            return total_vec

    class Unaligned_Effective_H(NpcLinearOperator):
        r"""Class defining the effective Hamiltonian for the multi-site excitation tensors `X`, where the `B` tensors are in left (LB) or right (RB) environment.

        Parameters
        ----------
        outer : :class:`PlaneWaveExcitationEngine`
            Parent engine for the plane wave excitation ansatz.
        p : float
            The momentum of the state; for unit cells larger than 1, we already include the
            factor of the smaller Brillouin zone: p*L.
        """

        def __init__(self, outer, p):
            self.ALs = outer.ALs
            self.ARs = outer.ARs
            self.VLs = outer.VLs
            self.LW = outer.LW
            self.RW = outer.RW
            self.Ws = outer.Ws
            self.p = p
            self.outer = outer

        def matvec(self, vec):
            size = self.outer.size
            L = self.outer.L

            total = [npc.Array.zeros_like(v) for v in vec]

            # sums where Bs are to the right
            inf_sum_TR = self.outer.infinite_sum_right(self.p, vec)
            for i in range(L):
                multiple_unit_cell = int(np.ceil(
                    (i + size) / L))  # number of extension of unit cells
                LP_VL = append_left_env([self.ALs[i]], [self.VLs[i]],
                                        self.outer.GS_env_L.get_LP(i),
                                        Ws=[self.Ws[i]])
                for j in range(1, size):
                    LP_VL = npc.tensordot(LP_VL, self.ALs[(i + j) % L], axes=(['vR'], ['vL']))
                    LP_VL = npc.tensordot(LP_VL,
                                          self.Ws[(i + j) % L],
                                          axes=(['wR', 'p'], ['wL', 'p*']))
                    LP_VL.ireplace_label('p', 'p' + str(j))
                RB = inf_sum_TR * np.exp(-1.0j * self.p * L * (multiple_unit_cell))

                for j in reversed(range(i + size, multiple_unit_cell * L)):
                    RP = self.outer.GS_env_R.get_RP((j + size - 1) % L)

                    RB = append_right_env([self.ALs[j % L]], [self.ARs[j % L]],
                                          RB,
                                          Ws=[self.Ws[j % L]])
                    RB += self.outer.attach_right(
                        self.VLs[j % L],
                        vec[j % L], [self.ARs[k % L] for k in range(j, j + size)],
                        RP,
                        Ws=[self.Ws[k % L]
                            for k in range(j, j + size)]) * np.exp(-1.0j * self.p * L *
                                                                   (multiple_unit_cell - 1))

                X_out_right = npc.tensordot(LP_VL, RB, axes=(['vR', 'wR'], ['vL', 'wL']))
                X_out_right.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                total[i] += X_out_right

            # sums where Bs are to the left
            inf_sum_TL = self.outer.infinite_sum_left(self.p, vec)
            for i in range(L):
                RP = self.outer.GS_env_R.get_RP((i + size - 1) % L)
                for j in reversed(range(0, size)):
                    RP = npc.tensordot(self.ARs[(i + j) % L], RP, axes=(['vR'], ['vL']))
                    RP = npc.tensordot(self.Ws[(i + j) % L], RP, axes=(['wR', 'p*'], ['wL', 'p']))
                    RP.ireplace_label('p', 'p' + str(j))

                LB = inf_sum_TL * np.exp(1.0j * self.p * L)

                # all Bs that fit completely to the left, but are not in inf_sum
                for j in range(-L * self.outer.shift_unit_cell, 0):
                    if j + size <= 0:
                        LP = self.outer.GS_env_L.get_LP(j % L)
                        LP_B = self.outer.attach_left(
                            self.VLs[j % L],
                            vec[j % L], [self.ALs[k % L] for k in range(j, j + size)],
                            LP,
                            Ws=[self.Ws[k % L]
                                for k in range(j, j + size)]) * np.exp(1.0j * self.p * L)
                        for k in range(j + size, 0):
                            LP_B = append_left_env([self.ARs[k % L]], [self.ALs[k % L]],
                                                   LP_B,
                                                   Ws=[self.Ws[k % L]])

                        LB += LP_B

                for j in range(i):
                    LP = self.outer.GS_env_L.get_LP((j - size + 1) % L)
                    phase = (j - size + 1) // L
                    LB = append_left_env([self.ARs[j % L]], [self.ALs[j % L]],
                                         LB,
                                         Ws=[self.Ws[j % L]])
                    LB += self.outer.attach_left(
                        self.VLs[(j - size + 1) % L],
                        vec[(j - size + 1) % L],
                        [self.ALs[k % L] for k in range(j - size + 1, j + 1)],
                        LP,
                        Ws=[self.Ws[k % L] for k in range(j - size + 1, j + 1)]) * np.exp(
                            -1.0j * self.p * L * phase)

                LB = npc.tensordot(LB, self.VLs[i].conj(), axes=(['vR*'], ['vL*']))

                X_out_left = npc.tensordot(LB, RP, axes=(['vR', 'wR', 'p*'], ['vL', 'wL', 'p0']))
                X_out_left.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                total[i] += X_out_left
            return total

    def initial_guess(self, qtotal_change):
        """
        Initial guess for the `X` tensors within a fixed charge sector.

        Parameters
        ----------
        qtotal_change : list of int
            For each charge sector specify how `X` should change the charge.

        Returns
        -------
        X_init : list of :class:`~tenpy.linalg.np_conserved.Array`
            Initial guess for excitation tensors for each site of the unit cell.
        """
        X_init = []
        valid_charge = False
        for i in range(self.L):
            # start with random complex state
            vL = self.VLs[i].get_leg('vR').conj()
            vL_label = 'vL'
            if self.size > 1:
                p_legs = [self.ALs[(i + j) % self.L].get_leg('p') for j in range(1, self.size)]
                plabels = [f'.p{i+1}' for i in range(self.size - 1)]
                vL = LegPipe([vL] + p_legs)
                vL_label = "(" + vL_label + "".join(plabels) + ")"
            vR = self.ALs[(i + self.size) % self.L].get_leg('vL').conj()
            th0 = npc.Array.from_func(np.random.standard_normal, [vL, vR],
                                      dtype=self.psi.dtype,
                                      qtotal=qtotal_change,
                                      labels=[vL_label, 'vR'])
            th0 += 1j * npc.Array.from_func(np.random.standard_normal, [vL, vR],
                                            dtype=self.psi.dtype,
                                            qtotal=qtotal_change,
                                            labels=[vL_label, 'vR'])
            if self.size > 1:
                th0 = th0.split_legs()
            if np.isclose(npc.norm(th0), 0):
                logger.warn("Initial guess for an X is zero; charges not be allowed on site %d.",
                            i)
            else:
                valid_charge = True
                th0 /= npc.norm(th0)

            X_init.append(th0)

        logger.info("Norms of the initial guess: %r.", [npc.norm(x) for x in X_init])
        assert valid_charge, "No X is non-zero; charge is not valid for gluing."
        return X_init
