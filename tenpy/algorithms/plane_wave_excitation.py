# Copyright 2022 TeNPy Developers, GNU GPLv3

import numpy as np
import time
import warnings
import logging
logger = logging.getLogger(__name__)

from ..linalg import np_conserved as npc
from ..networks.momentum_mps import MomentumMPS
from ..networks.mpo import MPOEnvironment, MPOTransferMatrix
from ..linalg.lanczos import GMRES, LanczosGroundState, norm, inner
from ..linalg.sparse import NpcLinearOperator, SumNpcLinearOperator, BoostNpcLinearOperator, ShiftNpcLinearOperator
from ..tools.params import asConfig
from ..tools.math import entropy
from ..algorithms.algorithm import Algorithm
from ..algorithms.mps_common import ZeroSiteH

__all__ = ['TR_general', 'LT_general', 'construct_orthogonal', 'PlaneWaveExcitationEngine', 'TopologicalPlaneWaveExcitationEngine']

"""
TODO - 04/01/2022
(1) Regulated transfer matrix for unit cell > 1
(2) Multi site excitation tensor
(3) Restarted Lanczos
(5) DMRG over the segment
"""

def TR_general(As, Bs, R, Ws=None):
    temp = R.copy()
    for i in reversed(range(len(As))):
        temp = npc.tensordot(Bs[i].conj(), temp, axes=(['vR*'], ['vL*']))
        if Ws is not None:
            temp = npc.tensordot(Ws[i], temp, axes=(['wR', 'p'], ['wL', 'p*']))
        temp = npc.tensordot(As[i], temp, axes=(['vR', 'p'], ['vL', 'p*']))
    return temp

def LT_general(As, Bs, L, Ws=None):
    temp = L.copy()
    for i in range(len(As)):
        temp = npc.tensordot(temp, Bs[i].conj(), axes=(['vR*'], ['vL*']))
        if Ws is not None:
            temp = npc.tensordot(temp, Ws[i], axes=(['wR', 'p*'], ['wL', 'p']))
        temp = npc.tensordot(temp, As[i], axes=(['vR', 'p*'], ['vL', 'p']))
    return temp

def construct_orthogonal(M, cutoff=1.e-13, left=True):
    if left:
        M = M.copy().combine_legs([['vL', 'p'], ['vR']], qconj=[+1, -1])
        Q = npc.orthogonal_columns(M, 'vR')
        assert npc.norm(npc.tensordot(Q, M.conj(), axes=(['(vL.p)'], ['(vL*.p*)']))) < 1.e-14
    else:
        M = M.copy().combine_legs([['vL'], ['p', 'vR']], qconj=[+1, -1])
        Q = npc.orthogonal_columns(M.transpose(['(p.vR)', '(vL)']), 'vL').itranspose(['vL', '(p.vR)'])
        assert npc.norm(npc.tensordot(Q, M.conj(), axes=(['(p.vR)'], ['(p*.vR*)']))) < 1.e-14
    return Q.split_legs()

class PlaneWaveExcitationEngine(Algorithm):
    def __init__(self, psi, model, options, **kwargs):
        super().__init__(psi, model, options, **kwargs)

        assert self.psi.L == self.model.H_MPO.L
        self.L = self.psi.L

        self.ALs = [self.psi.get_AL(i) for i in range(self.L)]
        self.ARs = [self.psi.get_AR(i) for i in range(self.L)]
        self.ACs = [self.psi.get_AC(i) for i in range(self.L)]
        self.Cs = [self.psi.get_C(i) for i in range(self.L)] # C on the left
        self.H = self.model.H_MPO
        self.Ws = [self.H.get_W(i) for i in range(self.L)]
        if len(self.Ws) < len(self.ALs):
            assert len(self.ALs) % len(self.Ws)
            self.Ws = self.Ws * len(self.ALs) // len(self.Ws)

        self.IdL = self.H.get_IdL(0)
        self.IdR = self.H.get_IdR(-1)

        self.guess_init_env_data = self.options.get('init_data',None)
        self.dW = self.Ws[0].get_leg('wR').ind_len # [TODO] this assumes a single site
        self.chi = self.ALs[0].get_leg('vL').ind_len
        self.d = self.ALs[0].get_leg('p').ind_len

        # Construct VL, needed to parametrize - B - as - VL - X -
        #                                       |        |
        # Use prescription under Eq. 85 in Tangent Space lecture notes.
        self.VLs = [construct_orthogonal(self.ALs[i]) for i in range(self.L)]

        # Get left and right generalized eigenvalues
        self.gauge = self.options.get('gauge', 'trace')
        self.boundary_env_data, self.energy_density, _ = MPOTransferMatrix.find_init_LP_RP(self.H, self.psi, calc_E=True, _subtraction_gauge=self.gauge, guess_init_env_data=self.guess_init_env_data)
        self.energy_density = np.mean(self.energy_density)
        self.LW = self.boundary_env_data['init_LP']
        self.RW = self.boundary_env_data['init_RP']

        # We create GS_env_L and GS_env_R to make topological easier.
        self.GS_env = self.GS_env_L = self.GS_env_R = MPOEnvironment(self.psi, self.H, self.psi, **self.boundary_env_data)
        self.lambda_C1 = options.get('lambda_C1', None)
        if self.lambda_C1 is None:
            C0_L = self.Cs[0]
            norm = npc.tensordot(C0_L, C0_L.conj(), axes=(['vL', 'vR'], ['vL*', 'vR*']))
            self.lambda_C1 = npc.tensordot(C0_L, self.RW, axes=(['vR'], ['vL']))
            self.lambda_C1 = npc.tensordot(self.LW, self.lambda_C1, axes=(['wR', 'vR'], ['wL', 'vL']))
            self.lambda_C1 = npc.tensordot(self.lambda_C1, C0_L.conj(), axes=(['vR*', 'vL*'], ['vL*', 'vR*'])) / norm
        # print('L:', self.lambda_C1)

        """
        # Tw[Al,AR]
        self.l_LR = npc.Array.zeros_like(self.LW).itranspose(['vR*','wR', 'vR']) # [TODO] check default ordering to potentially remove transpose
        # Original ordering of boundary vectors ['vR*', 'wR', 'vR']
        self.l_LR[:,self.IdR,:] = self.Cs[0].conj().transpose(['vR*', 'vL*'])

        self.CRW = npc.tensordot(self.Cs[0], self.RW, axes=(['vR'], ['vL'])) # ['vR','wR', 'vR*']
        self.LWCc= npc.tensordot(self.LW, self.Cs[0].conj(), axes=(['vR*'], ['vL*'])) # ['vL*','wL', 'vL']
        self.r_LR = npc.Array.zeros_like(self.RW) # ['vR','wR', 'vR*']
        self.r_LR[:,self.IdL,:] = self.Cs[0]

        # tTr (l_LR * CRW) = tTr (C C^\dag) = 1 usually
        self.CRW = self.CRW / npc.tensordot(self.l_LR, self.CRW, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*']))

        # LWCc -> LWCc / self.e_LR
        self.LWCc = self.LWCc / npc.tensordot(self.LWCc, self.CRW, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*']))

        self.r_LR = self.r_LR / npc.tensordot(self.LWCc, self.r_LR, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*'])) # Dividing by E_C, GS eigenvalue of HC

        Tr = TR_general(self.ALs, self.ARs, self.CRW, Ws=self.Ws)
        #Should be energy density / E_C
        self.e_LR = (Tr[0,self.IdL,0] - self.CRW[0,self.IdL,0])/self.r_LR[0,self.IdL,0] # tTr( LWCc * CRW) for original tensors

        # Tw[AR,AL]
        self.r_RL = npc.Array.zeros_like(self.RW).itranspose(['vL', 'wL', 'vL*']) # [TODO] check default ordering to potentially remove transpose
        # Original ordering of boundary vectors ['vL','wL', 'vL*']
        self.r_RL[:,self.IdL,:] = self.Cs[0].conj().transpose(['vR*', 'vL*'])
        self.LWC = npc.tensordot(self.LW, self.Cs[0], axes=(['vR'], ['vL']))
        self.CcRW= npc.tensordot(self.RW, self.Cs[0].conj(), axes=(['vL*'], ['vR*']))
        self.l_RL = npc.Array.zeros_like(self.LW)
        self.l_RL[:,self.IdR,:] = self.Cs[0]
        self.LWC = self.LWC / npc.tensordot(self.LWC, self.r_RL, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*']))
        self.CcRW = self.CcRW / npc.tensordot(self.LWC, self.CcRW, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*']))
        self.l_RL = self.l_RL / npc.tensordot(self.l_RL, self.CcRW, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*']))
        lT = LT_general(self.ARs, self.ALs, self.LWC, Ws=self.Ws)

        #Should be energy density / E_C
        self.e_RL = (lT[0,self.IdR,0] - self.LWC[0,self.IdR,0])/self.l_RL[0,self.IdR,0]
        """
        self.aligned_H = self.Aligned_Effective_H(self, self.ALs, self.ARs, self.VLs,
                                                  self.LW, self.RW, self.Ws,
                                                  self.chi, d=self.d)

        strange = []
        for i in range(self.L):
            temp_L = self.GS_env.get_LP(i) # LT_general(self.ALs[:i], self.ALs[:i], self.LW, Ws=self.Ws[:i])
            temp_R = self.GS_env.get_RP(i) # TR_general(self.ARs[i+1:], self.ARs[i+1:], self.RW, Ws=self.Ws[i+1:])
            temp = LT_general([self.VLs[i]], [self.ACs[i]], temp_L, Ws=[self.Ws[i]])
            temp = npc.tensordot(temp, temp_R, axes=(['wR', 'vR*'], ['wL', 'vL*']))
            strange.append(npc.norm(temp))
        logger.info("Norm of H|psi> projected into the tangent space on each site: %r.", strange)

        """
        if 1 >= 1:
            print("-"*20, "initializing excitation", "-"*20)
            # Don't know how to subtract these with charges
            #print("norm(LW-LW.dag):", npc.norm(self.LW - self.LW.transpose(['vR','wR','vR*']).conj()))
            #print("norm(RW-RW.dag):", npc.norm(self.RW - self.RW.transpose(['vL*','wL','vL']).complex_conj()))
            assert self.psi.valid_umps
            print("Energy density   :", self.energy_density)
            print("Lambda C1        :", self.lambda_C1)
            print("L norm           :", npc.norm(self.LW))
            print("R norm           :", npc.norm(self.RW))
            print("(L|R)            :", npc.tensordot(self.LW, self.RW, axes=(['vR*', 'wR', 'vR'],['vL*', 'wL', 'vL'])))

            lT  = LT_general(self.ALs, self.ARs, self.l_LR, Ws=self.Ws)
            Tr  = TR_general(self.ALs, self.ARs, self.r_LR, Ws=self.Ws)
            print("LR norm(l-lT)    :", npc.norm(self.l_LR-lT))
            print("LR norm(r-Tr)    :", npc.norm(self.r_LR-Tr))

            lT  = LT_general(self.ARs, self.ALs, self.l_RL, Ws=self.Ws)
            Tr  = TR_general(self.ARs, self.ALs, self.r_RL, Ws=self.Ws)
            print("RL norm(l-lT)    :", npc.norm(self.l_RL-lT))
            print("RL norm(r-Tr)    :", npc.norm(self.r_RL-Tr))

            print("l*rr             :", npc.tensordot(self.l_LR, self.CRW,axes=(['vR*', 'wR', 'vR'],['vL*', 'wL', 'vL'])))
            print("ll*rr            :", npc.tensordot(self.LWCc, self.CRW,axes=(['vR*', 'wR', 'vR'],['vL*', 'wL', 'vL'])))
            print("ll*r             :", npc.tensordot(self.LWCc, self.r_LR,axes=(['vR*', 'wR', 'vR'],['vL*', 'wL', 'vL'])))
            print("e_LR             :", self.e_LR)

            print("l*rr             :", npc.tensordot(self.l_RL, self.CcRW,axes=(['vR*', 'wR', 'vR'],['vL*', 'wL', 'vL'])))
            print("ll*rr            :", npc.tensordot(self.LWC, self.CcRW,axes=(['vR*', 'wR', 'vR'],['vL*', 'wL', 'vL'])))
            print("ll*r             :", npc.tensordot(self.LWC, self.r_RL,axes=(['vR*', 'wR', 'vR'],['vL*', 'wL', 'vL'])))
            print("e_RL             :", self.e_RL)
        """

    def run(self, p, qtotal_change=None, orthogonal_to=[], E_boosts=[]):
        self.unaligned_H = self.Unaligned_Effective_H(self, self.ALs, self.ARs, self.VLs,
                                                      self.LW, self.RW, self.Ws, p, self.chi, self.d)
        effective_H = SumNpcLinearOperator(self.aligned_H, self.unaligned_H)
        lanczos_params = self.options.subconfig('lanczos_params')
        X_init = self.initial_guess(qtotal_change)
        if len(E_boosts) != len(orthogonal_to):
            E_boost = self.options.get('E_boost', 100)
            E_boosts = [E_boost] * len(orthogonal_to)
        ortho_H = BoostNpcLinearOperator(effective_H, E_boosts, orthogonal_to)

        E, X, N = LanczosGroundState(ortho_H, X_init, lanczos_params).run()

        if N == lanczos_params.get('N_max', 20):
            import warnings
            warnings.warn('Maximum Lanczos iterations needed; be wary of results.')

        psi = MomentumMPS(X, self.psi, p, 1)
        return E - self.energy_density * self.L - self.lambda_C1, psi, N

    def resume_run(self):
        raise NotImplementedError()

    def energy(self, p, X):
        self.unaligned_H = self.Unaligned_Effective_H(self, self.ALs, self.ARs, self.VLs,
                                                      self.LW, self.RW, self.Ws, p, self.chi, self.d)
        effective_H = SumNpcLinearOperator(self.aligned_H, self.unaligned_H)
        HX = effective_H.matvec(X)
        E = np.real(inner(X, HX)).item()
        return E - self.energy_density * self.L - self.lambda_C1

    def infinite_sum_TLR(self, X, p):
        sum_tol = self.options.get('sum_tol', 1.e-10)
        sum_method = self.options.get('sum_method', 'explicit')

        B = npc.tensordot(self.VLs[self.L-1], X[self.L-1], axes=(['vR'], ['vL']))
        RB = TR_general([B], [self.ARs[self.L-1]], self.RW, Ws=[self.Ws[self.L-1]])
        for i in reversed(range(0, self.L-1)):
            B = npc.tensordot(self.VLs[i], X[i], axes=(['vR'], ['vL']))
            RB = TR_general([B], [self.ARs[i]], self.GS_env_R.get_RP(i), Ws=[self.Ws[i]]) + \
                 TR_general([self.ALs[i]], [self.ARs[i]], RB, Ws=[self.Ws[i]])
        R = RB

        assert not np.isclose(npc.norm(R), 0)
        if sum_method=='explicit':
            R_sum = R.copy()
            for _ in range(100):
                R = np.exp(-1.0j * p * self.L) * TR_general(self.ALs, self.ARs, R, Ws=self.Ws)
                R_sum.iadd_prefactor_other(1., R)
                if npc.norm(R) < sum_tol:
                    break
            return R_sum
        elif 'GMRES' in sum_method:
            class helper_matvec(NpcLinearOperator):
                def __init__ (self, excit, ALs, ARs, Ws, sum_method):
                    self.ALs = ALs
                    self.ARs = ARs
                    self.Ws = Ws
                    self.sum_method = sum_method
                    self.excit = excit
                def matvec(self, vec):
                    Tr = TR_general(self.ALs, self.ARs, vec, Ws=self.Ws)
                    if 'reg' in self.sum_method:
                        raise NotImplementedError('GMRES-reg not implemented for multi-site unit cell.')
                        lr = npc.tensordot(self.excit.l_LR, vec, axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        llr = npc.tensordot(self.excit.LWCc, vec, axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        T1_r = self.excit.r_LR * ((self.excit.e_LR-1) * lr + llr) + self.excit.CRW * lr
                        Tr = Tr - T1_r
                    return vec - np.exp(-1.0j * p * self.excit.L) * Tr

            tm_op = helper_matvec(self, self.ALs, self.ARs, self.Ws, sum_method)
            GMRES_params = self.options.subconfig('GMRES_params')
            R_sum, _, _, _ = GMRES(tm_op, npc.Array.zeros_like(R)*1.j, R, GMRES_params).run()
            return R_sum
        else:
            raise ValueError('Sum method', sum_method, 'not recognized!')


    def infinite_sum_TRL(self, X, p):
        sum_tol = self.options.get('sum_tol', 1.e-10)
        sum_method = self.options.get('sum_method', 'explicit')

        B = npc.tensordot(self.VLs[0], X[0], axes=(['vR'], ['vL']))
        LB = LT_general([B], [self.ALs[0]], self.LW, Ws=[self.Ws[0]])
        for i in range(1, self.L):
            B = npc.tensordot(self.VLs[i], X[i], axes=(['vR'], ['vL']))
            LB = LT_general([B], [self.ALs[i]], self.GS_env_L.get_LP(i), Ws=[self.Ws[i]]) + \
                 LT_general([self.ARs[i]], [self.ALs[i]], LB, Ws=[self.Ws[i]])
        L = LB

        assert not np.isclose(npc.norm(L), 0)
        if sum_method=='explicit':
            L_sum = L.copy()
            for i in range(100):
                L = np.exp(1.0j * p * self.L) * LT_general(self.ARs, self.ALs, L, Ws=self.Ws)
                L_sum.iadd_prefactor_other(1., L)
                if npc.norm(L) < sum_tol:
                    break
            return L_sum
        elif 'GMRES' in sum_method:
            class helper_matvec(NpcLinearOperator):
                def __init__ (self, excit, ALs, ARs, Ws, sum_method):
                    self.ALs = ALs
                    self.ARs = ARs
                    self.Ws = Ws
                    self.sum_method = sum_method
                    self.excit = excit

                def matvec(self, vec):
                    lT = LT_general(self.ARs, self.ALs, vec, Ws=self.Ws)
                    if 'reg' in self.sum_method:
                        raise NotImplementedError('GMRES-reg not implemented for multi-site unit cell.')
                        lr = npc.tensordot(vec, self.excit.r_RL, axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        lrr = npc.tensordot(vec, self.excit.CcRW, axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        T1_l = self.excit.l_RL * ((self.excit.e_RL-1) * lr + lrr) + self.excit.LWC * lr
                        lT = lT - T1_l
                    return vec - np.exp(1.0j * p * self.excit.L) * lT

            tm_op = helper_matvec(self, self.ALs, self.ARs, self.Ws, sum_method)
            GMRES_params = self.options.subconfig('GMRES_params')
            L_sum, _, _, _ = GMRES(tm_op, npc.Array.zeros_like(L)*1.j, L, GMRES_params).run()
            return L_sum
        else:
            raise ValueError('Sum method', sum_method, 'not recognized!')

    class Aligned_Effective_H(NpcLinearOperator):
        def __init__(self, outer, ALs, ARs, VLs, LW, RW, Ws, chi, d=2):
            self.ALs = ALs
            self.ARs = ARs
            self.VLs = VLs
            self.LW = LW
            self.RW = RW
            self.Ws = Ws
            self.chi = chi
            self.d = d
            self.outer = outer

        def matvec(self, vec):

            total_vec = [npc.Array.zeros_like(v) for v in vec]

            for i in range(self.outer.L):
                LB = npc.Array.zeros_like(self.LW)
                RB = npc.Array.zeros_like(self.RW)
                for j in range(i):
                    B = npc.tensordot(self.VLs[j], vec[j], axes=(['vR'], ['vL']))
                    if j > 0:
                        LB = LT_general([B], [self.ALs[j]], self.outer.GS_env_L.get_LP(j), Ws=[self.Ws[j]]) + \
                             LT_general([self.ARs[j]], [self.ALs[j]], LB, Ws=[self.Ws[j]]) # Does one extra multiplication when i = 0
                    else:
                        LB = LT_general([B], [self.ALs[j]], self.outer.GS_env_L.get_LP(j), Ws=[self.Ws[j]])

                B = npc.tensordot(self.VLs[i], vec[i], axes=(['vR'], ['vL']))
                LB = LT_general([self.ARs[i]], [self.VLs[i]], LB, Ws=[self.Ws[i]])
                LP1 = LT_general([self.ALs[i]], [self.VLs[i]], self.outer.GS_env_L.get_LP(i), Ws=[self.Ws[i]])
                LP2 = LT_general([B], [self.VLs[i]], self.outer.GS_env_L.get_LP(i), Ws=[self.Ws[i]])

                for j in reversed(range(i+1, self.outer.L)):
                    B = npc.tensordot(self.VLs[j], vec[j], axes=(['vR'], ['vL']))
                    if j < self.outer.L - 1:
                        RB = TR_general([B], [self.ARs[j]], self.outer.GS_env_R.get_RP(j), Ws=[self.Ws[j]]) + \
                             TR_general([self.ALs[j]], [self.ARs[j]], RB, Ws=[self.Ws[j]])
                    else:
                        RB = TR_general([B], [self.ARs[j]], self.outer.GS_env_R.get_RP(j), Ws=[self.Ws[j]])
                if i > 0:
                    total_vec[i] += npc.tensordot(LB, self.outer.GS_env_R.get_RP(i), axes=(['vR', 'wR'], ['vL', 'wL']))
                if i < self.outer.L-1:
                    total_vec[i] += npc.tensordot(LP1, RB, axes=(['vR', 'wR'], ['vL', 'wL']))
                total_vec[i] += npc.tensordot(LP2, self.outer.GS_env_R.get_RP(i), axes=(['vR', 'wR'], ['vL', 'wL']))

            return total_vec

    class Unaligned_Effective_H(NpcLinearOperator):
        def __init__(self, outer, ALs, ARs, VLs, LW, RW, Ws, p, chi, d=2):
            self.ALs = ALs
            self.ARs = ARs
            self.VLs = VLs
            self.LW = LW
            self.RW = RW
            self.Ws = Ws
            self.p = p
            self.outer = outer
            self.chi = chi
            self.d = d

        def matvec(self, vec):

            total = [npc.Array.zeros_like(v) for v in vec]

            inf_sum_TLR = self.outer.infinite_sum_TLR(vec, self.p)
            cached_TLR = [inf_sum_TLR]
            for i in reversed(range(1, self.outer.L)):
                cached_TLR.insert(0, TR_general([self.ALs[i]], [self.ARs[i]], cached_TLR[0], Ws=[self.Ws[i]]))
            for i in range(self.outer.L):
                LP_VL = LT_general([self.ALs[i]], [self.VLs[i]], self.outer.GS_env_L.get_LP(i), Ws=[self.Ws[i]])
                X_out_left = np.exp(-1.0j*self.p*self.outer.L) * npc.tensordot(LP_VL, cached_TLR[i], axes=(['vR', 'wR'], ['vL', 'wL']))
                X_out_left.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                total[i] += X_out_left
            cached_TLR = []

            inf_sum_TRL = self.outer.infinite_sum_TRL(vec, self.p)
            cached_TRL = [inf_sum_TRL]
            for i in range(0, self.outer.L-1):
                cached_TRL.append(LT_general([self.ARs[i]], [self.ALs[i]], cached_TRL[-1], Ws=[self.Ws[i]]))
            for i in reversed(range(self.outer.L)):
                TRL_VL = LT_general([self.ARs[i]], [self.VLs[i]], cached_TRL[i], Ws=[self.Ws[i]])
                X_out_left = np.exp(1.0j*self.p*self.outer.L) * npc.tensordot(TRL_VL, self.outer.GS_env_R.get_RP(i), axes=(['vR', 'wR'], ['vL', 'wL']))
                X_out_left.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                total[i] += X_out_left
            cached_TRL = []

            return total

    def initial_guess(self, qtotal_change):
        X_init = []
        valid_charge = False
        for i in range(self.L):
            vL = self.VLs[i].get_leg('vR').conj()
            vR = self.ALs[(i+1)% self.L].get_leg('vL').conj()
            th0 = npc.Array.from_func(np.ones, [vL, vR],
                                      dtype=self.psi.dtype,
                                      qtotal=qtotal_change,
                                      labels=['vL', 'vR'])

            if np.isclose(npc.norm(th0), 0):
                logger.info("Initial guess for an X is zero; charges not be allowed on site %d.", i)
                #warnings.warn('Initial guess for an X is zero; charges not be allowed on site ' + str(i) +  '.')
            else:
                valid_charge = True
                LP = self.GS_env_L.get_LP(i, store=True)
                RP = self.GS_env_R.get_RP(i, store=True)
                LP = LT_general([self.VLs[i]], [self.VLs[i]], LP, Ws=[self.Ws[i]])

                H0 = ZeroSiteH.from_LP_RP(LP, RP)
                if self.model.H_MPO.explicit_plus_hc:
                    H0 = SumNpcLinearOperator(H0, H0.adjoint())

                lanczos_params = self.options.subconfig('lanczos_params')
                _, th0, _ = LanczosGroundState(H0, th0, lanczos_params).run()

            X_init.append(th0)

        logger.info("Norms of the initial guess: %r.", [npc.norm(x) for x in X_init])
        #print('Norm of initial guess:', [npc.norm(x) for x in X_init])
        assert valid_charge, "No X is non-zero; charge is not valid for gluing."
        return X_init
    
class MultiSitePlaneWaveExcitationEngine(Algorithm):
    def __init__(self, psi, model, options, **kwargs):
        super().__init__(psi, model, options, **kwargs)

        assert self.psi.L == self.model.H_MPO.L
        self.L = self.psi.L

        self.size = self.options.get('excitation_size', 1)
        assert size >= 1
        self.stretch = (size - 1) // L # How many unit cells an excitation tensor crosses into beyond its own.
        
        self.ALs = [self.psi.get_AL(i) for i in range(self.L)]
        self.ARs = [self.psi.get_AR(i) for i in range(self.L)]
        self.ACs = [self.psi.get_AC(i) for i in range(self.L)]
        self.Cs = [self.psi.get_C(i) for i in range(self.L)] # C on the left
        self.H = self.model.H_MPO
        self.Ws = [self.H.get_W(i) for i in range(self.L)]
        if len(self.Ws) < len(self.ALs):
            assert len(self.ALs) % len(self.Ws)
            self.Ws = self.Ws * len(self.ALs) // len(self.Ws)

        self.IdL = self.H.get_IdL(0)
        self.IdR = self.H.get_IdR(-1)

        self.guess_init_env_data = self.options.get('init_data',None)
        self.dW = self.Ws[0].get_leg('wR').ind_len # [TODO] this assumes a single site
        self.chi = self.ALs[0].get_leg('vL').ind_len
        self.d = self.ALs[0].get_leg('p').ind_len

        # Construct VL, needed to parametrize - B - as - VL - X -
        #                                       |        |
        # Use prescription under Eq. 85 in Tangent Space lecture notes.
        self.VLs = [construct_orthogonal(self.ALs[i]) for i in range(self.L)]

        # Get left and right generalized eigenvalues
        self.gauge = self.options.get('gauge', 'trace')
        self.boundary_env_data, self.energy_density, _ = MPOTransferMatrix.find_init_LP_RP(self.H, self.psi, calc_E=True, _subtraction_gauge=self.gauge, guess_init_env_data=self.guess_init_env_data)
        self.energy_density = np.mean(self.energy_density)
        self.LW = self.boundary_env_data['init_LP']
        self.RW = self.boundary_env_data['init_RP']

        # We create GS_env_L and GS_env_R to make topological easier.
        self.GS_env = self.GS_env_L = self.GS_env_R = MPOEnvironment(self.psi, self.H, self.psi, **self.boundary_env_data)
        self.lambda_C1 = options.get('lambda_C1', None)
        if self.lambda_C1 is None:
            C0_L = self.Cs[0]
            norm = npc.tensordot(C0_L, C0_L.conj(), axes=(['vL', 'vR'], ['vL*', 'vR*']))
            self.lambda_C1 = npc.tensordot(C0_L, self.RW, axes=(['vR'], ['vL']))
            self.lambda_C1 = npc.tensordot(self.LW, self.lambda_C1, axes=(['wR', 'vR'], ['wL', 'vL']))
            self.lambda_C1 = npc.tensordot(self.lambda_C1, C0_L.conj(), axes=(['vR*', 'vL*'], ['vL*', 'vR*'])) / norm
        self.aligned_H = self.Aligned_Effective_H(self, self.ALs, self.ARs, self.VLs,
                                                  self.LW, self.RW, self.Ws,
                                                  self.chi, d=self.d)

        strange = []
        for i in range(self.L):
            temp_L = self.GS_env.get_LP(i) # LT_general(self.ALs[:i], self.ALs[:i], self.LW, Ws=self.Ws[:i])
            temp_R = self.GS_env.get_RP(i) # TR_general(self.ARs[i+1:], self.ARs[i+1:], self.RW, Ws=self.Ws[i+1:])
            temp = LT_general([self.VLs[i]], [self.ACs[i]], temp_L, Ws=[self.Ws[i]])
            temp = npc.tensordot(temp, temp_R, axes=(['wR', 'vR*'], ['wL', 'vL*']))
            strange.append(npc.norm(temp))
        logger.info("Norm of H|psi> projected into the tangent space on each site: %r.", strange)

        
    def run(self, p, qtotal_change=None, orthogonal_to=[], E_boosts=[]):
        self.unaligned_H = self.Unaligned_Effective_H(self, self.ALs, self.ARs, self.VLs,
                                                      self.LW, self.RW, self.Ws, p, self.chi, self.d)
        effective_H = SumNpcLinearOperator(self.aligned_H, self.unaligned_H)
        lanczos_params = self.options.subconfig('lanczos_params')
        X_init = self.initial_guess(qtotal_change)
        if len(E_boosts) != len(orthogonal_to):
            E_boost = self.options.get('E_boost', 100)
            E_boosts = [E_boost] * len(orthogonal_to)
        ortho_H = BoostNpcLinearOperator(effective_H, E_boosts, orthogonal_to)

        E, X, N = LanczosGroundState(ortho_H, X_init, lanczos_params).run()

        if N == lanczos_params.get('N_max', 20):
            import warnings
            warnings.warn('Maximum Lanczos iterations needed; be wary of results.')

        psi = MomentumMPS(X, self.psi, p, 1)
        return E - self.energy_density * self.L - self.lambda_C1, psi, N

    def resume_run(self):
        raise NotImplementedError()

    def energy(self, p, X):
        self.unaligned_H = self.Unaligned_Effective_H(self, self.ALs, self.ARs, self.VLs,
                                                      self.LW, self.RW, self.Ws, p, self.chi, self.d)
        effective_H = SumNpcLinearOperator(self.aligned_H, self.unaligned_H)
        HX = effective_H.matvec(X)
        E = np.real(inner(X, HX)).item()
        return E - self.energy_density * self.L - self.lambda_C1
    
    def attach_right(self, VL, X, As, R, Ws=None):
        B = npc.tensordot(VL, X, axes=(['vR'], ['vL']))
        RB = npc.tensordot(B, R, axes=(['vR'], ['vL']))
        for i in reversed(range(len(As))):
            p = 'p' + str(i)
            if Ws is not None:
                RB = npc.tensordot(RB, Ws[i], axes=([p, 'wL'], ['p*', 'wR']))
            RB = npc.tensordot(RB, As[i].conj(), axes=(['p', 'vL*'], ['p*', 'vR*']))
        print(RB)
        return RB
    
    def starting_right_TLR(self, X):
        R = npc.Array.zeros_like(self.RW)
        for i in range(self.L):
            RB = self.GS_env_R.get_RP(i+self.size-1)
            RB = self.attach_right(self.VLs[i], X[i], [self.ARs[j % self.L] for j in range(i, i+size)], 
                                   RP, Ws=[self.Ws[j % self.L] for j in range(i, i+size)])
            RB = TR_general(self.ALs[:i], self.ARs[:i], RB, Ws=[self.Ws[:i]])
            R += RB
        return R
    
    def infinite_sum_TLR(self, X, p):
        sum_tol = self.options.get('sum_tol', 1.e-10)
        sum_method = self.options.get('sum_method', 'explicit')

        R = self.starting_right_TLR(X)
    
        assert not np.isclose(npc.norm(R), 0)
        if sum_method=='explicit':
            R_sum = R.copy()
            for _ in range(100):
                R = np.exp(-1.0j * p * self.L) * TR_general(self.ALs, self.ARs, R, Ws=self.Ws)
                R_sum.iadd_prefactor_other(1., R)
                if npc.norm(R) < sum_tol:
                    break
            return R_sum
        elif 'GMRES' in sum_method:
            class helper_matvec(NpcLinearOperator):
                def __init__ (self, excit, ALs, ARs, Ws, sum_method):
                    self.ALs = ALs
                    self.ARs = ARs
                    self.Ws = Ws
                    self.sum_method = sum_method
                    self.excit = excit
                def matvec(self, vec):
                    Tr = TR_general(self.ALs, self.ARs, vec, Ws=self.Ws)
                    if 'reg' in self.sum_method:
                        raise NotImplementedError('GMRES-reg not implemented for multi-site unit cell.')
                        lr = npc.tensordot(self.excit.l_LR, vec, axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        llr = npc.tensordot(self.excit.LWCc, vec, axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        T1_r = self.excit.r_LR * ((self.excit.e_LR-1) * lr + llr) + self.excit.CRW * lr
                        Tr = Tr - T1_r
                    return vec - np.exp(-1.0j * p * self.excit.L) * Tr

            tm_op = helper_matvec(self, self.ALs, self.ARs, self.Ws, sum_method)
            GMRES_params = self.options.subconfig('GMRES_params')
            R_sum, _, _, _ = GMRES(tm_op, npc.Array.zeros_like(R)*1.j, R, GMRES_params).run()
            return R_sum
        else:
            raise ValueError('Sum method', sum_method, 'not recognized!')
    
    def attach_left(self, VL, X, As, L, Ws=None):
        B = npc.tensordot(VL, X, axes=(['vR'], ['vL'])) # What p labels does B have?
        LB = npc.tensordot(L, B, axes=(['vR'], ['vL']))
        for i in range(len(As)):
            p = 'p' + str(i)
            if Ws is not None:
                LB = npc.tensordot(Ws[i], LB, axes=(['p*', 'wL'], [p, 'wR']))
            LB = npc.tensordot(As[i].conj(), LB, axes=(['p*', 'vL*'], ['p', 'vR*']))
        print(LB)
        return LB
    
    def starting_left_TRL(self, X):
        L = npc.Array.zeros_like(self.LW)
        for i in range(self.L):
            LB = self.GS_env_L.get_LP(i)
            LB = self.attach_left(self.VLs[i], X[i], [self.ALs[j % self.L] for j in range(i, i+size)], 
                                   LP, Ws=[self.Ws[j % self.L] for j in range(i, i+size)])
            
            LB = TR_general(self.ARs[(i+size)%L:], self.ALs[(i+size)%L:], RB, Ws=[self.Ws[(i+size)%L]])
            L += RB
        return L

    def infinite_sum_TRL(self, X, p):
        sum_tol = self.options.get('sum_tol', 1.e-10)
        sum_method = self.options.get('sum_method', 'explicit')

        L = np.exp(1.0j * p * self.L * self.stretch) * self.starting_left_TRL(X)

        assert not np.isclose(npc.norm(L), 0)
        if sum_method=='explicit':
            L_sum = L.copy()
            for i in range(100):
                L = np.exp(1.0j * p * self.L) * LT_general(self.ARs, self.ALs, L, Ws=self.Ws)
                L_sum.iadd_prefactor_other(1., L)
                if npc.norm(L) < sum_tol:
                    break
            return L_sum
        elif 'GMRES' in sum_method:
            class helper_matvec(NpcLinearOperator):
                def __init__ (self, excit, ALs, ARs, Ws, sum_method):
                    self.ALs = ALs
                    self.ARs = ARs
                    self.Ws = Ws
                    self.sum_method = sum_method
                    self.excit = excit

                def matvec(self, vec):
                    lT = LT_general(self.ARs, self.ALs, vec, Ws=self.Ws)
                    if 'reg' in self.sum_method:
                        raise NotImplementedError('GMRES-reg not implemented for multi-site unit cell.')
                        lr = npc.tensordot(vec, self.excit.r_RL, axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        lrr = npc.tensordot(vec, self.excit.CcRW, axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        T1_l = self.excit.l_RL * ((self.excit.e_RL-1) * lr + lrr) + self.excit.LWC * lr
                        lT = lT - T1_l
                    return vec - np.exp(1.0j * p * self.excit.L) * lT

            tm_op = helper_matvec(self, self.ALs, self.ARs, self.Ws, sum_method)
            GMRES_params = self.options.subconfig('GMRES_params')
            L_sum, _, _, _ = GMRES(tm_op, npc.Array.zeros_like(L)*1.j, L, GMRES_params).run()
            return L_sum
        else:
            raise ValueError('Sum method', sum_method, 'not recognized!')

    class Aligned_Effective_H(NpcLinearOperator):
        def __init__(self, outer, ALs, ARs, VLs, LW, RW, Ws, chi, d=2):
            self.ALs = ALs
            self.ARs = ARs
            self.VLs = VLs
            self.LW = LW
            self.RW = RW
            self.Ws = Ws
            self.chi = chi
            self.d = d
            self.outer = outer

        def matvec(self, vec):

            total_vec = [npc.Array.zeros_like(v) for v in vec]
            
            for i in range(self.outer.stretch):
                for j in range(self.outer.L):
                    fe
            
            
            for i in range(self.outer.L):
                LB = npc.Array.zeros_like(self.LW)
                RB = npc.Array.zeros_like(self.RW)
                for j in range(i):
                    B = npc.tensordot(self.VLs[j], vec[j], axes=(['vR'], ['vL']))
                    if j > 0:
                        LB = LT_general([B], [self.ALs[j]], self.outer.GS_env_L.get_LP(j), Ws=[self.Ws[j]]) + \
                             LT_general([self.ARs[j]], [self.ALs[j]], LB, Ws=[self.Ws[j]]) # Does one extra multiplication when i = 0
                    else:
                        LB = LT_general([B], [self.ALs[j]], self.outer.GS_env_L.get_LP(j), Ws=[self.Ws[j]])

                B = npc.tensordot(self.VLs[i], vec[i], axes=(['vR'], ['vL']))
                LB = LT_general([self.ARs[i]], [self.VLs[i]], LB, Ws=[self.Ws[i]])
                LP1 = LT_general([self.ALs[i]], [self.VLs[i]], self.outer.GS_env_L.get_LP(i), Ws=[self.Ws[i]])
                LP2 = LT_general([B], [self.VLs[i]], self.outer.GS_env_L.get_LP(i), Ws=[self.Ws[i]])

                for j in reversed(range(i+1, self.outer.L)):
                    B = npc.tensordot(self.VLs[j], vec[j], axes=(['vR'], ['vL']))
                    if j < self.outer.L - 1:
                        RB = TR_general([B], [self.ARs[j]], self.outer.GS_env_R.get_RP(j), Ws=[self.Ws[j]]) + \
                             TR_general([self.ALs[j]], [self.ARs[j]], RB, Ws=[self.Ws[j]])
                    else:
                        RB = TR_general([B], [self.ARs[j]], self.outer.GS_env_R.get_RP(j), Ws=[self.Ws[j]])
                if i > 0:
                    total_vec[i] += npc.tensordot(LB, self.outer.GS_env_R.get_RP(i), axes=(['vR', 'wR'], ['vL', 'wL']))
                if i < self.outer.L-1:
                    total_vec[i] += npc.tensordot(LP1, RB, axes=(['vR', 'wR'], ['vL', 'wL']))
                total_vec[i] += npc.tensordot(LP2, self.outer.GS_env_R.get_RP(i), axes=(['vR', 'wR'], ['vL', 'wL']))

            return total_vec

    class Unaligned_Effective_H(NpcLinearOperator):
        def __init__(self, outer, ALs, ARs, VLs, LW, RW, Ws, p, chi, d=2):
            self.ALs = ALs
            self.ARs = ARs
            self.VLs = VLs
            self.LW = LW
            self.RW = RW
            self.Ws = Ws
            self.p = p
            self.outer = outer
            self.chi = chi
            self.d = d

        def matvec(self, vec):

            total = [npc.Array.zeros_like(v) for v in vec]

            inf_sum_TLR = self.outer.infinite_sum_TLR(vec, self.p)
            end = (self.outer.size-1 % self.outer.L)
            if start > 0:
                inf_sum_TLR = TR_general(self.ALs[start:], self.ARs[start:], inf_sum_TLR, Ws=self.Ws[start:])
            for i in reversed(range(self.outer.L)):    
                LP_VL = LT_general([self.ALs[i]], [self.VLs[i]], self.outer.GS_env_L.get_LP(i), Ws=[self.Ws[i]])
                for j in range(self.outer.size-1):
                    LP_VL = npc.tensordot(LP_VL, self.ALs[j], axes=(['vR'], ['vL']))
                    LP_VL = npc.tensordot(LP_VL, self.Ws[j], axes=(['wR', 'p'], ['wL', 'p*']))
                    LP_VL.ireplace_label('p', 'p'+str(j))
                X_out_left = np.exp(-1.0j*self.p*self.outer.L*(self.outer.stretch+1)) * npc.tensordot(LP_VL, inf_sum_TLR, axes=(['vR', 'wR'], ['vL', 'wL']))
                X_out_left.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                total[i] += X_out_left
                
                inf_sum_TLR = TR_general([self.ALs[i]], [self.ARs[i]], inf_sum_TLR, Ws=[self.Ws[i]])
                
            for i in range(self.outer.stretch):
                # How many unit cells excitations extend into
                for j in range(self.outer.L):
                    for k in range(self.outer.L):
                        LP = self.outer.GS_env_L.get_LP(j)
                        LP = npc.tensordot(LP, )
                        
                        
            inf_sum_TRL = self.outer.infinite_sum_TRL(vec, self.p)
            for i in range(self.outer.L):
                TRL_VL = LT_general([self.ARs[i]], [self.VLs[i]], inf_sum_TRL, Ws=[self.Ws[i]])
                for j in range(self.outer.size-1):
                    TRL_VL = npc.tensordot(TRL_VL, self.ARs[j], axes=(['vR'], ['vL']))
                    TRL_VL = npc.tensordot(TRL_VL, self.Ws[j], axes=(['wR', 'p'], ['wL', 'p*']))
                    TRL_VL.ireplace_label('p', 'p'+str(j))
                X_out_right = np.exp(1.0j*self.p*self.outer.L) * npc.tensordot(TRL_VL, self.outer.GS_env_R.get_RP(i+self.outer.size-1), axes=(['vR', 'wR'], ['vL', 'wL']))
                X_out_right.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                total[i] += X_out_right
                
                inf_sum_TRL = TR_general([self.ARs[i]], [self.ALs[i]], inf_sum_TRL, Ws=[self.Ws[i]])

            return total

    def initial_guess(self, qtotal_change):
        X_init = []
        valid_charge = False
        for i in range(self.L):
            vL = self.VLs[i].get_leg('vR').conj()
            vR = self.ALs[(i+1)% self.L].get_leg('vL').conj()
            th0 = npc.Array.from_func(np.ones, [vL, vR],
                                      dtype=self.psi.dtype,
                                      qtotal=qtotal_change,
                                      labels=['vL', 'vR'])

            if np.isclose(npc.norm(th0), 0):
                logger.info("Initial guess for an X is zero; charges not be allowed on site %d.", i)
                #warnings.warn('Initial guess for an X is zero; charges not be allowed on site ' + str(i) +  '.')
            else:
                valid_charge = True
                LP = self.GS_env_L.get_LP(i, store=True)
                RP = self.GS_env_R.get_RP(i, store=True)
                LP = LT_general([self.VLs[i]], [self.VLs[i]], LP, Ws=[self.Ws[i]])

                H0 = ZeroSiteH.from_LP_RP(LP, RP)
                if self.model.H_MPO.explicit_plus_hc:
                    H0 = SumNpcLinearOperator(H0, H0.adjoint())

                lanczos_params = self.options.subconfig('lanczos_params')
                _, th0, _ = LanczosGroundState(H0, th0, lanczos_params).run()

            X_init.append(th0)

        logger.info("Norms of the initial guess: %r.", [npc.norm(x) for x in X_init])
        #print('Norm of initial guess:', [npc.norm(x) for x in X_init])
        assert valid_charge, "No X is non-zero; charge is not valid for gluing."
        return X_init


class TopologicalPlaneWaveExcitationEngine(PlaneWaveExcitationEngine):
    def __init__(self, psi_L, psi_R, model, options, **kwargs):
        Algorithm.__init__(self, psi_L, model, options, **kwargs)
        self.psi_L, self.psi_R = psi_L, psi_R
        assert self.psi_L.L == self.psi_R.L == self.model.H_MPO.L
        self.L = self.psi_L.L

        self.ALs = [self.psi_L.get_AL(i) for i in range(self.L)]
        self.ARs = [self.psi_R.get_AR(i) for i in range(self.L)]
        self.ACs = [self.psi_L.get_AC(i) for i in range(self.L)]
        self.Cs = [self.psi_L.get_C(i) for i in range(self.L)] # C on the left
        self.H = self.model.H_MPO
        self.Ws = [self.H.get_W(i) for i in range(self.L)]
        if len(self.Ws) < len(self.ALs):
            assert len(self.ALs) % len(self.Ws)
            self.Ws = self.Ws * len(self.ALs) // len(self.Ws)

        self.IdL = self.H.get_IdL(0)
        self.IdR = self.H.get_IdR(-1)

        self.guess_init_env_data_L = self.options.get('init_data_L',None)
        self.guess_init_env_data_R = self.options.get('init_data_R',None)
        self.dW = self.Ws[0].get_leg('wR').ind_len # [TODO] this assumes a single site
        self.chi = self.ALs[0].get_leg('vL').ind_len
        self.d = self.ALs[0].get_leg('p').ind_len

        # Construct VL, needed to parametrize - B - as - VL - X -
        #                                       |        |
        # Use prescription under Eq. 85 in Tangent Space lecture notes.
        self.VLs = [construct_orthogonal(self.ALs[i]) for i in range(self.L)]

        # Get left and right generalized eigenvalues
        self.gauge = self.options.get('gauge', 'trace')
        self.boundary_env_data_L, self.energy_density_L, _ = MPOTransferMatrix.find_init_LP_RP(self.H, self.psi_L, calc_E=True, subtraction_gauge=self.gauge, guess_init_env_data=self.guess_init_env_data_L)
        self.boundary_env_data_R, self.energy_density_R, _ = MPOTransferMatrix.find_init_LP_RP(self.H, self.psi_R, calc_E=True, subtraction_gauge=self.gauge, guess_init_env_data=self.guess_init_env_data_R)
        self.energy_density_L = np.mean(self.energy_density_L)
        self.energy_density_R = np.mean(self.energy_density_R)
        assert np.abs(self.energy_density_L - self.energy_density_R) < 1.e-10
        self.energy_density = np.mean([self.energy_density_L, self.energy_density_R])

        self.LW = self.boundary_env_data_L['init_LP']
        self.RW = self.boundary_env_data_R['init_RP']
        self.GS_env_L = MPOEnvironment(self.psi_L, self.H, self.psi_L, **self.boundary_env_data_L)
        self.GS_env_R = MPOEnvironment(self.psi_R, self.H, self.psi_R, **self.boundary_env_data_R)

        self.lambda_C1_L = options.get('lambda_C1_L', None)
        if self.lambda_C1_L is None:
            C0_L = self.Cs[0]
            norm = npc.tensordot(C0_L, C0_L.conj(), axes=(['vL', 'vR'], ['vL*', 'vR*']))
            self.lambda_C1_L = npc.tensordot(C0_L, self.boundary_env_data_L['init_RP'], axes=(['vR'], ['vL']))
            self.lambda_C1_L = npc.tensordot(self.LW, self.lambda_C1_L, axes=(['wR', 'vR'], ['wL', 'vL']))
            self.lambda_C1_L = npc.tensordot(self.lambda_C1_L, C0_L.conj(), axes=(['vR*', 'vL*'], ['vL*', 'vR*'])) / norm
        print('L:', self.lambda_C1_L)
        self.lambda_C1 = self.lambda_C1_L
        """
        self.lambda_C1_R = options.get('lambda_C1_R', None)
        if self.lambda_C1_R is None:
            C0_R = self.psi_R.get_C(0)
            norm = npc.tensordot(C0_R, C0_R.conj(), axes=(['vL', 'vR'], ['vL*', 'vR*']))
            self.lambda_C1_R = npc.tensordot(C0_R, self.RW, axes=(['vR'], ['vL']))
            self.lambda_C1_R = npc.tensordot(self.boundary_env_data_R['init_LP'], self.lambda_C1_R, axes=(['wR', 'vR'], ['wL', 'vL']))
            self.lambda_C1_R = npc.tensordot(self.lambda_C1_R, C0_R.conj(), axes=(['vR*', 'vL*'], ['vL*', 'vR*'])) / norm
        print('R:', self.lambda_C1_R)
        print(np.abs(self.lambda_C1_L - self.lambda_C1_R))
        assert np.abs(self.lambda_C1_L - self.lambda_C1_R) < 1.e-10
        self.lambda_C1 = np.mean([self.lambda_C1_L, self.lambda_C1_R])
        """
        self.aligned_H = self.Aligned_Effective_H(self, self.ALs, self.ARs, self.VLs,
                                                  self.LW, self.RW, self.Ws,
                                                  self.chi, d=self.d)

        strange = []
        for i in range(self.L):
            temp_L = self.GS_env_L.get_LP(i) # LT_general(self.ALs[:i], self.ALs[:i], self.LW, Ws=self.Ws[:i])
            temp_R = self.GS_env_L.get_RP(i) # TR_general(self.ARs[i+1:], self.ARs[i+1:], self.RW, Ws=self.Ws[i+1:])
            AC = self.ACs[i]
            temp = LT_general([self.VLs[i]], [AC], temp_L, Ws=[self.Ws[i]])
            temp = npc.tensordot(temp, temp_R, axes=(['wR', 'vR*'], ['wL', 'vL*']))
            strange.append(npc.norm(temp))
        logger.info("Norm of H|psi_L> projected into the tangent space on each site: %r.", strange)

        strange = []
        for i in range(self.L):
            temp_L = self.GS_env_R.get_LP(i) # LT_general(self.ALs[:i], self.ALs[:i], self.LW, Ws=self.Ws[:i])
            temp_R = self.GS_env_R.get_RP(i) # TR_general(self.ARs[i+1:], self.ARs[i+1:], self.RW, Ws=self.Ws[i+1:])
            AC = self.psi_R.get_AC(i)
            AL = self.psi_R.get_AL(i)
            VL_R = construct_orthogonal(AL)
            temp = LT_general([VL_R], [AC], temp_L, Ws=[self.Ws[i]])
            temp = npc.tensordot(temp, temp_R, axes=(['wR', 'vR*'], ['wL', 'vL*']))
            strange.append(npc.norm(temp))
        logger.info("Norm of H|psi_R> projected into the tangent space on each site: %r.", strange)


    def gauge_ground_states():
        # Shift charges on bond so that left and right match up.
        H0 = ZeroSiteH.from_LP_RP(self.LW, self.RW)
        if self.model.H_MPO.explicit_plus_hc:
            H0 = SumNpcLinearOperator(H0, H0.adjoint())
        vL, vR = LP.get_leg('vR').conj(), RP.get_leg('vL').conj()

        # TODO make VUMPS default to diagonal gauge or this won't work.
        if self.psi_L.chinfo.qnumber == 0:    # Handles the case of no charge-conservation
            desired_Q = None
        else:
            if join_method == "average charge":
                Q_bar_L = sself.psi_L.average_charge(0)
                for i in range(1, self.L):
                    Q_bar_L += self.psi_L.average_charge(i)
                Q_bar_L = vL.chinfo.make_valid(np.around(Q_bar_L / self.L))
                self.logger.info("Charge of left BC, averaged over site and unit cell: %r", Q_bar_L)

                Q_bar_R = self.psi_R.average_charge(0)
                for i in range(1, self.L):
                    Q_bar_R += self.psi_R.average_charge(i)
                Q_bar_R = vR.chinfo.make_valid(-1 * np.around(Q_bar_R / self.L))
                self.logger.info("Charge of right BC, averaged over site and unit cell: %r", -1*Q_bar_R)
                desired_Q = list(vL.chinfo.make_valid(Q_bar_L + Q_bar_R))
            elif join_method == "most probable charge":
                posL = self.L
                posR = 0
                QsL, psL = left_segment.probability_per_charge(posL)
                QsR, psR = right_segment.probability_per_charge(posR)

                self.logger.info(side_by_side)
                side_by_side = vert_join(["left seg\n" + str(QsL), "prob\n" + str(np.array([psL]).T), "right seg\n" + str(QsR),"prob\n" +str(np.array([psR]).T)], delim=' | ')
                self.logger.info(side_by_side)

                Qmostprobable_L = QsL[np.argmax(psL)]
                Qmostprobable_R = -1 * QsR[np.argmax(psR)]
                self.logger.info("Most probable left:" + str(Qmostprobable_L))
                self.logger.info("Most probable right:" + str(Qmostprobable_R))
                desired_Q = list(vL.chinfo.make_valid(Qmostprobable_L + Qmostprobable_R))
            else:
                raise ValueError("Invalid `join_method` %s " % join_method)

        self.logger.info("Desired gluing charge: %r", desired_Q)

        # We need a tensor that is non-zero only when Q = (Q^i_L - bar(Q_L)) + (Q^i_R - bar(Q_R))
        # Q is the the charge we insert. Here we only do charge gluing to get a valid segment.
        # Changing charge sector is done below by basically identical code when the segment is already formed.

        th0 = npc.Array.from_func(np.ones, [vL, vR],
                                  dtype=left_segment.dtype,
                                  qtotal=desired_Q,
                                  labels=['vL', 'vR'])
        lanczos_params = self.options.get("lanczos_params", {}) # See if lanczos_params is in yaml, if not use empty dictionary
        _, th0, _ = lanczos.LanczosGroundState(H0, th0, lanczos_params).run()
        U, s, Vh = npc.svd(th0, inner_labels=['vR', 'vL'])
        left_segment.set_B(llast, npc.tensordot(left_segment.get_B(llast, 'A'), U, axes=['vR', 'vL']), form='A') # Put AU into last site of left segment
        lA[llast] = left_segment.get_B(llast, 'A')
        right_segment.set_B(0, npc.tensordot(Vh, right_segment.get_B(0, 'B'), axes=['vR', 'vL']), form='B') # Put Vh B into first site of right segment
        right_segment.set_SL(0, s)

        rB[0] = right_segment.get_B(0)
        rS[0] = right_segment.get_SL(0)
        lS = lS[0:-1] # Remove last singular values from list of singular values in A part of segment.
