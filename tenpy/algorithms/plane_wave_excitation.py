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

__all__ = ['TR_general', 'LT_general', 'construct_orthogonal', 'PlaneWaveExcitations', ]

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
        self.boundary_env_data, self.energy_density, _ = MPOTransferMatrix.find_init_LP_RP(self.H, self.psi, calc_E=True, subtraction_gauge=self.gauge, guess_init_env_data=self.guess_init_env_data)
        self.energy_density = np.mean(self.energy_density)
        self.LW = self.boundary_env_data['init_LP']
        self.RW = self.boundary_env_data['init_RP']
        
        self.GS_env = MPOEnvironment(self.psi, self.H, self.psi, **self.boundary_env_data)
        self.lambda_C1 = options.get('lambda_C1', None)
        if self.lambda_C1 is None:
            self.lambda_C1 = npc.tensordot(self.Cs[0], self.RW, axes=(['vR'], ['vL']))
            self.lambda_C1 = npc.tensordot(self.LW, self.lambda_C1, axes=(['wR', 'vR'], ['wL', 'vL']))
            self.lambda_C1 = self.lambda_C1[0,0] / self.Cs[0][0,0]
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
            RB = TR_general([B], [self.ARs[i]], self.GS_env.get_RP(i), Ws=[self.Ws[i]]) + \
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
            LB = LT_general([B], [self.ALs[i]], self.GS_env.get_LP(i), Ws=[self.Ws[i]]) + \
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
                        LB = LT_general([B], [self.ALs[j]], self.outer.GS_env.get_LP(j), Ws=[self.Ws[j]]) + \
                             LT_general([self.ARs[j]], [self.ALs[j]], LB, Ws=[self.Ws[j]]) # Does one extra multiplication when i = 0
                    else:
                        LB = LT_general([B], [self.ALs[j]], self.outer.GS_env.get_LP(j), Ws=[self.Ws[j]])
                
                B = npc.tensordot(self.VLs[i], vec[i], axes=(['vR'], ['vL']))
                LB = LT_general([self.ARs[i]], [self.VLs[i]], LB, Ws=[self.Ws[i]])
                LP1 = LT_general([self.ALs[i]], [self.VLs[i]], self.outer.GS_env.get_LP(i), Ws=[self.Ws[i]])
                LP2 = LT_general([B], [self.VLs[i]], self.outer.GS_env.get_LP(i), Ws=[self.Ws[i]])
                
                for j in reversed(range(i+1, self.outer.L)):
                    B = npc.tensordot(self.VLs[j], vec[j], axes=(['vR'], ['vL']))
                    if j < self.outer.L - 1:
                        RB = TR_general([B], [self.ARs[j]], self.outer.GS_env.get_RP(j), Ws=[self.Ws[j]]) + \
                             TR_general([self.ALs[j]], [self.ARs[j]], RB, Ws=[self.Ws[j]])
                    else:
                        RB = TR_general([B], [self.ARs[j]], self.outer.GS_env.get_RP(j), Ws=[self.Ws[j]])
                if i > 0:
                    total_vec[i] += npc.tensordot(LB, self.outer.GS_env.get_RP(i), axes=(['vR', 'wR'], ['vL', 'wL']))
                if i < self.outer.L-1:
                    total_vec[i] += npc.tensordot(LP1, RB, axes=(['vR', 'wR'], ['vL', 'wL']))
                total_vec[i] += npc.tensordot(LP2, self.outer.GS_env.get_RP(i), axes=(['vR', 'wR'], ['vL', 'wL']))
            
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
                LP_VL = LT_general([self.ALs[i]], [self.VLs[i]], self.outer.GS_env.get_LP(i), Ws=[self.Ws[i]])
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
                X_out_left = np.exp(1.0j*self.p*self.outer.L) * npc.tensordot(TRL_VL, self.outer.GS_env.get_RP(i), axes=(['vR', 'wR'], ['vL', 'wL']))
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
                warnings.warn('Initial guess for an X is zero; charges not be allowed on site ' + str(i) +  '.')
            else:
                valid_charge = True
                LP = self.GS_env.get_LP(i, store=True)
                RP = self.GS_env.get_RP(i, store=True)
                LP = LT_general([self.VLs[i]], [self.VLs[i]], LP, Ws=[self.Ws[i]])

                H0 = ZeroSiteH.from_LP_RP(LP, RP)
                if self.model.H_MPO.explicit_plus_hc:
                    H0 = SumNpcLinearOperator(H0, H0.adjoint())

                lanczos_params = self.options.subconfig('lanczos_params')
                _, th0, _ = LanczosGroundState(H0, th0, lanczos_params).run()

            X_init.append(th0)
        print('Norm of initial guess:', [npc.norm(x) for x in X_init])
        assert valid_charge, "No X is non-zero; charge is not valid for gluing."
        return X_init