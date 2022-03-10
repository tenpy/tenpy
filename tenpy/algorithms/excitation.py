import numpy as np
import time
import warnings
import logging
logger = logging.getLogger(__name__)

from ..linalg import np_conserved as npc
from ..networks.mpo import MPOEnvironment, MPOTransferMatrix
from ..linalg.lanczos import GMRES, LanczosGroundState
from ..linalg.sparse import NpcLinearOperator, SumNpcLinearOperator, BoostNpcLinearOperator, ShiftNpcLinearOperator
from ..tools.params import asConfig
from ..tools.math import entropy
from ..algorithms.algorithm import Algorithm
from ..algorithms.mps_common import ZeroSiteH
#from ..tools.misc import find_subclass
#from ..tools.process import memory_usage
#from .mps_common import Sweep, ZeroSiteH, OneSiteH


"""
TODO - 02/25/2022
(1) Regulated transfer matrix for unit cell > 1
(2) construct_orthogonal in npc way
(3) restarted Lanczos
(4) Proper way to get multiple excitations out
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
        #temp.itranspose(['vR*', 'wR', 'vR'])
    return temp

# TODO - Make npc version of this as converting R to a dense array is very memory intensive.
# Do complete QR on blocks of M and construct VL from this a la npc.qr.
def construct_orthogonal(M, cutoff=1.e-13, left=True):
    if left:
        M = M.copy().combine_legs([['vL', 'p'], ['vR']], qconj=[+1, -1])
        Q, R = npc.qr(M, mode='complete', inner_labels=['vR', 'vL'], qtotal_Q=M.qtotal)
        proj = np.linalg.norm(R.to_ndarray(), axis=-1)
        #proj = np.concatenate([np.linalg.norm(r, axis=-1) for r in R._data])
        proj = proj < cutoff
        Q.iproject(proj, 'vR')
        assert npc.norm(npc.tensordot(Q, M.conj(), axes=(['(vL.p)'], ['(vL*.p*)']))) < 1.e-14
    else:
        M = M.copy().combine_legs([['vL'], ['p', 'vR']], qconj=[+1, -1])
        Q, R = npc.qr(M.transpose(['(p.vR)', '(vL)']), mode='complete', inner_labels=['vL', 'vR'], qtotal_Q=M.qtotal)
        Q.itranspose(['vL', '(p.vR)'])
        R.itranspose(['(vL)', 'vR'])
        proj = np.linalg.norm(R.to_ndarray(), axis=0)
        #proj = np.concatenate([np.linalg.norm(r, axis=0) for r in R._data])
        proj = proj < cutoff
        Q.iproject(proj, 'vL')
        assert npc.norm(npc.tensordot(Q, M.conj(), axes=(['(p.vR)'], ['(p*.vR*)']))) < 1.e-14
    return Q.split_legs()

def make_Bs(VLs, Xs):
    Bs = [npc.tensordot(VL, X, axes=(['vR'], ['vL'])).combine_legs(['vL', 'p'], qconj=+1) for VL, X in zip(VLs, Xs)]
    return Bs

def get_Xs(VLs, Bs):
    Xs = [npc.tensordot(B.split_legs(), VL.conj(), axes=(['vL', 'p'], ['vL*', 'p*'])).ireplace_label('vR*', 'vL').itranspose(['vL', 'vR']) for VL, B in zip(VLs, Bs)]
    return Xs

"""
def construct_orthogonal(orig_AL):
        chi = orig_AL.get_leg('vL').ind_len
        AL = orig_AL.combine_legs(['vL', 'p'], qconj=[+1])
        Q, R = npc.qr(AL, mode='complete', inner_labels=['vR', 'vL'])
        n_rows = R.shape[1]
        V_grouped = Q[:,chi:] # [TODO] Karthik did this differently, but I think this is right?
        X = R[chi:,:]
        VL = V_grouped.split_legs()
        
        print(npc.norm(npc.tensordot(Q[:,:chi].split_legs(), orig_AL.conj(), axes=(['vL', 'p'], ['vL*', 'p*']))))
        
        print(npc.norm(npc.tensordot(VL, orig_AL.conj(), axes=(['vL', 'p'], ['vL*', 'p*']))))
        assert npc.norm(npc.tensordot(VL, orig_AL.conj(), axes=(['vL', 'p'], ['vL*', 'p*']))) < 1.e-14
        return VL, X
"""

class PlaneWaveExcitations(Algorithm):
    def __init__(self, psi, model, options, **kwargs):
        #options = asConfig(options, self.__class__.__name__)
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
        #self.guess_init_env_data = options.get('init_data', {'init_RP': npc.tensordot(self.C, self.C.conj(), axes=(['vR', 'vL*'])) ,
        #                                                    'init_LP': npc.tensordot(self.C.conj(), self.C, axes=(['vR*', 'vR']))})
        self.guess_init_env_data = self.options.get('init_data',None)
        self.dW = self.Ws[0].get_leg('wR').ind_len # [TODO] this assumes a single site
        self.chi = self.ALs[0].get_leg('vL').ind_len
        self.d = self.ALs[0].get_leg('p').ind_len
        
        # Construct VL, needed to parametrize - B - as - VL - X -
        #                                       |        |
        # Use prescription under Eq. 85 in Tangent Space lecture notes.
        self.VLs = [construct_orthogonal(self.ALs[i]) for i in range(self.L)]
        
        self.do_B = self.options.get('do_B', False)
        
        # Get left and right generalized eigenvalues
        self.gauge = self.options.get('gauge', 'trace')
        self.boundary_env_data, self.energy_density, _ = MPOTransferMatrix.find_init_LP_RP(self.H, self.psi, calc_E=True, subtraction_gauge=self.gauge, guess_init_env_data=self.guess_init_env_data)
        self.energy_density = np.mean(self.energy_density)
        self.LW = self.boundary_env_data['init_LP']
        self.RW = self.boundary_env_data['init_RP']
        self.lambda_C1 = options.get('lambda_C1', None)
        if self.lambda_C1 is None:
            self.lambda_C1 = npc.tensordot(self.Cs[0], self.RW, axes=(['vR'], ['vL']))
            self.lambda_C1 = npc.tensordot(self.LW, self.lambda_C1, axes=(['wR', 'vR'], ['wL', 'vL']))
            self.lambda_C1 = self.lambda_C1[0,0] / self.Cs[0][0,0]
        #In 'trace' gauge, $Tr(RW[ID_L] = 0$.
        #print(npc.trace(self.RW[:,0,:], 'vL', 'vL*'))

        #temp = npc.tensordot(self.LW, self.C, axes=(['vR'], ['vL']))
        #temp = npc.tensordot(self.C.conj(), temp, axes=(['vL*'], ['vR*']))
        #temp = npc.trace(temp[:,0,:], 'vR', 'vR*')
        #print(temp) -> 1
        # This checks $tTr(C C^\conj) = 1.
        
        # Tw[Al,AR]
        self.l_LR = npc.Array.zeros_like(self.LW).itranspose(['vR*','wR', 'vR']) # [TODO] check default ordering to potentially remove transpose
        # Original ordering of boundary vectors ['vR*', 'wR', 'vR']
        self.l_LR[:,self.IdR,:] = self.Cs[0].conj().transpose(['vR*', 'vL*'])
        
        #wL = self.H.get(0).get_leg('wL')
        #wR = wL.conj()
        #self.l_LR = self.C.conj().add_leg(wR, self.IdR, axis=1, label='wR'
        
        self.CRW = npc.tensordot(self.Cs[0], self.RW, axes=(['vR'], ['vL'])) # ['vR','wR', 'vR*']
        self.LWCc= npc.tensordot(self.LW, self.Cs[0].conj(), axes=(['vR*'], ['vL*'])) # ['vL*','wL', 'vL']
        self.r_LR = npc.Array.zeros_like(self.RW) # ['vR','wR', 'vR*']
        self.r_LR[:,self.IdL,:] = self.Cs[0]
        
        # tTr (l_LR * CRW) = tTr (C C^\dag) = 1 usually
        self.CRW = self.CRW / npc.tensordot(self.l_LR, self.CRW, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*']))
        
        # LWCc -> LWCc / self.e_LR
        self.LWCc = self.LWCc / npc.tensordot(self.LWCc, self.CRW, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*']))
        #print('l_LR * CRW', npc.tensordot(self.l_LR, self.CRW, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*'])))
        #print('LWCc * CRW', npc.tensordot(self.LWCc, self.CRW, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*'])))
        #print('LWCc * r_LR', npc.tensordot(self.LWCc, self.r_LR, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*'])))
        self.r_LR = self.r_LR / npc.tensordot(self.LWCc, self.r_LR, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*'])) # Dividing by E_C, GS eigenvalue of HC
        #print('l_LR * CRW', npc.tensordot(self.l_LR, self.CRW, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*'])))
        #print('LWCc * CRW', npc.tensordot(self.LWCc, self.CRW, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*'])))
        #print('LWCc * r_LR', npc.tensordot(self.LWCc, self.r_LR, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*'])))
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
        
        self.aligned_H = self.Aligned_Effective_H(self, self.ALs, self.ARs, self.VLs,
                                                  self.LW, self.RW, self.Ws, 
                                                  self.chi, d=self.d)
        
        strange = []
        for i in range(self.L):
            temp_L = LT_general(self.ALs[:i], self.ALs[:i], self.LW, Ws=self.Ws[:i])
            temp_R = TR_general(self.ARs[i+1:], self.ARs[i+1:], self.RW, Ws=self.Ws[i+1:])
            temp = LT_general([self.VLs[i]], [self.ACs[i]], temp_L, Ws=[self.Ws[i]])
            temp = npc.tensordot(temp, temp_R, axes=(['wR', 'vR*'], ['wL', 'vL*']))
            strange.append(npc.norm(temp))
        print('Strange Cancellation Term:', strange)
        
        if 1 >= 1:
            print("-"*20, "initializing excitation", "-"*20)
            # Don't know how to subtract these with charges
            #print("norm(LW-LW.dag):", npc.norm(self.LW - self.LW.transpose(['vR','wR','vR*']).conj()))
            #print("norm(RW-RW.dag):", npc.norm(self.RW - self.RW.transpose(['vL*','wL','vL']).complex_conj()))
            assert self.psi.valid_umps
            print("Energy density:", self.energy_density)
            print("Lambda C1:", self.lambda_C1)
            print("L norm:", npc.norm(self.LW))
            print("R norm:", npc.norm(self.RW))
            print("LR:", npc.tensordot(self.LW, self.RW, axes=(['vR*', 'wR', 'vR'],['vL*', 'wL', 'vL'])))

            lT  = LT_general(self.ALs, self.ARs, self.l_LR, Ws=self.Ws)  
            Tr  = TR_general(self.ALs, self.ARs, self.r_LR, Ws=self.Ws)
            print("LR norm(l-lT):", npc.norm(self.l_LR-lT))
            print("LR norm(r-Tr):", npc.norm(self.r_LR-Tr))
            
            lT  = LT_general(self.ARs, self.ALs, self.l_RL, Ws=self.Ws)  
            Tr  = TR_general(self.ARs, self.ALs, self.r_RL, Ws=self.Ws)
            print("RL norm(l-lT):", npc.norm(self.l_RL-lT))
            print("RL norm(r-Tr):", npc.norm(self.r_RL-Tr))

            #Tr  = TR_general(self.AL, self.AR, self.CRW, W=self.H.get_W(0)) 
            #print("Trr-rr:\n", Tr-self.CRW)
            #print("(Trr-rr)/rr[-1]:\n", (Tr-self.CRW)[-1,:,:]/self.r_RL[-1])

            #lT  = LT_general(self.AL, self.AR, self.LWCc, W=self.H.get_W(0))            
            #print("llT-ll:\n", lT-self.LWCc)
            #print("(llT-ll)/ll[0]:\n", (lT-self.LWCc)[0,:,:]/self.l[0])
            print("l*rr:", npc.tensordot(self.l_LR, self.CRW,axes=(['vR*', 'wR', 'vR'],['vL*', 'wL', 'vL'])))
            print("ll*rr:", npc.tensordot(self.LWCc, self.CRW,axes=(['vR*', 'wR', 'vR'],['vL*', 'wL', 'vL'])))
            print("ll*r:", npc.tensordot(self.LWCc, self.r_LR,axes=(['vR*', 'wR', 'vR'],['vL*', 'wL', 'vL'])))
            print("e_LR:", self.e_LR)
            
            #TR  = LT_general(self.AL, self.AR, self.LWCc, W=self.H.get_W(0))            
            #print("llT-ll:\n", lT-self.LWCc)
            #print("(llT-ll)/ll[0]:\n", (lT-self.LWCc)[0,:,:]/self.l[0])
            print("l*rr:", npc.tensordot(self.l_RL, self.CcRW,axes=(['vR*', 'wR', 'vR'],['vL*', 'wL', 'vL'])))
            print("ll*rr:", npc.tensordot(self.LWC, self.CcRW,axes=(['vR*', 'wR', 'vR'],['vL*', 'wL', 'vL'])))
            print("ll*r:", npc.tensordot(self.LWC, self.r_RL,axes=(['vR*', 'wR', 'vR'],['vL*', 'wL', 'vL'])))
            print("e_RL:", self.e_RL)    
        
    def infinite_sum_TLR(self, X, p):
        tol = self.options.get('tol', 1.e-10)
        sum_method = self.options.get('sum_method', 'explicit')
        
        RP = TR_general([self.ARs[self.L-1]], [self.ARs[self.L-1]], self.RW, Ws=[self.Ws[self.L-1]])
        #B = npc.tensordot(self.VLs[self.L-1], X[self.L-1], axes=(['vR'], ['vL']))
        B = X[self.L-1].split_legs() if self.do_B else npc.tensordot(self.VLs[self.L-1], X[self.L-1], axes=(['vR'], ['vL']))
        RB = TR_general([B], [self.ARs[self.L-1]], self.RW, Ws=[self.Ws[self.L-1]])
        for i in reversed(range(1, self.L-1)):
            #B = npc.tensordot(self.VLs[i], X[i], axes=(['vR'], ['vL']))
            B = X[i].split_legs() if self.do_B else npc.tensordot(self.VLs[i], X[i], axes=(['vR'], ['vL']))
            RB = TR_general([B], [self.ARs[i]], RP, Ws=[self.Ws[i]]) + \
                 TR_general([self.ALs[i]], [self.ARs[i]], RB, Ws=[self.Ws[i]])
            
            RP = TR_general([self.ARs[i]], [self.ARs[i]], RP, Ws=[self.Ws[i]])
        #B = npc.tensordot(self.VLs[0], X[0], axes=(['vR'], ['vL']))
        B = X[0].split_legs() if self.do_B else npc.tensordot(self.VLs[0], X[0], axes=(['vR'], ['vL']))
        RB = TR_general([B], [self.ARs[0]], RP, Ws=[self.Ws[0]]) + \
             TR_general([self.ALs[0]], [self.ARs[0]], RB, Ws=[self.Ws[0]])
        R = RB
        """     
        B0 = npc.tensordot(self.VLs[0], X[0], axes=(['vR'], ['vL']))
        R = TR_general(self.ALs[:0] + [B0] + self.ARs[0+1:], 
                        self.ARs, 
                        self.RW, Ws = self.Ws)
        for i in range(1, self.L):
            Bi = npc.tensordot(self.VLs[i], X[i], axes=(['vR'], ['vL']))
            R.iadd_prefactor_other(1., TR_general(self.ALs[:i] + [Bi] + self.ARs[i+1:], 
                                                  self.ARs, 
                                                  self.RW, Ws = self.Ws))
        """
        
        assert not np.isclose(npc.norm(R), 0)
        if sum_method=='explicit':
            R_sum = R.copy()
            for _ in range(100):
                R = np.exp(-1.0j * p * self.L) * TR_general(self.ALs, self.ARs, R, Ws=self.Ws)
                R_sum.iadd_prefactor_other(1., R)
                if npc.norm(R) < tol:
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
                        assert False
                        lr = npc.tensordot(self.excit.l_LR, vec, axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        llr = npc.tensordot(self.excit.LWCc, vec, axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        T1_r = self.excit.r_LR * ((self.excit.e_LR-1) * lr + llr) + self.excit.CRW * lr
                        Tr = Tr - T1_r
                    return vec - np.exp(-1.0j * p * self.excit.L) * Tr
                
            tm_op = helper_matvec(self, self.ALs, self.ARs, self.Ws, sum_method)
            GMRES_options = self.options.subconfig('GMRES_options')
            R_sum, _, _, _ = GMRES(tm_op, npc.Array.zeros_like(R)*1.j, R, GMRES_options).run()
            return R_sum
        else:
            raise ValueError('Sum method', sum_method, 'not recognized!')
            
        
    def infinite_sum_TRL(self, X, p):
        tol = self.options.get('tol', 1.e-10)
        sum_method = self.options.get('sum_method', 'explicit')
        
        LP = LT_general([self.ALs[0]], [self.ALs[0]], self.LW, Ws=[self.Ws[0]])
        #B = npc.tensordot(self.VLs[0], X[0], axes=(['vR'], ['vL']))
        B = X[0].split_legs() if self.do_B else npc.tensordot(self.VLs[0], X[0], axes=(['vR'], ['vL']))
        LB = LT_general([B], [self.ALs[0]], self.LW, Ws=[self.Ws[0]])
        for i in range(1, self.L-1):
            #B = npc.tensordot(self.VLs[i], X[i], axes=(['vR'], ['vL']))
            B = X[i].split_legs() if self.do_B else npc.tensordot(self.VLs[i], X[i], axes=(['vR'], ['vL']))
            LB = LT_general([B], [self.ALs[i]], LP, Ws=[self.Ws[i]]) + \
                 LT_general([self.ARs[i]], [self.ALs[i]], LB, Ws=[self.Ws[i]])
            
            LP = LT_general([self.ALs[i]], [self.ALs[i]], LP, Ws=[self.Ws[i]])
        #B = npc.tensordot(self.VLs[self.L-1], X[self.L-1], axes=(['vR'], ['vL']))
        B = X[self.L-1].split_legs() if self.do_B else npc.tensordot(self.VLs[self.L-1], X[self.L-1], axes=(['vR'], ['vL']))
        LB = LT_general([B], [self.ALs[self.L-1]], LP, Ws=[self.Ws[self.L-1]]) + \
             LT_general([self.ARs[self.L-1]], [self.ALs[self.L-1]], LB, Ws=[self.Ws[self.L-1]])
        L = LB
        
        """
        B0 = npc.tensordot(self.VLs[0], X[0], axes=(['vR'], ['vL']))
        L = LT_general(self.ALs[:0] + [B0] + self.ARs[0+1:], 
                        self.ALs, 
                        self.LW, Ws = self.Ws)
        for i in range(1, self.L):
            Bi = npc.tensordot(self.VLs[i], X[i], axes=(['vR'], ['vL']))
            L.iadd_prefactor_other(1., LT_general(self.ALs[:i] + [Bi] + self.ARs[i+1:], 
                                                  self.ALs, 
                                                  self.LW, Ws = self.Ws))
        """
        assert not np.isclose(npc.norm(L), 0)
        if sum_method=='explicit':
            L_sum = L.copy()
            for _ in range(100):
                L = np.exp(1.0j * p * self.L) * LT_general(self.ARs, self.ALs, L, Ws=self.Ws)
                L_sum.iadd_prefactor_other(1., L)
                if npc.norm(L) < tol:
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
                        assert False
                        lr = npc.tensordot(vec, self.excit.r_RL, axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        lrr = npc.tensordot(vec, self.excit.CcRW, axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
                        T1_l = self.excit.l_RL * ((self.excit.e_RL-1) * lr + lrr) + self.excit.LWC * lr
                        lT = lT - T1_l
                    return vec - np.exp(1.0j * p * self.excit.L) * lT
        
            tm_op = helper_matvec(self, self.ALs, self.ARs, self.Ws, sum_method)
            GMRES_options = self.options.subconfig('GMRES_options')
            L_sum, _, _, _ = GMRES(tm_op, npc.Array.zeros_like(L)*1.j, L, GMRES_options).run()
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
            #assert len(vec) == self.outer.L
            #assert np.sum([v.get_leg('vL').ind_len * v.get_leg('vR').ind_len for v in vec]) == self.outer.L * self.chi*(self.d-1)*self.chi
            
            total_vec = [npc.Array.zeros_like(v) for v in vec]
            
            for i in range(len(self.VLs)):
                if not self.outer.do_B:
                    Bi = npc.tensordot(self.VLs[i], vec[i], axes=(['vR'], ['vL']))
                else:
                    Bi = vec[i].split_legs()
                for j in range(len(self.VLs)):
                    if not self.outer.do_B:
                        Bj = npc.tensordot(self.VLs[j], vec[j], axes=(['vR'], ['vL']))
                        if i <= j:
                            X_out = LT_general(self.ALs[:i] + [Bi] + self.ARs[i+1:j+1], 
                                              self.ALs[:j] + [self.VLs[j]], 
                                              self.LW, Ws=self.Ws[:j+1])
                            T_RW = TR_general(self.ARs[j+1:], self.ARs[j+1:], self.RW, Ws=self.Ws[j+1:])
                            X_out = npc.tensordot(X_out, T_RW, axes=(['vR', 'wR'], ['vL', 'wL']))
                        else:
                            X_out = LT_general(self.ALs[:j+1], 
                                              self.ALs[:j] + [self.VLs[j]], 
                                              self.LW, Ws=self.Ws[:j+1])
                            T_RW = TR_general(self.ALs[j+1:i] + [Bi] + self.ARs[i+1:], 
                                              self.ARs[j+1:], 
                                              self.RW, Ws=self.Ws[j+1:])                        
                            X_out = npc.tensordot(X_out, T_RW, axes=(['vR', 'wR'], ['vL', 'wL']))
                        X_out.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                    else:
                        if i < j:                        
                            X_out = LT_general(self.ALs[:i] + [Bi] + self.ARs[i+1:j], 
                                              self.ALs[:j], 
                                              self.LW, Ws=self.Ws[:j])
                            X_out = npc.tensordot(X_out, self.ARs[j], axes=(['vR'], ['vL']))
                            X_out = npc.tensordot(X_out, self.Ws[j], axes=(['wR', 'p'], ['wL', 'p*']))
                            T_RW = TR_general(self.ARs[j+1:], self.ARs[j+1:], self.RW, Ws=self.Ws[j+1:])
                            X_out = npc.tensordot(X_out, T_RW, axes=(['vR', 'wR'], ['vL', 'wL']))
                        elif i > j:
                            X_out = LT_general(self.ALs[:j],
                                              self.ALs[:j],
                                              self.LW, Ws=self.Ws[:j])
                            X_out = npc.tensordot(X_out, self.ALs[j], axes=(['vR'], ['vL']))
                            X_out = npc.tensordot(X_out, self.Ws[j], axes=(['wR', 'p'], ['wL', 'p*']))
                            T_RW = TR_general(self.ALs[j+1:i] + [Bi] + self.ARs[i+1:], 
                                              self.ARs[j+1:], 
                                              self.RW, Ws=self.Ws[j+1:])                        
                            X_out = npc.tensordot(X_out, T_RW, axes=(['vR', 'wR'], ['vL', 'wL']))
                        else:
                            LW_T = LT_general(self.ALs[:j], 
                                              self.ALs[:j], 
                                              self.LW, Ws=self.Ws[:j])
                            LW_T = npc.tensordot(LW_T, Bi, axes=(['vR'], ['vL']))

                            LW_T = npc.tensordot(LW_T, self.Ws[i], axes=(['wR', 'p'], ['wL', 'p*']))
                            T_RW = TR_general(self.ARs[j+1:], 
                                              self.ARs[j+1:], 
                                              self.RW, Ws=self.Ws[j+1:])                        
                            X_out = npc.tensordot(LW_T, T_RW, axes=(['vR', 'wR'], ['vL', 'wL']))
                        
                        X_out.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                        X_out.itranspose(['vL', 'p', 'vR'])
                        X_out = X_out.combine_legs(['vL', 'p'], qconj=+1)
                    total_vec[j] += X_out
            #print('aligned:', [npc.norm(x) for x in total_vec])        
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
            #assert len(vec) == self.outer.L
            #assert np.sum([v.get_leg('vL').ind_len * v.get_leg('vR').ind_len for v in vec]) == self.outer.L * self.chi*(self.d-1)*self.chi
            total = [npc.Array.zeros_like(v) for v in vec]
            if not self.outer.do_B:
                
                inf_sum_TLR = self.outer.infinite_sum_TLR(vec, self.p)
                #env_LR = MPOEnvironment(self.psi, self.H, self.psi, **self.boundary_env_data)
                for i in range(self.outer.L):
                    X_out_left = TR_general(self.ALs[i+1:], self.ARs[i+1:], inf_sum_TLR, Ws=self.Ws[i+1:])
                    LW_T_VL = LT_general(self.ALs[:i+1], self.ALs[:i] + [self.VLs[i]], self.LW, Ws=self.Ws[:i+1])
                    X_out_left = np.exp(-1.0j*self.p*self.outer.L) * npc.tensordot(LW_T_VL, X_out_left, axes=(['vR', 'wR'], ['vL', 'wL']))
                    X_out_left.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                    total[i] += X_out_left

                #total_right = [npc.Array.zeros_like(v) for v in vec]
                inf_sum_TRL = self.outer.infinite_sum_TRL(vec, self.p)
                for i in range(self.outer.L):
                    X_out_right = LT_general(self.ARs[:i+1], self.ALs[:i] + [self.VLs[i]], inf_sum_TRL, Ws=self.Ws[:i+1])
                    T_RW = TR_general(self.ARs[i+1:], self.ARs[i+1:], self.RW, Ws=self.Ws[i+1:])
                    X_out_right = np.exp(1.0j*self.p*self.outer.L) * npc.tensordot(X_out_right, T_RW, axes=(['vR', 'wR'], ['vL', 'wL']))
                    X_out_right.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                    total[i] += X_out_right
            else:
                inf_sum_TLR = self.outer.infinite_sum_TLR(vec, self.p)
                #env_LR = MPOEnvironment(self.psi, self.H, self.psi, **self.boundary_env_data)
                for i in range(self.outer.L):
                    X_out_left = TR_general(self.ALs[i+1:], self.ARs[i+1:], inf_sum_TLR, Ws=self.Ws[i+1:])
                    LW_T_VL = LT_general(self.ALs[:i], self.ALs[:i], self.LW, Ws=self.Ws[:i+1])
                    LW_T_VL = npc.tensordot(LW_T_VL, self.ALs[i], axes=(['vR'], ['vL']))
                    LW_T_VL = npc.tensordot(LW_T_VL, self.Ws[i], axes=(['wR', 'p'], ['wL', 'p*']))
                    X_out_left = np.exp(-1.0j*self.p*self.outer.L) * npc.tensordot(LW_T_VL, X_out_left, axes=(['vR', 'wR'], ['vL', 'wL']))
                    X_out_left.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                    X_out_left.itranspose(['vL', 'p', 'vR'])
                    total[i] += X_out_left.combine_legs(['vL', 'p'], qconj=+1)

                #total_right = [npc.Array.zeros_like(v) for v in vec]
                inf_sum_TRL = self.outer.infinite_sum_TRL(vec, self.p)
                for i in range(self.outer.L):
                    X_out_right = LT_general(self.ARs[:i], self.ALs[:i], inf_sum_TRL, Ws=self.Ws[:i+1])
                    T_RW = TR_general(self.ARs[i+1:], self.ARs[i+1:], self.RW, Ws=self.Ws[i+1:])
                    T_RW = npc.tensordot(self.ARs[i], T_RW, axes=(['vR'], ['vL']))
                    T_RW = npc.tensordot(self.Ws[i], T_RW, axes=(['wR', 'p*'], ['wL', 'p']))
                    X_out_right = np.exp(1.0j*self.p*self.outer.L) * npc.tensordot(X_out_right, T_RW, axes=(['vR', 'wR'], ['vL', 'wL']))
                    X_out_right.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                    X_out_right.itranspose(['vL', 'p', 'vR'])
                    
                    total[i] += X_out_right.combine_legs(['vL', 'p'], qconj=+1)
                
            #print('unaligned:', [npc.norm(x) for x in total]) 
            return total #[a + b for a, b in zip(total_left, total_right)]

    def diagonalize(self, p, qtotal_change=None, n_bands=1):
        
        self.unaligned_H = self.Unaligned_Effective_H(self, self.ALs, self.ARs, self.VLs, 
                                                      self.LW, self.RW, self.Ws, p, self.chi, self.d)
        effective_H = SumNpcLinearOperator(self.aligned_H, self.unaligned_H)

        lanczos_options = self.options.subconfig('lanczos_options')
        lanczos_options['num_ev'] = n_bands
        #lanczos_options['E_shift'] = 0
        X_init = self.initial_guess(qtotal_change)
        B_init = make_Bs(self.VLs, X_init)
        #E, X, N = LanczosGroundState(effective_H, X_init, lanczos_options).run()
        #return E - self.energy_density * self.L - self.lambda_C1, X
        #E, X, N = LanczosExcitedState(effective_H, X_init, lanczos_options).run()
        Xs = []
        Es = []
        Ns = []
        for i in range(n_bands):
            # TODO getting poorely conditioned Krylov error for n_bands > 1;
            # 2nd excitation energy is not correct!
            # TODO why doesn't E_shift and OrthogonalNPC not work???
            #ortho_H = OrthogonalNpcLinearOperator(effective_H, Xs)
            
            ortho_H = BoostNpcLinearOperator(effective_H, [100]*i, Xs)
            #ortho_H = ShiftNpcLinearOperator(ortho_H, -100)
            if self.do_B:
                E, Bs, N = LanczosGroundState(ortho_H, B_init, lanczos_options).run()
                X = get_Xs(self.VLs, Bs)
            else:
                E, X, N = LanczosGroundState(ortho_H, X_init, lanczos_options).run()
            #
            #print(i, E, N)
            Xs.append(X)
            Es.append(E)
            Ns.append(N)
            print(p, E)
            if N == lanczos_options.get('N_max', 20):
                import warnings
                warnings.warn('Maximum Lanczos iterations needed; be wary of results.')
        return [E - self.energy_density * self.L - self.lambda_C1 for E in Es], Xs, Ns
    
    def initial_guess(self, qtotal_change):
        X_init = []
        env = MPOEnvironment(self.psi, self.H, self.psi, **self.boundary_env_data)
        for i in range(self.L):      
            vL = self.VLs[i].get_leg('vR').conj()
            vR = self.ALs[(i+1)% self.L].get_leg('vL').conj()
            th0 = npc.Array.from_func(np.ones, [vL, vR],
                                      dtype=self.psi.dtype,
                                      qtotal=qtotal_change,
                                      labels=['vL', 'vR'])
            
            if np.isclose(npc.norm(th0), 0):
                warnings.warn('Initial guess for X is zero; charges may not be allowed.')
                
            LP = env.get_LP(i, store=True)
            RP = env.get_RP(i, store=True)
            LP = LT_general([self.VLs[i]], [self.VLs[i]], LP, Ws=[self.Ws[i]])
            
            H0 = ZeroSiteH.from_LP_RP(LP, RP)
            if self.model.H_MPO.explicit_plus_hc:
                H0 = SumNpcLinearOperator(H0, H0.adjoint())
            
            lanczos_options = self.options.subconfig('lanczos_options')
            _, th0, _ = LanczosGroundState(H0, th0, lanczos_options).run()
            
            X_init.append(th0)
            #print('X' + str(i) + ' norm:', npc.norm(th0))
            
        return X_init      