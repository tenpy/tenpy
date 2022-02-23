import numpy as np
import time
import warnings
import logging
logger = logging.getLogger(__name__)

from ..linalg import np_conserved as npc
from ..networks.mpo import MPOEnvironment, MPOTransferMatrix
from ..linalg.lanczos import GMRES, LanczosGroundState
from ..linalg.sparse import NpcLinearOperator, SumNpcLinearOperator
from ..tools.params import asConfig
from ..tools.math import entropy
from ..algorithms.algorithm import Algorithm
#from ..tools.misc import find_subclass
#from ..tools.process import memory_usage
#from .mps_common import Sweep, ZeroSiteH, OneSiteH

#import sys
#sys.path.append("/home/sajant/vumps-tBLG/Nsite/")
#from misc import *
#from vumps_utils import *

# Generalize this to a unit cell?
def TR_general(A, B, R, W=None):
    temp = npc.tensordot(A, R, axes=(['vR'], ['vL']))
    if W is not None:
        temp = npc.tensordot(W, temp, axes=(['wR', 'p*'], ['wL', 'p']))
    temp = npc.tensordot(B.conj(), temp, axes=(['vR*', 'p*'], ['vL*', 'p']))
    return temp.itranspose(['vL', 'wL', 'vL*'])

def LT_general(A, B, L, W=None):
    temp = npc.tensordot(L, A, axes=(['vR'], ['vL']))
    if W is not None:
        temp = npc.tensordot(temp, W, axes=(['wR', 'p'], ['wL', 'p*']))
    temp = npc.tensordot(temp, B.conj(), axes=(['vR*', 'p'], ['vL*', 'p*']))
    return temp.itranspose(['vR*', 'wR', 'vR'])

class PlaneWaveExcitations(Algorithm):
    def __init__(self, psi, model, options, **kwargs):
        #options = asConfig(options, self.__class__.__name__)
        super().__init__(psi, model, options, **kwargs)
        
        assert self.psi.L == 1
        assert self.model.H_MPO.L == 1
        
        self.ALs = [self.psi.get_AL(i) for i in range(self.psi.L)]
        self.AR = [self.psi.get_AR(i) for i in range(self.psi.L)]
        self.AC = [self.psi.get_AC(i) for i in range(self.psi.L)]
        self.C = [self.psi.get_C(i) for i in range(self.psi.L)] # C on the left
        self.H = self.model.H_MPO
        self.IdL = self.H.get_IdL(0)
        self.IdR = self.H.get_IdR(-1)
        #self.guess_init_env_data = options.get('init_data', {'init_RP': npc.tensordot(self.C, self.C.conj(), axes=(['vR', 'vL*'])) ,
        #                                                    'init_LP': npc.tensordot(self.C.conj(), self.C, axes=(['vR*', 'vR']))})
        self.guess_init_env_data = options.get('init_data',None)
        self.dW = self.H.get_W(0).get_leg('wR').ind_len # [TODO] this assumes a single site
        self.chi = self.AL.get_leg('vL').ind_len
        
        # Construct VL, needed to parametrize - B - as - VL - X -
        #                                       |        |
        # Use prescription under Eq. 85 in Tangent Space lecture notes.
        self.construct_orthogonal() 
        
        # Get left and right generalized eigenvalues
        boundary_env_data, self.energy_density, _ = MPOTransferMatrix.find_init_LP_RP(self.H, self.psi, calc_E=True, subtraction_gauge='rho', guess_init_env_data=self.guess_init_env_data)
        self.LW = boundary_env_data['init_LP']
        self.RW = boundary_env_data['init_RP']
               
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
        self.l_LR[:,self.IdR,:] = self.C.conj().transpose(['vR*', 'vL*'])
        
        #wL = self.H.get(0).get_leg('wL')
        #wR = wL.conj()
        #self.l_LR = self.C.conj().add_leg(wR, self.IdR, axis=1, label='wR'
        
        self.CRW = npc.tensordot(self.C, self.RW, axes=(['vR'], ['vL'])) # ['vR','wR', 'vR*']
        self.LWCc= npc.tensordot(self.LW, self.C.conj(), axes=(['vR*'], ['vL*'])) # ['vL*','wL', 'vL']
        self.r_LR = npc.Array.zeros_like(self.RW) # ['vR','wR', 'vR*']
        self.r_LR[:,self.IdL,:] = self.C
        
        # tTr (l_LR * CRW) = tTr (C C^\dag) = 1 usually
        self.CRW = self.CRW / npc.tensordot(self.l_LR, self.CRW, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*']))
        
        # LWCc -> LWCc / self.e_LR
        self.LWCc = self.LWCc / npc.tensordot(self.LWCc, self.CRW, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*']))
        print('l_LR * CRW', npc.tensordot(self.l_LR, self.CRW, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*'])))
        print('LWCc * CRW', npc.tensordot(self.LWCc, self.CRW, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*'])))
        print('LWCc * r_LR', npc.tensordot(self.LWCc, self.r_LR, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*'])))
        self.r_LR = self.r_LR / npc.tensordot(self.LWCc, self.r_LR, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*'])) # Dividing by E_C, GS eigenvalue of HC
        print('l_LR * CRW', npc.tensordot(self.l_LR, self.CRW, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*'])))
        print('LWCc * CRW', npc.tensordot(self.LWCc, self.CRW, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*'])))
        print('LWCc * r_LR', npc.tensordot(self.LWCc, self.r_LR, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*'])))
        Tr = TR_general(self.AL, self.AR, self.CRW, W=self.H.get_W(0))
        
        #Should be energy density / E_C
        self.e_LR = (Tr - self.CRW)[0,self.IdL,0]/self.r_LR[0,self.IdL,0] # tTr( LWCc * CRW) for original tensors
        
        # Tw[AR,AL]
        self.r_RL = npc.Array.zeros_like(self.RW).itranspose(['vL', 'wL', 'vL*']) # [TODO] check default ordering to potentially remove transpose
        # Original ordering of boundary vectors ['vL','wL', 'vL*']
        self.r_RL[:,self.IdL,:] = self.C.conj().transpose(['vR*', 'vL*'])
        self.LWC = npc.tensordot(self.LW, self.C, axes=(['vR'], ['vL']))
        self.CcRW= npc.tensordot(self.RW, self.C.conj(), axes=(['vL*'], ['vR*']))
        self.l_RL = npc.Array.zeros_like(self.LW)
        self.l_RL[:,self.IdR,:] = self.C
        self.LWC = self.LWC / npc.tensordot(self.LWC, self.r_RL, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*']))
        self.CcRW = self.CcRW / npc.tensordot(self.LWC, self.CcRW, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*']))
        self.l_RL = self.l_RL / npc.tensordot(self.l_RL, self.CcRW, axes=(['wR', 'vR', 'vR*'], ['wL', 'vL', 'vL*']))
        lT = LT_general(self.AR, self.AL, self.LWC, W=self.H.get_W(0))

        #Should be energy density / E_C
        self.e_RL = (lT - self.LWC)[0,self.IdR,0]/self.l_RL[0,self.IdR,0]
        self.construct_orthogonal()
        
        self.aligned_H = self.Aligned_Effective_H(self.VL, self.LW, self.RW, self.H.get_W(0))
        
        temp = LT_general(self.VL, self.AC, self.LW, W=self.H.get_W(0))
        temp = npc.tensordot(temp, self.RW, axes=(['wR', 'vR*'], ['wL', 'vL*']))
        print('Strange Cancellation Term:', npc.norm(temp))
        
        if 1 >= 1:
            print("-"*20, "initializing excitation", "-"*20)
            print("norm(LW-LW.dag):", npc.norm(self.LW - self.LW.transpose(['vR','wR','vR*']).conj()))
            print("norm(RW-RW.dag):", npc.norm(self.RW - self.RW.transpose(['vL*','wL','vL']).conj()))
            assert self.psi.valid_umps
            print("Energy density:", self.energy_density)
            print("L norm:", npc.norm(self.LW))
            print("R norm:", npc.norm(self.RW))
            print("LR:", npc.tensordot(self.LW, self.RW, axes=(['vR*', 'wR', 'vR'],['vL*', 'wL', 'vL'])))

            lT  = LT_general(self.AL, self.AR, self.l_LR, W=self.H.get_W(0))  
            Tr  = TR_general(self.AL, self.AR, self.r_LR, W=self.H.get_W(0))
            print("LR norm(l-lT):", npc.norm(self.l_LR-lT))
            print("LR norm(r-Tr):", npc.norm(self.r_LR-Tr))
            
            lT  = LT_general(self.AR, self.AL, self.l_RL, W=self.H.get_W(0))  
            Tr  = TR_general(self.AR, self.AL, self.r_RL, W=self.H.get_W(0))
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
        
    def construct_orthogonal(self):
        AL = self.AL
        #chi = AL.get_leg['vL'].ind_len
        AL = AL.combine_legs(['vL', 'p'], qconj=[+1])
        Q, R = npc.qr(AL, mode='complete', inner_labels=['vR', 'vL'])
        n_rows = R.shape[1]
        V_grouped = Q[:,self.chi:] # [TODO] Karthik did this differently, but I think this is right?
        self.VL = V_grouped.split_legs()

        assert npc.norm(npc.tensordot(self.VL, self.AL.conj(), axes=(['vL', 'p'], ['vL*', 'p*']))) < 1.e-14
        
    def infinite_sum_TLR(self, X, p):
        tol = self.options.get('tol', 1.e-10)
        sum_method = self.options.get('sum_method', 'explicit')
        B = npc.tensordot(self.VL, X, axes=(['vR'], ['vL']))
        R = TR_general(B, self.AR, self.RW, W=self.H.get_W(0))
        
        if sum_method=='explicit':
            R_sum = R = TR_general(B, self.AR, self.RW, W=self.H.get_W(0))
            for _ in range(100):
                R = np.exp(-1.0j * p) * TR_general(self.AL, self.AR, R, W=self.H.get_W(0))
                R_sum.iadd_prefactor_other(1., R)
            return R_sum
        
    def infinite_sum_TRL(self, X, p):
        tol = self.options.get('tol', 1.e-10)
        sum_method = self.options.get('sum_method', 'explicit')
        B = npc.tensordot(self.VL, X, axes=(['vR'], ['vL']))
        L = LT_general(B, self.AL, self.LW, W=self.H.get_W(0))
        
        if sum_method=='explicit':
            L_sum = L
            for _ in range(100):
                L = np.exp(1.0j * p) * LT_general(self.AR, self.AL, L, W=self.H.get_W(0))
                L_sum.iadd_prefactor_other(1., L)
            return L_sum
        
    class Aligned_Effective_H(NpcLinearOperator):
        def __init__(self, VL, LW, RW, W):
            self.VL = VL
            self.LW = LW
            self.RW = RW
            self.W = W
        
        def matvec(self, vec):
            B = npc.tensordot(self.VL, vec, axes=(['vR'], ['vL']))
            X_out = LT_general(B, self.VL, self.LW, W=self.W)
            X_out = npc.tensordot(X_out, self.RW, axes=(['vR', 'wR'], ['vL', 'wL']))
            X_out.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
            
            return X_out
    
    class Unaligned_Effective_H(NpcLinearOperator):
        def __init__(self, outer, AL, AR, VL, LW, RW, W, p):
            self.AL = AL
            self.AR = AR
            self.VL = VL
            self.LW = LW
            self.RW = RW
            self.W = W
            self.p = p
            self.outer = outer
            
        def matvec(self, vec):
            X_out_left = self.outer.infinite_sum_TLR(vec, self.p)
            LW_VL = LT_general(self.AL, self.VL, self.LW, W=self.W)
            X_out_left = np.exp(-1.0j*self.p) * npc.tensordot(LW_VL, X_out_left, axes=(['vR', 'wR'], ['vL', 'wL']))
            X_out_left.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
            
            X_out_right = self.outer.infinite_sum_TRL(vec, self.p)
            X_out_right = LT_general(self.AR, self.VL, X_out_right, W=self.W)
            X_out_right = np.exp(1.0j*self.p) * npc.tensordot(X_out_right, self.RW, axes=(['vR', 'wR'], ['vL', 'wL']))
            X_out_right.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                        
            return X_out_left + X_out_right

    def diagonalize(self, p, n_bands=1):
        self.unaligned_H = self.Unaligned_Effective_H(self, self.AL, self.AR, self.VL, self.LW, self.RW, self.H.get_W(0), p)
        effective_H = SumNpcLinearOperator(self.aligned_H, self.unaligned_H)
        
        lanczos_options = self.options.subconfig('lanczos_options')
        E, theta, N = LanczosGroundState(effective_H, npc.Array.from_ndarray_trivial(np.random.rand(10,10), labels=['vL', 'vR']), lanczos_options).run()
        
        return E, theta
        