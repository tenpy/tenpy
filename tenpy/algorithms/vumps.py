import numpy as np
import time
import warnings
import logging
logger = logging.getLogger(__name__)

from ..linalg import np_conserved as npc
from ..networks.mpo import MPOEnvironment, MPOTransferMatrix
from ..linalg.lanczos import LanczosGroundState, lanczos_arpack
from ..tools.params import asConfig
from ..tools.math import entropy
from ..tools.misc import find_subclass
from ..tools.process import memory_usage
from .mps_common import Sweep, ZeroSiteH, OneSiteH

import sys
sys.path.append("/home/sajant/vumps-tBLG/Nsite/")
from misc import *
from vumps_utils import *

class OneSiteVUMPSEngine(Sweep):
    EffectiveH = OneSiteH
    
    def __init__(self, psi, model, options, **kwargs):
        #options = asConfig(options, self.__class__.__name__)
        super().__init__(psi, model, options, **kwargs)
        self.guess_init_env_data = self.env.get_initialization_data()
        self.env.clear()
        self._entropy_approx = [None] * psi.L  # always left of a given site
        
    def run(self):
        options = self.options
        start_time = self.time0
        self.shelve = False
        
        min_sweeps = options.get('min_sweeps', 1)
        max_sweeps = options.get('max_sweeps', 1000)
        max_E_err = options.get('max_E_err', 1.e-8)
        max_S_err = options.get('max_S_err', 1.e-5)
        split_err_tol = options.get('max_split_err', 1.e-6)
        #max_seconds = 3600 * options.get('max_hours', 24 * 365)
        E, Delta_E, Delta_S, Error = 1., 1., 1., 1.
        E_old, S_old = np.nan, np.mean(self.psi.entanglement_entropy())  # initial dummy values
        max_split_error = 1
        is_first_sweep = True
        self.psi.valid_umps = False

        while True:
            loop_start_time = time.time()
            #Check convergence criteria
            if self.sweeps >= max_sweeps:
                break
            if (self.sweeps > min_sweeps and -Delta_E < max_E_err * max(abs(E), 1.)
                    and abs(Delta_S) < max_S_err and max_split_error < split_err_tol):
                break     
            if not is_first_sweep:
                self.checkpoint.emit(self)
            # --------- the main work --------------
            logger.info('Running sweep with optimization')
            self.sweep()
            max_split_error = np.max(self.update_stats['split_err_L'][-self.psi.L:] + self.update_stats['split_err_R'][-self.psi.L:])
            # --------------------------------------
            # update values for checking the convergence
            entropy_bonds = self._entropy_approx
            max_S = max(entropy_bonds)
            S = np.mean(entropy_bonds)
            Delta_S = (S - S_old)
            S_old = S
            E = np.mean(self.update_stats['e_AC'][-self.psi.L:])
            Delta_E = (E - E_old)
            E_old = E
            self.psi.valid_umps = True
            norm_err = np.linalg.norm(self.psi.norm_test())
            self.psi.valid_umps = False
            # update statistics
            self.sweep_stats['sweep'].append(self.sweeps)
            self.sweep_stats['N_updates'].append(len(self.update_stats['i0']))
            self.sweep_stats['E_AC'].append(np.mean(self.update_stats['e_AC'][-self.psi.L:]))
            self.sweep_stats['E_L'].append(np.mean(self.update_stats['e_L'][-self.psi.L:]))
            self.sweep_stats['E_R'].append(np.mean(self.update_stats['e_R'][-self.psi.L:]))
            self.sweep_stats['E_C1'].append(np.mean(self.update_stats['e_C1'][-self.psi.L:]))
            self.sweep_stats['E_C2'].append(np.mean(self.update_stats['e_C2'][-self.psi.L:]))
            self.sweep_stats['S'].append(S)
            self.sweep_stats['max_S'].append(max_S)
            self.sweep_stats['time'].append(time.time() - start_time)
            self.sweep_stats['max_chi'].append(np.max(self.psi.chi))
            self.sweep_stats['norm_err'].append(norm_err)
            self.sweep_stats['max_split_err'].append(max_split_error)
            
            
            print(self.sweeps, Delta_E, norm_err, Delta_S, max_split_error, self.sweep_stats['E_AC'][-1], self.sweep_stats['E_L'][-1], 
                  self.sweep_stats['E_R'][-1], self.sweep_stats['E_C1'][-1], self.sweep_stats['E_C2'][-1])
            # status update
            logger.info(
                "checkpoint after sweep %(sweeps)d\n"
                "energy=%(E).16f, max S=%(S).16f, age=%(age)d, norm_err=%(norm_err).1e\n"
                "Current memory usage %(mem).1fMB, wall time: %(wall_time).1fs\n"
                "Delta E = %(dE).4e, Delta S = %(dS).4e (per sweep)\n"
                "max trunc_err = %(trunc_err).4e, max E_trunc = %(E_trunc).4e\n"
                "chi: %(chi)s\n"
                "%(sep)s", {
                    'sweeps': self.sweeps,
                    'E': E,
                    'S': max_S,
                    'norm_err': norm_err,
                    'mem': memory_usage(),
                    'wall_time': time.time() - loop_start_time,
                    'dE': Delta_E,
                    'dS': Delta_S,
                    'max_split_err': max_split_error,
                    'chi': self.psi.chi if self.psi.L < 40 else max(self.psi.chi),
                    'sep': "=" * 80,
                })
            is_first_sweep = False
        
        self.psi.test_validity()
        logger.info("DMRG finished after %d sweeps, max chi=%d", self.sweeps, max(self.psi.chi))
        return (self.sweep_stats['E_L'][-1] + self.sweep_stats['E_R'][-1])/2, self.psi

    def environment_sweeps(self, N_sweeps):
        # In VUMPS we don't want to do this as we regenerate the environment each time we do an update.
        pass
    
    def reset_stats(self, resume_data=None):
        super().reset_stats(resume_data)
        self.update_stats = {
            'i0': [],
            'e_L': [],
            'e_R': [],
            'e_C1': [],
            'e_C2': [],
            'e_AC': [],
            'N_lanczos': [],
            'split_err_L': [],
            'split_err_R': [],
            'time': [],
        }

        self.sweep_stats = {
            'sweep': [],
            'N_updates': [],
            'E_L': [],
            'E_R': [],
            'E_C1': [],
            'E_C2': [],
            'E_AC': [],
            'S': [],
            'max_S': [],
            'time': [],
            'max_chi': [],
            'norm_err': [],
            'max_split_err': [],
        }
        
    def get_sweep_schedule(self):
        """Sweep from site 0 to L-1"""
        L = self.psi.L
        i0s = list(range(0, L))
        move_right = [True] * L
        update_LP_RP = [[False, False]] * L
        return zip(i0s, move_right, update_LP_RP)
    
    def prepare_update(self):
        # For each update, we need to rebuild the environments from scratch using the most recent tensors        
        i0 = self.i0
        H = self.model.H_MPO
        psi = self.psi
        
        Es, boundary_env_data = MPOTransferMatrix.find_init_LP_RP(H, self.psi, calc_E=True, guess_init_env_data=self.guess_init_env_data) # E is already the energy density.
        self.env = MPOEnvironment(psi, H, psi, **boundary_env_data)
        self.transfer_matrix_energy = (Es[1], Es[0])
        
        self.make_eff_H()
        theta = self.psi.get_theta(i0, n=self.n_optimize, cutoff=self.S_inv_cutoff) #n_optimize will be 1
        assert self.eff_H1.combine == False
        theta = self.eff_H1.combine_theta(theta) #combine should be false.
        C1, C2 = self.psi.get_C(i0), self.psi.get_C(i0+1)
        return (theta, C1, C2)
    
    def make_eff_H(self):
        # Called by prepare_update; we need to make 3 H's.
        """
        self.LP1 = self.env.get_LP(self.i0, store=True)
        self.LP2 = self.env.get_LP(self.i0 + 1, store=False)
        self.RP2 = self.env.get_RP(self.i0, store=True)
        self.RP1 = self.env.get_RP(self.i0 - 1, store=False)
        self.W0 = self.env.H.get_W(self.i0)
        self.eff_H0_1 = ZeroSiteH.from_LP_RP(self.LP1, self.RP1, self.i0)
        self.eff_H0_2 = ZeroSiteH.from_LP_RP(self.LP2, self.RP2, self.i0+1)
        self.eff_H1 = OneSiteH.from_LP_W0_RP(self.LP1, self.env.H.get_W(self.i0), self.RP2, i0=self.i0, combine=False, move_right=True)
        """
        self.eff_H0_1 = ZeroSiteH(self.env, self.i0) # This saves more envs than optimal.
        self.eff_H0_2 = ZeroSiteH(self.env, self.i0 + 1) # This saves more envs than optimal.
        self.eff_H1 = self.EffectiveH(self.env, self.i0, self.combine, self.move_right)
        
        if hasattr(self.env, 'H') and self.env.H.explicit_plus_hc:
            self.eff_H1 = SumNpcLinearOperator(self.eff_H1, self.eff_H1.adjoint())
        if hasattr(self.env, 'H') and self.env.H.explicit_plus_hc:
            self.eff_H0_1 = SumNpcLinearOperator(self.eff_H0_1, self.eff_H0_1.adjoint())
        if hasattr(self.env, 'H') and self.env.H.explicit_plus_hc:
            self.eff_H0_2 = SumNpcLinearOperator(self.eff_H0_2, self.eff_H0_2.adjoint())

    def _wrap_ortho_eff_H(self):
        raise NotImplementedError("Do we want this for VUMPS?")
        
    def update_local(self, theta, **kwargs):
        # Update on site 
        #logger.info('Update on site: ', self.i0)
        #print('Update on site: ', self.i0)
        psi = self.psi
        i0 = self.i0
        H0_1, H0_2, H1 = self.eff_H0_1, self.eff_H0_2, self.eff_H1
        AC, C1, C2 = theta
        """
        W0 = self.W0.itranspose(['wL', 'wR', 'p*', 'p']).to_ndarray()
        print(self.LP1.get_leg_labels())
        lw_nn = self.LP1.itranspose(['wR', 'vR', 'vR*']).to_ndarray()
        lw_n = self.LP2.itranspose(['wR', 'vR', 'vR*']).to_ndarray()
        rw_n = self.RP2.itranspose(['wL', 'vL', 'vL*']).to_ndarray()
        rw_nn = self.RP1.itranspose(['wL', 'vL', 'vL*']).to_ndarray()

        AC_n, lamAC_n = Lanczos_AC(W0, lw_nn, rw_n, verbose=-1)    
        C_n,  lamC_n  = Lanczos_C(lw_n, rw_n, verbose=-1) 
        C_nn, lamC_nn = Lanczos_C(lw_nn, rw_nn, verbose=-1)
        E0_1, E0_2, E1 = lamC_nn, lamC_n, lamAC_n
        N0_1, N0_2, N1 = 0, 0, 0
        theta0_1_2 = npc.Array.from_ndarray_trivial(C_nn, labels=['vL', 'vR'])
        theta0_2_2 = npc.Array.from_ndarray_trivial(C_n, labels=['vL', 'vR'])
        theta1 = npc.Array.from_ndarray_trivial(AC_n, labels=['p0', 'vL', 'vR'])
        #theta0_1.iscale_prefactor(npc.inner())
        """
        #assert False
        #print(self.psi.norm_test())
        
        #lanczos_options = self.options.subconfig('lanczos_options')
        #E0_1, theta0_1 = lanczos_arpack(H0_1, C1, lanczos_options)
        #E0_2, theta0_2 = lanczos_arpack(H0_2, C2, lanczos_options)
        #E1, theta1 = lanczos_arpack(H1, AC, lanczos_options)
        #N0_1 = N0_2 = N1 = 0
        #lanczos_options = {'N_max': 1000, 'cutoff': 1.e-16, 'P_tol': 1.e-16}
        lanczos_options = self.options.subconfig('lanczos_options')
        E0_1, theta0_1, N0_1 = LanczosGroundState(H0_1, C1, lanczos_options).run()
        E0_2, theta0_2, N0_2 = LanczosGroundState(H0_2, C2, lanczos_options).run()
        E1, theta1, N1 = LanczosGroundState(H1, AC, lanczos_options).run()
        print(N0_1, N0_2, N1)
        
        """
        Updating number of Lanczos iterations, activate orthogonalize against
        """
        
        theta1.ireplace_label('p0', 'p')
        psi.set_C(i0, theta0_1)
        psi.set_C(i0+1, theta0_2)
        psi.set_B(i0, theta1, form='AC')
        AL, AR, eps_L, eps_R, entropy_1, entropy_2 = self.polar_max(theta1, theta0_1, theta0_2)
        psi.set_B(i0, AL, form='AL')
        psi.set_B(i0, AR, form='AR')
        self._entropy_approx[i0 % self.psi.L] = entropy_1
        self._entropy_approx[(i0+1) % self.psi.L] = entropy_2
        update_data = {
            'e_L': self.transfer_matrix_energy[0],
            'e_R': self.transfer_matrix_energy[1],
            'eps_L': eps_L,
            'eps_R': eps_R,
            'e_C1': E0_1,
            'e_C2': E0_2,
            'e_AC': E1,
            'N0_L': N0_1,
            'N0_R': N0_2,
            'N1': N1
        }
        
        self.trunc_err_list.append(0)
        
        return update_data
        
    def update_env(self, **update_data):
        # Get guesses for the next LP and RP
        self.guess_init_env_data = self.env.get_initialization_data()
        pass
    
    def post_update_local(self, e_L, e_R, eps_L, eps_R, e_C1, e_C2, e_AC, N0_L, N0_R, N1, **update_data):
        self.update_stats['i0'].append(self.i0)
        self.update_stats['e_L'].append(e_L)
        self.update_stats['e_R'].append(e_R)
        self.update_stats['e_C1'].append(e_C1)
        self.update_stats['e_C2'].append(e_C2)
        self.update_stats['e_AC'].append(e_AC)
        self.update_stats['N_lanczos'].append(N0_L + N0_R + N1)
        self.update_stats['split_err_L'].append(eps_L)
        self.update_stats['split_err_R'].append(eps_R)
        self.update_stats['time'].append(time.time() - self.time0)

    def free_no_longer_needed_envs(self):
        for env in self._all_envs:
            env.clear() # Can we do better?
    """        
    def split_AC2(self, AC, C, verbose=-1):
        '''
        Eq. (C29) in PRB 97, 045145 (2018)
        or 
        Eq. (143)-(145) in arxiv.1810.07006 
        '''
        AC_np = AC.to_ndarray().transpose([1, 0, 2])
        C_np = C.to_ndarray()
        
        E_AL = np.tensordot(AC_np.conj(), C_np, [[2],[1]])
        AL = polar_max_tensor(E_AL, in_inds=[0,1], out_inds=[2])
        E_AR = np.tensordot(AC_np.conj(), C_np, [[1],[0]]).transpose([0,2,1])
        AR = polar_max_tensor(E_AR, in_inds=[0,2], out_inds=[1])
        '''
        AC_l, pipe_l = group_legs(AC, [[0,1],[2]])
        AC_r, pipe_r = group_legs(AC, [[1],[0,2]])
        U_AC_l, _ = scipy.linalg.polar(AC_l, side='right')
        U_C_l,  _ = scipy.linalg.polar(C, side='right')
        U_C_r,  _ = scipy.linalg.polar(C, side='left')
        U_AC_r, _ = scipy.linalg.polar(AC_r, side='left')
        AL = ungroup_legs(U_AC_l @ U_C_l.conj().T, pipe_l)
        AR = ungroup_legs(U_C_r.conj().T @ U_AC_r, pipe_r)
        '''
        eps_L = np.linalg.norm(AC_np - mT(C_np, AL, 2, order='Tm'))
        eps_R = np.linalg.norm(AC_np - mT(C_np, AR, 1, order='mT'))
        if verbose >= 2:
            print("split_AC eps_L:", eps_L)
            print("split_AC eps_R:", eps_R)
        
        AL = npc.Array.from_ndarray_trivial(AL, labels=['p', 'vL', 'vR'])
        AR = npc.Array.from_ndarray_trivial(AR, labels=['p', 'vL', 'vR'])
        
        return AL, AR, abs(eps_L), abs(eps_R)
    """
        
    def polar_max(self, AC, C1, C2):
        # Given AC and C, find AL such that AL C = AC
        
        """
        AL_env = npc.tensordot(AC, C2.conj(), axes=['vR', 'vR*']).replace_label('vL*', 'vR').combine_legs(['vL', 'p'], qconj=[+1])
        U, s, V = npc.svd(AL_env, inner_labels=['vR', 'vL'])
        AL = npc.tensordot(U, V, axes=['vR', 'vL']).split_legs()
        """
        U_ACL, _, _ = npc.polar(AC.combine_legs(['vL', 'p'], qconj=[+1]), left=False)
        U_CL, _, s1 = npc.polar(C2, left=False)
        AL = npc.tensordot(U_ACL.split_legs(), U_CL.conj(), axes=(['vR'], ['vR*'])).replace_label('vL*', 'vR')
        
        #print(npc.norm(npc.tensordot(AL, AL.conj(), axes=[['vL', 'p'], ['vL*', 'p*']]) - npc.eye_like(AL, axis='vR')))
        """
        AR_env = npc.tensordot(C1.conj(), AC, axes=['vL*', 'vL']).replace_label('vR*', 'vL').combine_legs(['p', 'vR'], qconj=[+1])
        U, s, V = npc.svd(AR_env, inner_labels=['vR', 'vL'])
        AR = npc.tensordot(U, V, axes=['vR', 'vL']).split_legs()
        """
        U_ACR, _, _ = npc.polar(AC.combine_legs(['p', 'vR'], qconj=[+1]), left=True)
        U_CR, _, s2 = npc.polar(C1, left=True)
        AR = npc.tensordot(U_CR.conj(), U_ACR.split_legs(), axes=(['vL*'], ['vL'])).replace_label('vR*', 'vL')
        
        #print(npc.norm(npc.tensordot(AR, AR.conj(), axes=[['vR', 'p'], ['vR*', 'p*']]) - npc.eye_like(AR, axis='vL')))
        eps_L = npc.norm(AC - npc.tensordot(AL, C2, axes=['vR', 'vL']))
        eps_R = npc.norm(AC - npc.tensordot(C1, AR, axes=['vR', 'vL']))
        
        entropy_left = entropy(s1**2, n=1)
        entropy_right = entropy(s2**2, n=1)
        
        return AL, AR, eps_L, eps_R, entropy_left, entropy_right
        
    def resume_run(self):
        raise NotImplementedError("TODO")