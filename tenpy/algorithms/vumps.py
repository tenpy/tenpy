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
        split_err_tol = options.get('max_split_err', 1.e-8)

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
            norm_err = np.linalg.norm(self.psi.norm_test(force=True))

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
            self.sweep_stats['total_time'].append(time.time() - start_time)
            self.sweep_stats['sweep_time'].append(self.sweep_stats['total_time'][-1] - self.sweep_stats['total_time'][-2])
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
                "max split_err = %(max_split_err).4e\n"
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
        self.tangent_projector_test()
        logger.info("VUMPS finished after %d sweeps, max chi=%d", self.sweeps, max(self.psi.chi))
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
            'total_time': [0],
            'sweep_time': [],
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
        
        boundary_env_data, Es, _ = MPOTransferMatrix.find_init_LP_RP(H, self.psi, calc_E=True, guess_init_env_data=self.guess_init_env_data) # E is already the energy density.
        self.env = MPOEnvironment(psi, H, psi, **boundary_env_data)
        self.transfer_matrix_energy = Es

        self.make_eff_H()
        theta = self.psi.get_theta(i0, n=self.n_optimize, cutoff=self.S_inv_cutoff) #n_optimize will be 1
        assert self.eff_H1.combine == False
        theta = self.eff_H1.combine_theta(theta) #combine should be false.
        C1, C2 = self.psi.get_C(i0), self.psi.get_C(i0+1)
        return (theta, C1, C2)
    
    def make_eff_H(self):

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
        psi = self.psi
        i0 = self.i0
        H0_1, H0_2, H1 = self.eff_H0_1, self.eff_H0_2, self.eff_H1
        AC, C1, C2 = theta
       
        lanczos_options = self.options.subconfig('lanczos_options')
        E0_1, theta0_1, N0_1 = LanczosGroundState(H0_1, C1, lanczos_options).run()
        E0_2, theta0_2, N0_2 = LanczosGroundState(H0_2, C2, lanczos_options).run()
        E1, theta1, N1 = LanczosGroundState(H1, AC, lanczos_options).run()
        
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
            'e_L': self.transfer_matrix_energy[1],
            'e_R': self.transfer_matrix_energy[0],
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

    def free_no_longer_needed_envs(self):
        for env in self._all_envs:
            env.clear() # Can we do better?
        
    def polar_max(self, AC, C1, C2):
        # Given AC and C, find AL such that AL C = AC
        
        U_ACL, _, _ = npc.polar(AC.combine_legs(['vL', 'p'], qconj=[+1]), left=False)
        U_CL, _, s1 = npc.polar(C2, left=False)
        AL = npc.tensordot(U_ACL.split_legs(), U_CL.conj(), axes=(['vR'], ['vR*'])).replace_label('vL*', 'vR')
        
        U_ACR, _, _ = npc.polar(AC.combine_legs(['p', 'vR'], qconj=[+1]), left=True)
        U_CR, _, s2 = npc.polar(C1, left=True)
        AR = npc.tensordot(U_CR.conj(), U_ACR.split_legs(), axes=(['vL*'], ['vL'])).replace_label('vR*', 'vL')
        
        eps_L = npc.norm(AC - npc.tensordot(AL, C2, axes=['vR', 'vL']))
        eps_R = npc.norm(AC - npc.tensordot(C1, AR, axes=['vR', 'vL']))
        
        entropy_left = entropy(s1**2, n=1)
        entropy_right = entropy(s2**2, n=1)
        
        return AL, AR, eps_L, eps_R, entropy_left, entropy_right
        
    def resume_run(self):
        raise NotImplementedError("TODO")
        
    def tangent_projector_test(self):
        pass