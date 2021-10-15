import numpy as np
import time
import warnings
import logging
logger = logging.getLogger(__name__)

from ..algorithms.algorithm import Algorithm
from ..linalg import np_conserved as npc
from ..networks.mps import MPSEnvironment
from ..networks.mpo import MPOEnvironment, MPOTrnasferMatrix
from ..linalg.lanczos import lanczos, lanczos_arpack
from ..truncation import truncate, svd_theta
from ..tools.params import asConfig
from ..tools.math import entropy
from ..tools.misc import find_subclass
from ..tools.process import memory_usage
from .mps_common import Sweep, OneSiteH, TwoSiteH

class VUMPSEngine(Sweep):
    EffectiveH = OneSiteH
    
    def __init__(self, psi, model, options, **kwargs):
        options = asConfig(options, self.__class__.__name__)
        super().__init__(psi, model, options, **kwargs)
        
        
    def run(self):
        options = self.optoins
        start_time = self.time0
        self.shelve = False
        
        min_sweeps = options.get('min_sweeps', 1)
        max_sweeps = options.get('max_sweeps', 1000)
        max_E_err = options.get('max_E_err', 1.e-8)
        max_S_err = options.get('max_S_err', 1.e-5)
        max_split_err = options.get('max_split_err', 1.e-8)
        #max_seconds = 3600 * options.get('max_hours', 24 * 365)
        E, Delta_E, Delta_S, Error = 1., 1., 1.
        E_old, S_old = np.nan, np.mean(self.psi.entanglement_entropy())  # initial dummy values
        
        is_first_sweep = True
        
        while True:
            loop_start_time = time.time()
            #Check convergence criteria
            if self.sweeps >= max_sweeps:
                break
            if (self.sweeps > min_sweeps and -Delta_E < max_E_err * max(abs(E), 1.)
                    and abs(Delta_S) < max_S_err):
                break     
            if not is_first_sweep:
                self.checkpoint.emit(self)
            # --------- the main work --------------
            logger.info('Running sweep with optimization')
            max_trunc_err = self.sweep(meas_E_trunc=True)
            max_E_trunc = np.max(self.E_trunc_list)
            self.sweep_stats['max_trunc_err'].append(max_trunc_err)
            # --------------------------------------
            # update values for checking the convergence
            entropy_bonds = self._entropy_approx
            max_S = max(entropy_bonds)
            S = sum(entropy_bonds) / len(entropy_bonds)  # mean
            Delta_S = (S - S_old) / N_sweeps_check
            S_old = S
            E = TODO
            Delta_E = (E - E_old) / N_sweeps_check
            E_old = E
            norm_err = np.linalg.norm(self.psi.norm_test())

            # update statistics
            self.sweep_stats['sweep'].append(self.sweeps)
            self.sweep_stats['N_updates'].append(len(self.update_stats['i0']))
            self.sweep_stats['E'].append(E)
            self.sweep_stats['S'].append(S)
            self.sweep_stats['max_S'].append(max_S)
            self.sweep_stats['time'].append(time.time() - start_time)
            self.sweep_stats['max_split_err'].append(max_trunc_err)
            self.sweep_stats['max_trunc_err'].append(max_trunc_err)
            self.sweep_stats['max_E_trunc'].append(max_E_trunc)
            self.sweep_stats['max_chi'].append(np.max(self.psi.chi))
            self.sweep_stats['norm_err'].append(norm_err)

            
    def update(self):
        # Update on site 
        self.logger.info('Update on site: ', self.i0)
        i0 = self.i0
        H = self.model.H_MPO
        psi = self.psi
        
        # Left and Right eigenvector of MPO Transfer matrix for the boundary between unit cells.
        E, boundary_env_data = MPOTransferMatrix.find_init_LP_RP(H, self.psi, calc_E=True) # E is already the energy density.
        env = MPOEnvironment(psi, H, psi, **boundary_env_data)
        LP1 = env.get_LP(i0, store=True) # includes site i0
        RP2 = env.get_RP(i0, store=True) # includes site i0
        LP2 = env.get_LP(i0+1, store=False) # includes site i0
        RP1 = env.get_RP(i0-1, store=False) # includes site i0
        
        H0_1 = ZeroSiteH.from_LP_RP(LP1, H.get_W(n), RP2, i0=i0-1) # ZeroSiteH on bond left of site i0
        H0_2 = ZeroSiteH.from_LP_RP(LP2, RP2, i0=i0)   # ZeroSiteH on bond right of site i0
        H1 = OneSiteH.from_LP_W0_RP(LP, W0, RP, i0=i0)
        
        lanczos_options = self.options.subconfig('lanczos_options')
        E0_1, theta0_1, N0_1 = LanczosGroundState(H0_1, psi.get_SL(i0), lanczos_options)
        E0_2, theta0_2, N0_2 = LanczosGroundState(H0_2, psi.get_SR(i0), lanczos_options)
        E1, theta1, N1 = LanczosGroundState(H1, psi.get_B(i0, 'Th'), lanczos_options)
        
    def get_sweep_schedule(self):
        """Sweep from site 0 to L-1"""
        L = self.psi.L
        if self.finite:
            raise NotImplementedError("Only infinite VUMPS is implemented")
        else:
            i0s = list(range(0, L))
            move_right = [True] * L
            update_LP_RP = [[True, False]] * L
        return zip(i0s, move_right, update_LP_RP) 