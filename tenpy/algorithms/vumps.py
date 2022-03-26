# Copyright 2022 TeNPy Developers, GNU GPLv3


import numpy as np
import time
import warnings
import logging
logger = logging.getLogger(__name__)

from ..linalg import np_conserved as npc
from ..networks.mpo import MPOEnvironment, MPOTransferMatrix
from ..networks.mps import TransferMatrix
from ..linalg.lanczos import LanczosGroundState, lanczos_arpack
from ..tools.params import asConfig
from ..tools.math import entropy
from ..tools.misc import find_subclass
from ..tools.process import memory_usage
from .mps_common import Sweep, ZeroSiteH, OneSiteH, TwoSiteH
from .truncation import truncate, svd_theta
from .excitation import LT_general, TR_general, construct_orthogonal
#import sys
#sys.path.append("/home/sajant/vumps-tBLG/Nsite/")
#from misc import *
#from vumps_utils import *

__all__ = ['VUMPSEngine', 'OneSiteVUMPSEngine', 'TwoSiteVUMPSEngine']


class VUMPSEngine(Sweep):
    EffectiveH = None

    def __init__(self, psi, model, options, **kwargs):
        #options = asConfig(options, self.__class__.__name__)
        super().__init__(psi, model, options, **kwargs)
        self.allow_reduction = options.get('allow_reduction', False)
        self.guess_init_env_data = self.env.get_initialization_data()
        self.env.clear()
        self._entropy_approx = [None] * psi.L  # always left of a given site
        assert psi.L % model.H_MPO.L == 0
        self.tangent_projector_test(self.env.get_initialization_data())

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
        E_old, S_old = np.nan, np.nan  # initial dummy values
        max_split_error = 1
        is_first_sweep = True
        self.psi.valid_umps = False

        vumps_step = 0
        while True:
            print("-"*20, "vumps_step:", vumps_step, "-"*20)
            loop_start_time = time.time()
            #Check convergence criteria
            if self.sweeps >= max_sweeps:
                break
            if (self.sweeps >= min_sweeps and np.abs(Delta_E) < max_E_err * max(abs(E), 1.)
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
            #E = np.mean(self.update_stats['e_L'][-self.psi.L:] + self.update_stats['e_R'][-self.psi.L:])
            E = np.mean(self.update_stats['e_L'][-self.psi.L:] + self.update_stats['e_R'][-self.psi.L:])
            Delta_E = (E - E_old)
            E_old = E
            norm_err = np.linalg.norm(self.psi.norm_test(force=True))

            # update statistics
            self.sweep_stats['sweep'].append(self.sweeps)
            self.sweep_stats['N_updates'].append(len(self.update_stats['i0']))
            self.sweep_stats['E_theta'].append(np.mean(self.update_stats['e_theta'][-self.psi.L:]))
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

            #self.tangent_projector_test(self.env.get_initialization_data())
            #print(self.sweeps, Delta_E, norm_err, Delta_S, max_split_error, self.sweep_stats['E_theta'][-1], self.sweep_stats['E_L'][-1], self.sweep_stats['E_R'][-1], self.sweep_stats['E_C1'][-1], self.sweep_stats['E_C2'][-1])
            # status update
            logger.info(
                "checkpoint after sweep %(sweeps)d\n"
                "energy=%(E).16f, max S=%(S).16f, norm_err=%(norm_err).1e\n"
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
            vumps_step += 1

        self.psi.test_validity()
        #self.tangent_projector_test(self.env.get_initialization_data())
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
            'e_theta': [],
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
            'E_theta': [],
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

        #i0s = list(range(0, L)) + list(range(L, 0, -1))
        #move_right = [True] * L + [False] * L
        #update_LP_RP = [[False, False]] * (2*L)

        i0s = list(range(0, L))
        move_right = [True] * L
        update_LP_RP = [[False, False]] * L
        return zip(i0s, move_right, update_LP_RP)

    def prepare_update(self):
        # For each update, we need to rebuild the environments from scratch using the most recent tensors
        i0 = self.i0
        H = self.model.H_MPO
        psi = self.psi

        #boundary_env_data, Es, _ = MPOTransferMatrix.find_init_LP_RP(H, self.psi, calc_E=True, guess_init_env_data=self.guess_init_env_data) # E is already the energy density.
        #print('Converge fixed points.')
        boundary_env_data, Es, _ = MPOTransferMatrix.find_init_LP_RP(H, self.psi, calc_E=True, guess_init_env_data=self.guess_init_env_data) # E is already the energy density.
        self.env = MPOEnvironment(psi, H, psi, **boundary_env_data)
        self.transfer_matrix_energy = Es

        self.make_eff_H()
        theta = self.psi.get_theta(i0, n=self.n_optimize, cutoff=self.S_inv_cutoff) #n_optimize will be 1
        assert self.eff_H1.combine == False
        theta = self.eff_H1.combine_theta(theta) #combine should be false.
        C1, C2 = self.psi.get_C(i0), self.psi.get_C(i0+self.n_optimize)

        #print(theta.get_leg('vL').ind_len, theta.get_leg('vR').ind_len)
        #print(C1.get_leg('vL').ind_len, C1.get_leg('vR').ind_len)
        #print(C2.get_leg('vL').ind_len, C2.get_leg('vR').ind_len)

        return (theta, C1, C2)

    def make_eff_H(self):

        self.eff_H0_1 = ZeroSiteH(self.env, self.i0) # This saves more envs than optimal.
        self.eff_H0_2 = ZeroSiteH(self.env, self.i0 + self.n_optimize) # This saves more envs than optimal.
        self.eff_H1 = self.EffectiveH(self.env, self.i0, self.combine, self.move_right)

        if hasattr(self.env, 'H') and self.env.H.explicit_plus_hc:
            self.eff_H1 = SumNpcLinearOperator(self.eff_H1, self.eff_H1.adjoint())
        if hasattr(self.env, 'H') and self.env.H.explicit_plus_hc:
            self.eff_H0_1 = SumNpcLinearOperator(self.eff_H0_1, self.eff_H0_1.adjoint())
        if hasattr(self.env, 'H') and self.env.H.explicit_plus_hc:
            self.eff_H0_2 = SumNpcLinearOperator(self.eff_H0_2, self.eff_H0_2.adjoint())

    def _wrap_ortho_eff_H(self):
        raise NotImplementedError("Do we want this for VUMPS?")


    def update_env(self, **update_data):
        # Get guesses for the next LP and RP
        # TODO: Legs currently incompatible since we are reducing chi
        self.guess_init_env_data = None#self.env.get_initialization_data()
        pass

    def post_update_local(self, e_L, e_R, eps_L, eps_R, e_C1, e_C2, e_theta, N0_L, N0_R, N1, **update_data):
        self.update_stats['i0'].append(self.i0)
        self.update_stats['e_L'].append(e_L)
        self.update_stats['e_R'].append(e_R)
        self.update_stats['e_C1'].append(e_C1)
        self.update_stats['e_C2'].append(e_C2)
        self.update_stats['e_theta'].append(e_theta)
        self.update_stats['N_lanczos'].append(N0_L + N0_R + N1)
        self.update_stats['split_err_L'].append(eps_L)
        self.update_stats['split_err_R'].append(eps_R)

    def free_no_longer_needed_envs(self):
        for env in self._all_envs:
            env.clear() # Can we do better?

    def resume_run(self):
        raise NotImplementedError("TODO")

    def tangent_projector_test(self, env_data):
        LW = env_data['init_LP']
        RW = env_data['init_RP']

        VLs = [construct_orthogonal(self.psi.get_B(i, form='AL')) for i in range(self.psi.L)]
        VRs = [construct_orthogonal(self.psi.get_B(i, form='AR'), left=False) for i in range(self.psi.L)]
        ALs = self.psi._AL
        ARs = self.psi._AR
        ACs = self.psi._AC
        Ws = self.model.H_MPO._W
        strange_left = []
        strange_right = []
        for i in range(self.psi.L):
            temp_L = LT_general(ALs[:i], ALs[:i], LW, Ws=Ws[:i])
            temp_R = TR_general(ARs[i+1:], ARs[i+1:], RW, Ws=Ws[i+1:])

            temp_VL = LT_general([VLs[i]], [ACs[i]], temp_L, Ws=[Ws[i]])
            temp_VL = npc.tensordot(temp_VL, temp_R, axes=(['wR', 'vR*'], ['wL', 'vL*']))

            temp_VR = TR_general([VRs[i]], [ACs[i]], temp_R, Ws=[Ws[i]])
            temp_VR = npc.tensordot(temp_L, temp_VR, axes=(['wR', 'vR*'], ['wL', 'vL*']))

            strange_left.append(npc.norm(temp_VL))
            strange_right.append(npc.norm(temp_VR))
        print('Strange Cancellation left:', strange_left, "right:", strange_right)

        return strange_left, strange_right

    def _diagonal_gauge_C(self, theta, i0):
        U, S, VH = npc.svd(theta,
                           cutoff=self.S_inv_cutoff,
                           qtotal_LR=[theta.qtotal, None],
                           inner_labels=['vR', 'vL'])


        #print('S norm', np.linalg.norm(S))
        theta = npc.diag(S, VH.get_leg('vL'), labels=['vL', 'vR'])

        self.psi.set_B(i0-1, npc.tensordot(self.psi.get_B(i0-1, 'AL'), U, axes=(['vR'], ['vL'])), 'AL')
        self.psi.set_B(i0, npc.tensordot(U.conj(), self.psi.get_B(i0, 'AL'), axes=(['vL*'], ['vL'])).ireplace_label('vR*', 'vL'), 'AL')

        self.psi.set_B(i0, npc.tensordot(VH, self.psi.get_B(i0, 'AR'), axes=(['vR'], ['vL'])), 'AR')
        self.psi.set_B(i0-1, npc.tensordot(self.psi.get_B(i0-1, 'AR'), VH.conj(), axes=(['vR'], ['vR*'])).ireplace_label('vL*', 'vR'), 'AR')
        return theta, U, VH

    def _diagonal_gauge_AC(self, U, VH, i0):
        theta = self.psi.get_B(i0+1, 'AC')
        #C = self.psi.get_C(i0+1)
        theta = npc.tensordot(U.conj(), theta, axes=(['vL*'], ['vL'])).ireplace_label('vR*', 'vL')
        #C = npc.tensordot(U.conj(), C, axes=(['vL*'], ['vL'])).ireplace_label('vR*', 'vL')
        self.psi.set_B(i0+1, theta, 'AC')
        #self.psi.set_C(i0+1, C)

        theta = self.psi.get_B(i0-1, 'AC')
        #C = self.psi.get_C(i0-1)
        theta = npc.tensordot(theta, VH.conj(), axes=(['vR'], ['vR*'])).ireplace_label('vL*', 'vR')
        #C = npc.tensordot(C, VH.conj(), axes=(['vR'], ['vR*'])).ireplace_label('vL*', 'vR')
        self.psi.set_B(i0-1, theta, 'AC')
        #self.psi.set_C(i0-1, C)

class OneSiteVUMPSEngine(VUMPSEngine):
    EffectiveH = OneSiteH

    def __init__(self, psi, model, options, **kwargs):
        #options = asConfig(options, self.__class__.__name__)
        super().__init__(psi, model, options, **kwargs)

    def update_local(self, theta, **kwargs):

        TM = TransferMatrix(self.psi, self.psi, charge_sector=None, form='A')
        ov, _ = TM.eigenvectors()
#          print('T_{A:A} eigval_1:', ov)

        TM = TransferMatrix(self.psi, self.psi, charge_sector=None, form='B')
        ov, _ = TM.eigenvectors()
#          print('T_{B:B} eigval_1:', ov)

        # Update on site
        psi = self.psi
        i0 = self.i0
        H0_1, H0_2, H1 = self.eff_H0_1, self.eff_H0_2, self.eff_H1
        AC, C1, C2 = theta
#          print('Lanczos time')
        lanczos_options = self.options.subconfig('lanczos_options')
        E0_1, theta0_1, N0_1 = LanczosGroundState(H0_1, C1, lanczos_options).run()
        if self.psi.L > 1:
            E0_2, theta0_2, N0_2 = LanczosGroundState(H0_2, C2, lanczos_options).run()
            #E0_2 -= self.
        E1, theta1, N1 = LanczosGroundState(H1, AC, lanczos_options).run()

#          print(theta1.get_leg('vL').ind_len, theta1.get_leg('vR').ind_len)

        """
        #print(npc.norm(theta0_1 - theta0_2))
        print(theta1.get_leg('vL').ind_len, theta1.get_leg('vR').ind_len)
        U, S, VH = npc.svd(theta0_1,
                           cutoff=self.S_inv_cutoff,
                           qtotal_LR=[theta0_1.qtotal, None],
                           inner_labels=['vR', 'vL'])


        #mask = S>1.e-15
        #U.iproject(mask, 'vR')
        #VH.iproject(mask, 'vL')

        #print(npc.norm(npc.tensordot(U.iscale_axis(S[mask]), VH, axes=['vR', 'vL']) - theta0_1))
        theta0_1 = npc.diag(S, VH.get_leg('vL'), labels=['vL', 'vR'])

        psi.set_B(i0-1, npc.tensordot(psi.get_B(i0-1, 'AL'), U, axes=(['vR'], ['vL'])), 'AL')
        psi.set_B(i0, npc.tensordot(U.conj(), psi.get_B(i0, 'AL'), axes=(['vL*'], ['vL'])).ireplace_label('vR*', 'vL'), 'AL')

        psi.set_B(i0, npc.tensordot(VH, psi.get_B(i0, 'AR'), axes=(['vR'], ['vL'])), 'AR')
        psi.set_B(i0-1, npc.tensordot(psi.get_B(i0-1, 'AR'), VH.conj(), axes=(['vR'], ['vR*'])).ireplace_label('vL*', 'vR'), 'AR')
        """
        if self.allow_reduction:
            theta0_1, U1, VH1 = self._diagonal_gauge_C(theta0_1, i0)

            theta1 = npc.tensordot(U1.conj(), theta1, axes=(['vL*'], ['vL']))

            TM = TransferMatrix(self.psi, self.psi, charge_sector=None, form='A')
            ov, _ = TM.eigenvectors()
#              print('T_{A:A} eigval_1:', ov)

            TM = TransferMatrix(self.psi, self.psi, charge_sector=None, form='B')
            ov, _ = TM.eigenvectors()
#              print('T_{B:B} eigval_1:', ov)


            if self.psi.L == 1:
                theta1 = npc.tensordot(theta1, VH1.conj(), axes=(['vR'], ['vR*'])).ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                E0_2, theta0_2, N0_2 = E0_1, theta0_1, N0_1
            else: # self.psi.L > 1

                theta0_2, U2, VH2 = self._diagonal_gauge_C(theta0_2, i0+1)

                """
                U, S, VH = npc.svd(theta0_2,
                                   cutoff=self.S_inv_cutoff,
                                   qtotal_LR=[theta0_2.qtotal, None],
                                   inner_labels=['vR', 'vL'])

                #mask = S>1.e-14
                #U.iproject(mask, 'vR')
                #VH.iproject(mask, 'vL')

                #print(npc.norm(npc.tensordot(U.iscale_axis(S[mask]), VH, axes=['vR', 'vL']) - theta0_1))
                theta0_2 = npc.diag(S, VH.get_leg('vL'), labels=['vL', 'vR'])
                print(psi.get_B(0, 'AL'))
                print(psi.get_B(0, 'AR'))
                psi.set_B(i0, npc.tensordot(psi.get_B(i0, 'AL'), U, axes=(['vR'], ['vL'])), 'AL')
                psi.set_B(i0+1, npc.tensordot(U.conj(), psi.get_B(i0+1, 'AL'), axes=(['vL*'], ['vL'])).ireplace_label('vR*', 'vL'), 'AL')

                psi.set_B(i0+1, npc.tensordot(VH, psi.get_B(i0+1, 'AR'), axes=(['vR'], ['vL'])), 'AR')
                psi.set_B(i0, npc.tensordot(psi.get_B(i0, 'AR'), VH.conj(), axes=(['vR'], ['vR*'])).ireplace_label('vL*', 'vR'), 'AR')
                print(psi.get_B(0, 'AL'))
                print(psi.get_B(0, 'AR'))
                print('C2 SVs:', np.sum(mask))
                """
                theta1 = npc.tensordot(theta1, VH2.conj(), axes=(['vR'], ['vR*'])).ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])

#                  TM = TransferMatrix(self.psi, self.psi, charge_sector=None, form='A')
#                  ov, _ = TM.eigenvectors()
#                  print('T_{A:A} eigval_1:', ov)
#
#                  TM = TransferMatrix(self.psi, self.psi, charge_sector=None, form='B')
#                  ov, _ = TM.eigenvectors()
#                  print('T_{B:B} eigval_1:', ov)

                self._diagonal_gauge_AC(U2, VH1, i0)
        else:
            if self.psi.L == 1:
                E0_2, theta0_2, N0_2 = E0_1, theta0_1, N0_1

#          print(theta1.get_leg('vL').ind_len, theta1.get_leg('vR').ind_len)
#          print(theta0_1.get_leg('vL').ind_len, theta0_1.get_leg('vR').ind_len)
#          print(theta0_2.get_leg('vL').ind_len, theta0_2.get_leg('vR').ind_len)
        TM = TransferMatrix(self.psi, self.psi, charge_sector=None, form='A')
        ov, _ = TM.eigenvectors()
#          print('T_{A:A} eigval_1:', ov)

        TM = TransferMatrix(self.psi, self.psi, charge_sector=None, form='B')
        ov, _ = TM.eigenvectors()
#          print('T_{B:B} eigval_1:', ov)


        theta1.ireplace_label('p0', 'p')
        psi.set_C(i0, theta0_1)
        psi.set_C(i0+1, theta0_2)
        psi.set_B(i0, theta1, form='AC')
        AL, AR, eps_L, eps_R, entropy_1, entropy_2 = self.polar_max(theta1, theta0_1, theta0_2)
        psi.set_B(i0, AL, form='AL')
        psi.set_B(i0, AR, form='AR')
        self._entropy_approx[i0 % self.psi.L] = entropy_1
        self._entropy_approx[(i0+self.n_optimize) % self.psi.L] = entropy_2

#          print('Site:', i0)
#          print(self.psi.get_B(i0, 'AL').get_leg('vL').ind_len, self.psi.get_B(i0, 'AL').get_leg('vR').ind_len)
#          print(self.psi.get_B(i0, 'AR').get_leg('vL').ind_len, self.psi.get_B(i0, 'AR').get_leg('vR').ind_len)
#          print(self.psi.get_B(i0, 'AC').get_leg('vL').ind_len, self.psi.get_B(i0, 'AC').get_leg('vR').ind_len)
#          print(self.psi.get_C(i0).get_leg('vL').ind_len, self.psi.get_C(i0).get_leg('vR').ind_len)
#          print('Site:', i0+1)
#          print(self.psi.get_B(i0+1, 'AL').get_leg('vL').ind_len, self.psi.get_B(i0+1, 'AL').get_leg('vR').ind_len)
#          print(self.psi.get_B(i0+1, 'AR').get_leg('vL').ind_len, self.psi.get_B(i0+1, 'AR').get_leg('vR').ind_len)
#          print(self.psi.get_B(i0+1, 'AC').get_leg('vL').ind_len, self.psi.get_B(i0+1, 'AC').get_leg('vR').ind_len)
#          print(self.psi.get_C(i0+1).get_leg('vL').ind_len, self.psi.get_C(i0+1).get_leg('vR').ind_len)

#          TM = TransferMatrix(self.psi, self.psi, charge_sector=None, form='A')
#          ov, _ = TM.eigenvectors()
#          print('T_{A:A} eigval_1:', ov)
#
#          TM = TransferMatrix(self.psi, self.psi, charge_sector=None, form='B')
#          ov, _ = TM.eigenvectors()
#          print('T_{B:B} eigval_1:', ov)

        update_data = {
            'e_L': self.transfer_matrix_energy[1],
            'e_R': self.transfer_matrix_energy[0],
            'eps_L': eps_L,
            'eps_R': eps_R,
            'e_C1': E0_1,
            'e_C2': E0_2,
            'e_theta': E1,
            'N0_L': N0_1,
            'N0_R': N0_2,
            'N1': N1
        }

        self.trunc_err_list.append(0)

        print("one-site vumps:")
        print("e_L          :", update_data['e_L'])
        print("e_R          :", update_data['e_R'])
        print("eps_L        :", eps_L)
        print("eps_R        :", eps_R)
        print("AC,C1,C2 iter:", N1, N0_1, N0_2)
        print("AC,C1,C2 eigv:", E1, E0_1, E0_2)
        print("AL chi       :", AL.get_leg('vL').ind_len, AL.get_leg('vR').ind_len)


        return update_data

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

class TwoSiteVUMPSEngine(VUMPSEngine):
    EffectiveH = TwoSiteH

    def __init__(self, psi, model, options, **kwargs):
        #options = asConfig(options, self.__class__.__name__)
        super().__init__(psi, model, options, **kwargs)
        assert self.psi.L > 1, 'Two-site methods require a two-site unit cell.'

    def update_local(self, theta, **kwargs):
        # Update on site
        psi = self.psi
        i0 = self.i0
        H0_1, H0_2, H2 = self.eff_H0_1, self.eff_H0_2, self.eff_H1
        AC, C1, C2 = theta

        lanczos_options = self.options.subconfig('lanczos_options')
        E0_1, theta0_1, N0_1 = LanczosGroundState(H0_1, C1, lanczos_options).run()
        E0_2, theta0_2, N0_2 = LanczosGroundState(H0_2, C2, lanczos_options).run()
        E2, theta2, N2 = LanczosGroundState(H2, AC, lanczos_options).run()

        #theta2.ireplace_label(['p0' ,'p1'], ['p', )
        U, S, VH, err, S_approx = svd_theta(theta2.combine_legs([['vL', 'p0'], ['vR', 'p1']], qconj=[+1, -1]),
                                     self.trunc_params,
                                     qtotal_LR=[theta2.qtotal, None],
                                     inner_labels=['vR', 'vL'])
        AL1 = U.split_legs().ireplace_label('p0', 'p')
        AR2 = VH.split_legs().ireplace_label('p1', 'p')

        #print(AL1, AR2, S)
        S = npc.diag(S, AL1.get_leg('vR').conj(), labels=['vL', 'vR'])

        AC1 = npc.tensordot(AL1, S, axes=['vR', 'vL'])
        AC2 = npc.tensordot(S, AR2, axes=['vR', 'vL'])

        psi.set_C(i0, theta0_1)
        psi.set_C(i0+2, theta0_2)
        psi.set_C(i0+1, S)
        psi.set_B(i0, AL1, form='AL')
        psi.set_B(i0+1, AR2, form='AR')
        psi.set_B(i0, AC1, form='AC')
        psi.set_B(i0+1, AC2, form='AC')

        AL2, AR1, eps_L, eps_R, entropy_1, entropy_2 = self.polar_max(AC1, AC2, theta0_1, theta0_2)
        psi.set_B(i0, AR1, form='AR')
        psi.set_B(i0+1, AL2, form='AL')


        self._entropy_approx[i0 % self.psi.L] = entropy_1
        self._entropy_approx[(i0+1) % self.psi.L] = entropy(S_approx**2, n=1)
        self._entropy_approx[(i0+2) % self.psi.L] = entropy_2
        update_data = {
            'e_L': self.transfer_matrix_energy[1],
            'e_R': self.transfer_matrix_energy[0],
            'eps_L': eps_L,
            'eps_R': eps_R,
            'e_C1': E0_1,
            'e_C2': E0_2,
            'e_theta': E2,
            'N0_L': N0_1,
            'N0_R': N0_2,
            'N1': N2
        }
        print("two-site vumps:")
        print("e_L          :", update_data['e_L'])
        print("e_R          :", update_data['e_R'])
        print("eps_L        :", eps_L)
        print("eps_R        :", eps_R)
        print("lanczos iter :", N2)
        print("AC chi       :", AC1.get_leg('vL').ind_len, AC1.get_leg('vR').ind_len)


        self.trunc_err_list.append(err.eps)

        return update_data

    def polar_max(self, AC1, AC2, C1, C3):
        # Given AC and C, find AL such that AL C = AC

        U_ACL, _, _ = npc.polar(AC2.combine_legs(['vL', 'p'], qconj=[+1]), left=False)
        U_CL, _, s1 = npc.polar(C3, left=False)
        AL2 = npc.tensordot(U_ACL.split_legs(), U_CL.conj(), axes=(['vR'], ['vR*'])).replace_label('vL*', 'vR')

        U_ACR, _, _ = npc.polar(AC1.combine_legs(['p', 'vR'], qconj=[+1]), left=True)
        U_CR, _, s2 = npc.polar(C1, left=True)
        AR1 = npc.tensordot(U_CR.conj(), U_ACR.split_legs(), axes=(['vL*'], ['vL'])).replace_label('vR*', 'vL')

        eps_L = npc.norm(AC2 - npc.tensordot(AL2, C3, axes=['vR', 'vL']))
        eps_R = npc.norm(AC1 - npc.tensordot(C1, AR1, axes=['vR', 'vL']))

        entropy_left = entropy(s1**2, n=1)
        entropy_right = entropy(s2**2, n=1)

        return AL2, AR1, eps_L, eps_R, entropy_left, entropy_right
