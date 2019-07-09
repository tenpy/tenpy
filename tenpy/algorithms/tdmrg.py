r"""Time-dependent DMRG (tDMRG).

The summary of tDMRG method goes here.

"""
# Copyright 2019 TeNPy Developers
from tqdm import trange  # trange = range with waitbar

import numpy as np
import time
import copy

from tenpy.tools.params import get_parameter
from tenpy.tools.process import memory_usage
from tenpy.networks.mps import MPSEnvironment

# python 3.7 or above
#from tenpy.soshi.spectra_data import Crt, Srt, Xrt
#from tenpy.soshi.postprocess import FourierTransform


class Engine():
    """Time-dependent DMRG(tDMRG) 'engine'.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        the ground state of the model. It usually needs to be computed with DMRG in advance.
    model : :class:`~tenpy.models.MPOModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    tdmrg_params : dict
        macro params controlling tDMRG algorithm
        ============== ========= ===============================================================
        key            type      description
        ============== ========= ===============================================================
        Nt             int       Number of measurements for time-dependent correlation functions

        -------------- --------- ---------------------------------------------------------------
        E0             float     groundstate energy (energy of psi), which needs to be computed
                                 with DMRG in advance.
        -------------- --------- ---------------------------------------------------------------
        opA            operator  tDMRG will compute <psi|opA(r,t)opB(0,0)|psi>.
        opB                        (ex) opA=Sp, opB=Sm to compute <psi|S^+(r,t)S^-(0,0)|psi>
        ============== ========= ===============================================================

    tevol_params : dict
        micro params controlling the time-evolution during tDMRG. Need to be valid parameters for the tevol-method used.

    tevol_method : algorithm
        time-evol algorithm. Currently {tebd, tdvp1, tdvp2} are supported options.

    Attributes
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        The groundstate MPS, not changed after initialization.

    phi_t: :class:`~tenpy.networks.mps.MPS`
        The time-dependent MPS, |phi(t)>=e^{-iHt}opB(0,0)|psi>

    tDMRG_results: dict
        A dictionary with keys ``'Crt', 'Xrt', 'Srt'``
    """

    # tevol_method={tebd, tdvp1, tdvp2}
    def __init__(self, psi, model, tdmrg_params, tevol_params, tevol_method):

        self.psi = psi
        self.model = model
        self.tdmrg_params = tdmrg_params
        self.tevol_params = tevol_params
        self.tevol_method = tevol_method

        self.L = psi.L
        self.Nt = get_parameter(self.tdmrg_params, 'Nt', 50, 'tDMRG')
        self.E0 = get_parameter(self.tdmrg_params, 'E0', {}, 'tDMRG')
        self.opA = get_parameter(self.tdmrg_params, 'opA', {}, 'tDMRG')
        self.opB = get_parameter(self.tdmrg_params, 'opB', {}, 'tDMRG')

    # =====main function================
    def run(self):
        """run tDMRG calculation. C(r,t)

        Returns
        -------
        tDMRG_results: dict
            A dictionary with keys ``'Crt', 'Xrt', 'Srt'``
        """

        # setup tdmrg
        self._prepare_tdmrg()

        # =====run tDMRG=================
        # measurement at t=0
        self._measure()

        for nt in trange(self.Nt, desc='tDMRG'):  # trange=range + waitbar
            # evolve phi_t by 1 step (dt_measure=dt*N_steps)
            self.tevol.run()

            # measurement at t
            self._measure()

        self._store_result()
        # ===============================

        return self.tDMRG_result

    # =====helper functions================

    def _set_phi_t(self):
        """phi_t=opB(r=i0)|psi>, where i0 is a center site i0=L/2"""

        i0 = self.psi.L // 2
        phi_t = copy.deepcopy(self.psi)
        phi_t.apply_local_op(i0, self.opB, unitary=False)

        self.phi_t = phi_t

    def _set_tevol(self):
        """set time-evolution (TEBD) engine"""

        self.tevol = self.tevol_method.Engine(self.phi_t, self.model, self.tevol_params)

    def _prepare_tdmrg(self):
        """all preparation for tdmrg"""
        self._set_phi_t()
        self._set_tevol()

        self.t_list = []
        self.C_rt = []  # <opA(r, t) opB(0, 0)>
        self.S_rt = []  # S^{ent}(r, t)
        self.X_rt = []  # Chi(r, t)

        self.tevol_stats = {'time': [], 'memory': []}
        self.last_time = time.time()
        #self.tevol_stats['memory'].append(memory_usage())
        #self.tevol_stats['time'].append(self.now_time)

    # =========================
    def _update_env(self):
        """update environment (psi, phi_t)"""
        self.env = MPSEnvironment(self.psi, self.phi_t)

    def _get_Crt(self):
        """compute corr function, C(r,t)= env(opA)= <psi| opA| phi_t>"""

        return self.env.expectation_value(self.opA) * self.phase

    def _get_Srt(self):
        """compute Entanglement Entropy at each bond, S(r,t)"""

        return self.phi_t.entanglement_entropy()

    def _get_Xrt(self):
        """compute bond dimension at each bond, X(r,t)"""

        return self.phi_t.chi

    def _measure(self):
        """update env and do measurements at each t. """
        self._update_env()
        t = self.tevol.evolved_time
        self.phase = np.exp(1j * self.E0 * t)
        self.t_list.append(t)
        self.C_rt.append(self._get_Crt())
        self.S_rt.append(self._get_Srt())
        self.X_rt.append(self._get_Xrt())

        self.tevol_stats['memory'].append(memory_usage())
        self.now_time = time.time()
        self.tevol_stats['time'].append(self.now_time - self.last_time)
        self.last_time = self.now_time

    def _store_result(self):
        """store all tDMRG result"""
        tevol_name = (self.tevol_method.__name__).split('.')[-1]  # extract tevol name from tevol object
        r_list = np.arange(self.L) - int(self.L / 2)
        t_list = np.asarray(self.t_list)
        C_rt = np.asarray(self.C_rt)
        S_rt = np.asarray(self.S_rt)
        X_rt = np.asarray(self.X_rt)

        tDMRG_result = dict()
        tDMRG_result['tevol_name'] = tevol_name.upper()  # capitalize all letters
        tDMRG_result['tevol_stats'] = self.tevol_stats
        
        tDMRG_result['Crt'] = C_rt  # corr function
        tDMRG_result['Srt'] = S_rt  # entanglement entropy
        tDMRG_result['Xrt'] = X_rt  # bond dim
        tDMRG_result['r'] = r_list
        tDMRG_result['t'] = t_list
        
#        #for python 3.7 (@dataclass) with (spectra_data.py, postprocess.py)
#        tDMRG_result['Crt'] = Crt(C_rt, r_list, t_list)
#        tDMRG_result['Srt'] = Srt(S_rt, r_list[:-1], t_list)
#        tDMRG_result['Xrt'] = Xrt(X_rt, r_list[:-1], t_list)
#        tDMRG_result['Sqt'], tDMRG_result['Sqw'] = FourierTransform.Crt_to_Sqt_to_Sqw(
#            tDMRG_result['Crt'])

        self.tDMRG_result = tDMRG_result
