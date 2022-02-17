"""Examples using segment boundary conditions

This code shows the general structure of DMRG with segment boundary conditions, which allows
to find topologically non-trivial excitations on a "finite" segment between two different
degenerate ground states.

For production, you should probalby use the
:class:`~tenpy.simulations.GroundStateSearch.OrthogonalExcitations` class,
but this example might be helpful to see the general idea.
"""
# Copyright 2022 TeNPy Developers, GNU GPLv3

import numpy as np

from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import matplotlib.pyplot as plt
from tenpy.tools.params import Config

import tenpy.linalg.np_conserved as npc


def calc_infinite_groundstates(dmrg_params, g=0.1):
    L = 2
    model_params = dict(L=L, J=1., g=g, bc_MPS='infinite', conserve=None, verbose=0)

    model = TFIChain(model_params)
    plus_x = np.array([1., 1.]) / np.sqrt(2)
    minus_x = np.array([1., -1.]) / np.sqrt(2)
    psi_plus = MPS.from_product_state(model.lat.mps_sites(), [plus_x] * L, model.lat.bc_MPS)
    psi_minus = MPS.from_product_state(model.lat.mps_sites(), [minus_x] * L, model.lat.bc_MPS)

    engine_plus = dmrg.TwoSiteDMRGEngine(psi_plus, model, dmrg_params)
    engine_plus.run()
    print("<Sx> =", psi_plus.expectation_value("Sigmax"))
    engine_minus = dmrg.TwoSiteDMRGEngine(psi_minus, model, dmrg_params)
    engine_minus.run()
    print("<Sx> =", psi_minus.expectation_value("Sigmax"))

    data_plus = {'psi': psi_plus}
    data_plus.update(**engine_plus.env.get_initialization_data())
    data_minus = {'psi': psi_minus}
    data_minus.update(**engine_minus.env.get_initialization_data())

    return model, data_plus, data_minus


def prepare_segment(model, data_L, data_R, repeat_L=20, repeat_R=20):

    psi_L = data_L['psi'].copy()
    psi_R = data_R['psi'].copy()
    psi_L.convert_form("B")
    psi_R.convert_form("B")
    psi_L.enlarge_mps_unit_cell(repeat_L)
    psi_R.enlarge_mps_unit_cell(repeat_R)
    psi_L.bc = "segment"
    psi_R.bc = "segment"

    model.enlarge_mps_unit_cell(repeat_L + repeat_R)
    model.lat.bc_MPS = "segment"
    model.H_MPO.bc = "segment"

    init_env_data = {
        'init_LP': data_L['init_LP'],
        'age_LP': data_L['age_LP'],
        'init_RP': data_R['init_RP'],
        'age_RP': data_R['age_RP'],
    }
    Bs_L = [psi_L.get_B(i) for i in range(psi_L.L)]
    Bs_R = [psi_R.get_B(i) for i in range(psi_R.L)]

    joint = npc.Array.from_func(np.ones,
                                [Bs_L[-1].get_leg('vR').conj(), Bs_R[0].get_leg('vL').conj()],
                                psi_L.dtype,
                                qtotal=None,
                                labels=['vL', 'vR'])
    Bs_L[-1] = npc.tensordot(Bs_L[-1], joint, axes=['vR', 'vL'])
    S = psi_L._S[:-1] + psi_R._S

    psi = MPS(psi_L.sites + psi_R.sites, Bs_L + Bs_R, S, 'segment')
    # UL, UR = psi.canonical_form_finite()

    return psi, model, init_env_data


def calc_segment_groundstate(psi, model, dmrg_params):
    engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E, psi = engine.run()
    return psi


def plot(psi, filename):
    x = np.arange(psi.L)
    meas = psi.expectation_value("Sigmax")
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(x, meas)
    ax.set_ylabel("<sigma_x>")
    ax.set_xlabel('$x$')
    plt.savefig(filename)
    print("saved to " + filename)


if __name__ == "__main__":
    dmrg_params = Config(
        {
            'trunc_params': {
                'chi_max': 50,
                'svd_min': 1.e-10,
                'trunc_cut': None
            },
            'update_env': 0,
            'start_env': 2,
            'max_E_err': 1.e-6,
            'max_S_err': 1.e-5,
            'max_sweeps': 100,
            'verbose': 1,
            'mixer': False
        }, "DMRG")

    model, psi_plus, psi_minus = calc_infinite_groundstates(dmrg_params)
    psi, model, init_env_data = prepare_segment(model, psi_plus, psi_minus)
    dmrg_params['init_env_data'] = init_env_data
    results = calc_segment_groundstate(psi, model, dmrg_params)
    plot(results, 'tfi_segment.pdf')
