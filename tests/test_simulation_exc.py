import tenpy.linalg.np_conserved as npc
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms.exact_diag import ExactDiag
import tenpy
import pytest
import numpy as np


tenpy.tools.misc.skip_logging_setup = True  # skip logging setup

params = [
    ({}, 1),
    ({'switch_charge_sector': [1]}, 1),
    ({'apply_local_op': dict(i=3, op='Sx')}, 1),
    ({'switch_charge_sector': [1]}, 2),
]


@pytest.mark.slow
@pytest.mark.parametrize("switch, group", params)
@pytest.mark.filterwarnings('ignore:divide by zero encountered in scalar divide:RuntimeWarning')
def test_OrthogonalExcitations(tmp_path, switch, group, eps=1.e-10, N_exc=3):
    # checks ground state and 2 excited states (in same symmetry sector) for a small system
    # (without truncation)
    L, g = 8, 1.2
    assert L % 2 == 0 # to ensure ground state is charge sector [0]
    # Z2 charge!
    model_params = dict(L=L, J=1., g=g, bc_MPS='finite', conserve='parity', sort_charge=True)
    M = TFIChain(model_params)

    # get exact diagonalization reference excitations
    ED = ExactDiag(M)
    ED.build_full_H_from_mpo()
    ED.full_diagonalization()
    # Note: energies sorted by chargesector and then within charge sector -> perfect for comparison
    assert np.argmin(ED.E) == 0
    E_ED_gs = ED.E[0]  # finite size gap ensures 0 is true ground state!
    psi_ED_gs = ED.V.take_slice(0, 'ps*')
    print(f"ED gives ground state E = {E_ED_gs:.10f}")

    charge_exc = [1] if switch else [0]
    print(f"charge sector for excitations: ", charge_exc)
    leg = ED.V.get_leg('ps*')
    qi = list(leg.charges[:, 0]).index(charge_exc[0])
    i = leg.slices[qi] + (0 if switch else 1)
    E_ED_exc = ED.E[i:i + N_exc] - E_ED_gs
    psi_ED_exc = [ED.V.take_slice(i, 'ps*') for i in range(i, i + N_exc)]
    print("E_ED_exc =", E_ED_exc)
    assert all([all(psi.qtotal == charge_exc) for psi in psi_ED_exc])

    # find ground state with first simulation run
    fn_gs = 'finite_ground_state.pkl'
    sim_params_gs = {
        'simulation_class': 'GroundStateSearch',
        'directory': tmp_path,
        'output_filename': 'finite_ground_state.pkl',
        'model_class': M.__class__.__name__,
        'model_params': model_params,
        'initial_state_params': {'method': 'lat_product_state',
                                 'product_state' : [['up']]},
        'algorithm_params': {'trunc_params': {'chi_max': 30, 'svd_min': 1.e-8}},
        'group_sites': group,
    }
    data_gs = tenpy.run_simulation(**sim_params_gs)
    E_DMRG_gs = data_gs['energy']
    print(f"DMRG gives ground state E = {E_DMRG_gs:.10f}")
    assert abs(E_DMRG_gs - E_ED_gs) < eps

    ov = npc.inner(psi_ED_gs, ED.mps_to_full(data_gs['psi']), 'range', do_conj=True)
    assert abs(abs(ov) - 1.) < eps  # unique groundstate: finite size gap!

    sim_params_exc = {
        'simulation_class': 'OrthogonalExcitations',
        'directory': tmp_path,
        'ground_state_filename': fn_gs,
        'N_excitations': N_exc,
        'algorithm_params':  {'trunc_params': {'chi_max': 30, 'svd_min': 1.e-8},
                              #  'diag_method': 'ED_block'
                              },

    }
    sim_params_exc.update(**switch)

    # run excitations
    data_exc = tenpy.run_simulation(**sim_params_exc)

    E_DMRG_exc = data_exc['excitation_energies']
    ovs = [npc.inner(psi_ED, ED.mps_to_full(psi_DMRG), 'range', do_conj=True)
           for psi_ED, psi_DMRG in zip(psi_ED_exc, data_exc['excitations'])]
    for E_DMRG, E_ED, ov in zip(E_DMRG_exc, E_ED_exc, ovs):
        print(f"E_ED={E_ED:.10f}, E_DMRG={E_DMRG:.10f}: 1-|<psi_ED|psi_DMRG>|={1-abs(ov):.3e}")
    np.testing.assert_allclose(E_DMRG_exc, E_ED_exc)
    #  assert False


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_OrthogonalExcitations(tmpdir, {'switch_charge_sector': [1]}, 2)
