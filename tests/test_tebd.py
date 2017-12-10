"""A collection of tests to check the functionality of `tenpy.tebd`"""
from __future__ import division

import numpy.testing as npt
import tenpy.linalg.np_conserved as npc
import numpy as np
from tenpy.networks.mps import MPS
from tenpy.models.spins import SpinChain
import tenpy.algorithms.tebd as tebd
from tenpy.algorithms.exact_diag import ExactDiag
from nose.plugins.attrib import attr

from test_dmrg import e0_tranverse_ising


def test_trotter_decomposition():
    # check that the time steps sum up to what we expect
    for order in [1, 2, 4]:
        dt = tebd.Engine.suzuki_trotter_time_steps(order)
        for N in [1, 2, 5]:
            evolved = [0., 0.]
            for j, k in tebd.Engine.suzuki_trotter_decomposition(order, N):
                evolved[k] += dt[j]
            npt.assert_array_almost_equal_nulp(evolved, N * np.ones([2]), N * 2)


def check_tebd(bc_MPS='finite', g=0.5):
    L = 2 if bc_MPS == 'infinite' else 6
    #  xxz_pars = dict(L=L, Jxx=1., Jz=3., hz=0., bc_MPS=bc_MPS)
    #  M = XXZChain(xxz_pars)
    # factor of 4 (2) for J (h) to change spin-1/2 to Pauli matrices
    model_pars = dict(L=L, Jx=0., Jy=0., Jz=-4., hx=2. * g, bc_MPS=bc_MPS, conserve=None)
    M = SpinChain(model_pars)
    state = ([[1, -1.], [1, -1.]] * L)[:L]  # pointing in (-x)-direction
    psi = MPS.from_product_state(M.lat.mps_sites(), state, bc=bc_MPS)

    tebd_param = {
        'verbose': 2,
        'dt': 0.01,
        'order': 4,
        'delta_tau_list': [0.1, 1.e-4, 1.e-8, 1.e-10],
        'max_error_E': 1.e-10,
        'trunc_params': {
            'chi_max': 50,
            'trunc_cut': 1.e-13
        }
    }
    engine = tebd.Engine(psi, M, tebd_param)
    engine.run_GS()

    print "norm_test", psi.norm_test()
    if bc_MPS == 'finite':
        psi.canonical_form()
        ED = ExactDiag(M)
        ED.build_full_H_from_mpo()
        ED.full_diagonalization()
        psi_ED = ED.groundstate()
        Etebd = np.sum(M.bond_energies(psi))
        Eexact = np.min(ED.E)
        print "E_TEBD={Etebd:.14f} vs E_exact={Eex:.14f}".format(Etebd=Etebd, Eex=Eexact)
        assert (abs((Etebd - Eexact) / Eexact) < 1.e-8)
        ov = npc.inner(psi_ED, ED.mps_to_full(psi), do_conj=True)
        print "compare with ED: overlap = ", abs(ov)**2
        assert (abs(abs(ov) - 1.) < 1.e-8)

        # Test real time TEBD: should change on an eigenstate
        Sold = np.average(psi.entanglement_entropy())
        for i in range(3):
            engine.run()
        Enew = np.sum(M.bond_energies(psi))
        Snew = np.average(psi.entanglement_entropy())
        assert (abs(Enew - Etebd) < 1.e-8)
        assert (abs(Sold - Snew) < 1.e-6)  # somehow we need larger tolerance here....

    if bc_MPS == 'infinite':
        Etebd = np.average(M.bond_energies(psi))
        Eexact = e0_tranverse_ising(g)
        print "E_TEBD={Etebd:.14f} vs E_exact={Eex:.14f}".format(Etebd=Etebd, Eex=Eexact)

        Sold = np.average(psi.entanglement_entropy())
        for i in range(2):
            engine.run()
        Enew = np.average(M.bond_energies(psi))
        Snew = np.average(psi.entanglement_entropy())
        assert (abs(Etebd - Enew) < 1.e-10)
        assert (abs(Sold - Snew) < 1.e-6)  # somehow we need larger tolerance here....


@attr('slow')
def test_tebd():
    for bc_MPS in ['finite', 'infinite']:
        yield check_tebd, bc_MPS


if __name__ == "__main__":
    for f_args in test_tebd():
        f = f_args[0]
        print "=" * 80
        print ' '.join([str(a) for a in f_args])
        print "=" * 80
        f(*f_args[1:])
