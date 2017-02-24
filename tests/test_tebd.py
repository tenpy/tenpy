"""A collection of tests to check the functionality of `tenpy.tebd`"""
from __future__ import division

import numpy.testing as npt
import tenpy.linalg.np_conserved as npc
import numpy as np
from tenpy.networks.mps import MPS
from tenpy.models.xxz_chain import XXZChain
import tenpy.algorithms.tebd as tebd
from tenpy.algorithms.exact_diag import ExactDiag
from nose.plugins.attrib import attr


def test_trotter_decomposition():
    # check that the time steps sum up to what we expect
    for order in [1, 2, 4]:
        dt = tebd.Engine.suzuki_trotter_time_steps(order)
        for N in [1, 2, 5]:
            evolved = [0., 0.]
            for j, k in tebd.Engine.suzuki_trotter_decomposition(order, N):
                evolved[k] += dt[j]
            npt.assert_array_almost_equal_nulp(evolved, N * np.ones([2]), N * 2)


def check_tebd(bc_MPS='finite'):
    L = 2 if bc_MPS == 'infinite' else 6
    xxz_pars = dict(L=L, Jxx=1., Jz=3., hz=0., bc_MPS=bc_MPS)
    M = XXZChain(xxz_pars)
    state = ([0, 1] * L)[:L]  # Neel
    psi = MPS.from_product_state(M.lat.mps_sites(), state, bc=bc_MPS)

    tebd_param = {'verbose': 2, 'chi_max': 50, 'dt': 0.1, 'order': 4,
                  'delta_tau_list': [1., 0.1, 1.e-4, 1.e-8, 1.e-12]}
    engine = tebd.Engine(psi, M, tebd_param)
    engine.run_GS()

    if bc_MPS == 'finite':
        ED = ExactDiag(M)
        ED.build_full_H_from_mpo()
        ED.full_diagonalization()
        psi_ED = ED.groundstate()
        ov = npc.inner(psi_ED, ED.mps_to_full(psi), do_conj=True)
        print "compare with ED: overlap = ", abs(ov)**2

        # Test real time TEBD
        Eold = np.average(M.bond_energies(psi))
        Sold = np.average(psi.entanglement_entropy())
        for i in range(3):
            engine.run()
        Enew = np.average(M.bond_energies(psi))
        Snew = np.average(psi.entanglement_entropy())
        assert (abs(abs(ov) - 1.) < 1.e-10)
        assert (abs(Eold - Enew) < 1.e-10)
        # TODO: why does the test below fail??
        # assert (abs(Sold-Snew) < 1.e-10)

    if bc_MPS == 'infinite':
        Eold = np.average(M.bond_energies(psi))
        Sold = np.average(psi.entanglement_entropy())
        for i in range(2):
            engine.run()
        Enew = np.average(M.bond_energies(psi))
        Snew = np.average(psi.entanglement_entropy())
        assert (abs(Eold - Enew) < 1.e-10)
        # TODO: why does the test below fail??
        # assert (abs(Sold-Snew) < 1.e-10)

    # TODO: compare with known ground state (energy) / ED !


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
