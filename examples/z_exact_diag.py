"""A simple example comparing DMRG output with full diagonalization (ED).

Sorry that this is not well documented! ED is meant to be used for debugging only ;)
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import tenpy.linalg.np_conserved as npc
from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.mps import MPS

from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.algorithms import dmrg


def example_exact_diagonalization(L, Jz):
    xxz_pars = dict(L=L, Jxx=1., Jz=Jz, hz=0.0, bc_MPS='finite', sort_charge=True)
    M = XXZChain(xxz_pars)

    product_state = ["up", "down"] * (xxz_pars['L'] // 2)  # this selects a charge sector!
    psi_DMRG = MPS.from_product_state(M.lat.mps_sites(), product_state)
    charge_sector = psi_DMRG.get_total_charge(True)  # ED charge sector should match

    ED = ExactDiag(M, charge_sector=charge_sector, max_size=2.e6)
    ED.build_full_H_from_mpo()
    # ED.build_full_H_from_bonds()  # whatever you prefer
    print("start diagonalization")
    ED.full_diagonalization()  # the expensive part for large L
    E0_ED, psi_ED = ED.groundstate()  # return the ground state
    print("psi_ED =", psi_ED)

    print("run DMRG")
    dmrg.run(psi_DMRG, M, {'verbose': 0})  # modifies psi_DMRG in place!
    # first way to compare ED with DMRG: convert MPS to ED vector
    psi_DMRG_full = ED.mps_to_full(psi_DMRG)
    print("psi_DMRG_full =", psi_DMRG_full)
    ov = npc.inner(psi_ED, psi_DMRG_full, axes='range', do_conj=True)
    print("<psi_ED|psi_DMRG_full> =", ov)
    assert (abs(abs(ov) - 1.) < 1.e-13)

    # second way: convert ED vector to MPS
    psi_ED_mps = ED.full_to_mps(psi_ED)
    ov2 = psi_ED_mps.overlap(psi_DMRG)
    print("<psi_ED_mps|psi_DMRG> =", ov2)
    assert (abs(abs(ov2) - 1.) < 1.e-13)
    assert (abs(ov - ov2) < 1.e-13)
    # -> advantage: expectation_value etc. of MPS are available!
    print("<Sz> =", psi_ED_mps.expectation_value('Sz'))


if __name__ == "__main__":
    example_exact_diagonalization(10, 1.)
