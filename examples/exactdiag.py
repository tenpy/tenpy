import tenpy.linalg.np_conserved as npc
from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.mps import MPS

from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.algorithms.dmrg import run as run_DMRG

xxz_pars = dict(L=4, Jxx=1., Jz=1., hz=0.0, bc_MPS='finite')
M = XXZChain(xxz_pars)
ED = ExactDiag(M, [0])  # kn
ED.build_full_H_from_mpo()
# ED.build_full_H_from_bonds()  # whatever you prefer
print "start diagonalization"
ED.full_diagonalization()
psi_ED = ED.groundstate()
print psi_ED

print "start DMRG"
product_state = [0, 1]*(xxz_pars['L']//2)  # this selects a charge sector!
psi_DMRG = MPS.from_product_state(M.lat.mps_sites(), product_state)

res = run_DMRG(psi_DMRG, M, {'verbose': 0})
# first way to compare ED with DMRG: convert MPS to ED vector
psi_DMRG_full = ED.mps_to_full(psi_DMRG)
print psi_DMRG_full
print "|<psi_ED|psi_DMRG>| =", abs(npc.inner(psi_ED, psi_DMRG_full, do_conj=True))

# second way: convert ED vector to MPS
psi_ED_mps = ED.full_to_mps(psi_ED)
ov, _ = psi_ED_mps.overlap(psi_DMRG)
print "|<psi_ED_mps|psi_DMRG>| =", abs(ov)
# -> advantange: expectation_value etc. of MPS are available!
print "<Sz> =", psi_ED_mps.expectation_value('Sz')
