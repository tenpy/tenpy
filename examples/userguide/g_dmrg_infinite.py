"""Call of infinite DMRG."""

from tenpy.algorithms import dmrg
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS

N = 2  # number of sites in unit cell
model = TFIChain({'L': N, 'J': 1.0, 'g': 1.1, 'bc_MPS': 'infinite'})
sites = model.lat.mps_sites()
psi = MPS.from_product_state(sites, ['up'] * N, 'infinite', unit_cell_width=N)
dmrg_params = {'trunc_params': {'chi_max': 100, 'svd_min': 1.0e-10}, 'mixer': True}
info = dmrg.run(psi, model, dmrg_params)
print('E =', info['E'])
# E = -1.342864022725017
print('max. bond dimension =', max(psi.chi))
# max. bond dimension = 56
print('corr. length =', psi.correlation_length())
# corr. length = 4.915809146764157
