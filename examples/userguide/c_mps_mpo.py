"""Initialization of sites, MPS and MPO."""

from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO

spin = SpinHalfSite(conserve="Sz")

N = 6  # number of sites
sites = [spin] * N  # repeat entry of list N times
pstate = ["up", "down"] * (N // 2)  # Neel state
psi = MPS.from_product_state(sites, pstate, bc="finite")
print("<Sz> =", psi.expectation_value("Sz"))
# <Sz> = [ 0.5 -0.5  0.5 -0.5]
print("<Sp_i Sm_j> =", psi.correlation_function("Sp", "Sm"), sep="\n")
# <Sp_i Sm_j> =
# [[1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 0. 0.]]

# define an MPO
Id, Sp, Sm, Sz = spin.Id, spin.Sp, spin.Sm, spin.Sz
J, Delta, hz = 1., 1., 0.2
W_bulk = [[Id, Sp, Sm, Sz, -hz * Sz], [None, None, None, None, 0.5 * J * Sm],
          [None, None, None, None, 0.5 * J * Sp], [None, None, None, None, J * Delta * Sz],
          [None, None, None, None, Id]]
W_first = [W_bulk[0]]  # first row
W_last = [[row[-1]] for row in W_bulk]  # last column
Ws = [W_first] + [W_bulk] * (N - 2) + [W_last]
H = MPO.from_grids([spin] * N, Ws, bc='finite', IdL=0, IdR=-1)
print("<psi|H|psi> =", H.expectation_value(psi))
# <psi|H|psi> = -1.25
