"""Simplified version of `a_np_conserved.py` making use of other classes (like MPS, MPO).

This example includes the following steps:
1) create Arrays for an Neel MPS
2) create an MPO representing the nearest-neighbour AFM Heisenberg Hamiltonian
3) define 'environments' left and right
4) contract MPS and MPO to calculate the energy
5) extract two-site hamiltonian ``H2`` from the MPO
6) calculate ``exp(-1.j*dt*H2)`` by diagonalization of H2
7) apply ``exp(H2)`` to two sites of the MPS and truncate with svd

Note that this example performs the same steps as `a_np_conserved.py`,
but makes use of other predefined classes except npc.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import tenpy.linalg.np_conserved as npc
import numpy as np

# some more imports
from tenpy.networks.site import SpinHalfSite
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO, MPOEnvironment
from tenpy.algorithms.truncation import svd_theta

# model parameters
Jxx, Jz = 1., 1.
L = 20
dt = 0.1
cutoff = 1.e-10
print("Jxx={Jxx}, Jz={Jz}, L={L:d}".format(Jxx=Jxx, Jz=Jz, L=L))

print("1) create Arrays for an Neel MPS")
site = SpinHalfSite(conserve='Sz')  # predefined charges and Sp,Sm,Sz operators
p_leg = site.leg
chinfo = p_leg.chinfo
# make lattice from unit cell and create product state MPS
lat = Chain(L, site, bc_MPS='finite')
state = ["up", "down"] * (L // 2) + ["up"] * (L % 2)  # Neel state
print("state = ", state)
psi = MPS.from_product_state(lat.mps_sites(), state, lat.bc_MPS)

print("2) create an MPO representing the AFM Heisenberg Hamiltonian")

# predefined physical spin-1/2 operators Sz, S+, S-
Sz, Sp, Sm, Id = site.Sz, site.Sp, site.Sm, site.Id

mpo_leg = npc.LegCharge.from_qflat(chinfo, [[0], [2], [-2], [0], [0]])

W_grid = [[Id,   Sp,   Sm,   Sz,   None          ],
          [None, None, None, None, 0.5 * Jxx * Sm],
          [None, None, None, None, 0.5 * Jxx * Sp],
          [None, None, None, None, Jz * Sz       ],
          [None, None, None, None, Id            ]]  # yapf:disable

W = npc.grid_outer(W_grid, [mpo_leg, mpo_leg.conj()], grid_labels=['wL', 'wR'])
# wL/wR = virtual left/right of the MPO
Ws = [W] * L
Ws[0] = W[:1, :]
Ws[-1] = W[:, -1:]
H = MPO(psi.sites, Ws, psi.bc, IdL=0, IdR=-1)

print("3) define 'environments' left and right")

# this is automatically done during initialization of MPOEnvironment
env = MPOEnvironment(psi, H, psi)
envL = env.get_LP(0)
envR = env.get_RP(L - 1)

print("4) contract MPS and MPO to calculate the energy <psi|H|psi>")

E = env.full_contraction(L - 1)
print("E =", E)

print("5) calculate two-site hamiltonian ``H2`` from the MPO")
# label left, right physical legs with p, q
W0 = H.get_W(0).replace_labels(['p', 'p*'], ['p0', 'p0*'])
W1 = H.get_W(1).replace_labels(['p', 'p*'], ['p1', 'p1*'])
H2 = npc.tensordot(W0, W1, axes=('wR', 'wL')).itranspose(['wL', 'wR', 'p0', 'p1', 'p0*', 'p1*'])
H2 = H2[H.IdL[0], H.IdR[2]]  # (If H has single-site terms, it's not that simple anymore)
print("H2 labels:", H2.get_leg_labels())

print("6) calculate exp(H2) by diagonalization of H2")
# diagonalization requires to view H2 as a matrix
H2 = H2.combine_legs([('p0', 'p1'), ('p0*', 'p1*')], qconj=[+1, -1])
print("labels after combine_legs:", H2.get_leg_labels())
E2, U2 = npc.eigh(H2)
print("Eigenvalues of H2:", E2)
U_expE2 = U2.scale_axis(np.exp(-1.j * dt * E2), axis=1)  # scale_axis ~= apply a diagonal matrix
exp_H2 = npc.tensordot(U_expE2, U2.conj(), axes=(1, 1))
exp_H2.iset_leg_labels(H2.get_leg_labels())
exp_H2 = exp_H2.split_legs()  # by default split all legs which are `LegPipe`
# (this restores the originial labels ['p0', 'p1', 'p0*', 'p1*'] of `H2` in `exp_H2`)

# alternative way: use :func:`~tenpy.linalg.np_conserved.expm`
exp_H2_alternative = npc.expm(-1.j * dt * H2).split_legs()
assert (npc.norm(exp_H2_alternative - exp_H2) < 1.e-14)

print("7) apply exp(H2) to even/odd bonds of the MPS and truncate with svd")
# (this implements one time step of first order TEBD)
trunc_par = {'svd_min': cutoff, 'trunc_cut': None}
for even_odd in [0, 1]:
    for i in range(even_odd, L - 1, 2):
        theta = psi.get_theta(i, 2)  # handles canonical form (i.e. scaling with 'S')
        theta = npc.tensordot(exp_H2, theta, axes=(['p0*', 'p1*'], ['p0', 'p1']))
        # view as matrix for SVD
        theta = theta.combine_legs([('vL', 'p0'), ('p1', 'vR')], new_axes=[0, 1], qconj=[+1, -1])
        # now theta has labels '(vL.p0)', '(p1.vR)'
        U, S, V, err, invsq = svd_theta(theta, trunc_par, inner_labels=['vR', 'vL'])
        psi.set_SR(i, S)
        A_L = U.split_legs('(vL.p0)').ireplace_label('p0', 'p')
        B_R = V.split_legs('(p1.vR)').ireplace_label('p1', 'p')
        psi.set_B(i, A_L, form='A')  # left-canonical form
        psi.set_B(i + 1, B_R, form='B')  # right-canonical form
print("finished")
