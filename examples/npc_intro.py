"""An example code to demonstrate the usage of :class:`~tenpy.linalg.np_conserved.Array`.

This example includes the following steps:
1) create (non-canonical) Arrays for an Neel MPS
2) create an MPO representing the nearest-neighbour AFM Heisenberg Hamiltonian
3) define 'environments' left and right
4) contract MPS and MPO to calculate the energy
5) extract two-site hamiltonian ``H2`` from the MPO
6) calculate ``exp(-1.j*dt*H2)`` by diagonalization of H2
7) apply ``exp(H2)`` to two sites of the MPS and truncate with svd
"""

import tenpy.linalg.np_conserved as npc
import numpy as np

# model parameters
Jxx, Jz = 1., 1.
L = 20
dt = 0.1
cutoff = 1.e-10
print "Jxx={Jxx}, Jz={Jz}, L={L:d}".format(Jxx=Jxx, Jz=Jz, L=L)

print "1) create Arrays for an Neel MPS"

# create a ChargeInfo to specify the nature of the charge
ci = npc.ChargeInfo([1], ['2*Sz'])  # the second argument is just a descriptive name

# create LegCharges on physical leg and even/odd bonds
p_leg = npc.LegCharge.from_qflat(ci, [[1], [-1]])  # charges for up, down
v_leg_even = npc.LegCharge.from_qflat(ci, [[0]])
v_leg_odd = npc.LegCharge.from_qflat(ci, [[1]])

B_even = npc.zeros([v_leg_even, v_leg_odd.conj(), p_leg])
B_odd = npc.zeros([v_leg_odd, v_leg_even.conj(), p_leg])
B_even[0, 0, 0] = 1.  # up
B_odd[0, 0, 1] = 1.  # down

for B in [B_even, B_odd]:
    B.set_leg_labels(['vL', 'vR', 'p'])  # virtual left/right, physical

Bs = [B_even, B_odd] * (L // 2) + [B_even] * (L % 2)  # (right-canonical)
Ss = [np.ones(1)] * L  # Ss[i] are singular values between Bs[i-1] and Bs[i]

# Side remark:
# An MPS is expected to have non-zero entries everywhere compatible with the charges.
# In general, we recommend to use `sort_legcharge` (or `as_completely_blocked`)
# to ensure complete blocking. (But the code will also work, if you don't do it.)
# The drawback is that this might introduce permutations in the indices of single legs,
# which you have to keep in mind when converting dense numpy arrays to and from npc.Arrays.

print "2) create an MPO representing the AFM Heisenberg Hamiltonian"

# create physical spin-1/2 operators Sz, S+, S-
Sz = npc.Array.from_ndarray([[0.5, 0.], [0., -0.5]], [p_leg, p_leg.conj()])
Sp = npc.Array.from_ndarray([[0., 1.], [0., 0.]], [p_leg, p_leg.conj()])
Sm = npc.Array.from_ndarray([[0., 0.], [1., 0.]], [p_leg, p_leg.conj()])
Id = npc.eye_like(Sz)  # identity
for op in [Sz, Sp, Sm, Id]:
    op.set_leg_labels(['p2', 'p'])  # physical out, physical in

mpo_leg = npc.LegCharge.from_qflat(ci, [[0], [2], [-2], [0], [0]])

W_grid = [[Id, Sp, Sm, Sz, None],
          [None, None, None, None, 0.5 * Jxx * Sm],
          [None, None, None, None, 0.5 * Jxx * Sp],
          [None, None, None, None, Jz * Sz],
          [None, None, None, None, Id]]  # yapf:disable

W = npc.grid_outer(W_grid, [mpo_leg, mpo_leg.conj()])
W.set_leg_labels(['wL', 'wR', 'p2', 'p'])  # wL/wR = virtual left/right of the MPO
Ws = [W] * L

print "3) define 'environments' left and right"

envL = npc.zeros([W.get_leg('wL').conj(), Bs[0].get_leg('vL').conj(), Bs[0].get_leg('vL')])
envL.set_leg_labels(['wR', 'vR', 'vR*'])
envL[0, :, :] = npc.diag(1., envL.legs[1])
envR = npc.zeros([W.get_leg('wR').conj(), Bs[-1].get_leg('vR').conj(), Bs[-1].get_leg('vR')])
envR.set_leg_labels(['wL', 'vL', 'vL*'])
envR[-1, :, :] = npc.diag(1., envR.legs[1])

print "4) contract MPS and MPO to calculate the energy"
contr = envL
for i in range(L):
    # contr labels: wR, vR, vR*
    contr = npc.tensordot(contr, Bs[i], axes=('vR', 'vL'))
    # wR, vR*, vR, p
    contr = npc.tensordot(contr, Ws[i], axes=(['p', 'wR'], ['p', 'wL']))
    # vR*, vR, wR, p2
    contr = npc.tensordot(contr, Bs[i].conj(), axes=(['p2', 'vR*'], ['p*', 'vL*']))
    # vR, wR, vR*
    # (the order of the legs changed, but that's no problem with labels)
E = npc.inner(contr, envR, axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
print "E =", E

print "5) calculate two-site hamiltonian ``H2`` from the MPO"
# label left, right physical legs with p, q
W2 = W.replace_label('p', 'q').ireplace_label('p2', 'q2')
H2 = npc.tensordot(W, W2, axes=('wR', 'wL')).itranspose(['wL', 'wR', 'p2', 'q2', 'p', 'q'])
H2 = H2[0, -1]
print "H2 labels:", H2.get_leg_labels()

print "6) calculate exp(H2) by diagonalization of H2"
# diagonalization requires to view H2 as a matrix
H2 = H2.combine_legs([('p2', 'q2'), ('p', 'q')], qconj=[+1, -1])
print "labels after combine_legs:", H2.get_leg_labels()
E2, U2 = npc.eigh(H2)
print "Eigenvalues of H2:", E2
U_expE2 = U2.scale_axis(np.exp(-1.j * dt * E2), axis=1)  # scale_axis ~= apply an diagonal matrix
exp_H2 = npc.tensordot(U_expE2, U2.conj(), axes=(1, 1))
exp_H2.set_leg_labels(H2.get_leg_labels())
exp_H2 = exp_H2.split_legs()  # by default split all legs which are `LegPipe`
# (this restores the originial labels ['p2', 'q2', 'p', 'q'] of `H2` in `exp_H2`)

print "7) apply exp(H2) to even/odd bonds of the MPS and truncate with svd"
# (this implements one time step of first order TEBD)
for even_odd in [0, 1]:
    for i in range(even_odd, L - 1, 2):
        B_L = Bs[i].scale_axis(Ss[i], 'vL')
        B_R = Bs[i + 1].replace_label('p', 'q')
        theta = npc.tensordot(B_L, B_R, axes=('vR', 'vL'))
        theta = npc.tensordot(theta, exp_H2, axes=(['p', 'q'], ['p', 'q']))
        # view as matrix for SVD
        theta = theta.combine_legs([('vL', 'p2'), ('vR', 'q2')], qconj=[+1, -1])
        U, S, V = npc.svd(theta, inner_labels=['vR', 'vL'])
        keep = S > cutoff
        S = S[keep]
        invsq = np.linalg.norm(S)
        Ss[i + 1] = S / invsq
        U = U.iscale_axis(S / invsq, 'vR')
        Bs[i] = U.split_legs(0).iscale_axis(Ss[i]**-1, 'vL').ireplace_label('p2', 'p')
        Bs[i + 1] = V.split_legs(1).ireplace_label('q2', 'p')
print "finished"
