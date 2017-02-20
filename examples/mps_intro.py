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
from tenpy.networks.mps import MPS
from tenpy.models.lattice import Lattice
from tenpy.models.lattice import Site

# model parameters
Jxx, Jz = 1., 1.
L = 20
dt = 0.1
cutoff = 1.e-10
print "Jxx={Jxx}, Jz={Jz}, L={L:d}".format(Jxx=Jxx, Jz=Jz, L=L)

# create a ChargeInfo to specify the nature of the charge and physical leg
ci = npc.ChargeInfo([1], ['2*Sz'])
p_leg = npc.LegCharge.from_qflat(ci, [1, -1])
Sp = [[0, 1.], [0, 0]]
Sm = [[0, 0], [1., 0]]
Sz = [[0.5, 0], [0, -0.5]]
# create the single site unit cell
site = Site(p_leg, ['up', 'down'], Splus=Sp, Sminus=Sm, Sz=Sz)

# make lattice from unit cell and create product MPS 'on lattice'
print "1) create Arrays for an Neel MPS"
lat = Lattice([L, 1], [site], order='default', bc_MPS='finite')
psi = MPS.from_product_state(
    lat.mps_sites(),
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], )

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

envL = npc.zeros(
    [W.get_leg('wL').conj(), psi.get_B(0).get_leg('vL').conj(), psi.get_B(0).get_leg('vL')])
envL.set_leg_labels(['wR', 'vR', 'vR*'])
envL[0, :, :] = npc.diag(1., envL.legs[1])
envR = npc.zeros(
    [W.get_leg('wR').conj(), psi.get_B(-1).get_leg('vR').conj(), psi.get_B(-1).get_leg('vR')])
envR.set_leg_labels(['wL', 'vL', 'vL*'])
envR[-1, :, :] = npc.diag(1., envR.legs[1])

print "4) contract MPS and MPO to calculate the energy"
contr = envL
for i in range(L):
    # contr labels: wR, vR, vR*
    contr = npc.tensordot(contr, psi.get_B(i), axes=('vR', 'vL'))
    # wR, vR*, vR, p
    contr = npc.tensordot(contr, Ws[i], axes=(['p', 'wR'], ['p', 'wL']))
    # vR*, vR, wR, p2
    contr = npc.tensordot(contr, psi.get_B(i).conj(), axes=(['p2', 'vR*'], ['p*', 'vL*']))
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
U_expE2 = U2.scale_axis(np.exp(-1.j * dt * E2), axis=1)  # scale_axis ~= apply a diagonal matrix
exp_H2 = npc.tensordot(U_expE2, U2.conj(), axes=(1, 1))
exp_H2.set_leg_labels(H2.get_leg_labels())
exp_H2 = exp_H2.split_legs()  # by default split all legs which are `LegPipe`
# (this restores the originial labels ['p2', 'q2', 'p', 'q'] of `H2` in `exp_H2`)

print "7) apply exp(H2) to even/odd bonds of the MPS and truncate with svd"
# (this implements one time step of first order TEBD)
for even_odd in [0, 1]:
    for i in range(even_odd, L - 1, 2):
        # TODO: instead use psi.get_theta(2)
        B_L = psi.get_B(i).scale_axis(psi.get_SL(i), 'vL')
        B_R = psi.get_B(i + 1).replace_label('p', 'q')
        theta = npc.tensordot(B_L, B_R, axes=('vR', 'vL'))
        theta = npc.tensordot(theta, exp_H2, axes=(['p', 'q'], ['p', 'q']))
        # view as matrix for SVD
        theta = theta.combine_legs([('vL', 'p2'), ('vR', 'q2')], qconj=[+1, -1])
        U, S, V = npc.svd(theta, inner_labels=['vR', 'vL'])
        keep = S > cutoff
        S = S[keep]
        invsq = np.linalg.norm(S)
        psi.set_SR(i, S / invsq)
        U = U.iscale_axis(S / invsq, 'vR')
        B_L = U.split_legs(0).iscale_axis(psi.get_SL(i)**-1, 'vL').ireplace_label('p2', 'p')
        B_R = V.split_legs(1).ireplace_label('q2', 'p')
        psi.set_B(i, B_L)
        psi.set_B(i + 1, B_R)
print "finished"
