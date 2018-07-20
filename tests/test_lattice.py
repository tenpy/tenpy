"""A collection of tests for tenpy.models.lattice"""
# Copyright 2018 TeNPy Developers

from tenpy.models import lattice
import tenpy.linalg.np_conserved as npc
from tenpy.networks import site
import numpy as np
import numpy.testing as npt
import nose.tools as nst

from random_test import gen_random_legcharge


def test_lattice():
    chinfo = npc.ChargeInfo([1, 3])
    leg = gen_random_legcharge(chinfo, 8)
    leg2 = gen_random_legcharge(chinfo, 2)
    op1 = npc.Array.from_func(np.random.random, [leg, leg.conj()], shape_kw='size')
    op2 = npc.Array.from_func(np.random.random, [leg2, leg2.conj()], shape_kw='size')
    site1 = lattice.Site(leg, [('up', 0), ('down', -1)], op1=op1)
    site2 = lattice.Site(leg2, [('down', 0), ('up', -1)], op2=op2)
    for order in ['default', 'Fstyle', 'snake', 'snakeFstyle']:
        print("order =", order)
        Ls = [5, 2]
        basis = [[1., 1.], [0., 1.]]
        pos = [[0.1, 0.], [0.2, 0.]]
        lat = lattice.Lattice(Ls, [site1, site2], order=order, basis=basis, positions=pos)
        nst.eq_(lat.dim, len(Ls))
        nst.eq_(lat.N_sites, np.prod(Ls) * 2)
        for i in range(lat.N_sites):
            nst.eq_(lat.lat2mps_idx(lat.mps2lat_idx(i)), i)
        idx = (4, 1, 0)
        nst.eq_(lat.mps2lat_idx(lat.lat2mps_idx(idx)), idx)
        npt.assert_equal([4.1, 5.], lat.position(idx))
        # test lat.mps2lat_values
        A = np.random.random([lat.N_sites, 2, lat.N_sites])
        print(A.shape)
        Ares = lat.mps2lat_values(A, axes=[-1, 0])
        for i in range(lat.N_sites):
            idx_i = lat.mps2lat_idx(i)
            for j in range(lat.N_sites):
                idx_j = lat.mps2lat_idx(j)
                for k in range(2):
                    idx = idx_i + (k, ) + idx_j
                    nst.eq_(Ares[idx], A[i, k, j])
        # and again for fixed `u` within the unit cell
        for u in range(len(lat.unit_cell)):
            A_u = A[np.ix_(lat.mps_idx_fix_u(u), np.arange(2), lat.mps_idx_fix_u(u))]
            A_u_res = lat.mps2lat_values(A_u, axes=[-1, 0], u=u)
            npt.assert_equal(A_u_res, Ares[:, :, u, :, :, :, u])

def test_lattice_order():
    s = site.SpinHalfSite('Sz')
    # yapf: disable
    square = lattice.SquareLattice(2, 2, s, 'default')
    order_default = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    npt.assert_equal(square.order, order_default)
    square = lattice.SquareLattice(4, 3, s, 'snake')
    order_snake = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 2, 0], [1, 1, 0], [1, 0, 0],
                            [2, 0, 0], [2, 1, 0], [2, 2, 0], [3, 2, 0], [3, 1, 0], [3, 0, 0]])
    npt.assert_equal(square.order, order_snake)
    square = lattice.SquareLattice(2, 3, s, ((1, 0), (True, False)))
    order_Fsnake = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 2, 0], [1, 2, 0]])
    npt.assert_equal(square.order, order_Fsnake)
    # yapf: enable
