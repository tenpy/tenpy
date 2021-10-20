"""A collection of tests for tenpy.models.lattice."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

from tenpy.models import lattice
import tenpy.linalg.np_conserved as npc
from tenpy.networks import site
from tenpy.networks.mps import MPS
import numpy as np
import numpy.testing as npt
import pytest

from random_test import gen_random_legcharge


def test_bc_choices():
    assert int(lattice.bc_choices['open']) == 1  # this is used explicitly
    assert int(lattice.bc_choices['periodic']) == 0  # and this as well


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
        lat = lattice.Lattice(Ls, [site1, site2],
                              order=order,
                              basis=basis,
                              positions=pos,
                              bc='periodic',
                              bc_MPS='infinite')
        assert lat.dim == len(Ls)
        assert lat.N_sites == np.prod(Ls) * 2
        for i in range(lat.N_sites):
            assert lat.lat2mps_idx(lat.mps2lat_idx(i)) == i
        idx = [4, 1, 0]
        assert np.all(lat.mps2lat_idx(lat.lat2mps_idx(idx)) == idx)
        # index conversion should also work for arbitrary index arrays and indices outside bc_MPS
        # for bc_MPS=infinite
        i = np.arange(-3, lat.N_sites * 2 - 3).reshape((-1, 2))
        assert np.all(lat.lat2mps_idx(lat.mps2lat_idx(i)) == i)
        # test position
        npt.assert_equal([4.1, 5.], lat.position(idx))
        # test lat.mps2lat_values
        A = np.random.random([lat.N_sites, 2, lat.N_sites])
        print(A.shape)
        Ares = lat.mps2lat_values(A, axes=[-1, 0])
        Ares_ma = lat.mps2lat_values_masked(A, [-1, 0], [None] * 2, [None] * 2)
        for i in range(lat.N_sites):
            idx_i = lat.mps2lat_idx(i)
            for j in range(lat.N_sites):
                idx_j = lat.mps2lat_idx(j)
                for k in range(2):
                    idx = tuple(idx_i) + (k, ) + tuple(idx_j)
                    assert Ares[idx] == A[i, k, j]
                    assert Ares_ma[idx] == A[i, k, j]
        # and again for fixed `u` within the unit cell
        for u in range(len(lat.unit_cell)):
            mps_u = lat.mps_idx_fix_u(u)
            A_u = A[np.ix_(mps_u, np.arange(2), mps_u)]
            A_u_res = lat.mps2lat_values(A_u, axes=[-1, 0], u=u)
            A_u_res_masked = lat.mps2lat_values_masked(A_u,
                                                       axes=[-1, 0],
                                                       mps_inds=[mps_u] * 2,
                                                       include_u=[False] * 2)
            A_u_expected = Ares[:, :, u, :, :, :, u]
            npt.assert_equal(A_u_res, A_u_expected)
            npt.assert_equal(A_u_res_masked, A_u_expected)


def test_TrivialLattice():
    s1 = site.SpinHalfSite('Sz')
    s2 = site.SpinSite(0.5, 'Sz')
    s3 = site.SpinSite(1.0, 'Sz')
    lat = lattice.TrivialLattice([s1, s2, s3, s2, s1])
    lat.test_sanity()


def test_IrregularLattice():
    s1 = site.SpinHalfSite('Sz')
    s2 = site.SpinSite(0.5, 'Sz')
    reg = lattice.Honeycomb(3, 3, [s1, s2], bc=['open', 'periodic'])
    ir = lattice.IrregularLattice(reg, [[1, 1, 0], [1, 1, 1], [0, 0, 0]],
                                  ([[1, 1, 2], [1, 1, 3]], [7, 10]), [s2, s2],
                                  [[-0.1, 0.0], [0.1, 0.0]])
    known = {  # written down by hand for this particular case
        (0, 1, (0, 0)): {'i': [5, 11, 0, 12, 1, 7, 13], 'j': [8, 14, 3, 15, 4, 10, 16]},
        (1, 0, (1, 0)): {'i': [ 2,  8,  4, 10], 'j': [5, 11, 7, 13]},
        (1, 0, (0, 1)): {'i': [ 2,  14,  3, 15, 10, 16], 'j': [0, 12, 1, 13, 5, 11]},
    }
    for (u0, u1, dx), expect in known.items():
        i, j, lat, sh = ir.possible_couplings(u0, u1, dx)
        print(i, j)
        sort = np.lexsort(lat.T)
        i = i[sort]
        j = j[sort]
        npt.assert_equal(i, np.array(expect['i']))
        npt.assert_equal(j, np.array(expect['j']))

        ops = [(None, dx, u1), (None, [0, 0], u0)]
        m_ji, m_lat_indices, m_coupling_shape = ir.possible_multi_couplings(ops)
        sort = np.lexsort(m_lat_indices.T)
        npt.assert_equal(m_ji[sort, 1], np.array(expect['i']))
        npt.assert_equal(m_ji[sort, 0], np.array(expect['j']))


def test_HelicalLattice():
    s = site.SpinHalfSite()
    honey = lattice.Honeycomb(2, 3, s, bc=['periodic', -1], bc_MPS='infinite', order='Cstyle')
    hel = lattice.HelicalLattice(honey, 2)
    strength = np.array([[1.5, 2.5, 1.5], [2.5, 1.5, 2.5]])

    def assert_same(i1, j1, s1, ij2, s2):
        """check that coupling and multi_coupling agree up to sorting order"""
        assert len(i1) == len(ij2)
        sort1 = np.lexsort(np.stack([i1, j1]))
        sort2 = np.lexsort(ij2.T)
        for a, b in zip(sort1, sort2):
            assert (i1[a], j1[a]) == tuple(ij2[b])
            assert s1[a] == s2[b]

    i, j, s = hel.possible_couplings(0, 1, [0, 0], strength)
    ijm, sm = hel.possible_multi_couplings([('X', [0, 0], 0), ('X', [0, 0], 1)], strength)
    assert np.all(i == [0, 2]) and np.all(j == [1, 3])
    assert np.all(s == [1.5, 2.5])
    assert_same(i, j, s, ijm, sm)

    i, j, s = hel.possible_couplings(0, 0, [1, 0], strength)
    ijm, sm = hel.possible_multi_couplings([('X', [0, 0], 0), ('X', [1, 0], 0)], strength)
    assert np.all(i == [0, 2]) and np.all(j == [6, 8])
    assert np.all(s == [1.5, 2.5])
    assert_same(i, j, s, ijm, sm)

    i, j, s = hel.possible_couplings(0, 0, [-1, 1], strength)
    assert np.all(i == [4, 6]) and np.all(j == [0, 2])
    assert np.all(s == [2.5, 1.5])  # swapped!
    ijm, sm = hel.possible_multi_couplings([('X', [0, 0], 0), ('X', [-1, 1], 0)], strength)
    assert_same(i, j, s, ijm, sm)
    ijm, sm = hel.possible_multi_couplings([('X', [1, 0], 0), ('X', [0, 1], 0)], strength)
    assert_same(i, j, s, ijm, sm)

    # test that MPS.from_lat_product_state checks translation invariance
    p_state = [[['up', 'down'], ['down', 'down'], ['up', 'down']],
               [['down', 'down'], ['up', 'down'], ['down', 'down']]]
    psi = MPS.from_lat_product_state(hel, p_state)
    p_state[0][2] = ['down', 'down']
    with pytest.raises(ValueError, match='.* not translation invariant .*'):
        psi = MPS.from_lat_product_state(hel, p_state)


def test_number_nn():
    s = None
    chain = lattice.Chain(2, s)
    assert chain.count_neighbors() == 2
    assert chain.count_neighbors(key='next_nearest_neighbors') == 2
    ladd = lattice.Ladder(2, s)
    for u in [0, 1]:
        assert ladd.count_neighbors(u) == 3
        assert ladd.count_neighbors(u, key='next_nearest_neighbors') == 2
    square = lattice.Square(2, 2, s)
    assert square.count_neighbors() == 4
    assert square.count_neighbors(key='next_nearest_neighbors') == 4
    triang = lattice.Triangular(2, 2, s)
    assert triang.count_neighbors() == 6
    assert triang.count_neighbors(key='next_nearest_neighbors') == 6
    hc = lattice.Honeycomb(2, 2, s)
    for u in [0, 1]:
        assert hc.count_neighbors(u) == 3
        assert hc.count_neighbors(u, key='next_nearest_neighbors') == 6
    kag = lattice.Kagome(2, 2, s)
    for u in [0, 1, 2]:
        assert kag.count_neighbors(u) == 4
        assert kag.count_neighbors(u, key='next_nearest_neighbors') == 4


def pairs_with_reversed(coupling_pairs):
    res = set([])
    for u1, u2, dx in coupling_pairs:
        res.add((u1, u2, tuple(dx)))
        res.add((u2, u1, tuple(-dx)))
    return res


def test_pairs():
    lattices = [
        lattice.Chain(2, None),
        lattice.Ladder(2, None),
        lattice.Square(2, 2, None),
        lattice.Triangular(2, 2, None),
        lattice.Honeycomb(2, 2, None),
        lattice.Kagome(2, 2, None)
    ]
    for lat in lattices:
        print(lat.__class__.__name__)
        found_dist_pairs = lat.find_coupling_pairs(5, 3.)
        dists = sorted(found_dist_pairs.keys())
        for i, name in enumerate([
                'nearest_neighbors', 'next_nearest_neighbors', 'next_next_nearest_neighbors',
                'fourth_nearest_neighbors', 'fifth_nearest_neighbors'
        ]):
            if name not in lat.pairs:
                assert i > 2  # all of them should define up to next_next_nearest_neighbors
                continue
            print(name)
            defined_pairs = lat.pairs[name]
            found_pairs = found_dist_pairs[dists[i]]
            assert len(defined_pairs) == len(found_pairs)
            defined_pairs = pairs_with_reversed(defined_pairs)
            found_pairs = pairs_with_reversed(found_pairs)
            assert defined_pairs == found_pairs
    # done


def test_lattice_order():
    s = site.SpinHalfSite('Sz')
    # yapf: disable
    chain = lattice.Chain(4, s)
    order_default = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    npt.assert_equal(chain.order, order_default)
    chain = lattice.Chain(4, s, order='folded')
    order_folded = np.array([[0, 0], [3, 0], [1, 0], [2, 0]])
    npt.assert_equal(chain.order, order_folded)
    chain = lattice.Chain(5, s, order='folded')
    order_folded = np.array([[0, 0], [4, 0], [1, 0], [3, 0], [2, 0]])
    npt.assert_equal(chain.order, order_folded)
    square = lattice.Square(2, 2, s, order='default')
    order_default = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    npt.assert_equal(square.order, order_default)
    square = lattice.Square(4, 3, s, order='snake')
    order_snake = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 2, 0], [1, 1, 0], [1, 0, 0],
                            [2, 0, 0], [2, 1, 0], [2, 2, 0], [3, 2, 0], [3, 1, 0], [3, 0, 0]])
    npt.assert_equal(square.order, order_snake)
    square = lattice.Square(2, 3, s, order=("standard", (True, False), (1, 0)))
    order_Fsnake = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 2, 0], [1, 2, 0]])
    npt.assert_equal(square.order, order_Fsnake)

    hc = lattice.Honeycomb(2, 3, s, order='default')
    order_hc_def = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 1], [0, 1, 1], [0, 2, 1],
                             [1, 0, 0], [1, 1, 0], [1, 2, 0], [1, 0, 1], [1, 1, 1], [1, 2, 1]])
    npt.assert_equal(hc.order, order_hc_def)
    hc = lattice.Honeycomb(2, 3, s, order=('standard', (True, False, False), (0.3, 0.1, -1.)))
    order_hc_mix = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 2, 0], [1, 2, 0],
                             [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 2, 1], [1, 2, 1]])
    npt.assert_equal(hc.order, order_hc_mix)

    kag = lattice.Kagome(2, 3, s, order=('grouped', [[1], [0, 2]]))
    order_kag_gr = np.array([[0, 0, 1], [0, 1, 1], [0, 2, 1], [0, 0, 0], [0, 0, 2], [0, 1, 0],
                             [0, 1, 2], [0, 2, 0], [0, 2, 2],
                             [1, 0, 1], [1, 1, 1], [1, 2, 1], [1, 0, 0], [1, 0, 2], [1, 1, 0],
                             [1, 1, 2], [1, 2, 0], [1, 2, 2]])
    npt.assert_equal(kag.order, order_kag_gr)
    kag = lattice.Kagome(2, 4, s, order=('grouped', [[0], [2, 1]], [1, 0, 2]))
    order_kag_gr = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 2], [0, 0, 1], [1, 0, 2], [1, 0, 1],
                             [0, 1, 0], [1, 1, 0], [0, 1, 2], [0, 1, 1], [1, 1, 2], [1, 1, 1],
                             [0, 2, 0], [1, 2, 0], [0, 2, 2], [0, 2, 1], [1, 2, 2], [1, 2, 1],
                             [0, 3, 0], [1, 3, 0], [0, 3, 2], [0, 3, 1], [1, 3, 2], [1, 3, 1]])
    npt.assert_equal(kag.order, order_kag_gr)
    # yapf: enable


def test_possible_couplings():
    lat_reg = lattice.Honeycomb(2,
                                3, [None, None],
                                order="snake",
                                bc="periodic",
                                bc_MPS="infinite")
    lat_irreg = lattice.IrregularLattice(lat_reg, remove=[[0, 0, 0]])
    u0, u1 = 0, 1
    for lat in [lat_reg, lat_irreg]:
        for dx in [(0, 0), (0, 1), (2, 1), (-1, -1)]:
            print("dx =", dx)
            mps0, mps1, lat_indices, coupling_shape = lat.possible_couplings(u0, u1, dx)
            ops = [(None, [0, 0], u0), (None, dx, u1)]
            m_ijkl, m_lat_indices, m_coupling_shape = lat.possible_multi_couplings(ops)
            assert coupling_shape == m_coupling_shape
            if len(lat_indices) == 0:
                continue
            sort = np.lexsort(lat_indices.T)
            mps0, mps1, lat_indices = mps0[sort], mps1[sort], lat_indices[sort, :]
            assert m_ijkl.shape == (len(mps0), 2)
            m_sort = np.lexsort(m_lat_indices.T)
            m_ijkl, m_lat_indices = m_ijkl[m_sort, :], m_lat_indices[m_sort, :]
            npt.assert_equal(m_lat_indices, lat_indices)
            npt.assert_equal(mps0, m_ijkl[:, 0])
            npt.assert_equal(mps1, m_ijkl[:, 1])


def test_index_conversion():
    from tenpy.networks.mps import MPS
    s = site.SpinHalfSite(conserve=None)
    state1 = [[[0, 1]]]  # 0=up, 1=down
    for order in ["snake", "default"]:
        lat = lattice.Honeycomb(2, 3, [s, s], order=order, bc_MPS="finite")
        psi1 = MPS.from_lat_product_state(lat, state1)
        sz1_mps = psi1.expectation_value("Sigmaz")
        sz1_lat = lat.mps2lat_values(sz1_mps)
        npt.assert_equal(sz1_lat[:, :, 0], +1.)
        npt.assert_equal(sz1_lat[:, :, 1], -1.)
        # and a random state
        state2 = np.random.random(lat.shape + (2, ))
        psi2 = MPS.from_lat_product_state(lat, state2)
        sz2_mps = psi2.expectation_value("Sigmaz")
        sz2_lat = lat.mps2lat_values(sz2_mps)
        expect_sz2 = np.sum(state2**2 * np.array([1., -1]), axis=-1)
        npt.assert_array_almost_equal_nulp(sz2_lat, expect_sz2, 100)
    # doen
