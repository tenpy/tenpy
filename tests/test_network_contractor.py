"""a collection of tests to check the functionality of network_contractor.py."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

from tenpy.algorithms.network_contractor import contract, outer_product
import numpy as np
from tenpy.linalg import np_conserved as npc
import pytest
import warnings

# Contruct toy tensors
# ====================

Sx = npc.Array.from_ndarray_trivial([[0., 1.], [1., 0.]])
Sy = npc.Array.from_ndarray_trivial([[0., -1.j], [1.j, 0.]], dtype=complex)
Sz = npc.Array.from_ndarray_trivial([[1., 0.], [0., -1.]])
S0 = npc.Array.from_ndarray_trivial([[1., 0.], [0., 1.]])


def two_site_hamiltonian(coupling=1, ferro=True):
    """ returns the two site gate h = -ZZ - gX for ferro=True and h = ZZ - gX otherwise
        the gX term is symmetrised as 1/2 (1*Sx + Sx*1)

        Index convention

          p1*   p2*
          ^     ^
          |     |
        |---------|
        |    h    |
        |---------|
          |     |
          ^     ^
          p1    p2

    """
    h = -npc.outer(Sz, Sz)
    sign = -1. if ferro else 1.
    h = h + sign * coupling * .5 * (npc.outer(Sx, S0) + npc.outer(S0, Sx))
    h.iset_leg_labels(['p1*', 'p1', 'p2*', 'p2'])
    return h


# Tests
# =======


def test_contract_to_real_number():
    # 1 contract to real number
    # ==========================
    v = npc.Array.from_ndarray_trivial([[1., .5], [0, -1.6]])
    v.iset_leg_labels(['L1', 'L2'])
    w = npc.Array.from_ndarray_trivial([[1.2, .6], [0.1, -1.2]])
    w.iset_leg_labels(['U1', 'U2'])
    h2 = two_site_hamiltonian()
    h = two_site_hamiltonian(coupling=.3)
    S = Sz
    S.iset_leg_labels(['U', 'L'])

    res = contract(tensor_list=[v, h2, S, h, w],
                   leg_contractions=[['v', 'L1', 'h2', 'p1*'], ['v', 'L2', 'h2', 'p2*'],
                                     ['h2', 'p1', 'h', 'p1*'], ['h2', 'p2', 'S', 'U'],
                                     ['S', 'L', 'h', 'p2*'], ['h', 'p1', 'w', 'U1'],
                                     ['h', 'p2', 'w', 'U2']],
                   open_legs=None,
                   tensor_names=['v', 'h2', 'S', 'h', 'w'])
    expected_result = -0.2970000000000002  # from MatLab

    assert np.abs(res - expected_result) < 1.e-10


def test_contract_to_complex_number():
    # 2 contract to complex number
    # ==========================
    v = npc.Array.from_ndarray_trivial([[1. + .2j, .5], [0 + .1j, -1.6]], dtype=complex)
    v.iset_leg_labels(['L1', 'L2'])
    w = npc.Array.from_ndarray_trivial([[1.2 + .3j, .6], [0.1 + .2j, -1.2]], dtype=complex)
    w.iset_leg_labels(['U1', 'U2'])
    h2 = two_site_hamiltonian()
    h = two_site_hamiltonian(coupling=.3)
    S = Sy
    S.iset_leg_labels(['U', 'L'])

    res = contract(tensor_list=[v, h2, S, h, w],
                   tensor_names=['v', 'h2', 'S', 'h', 'w'],
                   leg_contractions=[['v', 'L1', 'h2', 'p1*'], ['v', 'L2', 'h2', 'p2*'],
                                     ['h2', 'p1', 'h', 'p1*'], ['h2', 'p2', 'S', 'U'],
                                     ['S', 'L', 'h', 'p2*'], ['h', 'p1', 'w', 'U1'],
                                     ['h', 'p2', 'w', 'U2']])
    expected_result = -0.1735 - 0.5015j  # from MatLab

    assert np.abs(res - expected_result) < 1.e-10


def test_contract_with_sequence():
    # 3 contract to complex number using a sequence
    # ==========================
    v = npc.Array.from_ndarray_trivial([[1. + .2j, .5], [0 + .1j, -1.6]], dtype=complex)
    v.iset_leg_labels(['L1', 'L2'])
    w = npc.Array.from_ndarray_trivial([[1.2 + .3j, .6], [0.1 + .2j, -1.2]], dtype=complex)
    w.iset_leg_labels(['U1', 'U2'])
    h2 = two_site_hamiltonian()
    h = two_site_hamiltonian(coupling=.3)
    S = Sy
    S.iset_leg_labels(['U', 'L'])

    # the supplied sequence should generate three suboptimal warnings
    with warnings.catch_warnings(record=True) as cw:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        res = contract(tensor_list=[v, h2, S, h, w],
                       tensor_names=['v', 'h2', 'S', 'h', 'w'],
                       leg_contractions=[['v', 'L1', 'h2', 'p1*'], ['v', 'L2', 'h2', 'p2*'],
                                         ['h2', 'p1', 'h', 'p1*'], ['h2', 'p2', 'S', 'U'],
                                         ['S', 'L', 'h', 'p2*'], ['h', 'p1', 'w', 'U1'],
                                         ['h', 'p2', 'w', 'U2']],
                       sequence=[1, 3, 5, 6, 4, 2, 0])

        assert len(cw) == 3
        assert "Suboptimal contraction sequence" in str(cw[0].message)
        assert "Suboptimal contraction sequence" in str(cw[1].message)
        assert "Suboptimal contraction sequence" in str(cw[2].message)

    expected_result = -0.1735 - 0.5015j  # from MatLab

    assert np.abs(res - expected_result) < 1.e-10


def test_contract_to_tensor():
    h2 = two_site_hamiltonian()
    h = two_site_hamiltonian(coupling=.3)
    S = Sy
    S.iset_leg_labels(['U', 'L'])

    res = contract(tensor_list=[h2, S, h],
                   tensor_names=['h2', 'S', 'h'],
                   leg_contractions=[['h2', 'p1', 'h', 'p1*'], ['h2', 'p2', 'S', 'U'],
                                     ['S', 'L', 'h', 'p2*']],
                   open_legs=[['h2', 'p1*', 'U1'], ['h2', 'p2*', 'U2'], ['h', 'p1', 'L1'],
                              ['h', 'p2', 'L2']])

    expected_result = np.ones([2, 2, 2, 2], dtype=complex)
    expected_result[:, :, 0, 0] = [[.35j, -1j], [0, .65j]]
    expected_result[:, :, 1, 0] = [[0, -.65j], [-.35j, -1j]]
    expected_result[:, :, 0, 1] = [[1j, +.35j], [.65j, 0]]
    expected_result[:, :, 1, 1] = [[-.65j, 0], [1j, -.35j]]

    assert np.linalg.norm(res.to_ndarray() - expected_result) < 1.e-10


def test_outer_product():
    S = Sy
    S.iset_leg_labels(['U', 'L'])
    S2 = Sz
    S2.iset_leg_labels(['U', 'L'])

    res = contract(tensor_list=[S, S2],
                   tensor_names=['Sy', 'Sz'],
                   open_legs=[['Sy', 'U', 'U2'], ['Sy', 'L', 'L2'], ['Sz', 'U', 'U1'],
                              ['Sz', 'L', 'L1']])
    expected_result = np.kron(np.array([[1., 0.], [0., -1.]]), np.array([[0., -1.j], [1.j, 0.]]))
    expected_result = expected_result.reshape([2, 2, 2, 2])
    expected_result = expected_result.transpose([1, 3, 0, 2])

    assert np.linalg.norm(res.to_ndarray() - expected_result) < 1.e-10
