"""A collection of tests for tenpy.linalg.trees."""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import pytest

from tenpy.linalg import trees
from tenpy.linalg.symmetries import Symmetry, SymmetryError
from tenpy.linalg.spaces import ElementarySpace, ProductSpace
from tenpy.linalg.dtypes import Dtype
from tenpy.linalg.backends.backend_factory import get_backend


@pytest.mark.xfail(reason='Test not implemented yet')
def test_FusionTree_class():
    # TODO test hash, eq, str, repr, copy
    raise NotImplementedError


@pytest.mark.xfail(reason='Test not implemented yet')
def test_FusionTree_manipulations():
    # TODO test insert, insert_at, split
    raise NotImplementedError


def check_fusion_trees(it: trees.fusion_trees, expect_len: int = None):
    if expect_len is None:
        expect_len = len(it)
    else:
        assert len(it) == expect_len

    # make sure they run.
    _ = str(it)
    _ = repr(it)

    num_trees = 0
    for tree in it:
        assert np.all(tree.are_dual == it.are_dual)
        tree.test_sanity()
        assert it.index(tree) == num_trees
        num_trees += 1
    assert num_trees == expect_len
        
    
def test_fusion_trees(any_symmetry: Symmetry, make_any_sectors, np_random):
    """test the ``fusion_trees`` iterator"""
    some_sectors = make_any_sectors(20)  # generates unique sectors
    non_trivial_sectors = some_sectors[np.any(some_sectors != any_symmetry.trivial_sector[None, :], axis=1)]
    i = any_symmetry.trivial_sector

    print('consistent fusion: [] -> i')
    check_fusion_trees(trees.fusion_trees(any_symmetry, [], i), expect_len=1)
    
    print('consistent fusion: i -> i')
    check_fusion_trees(trees.fusion_trees(any_symmetry, [i], i, [False]), expect_len=1)
    check_fusion_trees(trees.fusion_trees(any_symmetry, [i], i, [True]), expect_len=1)

    print('large consistent fusion')
    uncoupled = some_sectors[:5]
    are_dual = np_random.choice([True, False], size=len(uncoupled), replace=True)
    # find the allowed coupled sectors
    allowed = ProductSpace([ElementarySpace(any_symmetry, [a]) for a in uncoupled]).sectors
    some_allowed = np_random.choice(allowed, axis=0)
    print(f'  uncoupled={", ".join(map(str, uncoupled))}   coupled={some_allowed}')
    it = trees.fusion_trees(any_symmetry, uncoupled, some_allowed, are_dual=are_dual)
    assert len(it) > 0
    check_fusion_trees(it)

    print('large inconsistent fusion')
    # find a forbidden coupled sector
    are_allowed = np.any(np.all(some_sectors[:, None, :] == allowed[None, :, :], axis=2), axis=1)
    forbidden_idcs = np.where(np.logical_not(are_allowed))[0]
    if len(forbidden_idcs) > 0:
        forbidden = some_sectors[np_random.choice(forbidden_idcs)]
        it = trees.fusion_trees(any_symmetry, uncoupled, forbidden, are_dual=are_dual)
        check_fusion_trees(it, expect_len=0)

    # rest of the checks assume we have access to at least one non-trivial sector
    if len(non_trivial_sectors) == 0:
        return
    c = non_trivial_sectors[0]
    c_dual = any_symmetry.dual_sector(c)

    print(f'consistent fusion: c -> c')
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c], c, [True]), expect_len=1)
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c], c, [False]), expect_len=1)

    print(f'consistent fusion: [c, dual(c)] -> i')
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c, c_dual], i, [False, False]), expect_len=1)
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c, c_dual], i, [False, True]), expect_len=1)

    # rest of the checks assume we have access to at least two non-trivial sector
    if len(non_trivial_sectors) == 1:
        return
    d = non_trivial_sectors[1]

    print(f'inconsistent fusion: c -> d')
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c], d, [True]), expect_len=0)
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c], d, [False]), expect_len=0)

    print('consistent fusion: [c, d] -> ?')
    e = any_symmetry.fusion_outcomes(c, d)[0]
    N = any_symmetry.n_symbol(c, d, e)
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c, d], e, [False, False]), expect_len=N)
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c, d], e, [False, True]), expect_len=N)


def check_to_block(symmetry, backend, uncoupled, np_random, dtype):
    """Common implementation for test_to_block and test_to_block_no_backend"""
    spaces = [ElementarySpace(symmetry, [a]) for a in uncoupled]
    domain = ProductSpace(spaces, backend=backend)
    coupled = np_random.choice(domain.sectors)
    all_trees = list(trees.fusion_trees(symmetry, uncoupled, coupled))

    if not symmetry.can_be_dropped:
        with pytest.raises(SymmetryError, match='Can not convert to block for symmetry .*'):
            _ = all_trees[0].as_block(backend, dtype)
        return
    
    coupled_dim = symmetry.sector_dim(coupled)
    uncoupled_dims = symmetry.batch_sector_dim(uncoupled)
    all_blocks = [t.as_block(backend, dtype) for t in all_trees]
    axes = list(range(len(uncoupled)))
    if symmetry.fusion_tensor_dtype.is_complex:
        expect_dtype = dtype.to_complex()
    else:
        expect_dtype = dtype
        
    if backend is None:
        backend = get_backend()
    coupled_eye = backend.block_backend.eye_block([coupled_dim], dtype)
    coupled_zero = backend.block_backend.zero_block([coupled_dim, coupled_dim], dtype)
    for i, X in enumerate(all_blocks):
        assert backend.block_backend.block_shape(X) == (*uncoupled_dims, coupled_dim)
        assert backend.block_backend.block_dtype(X) == expect_dtype
        for j, Y in enumerate(all_blocks):
            if i < j:
                continue  # redundant with (i, j) <-> (j, i)
            X_Y = backend.block_backend.block_tdot(backend.block_backend.block_conj(X), Y, axes, axes)
            expect = coupled_eye if i == j else coupled_zero
            assert backend.block_backend.block_allclose(X_Y, expect, rtol=1e-8, atol=1e-5)


@pytest.mark.parametrize('dtype', [Dtype.float64, Dtype.complex128])
def test_to_block(compatible_symmetry, compatible_backend, make_compatible_sectors, np_random, dtype):
    # need two test_* functions to generate the cases, implement actual test in check_to_block...
    uncoupled = make_compatible_sectors(4)
    check_to_block(compatible_symmetry, compatible_backend, uncoupled, np_random, dtype)


@pytest.mark.parametrize('dtype', [Dtype.float64, Dtype.complex128])
def test_to_block_no_backend(any_symmetry, make_any_sectors, np_random, dtype):
    # need two test_* functions to generate the cases, implement actual test in check_to_block
    coupled = make_any_sectors(4)
    check_to_block(any_symmetry, None, coupled, np_random, dtype)
