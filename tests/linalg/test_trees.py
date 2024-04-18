"""A collection of tests for tenpy.linalg.trees."""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import pytest

from tenpy.linalg import trees
from tenpy.linalg.symmetries import Symmetry
from tenpy.linalg.spaces import VectorSpace, ProductSpace


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
    allowed = ProductSpace([VectorSpace(any_symmetry, [a]) for a in uncoupled]).sectors
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
