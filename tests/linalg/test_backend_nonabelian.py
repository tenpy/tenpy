"""A collection of tests for tenpy.linalg.backends.fusion_tree_backend"""
# Copyright (C) TeNPy Developers, GNU GPLv3
import pytest

from tenpy.linalg.backends import fusion_tree_backend, get_backend
from tenpy.linalg.spaces import ProductSpace


@pytest.mark.parametrize('num_spaces', [3, 4, 5])
def test_block_sizes(any_symmetry, make_any_space, make_any_sectors, block_backend, num_spaces):
    backend = get_backend('fusion_tree', block_backend)
    spaces = [make_any_space() for _ in range(num_spaces)]
    domain = ProductSpace(spaces, symmetry=any_symmetry, backend=backend)

    for coupled in make_any_sectors(10):
        expect = sum(fusion_tree_backend.forest_block_size(domain, uncoupled, coupled)
                    for uncoupled in domain.iter_uncoupled())
        res = fusion_tree_backend.block_size(domain, coupled)
        assert res == expect
