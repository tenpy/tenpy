"""A collection of tests for cyten.tensors."""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

import operator
from contextlib import nullcontext
from functools import partial

import numpy as np
import numpy.testing as npt
import pytest

from cyten import BraidChiralityUnspecifiedError, backends, tensors
from cyten.backends import conventional_leg_order, get_backend
from cyten.block_backends import NumpyBlockBackend
from cyten.block_backends.dtypes import Dtype
from cyten.symmetries import (
    AbelianLegPipe,
    BraidingStyle,
    ElementarySpace,
    LegPipe,
    SU2Symmetry,
    SymmetryError,
    TensorProduct,
    u1_symmetry,
    z3_symmetry,
    z4_symmetry,
)
from cyten.tensors import ChargedTensor, DiagonalTensor, Mask, SymmetricTensor, Tensor
from cyten.testing import assert_tensors_almost_equal
from cyten.tools.misc import duplicate_entries, inverse_permutation, iter_common_noncommon_sorted_arrays, to_valid_idx

# TENSOR CLASSES


class DummyTensor(tensors.Tensor):
    """Want to test the Tensor class directly.

    This overrides the abstractmethods, so we can actually make instances.
    """

    def __init__(self, codomain, domain, backend, labels, dtype):
        super().__init__(codomain, domain, backend, labels, dtype, device='cpu')

    def copy(self, deep=True) -> tensors.Tensor:
        raise NotImplementedError

    def to_dense_block(self, leg_order: list[int | str] = None, dtype: Dtype = None, understood_braiding: bool = False):
        raise NotImplementedError

    def as_SymmetricTensor(self) -> SymmetricTensor:
        raise NotImplementedError

    def _get_item(self, idx: list[int]) -> bool | float | complex:
        raise NotImplementedError

    def move_to_device(self, device: str):
        raise NotImplementedError


def test_base_Tensor(make_compatible_space, compatible_backend):
    a, b, c, d, e = [make_compatible_space() for _ in range(5)]

    print('checking different labels input formats')
    tens1 = DummyTensor([a, b, c], [d, e], backend=compatible_backend, labels=None, dtype=Dtype.float64)
    tens1.test_sanity()
    assert tens1._labels == [None] * 5

    tens2 = DummyTensor(
        [a, b, c], [d, e], backend=compatible_backend, labels=['a', 'b', 'c', 'e*', 'd*'], dtype=Dtype.float64
    )
    tens2.test_sanity()
    assert tens2._labels == ['a', 'b', 'c', 'e*', 'd*']

    tens3 = DummyTensor(
        [a, b, c], [d, e], backend=compatible_backend, labels=[['a', None, None], ['d*', 'e*']], dtype=Dtype.float64
    )
    tens3.test_sanity()
    assert tens3._labels == ['a', None, None, 'e*', 'd*']

    tens4 = DummyTensor(
        [a, b, c], [d, e], backend=compatible_backend, labels=[['a', None, None], None], dtype=Dtype.float64
    )
    tens4.test_sanity()
    assert tens4._labels == ['a', None, None, None, None]

    print('checking .legs , .num_(co)domain_legs')
    for t in [tens1, tens2, tens3, tens4]:
        assert t.legs == [a, b, c, e.dual, d.dual]
        assert t.num_legs == 5
        assert t.num_codomain_legs == 3
        assert t.num_domain_legs == 2

        if t.symmetry.can_be_dropped:
            assert t.num_parameters <= t.size
        else:
            with pytest.raises(SymmetryError, match='not defined'):
                _ = t.size

    with pytest.raises(TypeError, match='does not support == comparison'):
        _ = tens1 == tens2

    print('checking .(co)domain_labels')
    assert tens1.codomain_labels == [None] * 3
    assert tens1.domain_labels == [None] * 2
    assert tens2.codomain_labels == ['a', 'b', 'c']
    assert tens2.domain_labels == ['d*', 'e*']
    assert tens3.codomain_labels == ['a', None, None]
    assert tens3.domain_labels == ['d*', 'e*']
    assert tens4.codomain_labels == ['a', None, None]
    assert tens4.domain_labels == [None, None]

    print('checking .is_fully_labelled')
    assert not tens1.is_fully_labelled
    assert tens2.is_fully_labelled
    assert not tens3.is_fully_labelled
    assert not tens4.is_fully_labelled

    print('check setting labels')
    tens1.labels = ['e', 'b', 'f', 'c', 'x']
    assert tens1._labels == ['e', 'b', 'f', 'c', 'x']

    print('check relabel')
    tens1.relabel(dict(e='xx', x='e'))
    assert tens1._labels == ['xx', 'b', 'f', 'c', 'e']

    print('check _parse_leg_idx')
    assert tens1._parse_leg_idx(1) == (False, 1, 1)
    assert tens1._parse_leg_idx(3) == (True, 1, 3)
    assert tens1._parse_leg_idx(-1) == (True, 0, 4)
    assert tens1._parse_leg_idx(-3) == (False, 2, 2)
    assert tens2._parse_leg_idx('a') == (False, 0, 0)
    assert tens2._parse_leg_idx('e*') == (True, 1, 3)

    print('check get_leg')
    assert tens2.get_leg(0) == a
    assert tens2.get_leg('b') == b
    assert tens2.get_leg('e*') == e.dual

    print('check has_label')
    assert tens2.has_label('a')
    assert tens2.has_label('a', 'b', 'e*')
    assert not tens2.has_label('foo')
    assert not tens2.has_label('a', 'b', '42')


@pytest.mark.parametrize('leg_nums', [(1, 1), (2, 1), (3, 0), (0, 3)], ids=['1->1', '1->2', '0->3', '3->0'])
@pytest.mark.parametrize('use_pipes', [True, 0.3])
def test_SymmetricTensor(make_compatible_tensor, leg_nums, use_pipes):
    T: SymmetricTensor = make_compatible_tensor(*leg_nums, use_pipes=use_pipes)

    print(T.ascii_diagram)

    T.test_sanity()
    assert T.num_codomain_legs == leg_nums[0]
    assert T.num_domain_legs == leg_nums[1]

    if not T.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    print('checking to_numpy')
    numpy_block = T.to_numpy(understood_braiding=True)
    dense_block = T.backend.block_backend.block_from_numpy(numpy_block)

    print('checking from_dense_block')
    tens = SymmetricTensor.from_dense_block(
        dense_block, codomain=T.codomain, domain=T.domain, backend=T.backend, understood_braiding=True
    )
    tens.test_sanity()
    tens_np = tens.to_numpy(understood_braiding=True)
    npt.assert_allclose(tens_np, numpy_block)

    can_have_non_symmetric_dense_blocks = T.num_parameters < T.size

    if can_have_non_symmetric_dense_blocks:  # otherwise all blocks are symmetric
        pert = tens.backend.block_backend.random_uniform(T.shape, dtype=T.dtype)
        non_symmetric_block = dense_block + pert
        with pytest.raises(ValueError, match='Block is not symmetric'):
            _ = SymmetricTensor.from_dense_block(
                non_symmetric_block, codomain=T.codomain, domain=T.domain, backend=T.backend, understood_braiding=True
            )

    # TODO: missing coverage:
    # - from_block_func / from_sector_block_func
    # - from_sector_projection
    # - diagonal

    print('checking from_zero')
    zero_tens = SymmetricTensor.from_zero(codomain=T.codomain, domain=T.domain, backend=T.backend)
    zero_tens.test_sanity()
    zero_tens_np = zero_tens.to_numpy(understood_braiding=True)
    npt.assert_array_almost_equal_nulp(zero_tens_np, np.zeros(T.shape), 10)

    print('checking repr and str')
    _ = str(T)
    _ = repr(T)
    _ = str(zero_tens)
    _ = repr(zero_tens)


@pytest.mark.parametrize('leg_num', [1, 2, 3])
def test_SymmetricTensor_from_eye(make_compatible_space, make_compatible_tensor, compatible_backend, leg_num):
    legs = [make_compatible_space() for _ in range(leg_num)]
    labels = list('abcdefg')[:leg_num]
    tens = SymmetricTensor.from_eye(legs, backend=compatible_backend, labels=labels)

    T1 = make_compatible_tensor(codomain=legs)
    eye_T1 = tensors.compose(tens, T1)
    assert tensors.almost_equal(eye_T1, T1)

    T2 = make_compatible_tensor(domain=legs)
    T2_eye = tensors.compose(T2, tens)
    assert tensors.almost_equal(T2_eye, T2)

    if tens.symmetry.can_be_dropped:
        expect_from_backend = compatible_backend.block_backend.to_numpy(
            compatible_backend.block_backend.eye_block([leg.dim for leg in legs], dtype=tens.dtype)
        )
        res = tens.to_numpy(understood_braiding=True)
        if leg_num == 1:
            expect_explicit = np.eye(legs[0].dim)
        elif leg_num == 2:
            expect_explicit = np.eye(legs[0].dim)[:, None, None, :] * np.eye(legs[1].dim)[None, :, :, None]
        elif leg_num == 3:
            expect_explicit = (
                np.eye(legs[0].dim)[:, None, None, None, None, :]
                * np.eye(legs[1].dim)[None, :, None, None, :, None]
                * np.eye(legs[2].dim)[None, None, :, :, None, None]
            )
        else:
            raise RuntimeError('Need to adjust test design')
        npt.assert_allclose(expect_from_backend, expect_explicit)
        npt.assert_allclose(res, expect_explicit, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize('leg_nums', [(1, 1), (2, 1), (3, 0), (0, 3)], ids=['1->1', '1->2', '0->3', '3->0'])
@pytest.mark.parametrize('dtype', [Dtype.float64, Dtype.complex128], ids=['real', 'complex'])
def test_SymmetricTensor_from_random_uniform(leg_nums, dtype, make_compatible_tensor):
    # use larger block size to reliably check distributions
    # based on some test runs, this corresponds to up to 9e5 samples
    # but may also only be 5e2 samples for a single leg in codomain and domain
    T: SymmetricTensor = make_compatible_tensor(*leg_nums, max_block_size=80)

    if T.symmetry.has_complex_topological_data and dtype.is_real:
        with pytest.raises(ValueError, match='must have complex dtype'):
            _ = SymmetricTensor.from_random_uniform(
                codomain=T.codomain, domain=T.domain, backend=T.backend, dtype=dtype
            )
        return

    rand_tens = SymmetricTensor.from_random_uniform(
        codomain=T.codomain, domain=T.domain, backend=T.backend, dtype=dtype
    )
    rand_tens.test_sanity()

    if not T.symmetry.can_be_dropped:
        # TODO what can we even check...?
        return

    if isinstance(T.backend, backends.NoSymmetryBackend):
        samples = np.asarray(rand_tens.data).flatten()
    else:
        samples = np.concatenate([np.asarray(block).flatten() for block in rand_tens.data.blocks])
    # test distribution if there are enough samples
    if samples.size > 1e4:
        true_mean = 0
        true_var = 1 / 3
        tol = 5 * samples.size**-0.5
        mean = np.mean(samples)
        if dtype.is_real:
            npt.assert_allclose(mean, true_mean, atol=tol)
            npt.assert_allclose(np.var(samples), true_var, atol=tol)
            assert samples[samples < -1].size == 0
            assert samples[samples > 1].size == 0
            # probability of happening: 0.999 ** samples.size; smaller than 4.5e-5
            assert samples[samples < -0.998].size > 0
            assert samples[samples > 0.998].size > 0
        else:
            npt.assert_allclose(mean.real, true_mean, atol=tol)
            npt.assert_allclose(mean.imag, true_mean, atol=tol)
            npt.assert_allclose(np.var(samples.real), true_var, atol=tol)
            npt.assert_allclose(np.var(samples.imag), true_var, atol=tol)
            assert samples[samples.real < -1].size == 0
            assert samples[samples.imag < -1].size == 0
            assert samples[samples.real > 1].size == 0
            assert samples[samples.imag > 1].size == 0
            assert samples[samples.real < -0.998].size > 0
            assert samples[samples.imag < -0.998].size > 0
            assert samples[samples.real > 0.998].size > 0
            assert samples[samples.imag > 0.998].size > 0


@pytest.mark.parametrize('leg_nums', [(1, 1), (2, 1), (3, 0), (0, 3)], ids=['1->1', '1->2', '0->3', '3->0'])
@pytest.mark.parametrize('dtype', [Dtype.float64, Dtype.complex128], ids=['real', 'complex'])
def test_SymmetricTensor_from_random_normal(leg_nums, dtype, make_compatible_tensor, np_random):
    # use larger block size to reliably check distributions
    # based on some test runs, this corresponds to up to 9e5 samples
    # but may also only be 5e2 samples for a single leg in codomain and domain
    T: SymmetricTensor = make_compatible_tensor(*leg_nums, max_block_size=80)
    # TODO do we want to test nontrivial means?
    sigma = np_random.uniform(high=3.0)

    if T.symmetry.has_complex_topological_data and dtype.is_real:
        with pytest.raises(ValueError, match='must have complex dtype'):
            _ = SymmetricTensor.from_random_normal(
                codomain=T.codomain, domain=T.domain, sigma=sigma, backend=T.backend, dtype=dtype
            )
        return

    rand_tens = SymmetricTensor.from_random_normal(
        codomain=T.codomain, domain=T.domain, sigma=sigma, backend=T.backend, dtype=dtype
    )
    rand_tens.test_sanity()

    if not T.symmetry.can_be_dropped:
        return  # TODO what can we even test?

    if isinstance(T.backend, backends.NoSymmetryBackend):
        samples = np.asarray(rand_tens.data).flatten()
    else:
        samples = np.concatenate([np.asarray(block).flatten() for block in rand_tens.data.blocks])
    # test distribution if there are enough samples
    if samples.size > 1e4:
        true_mean = 0
        true_var = sigma**2
        tol = 10 * samples.size**-0.5
        mean = np.mean(samples)
        if dtype.is_real:
            npt.assert_allclose(mean, true_mean, atol=tol)
            npt.assert_allclose(np.var(samples), true_var, rtol=tol)
        else:
            npt.assert_allclose(mean.real, true_mean, atol=tol)
            npt.assert_allclose(mean.imag, true_mean, atol=tol)
            npt.assert_allclose(np.var(samples.real), true_var / 2, rtol=tol)
            npt.assert_allclose(np.var(samples.imag), true_var / 2, rtol=tol)
            # absolute values are Rayleigh distributed
            npt.assert_allclose(np.mean(np.abs(samples)), np.sqrt(np.pi / 4) * sigma, atol=tol)
            npt.assert_allclose(np.var(np.abs(samples)), (4 - np.pi) / 4 * true_var, rtol=tol)


@pytest.mark.parametrize('leg_nums', [(1, 1), (2, 1), (3, 0), (0, 3)], ids=['1->1', '1->2', '0->3', '3->0'])
def test_SymmetricTensor_from_tree_pairs(make_compatible_tensor, leg_nums, np_random):
    T: SymmetricTensor = make_compatible_tensor(*leg_nums, use_pipes=False)

    numpy_block_backend = NumpyBlockBackend()
    # build two valid dicts
    trees1 = {}
    trees2 = {}
    for coupled in T.codomain.sector_decomposition:
        for X, _, mults1, _ in T.codomain.iter_tree_blocks([coupled]):
            for Y, _, mults2, _ in T.domain.iter_tree_blocks([coupled]):
                shape = [*mults1, *reversed(mults2)]
                if len(trees1) == 0 or np_random.choice([True, False]):
                    trees1[X, Y] = np_random.uniform(size=shape) + 1j * np_random.uniform(size=shape)
                if len(trees2) == 0 or np_random.choice([True, False]):
                    trees2[X, Y] = np_random.uniform(size=shape) + 1j * np_random.uniform(size=shape)
    # build the dict for the linear combination T3 = a * T1 + b * T2
    a = 1.3
    b = -3 + 4.7j
    trees3 = {}
    for pair, block in trees1.items():
        trees3[pair] = a * block
    for pair, block in trees2.items():
        trees3[pair] = trees3.get(pair, 0) + b * block

    T_from_trees1 = SymmetricTensor.from_tree_pairs(trees1, T.codomain, T.domain, backend=T.backend)
    T_from_trees1.test_sanity()
    T_from_trees2 = SymmetricTensor.from_tree_pairs(trees2, T.codomain, T.domain, backend=T.backend)
    T_from_trees2.test_sanity()
    T_from_trees3 = SymmetricTensor.from_tree_pairs(trees3, T.codomain, T.domain, backend=T.backend)
    T_from_trees3.test_sanity()

    assert tensors.almost_equal(T_from_trees3, a * T_from_trees1 + b * T_from_trees2)

    # compare to numpy
    if T.symmetry.can_be_dropped:
        for T_res, trees_dict in zip([T_from_trees1, T_from_trees2, T_from_trees3], [trees1, trees2, trees3]):
            T_np = T_res.to_numpy(understood_braiding=True)
            expect = np.zeros_like(T_np)
            for (X, Y), block in trees_dict.items():
                # [a1...aJ,b1...bK]
                symmetry_data = np.tensordot(X.as_block().conj(), Y.as_block(), (-1, -1))
                # [a1...aJ,b1...bK] & [a1...aJ,bK...b1]
                symmetry_data = np.transpose(
                    symmetry_data, [*range(T.num_codomain_legs), *reversed(range(T.num_codomain_legs, T.num_legs))]
                )
                contribution = np.kron(symmetry_data, block)
                codom_idcs = [
                    slice(*l.slices[l.sector_decomposition_where(a)]) for l, a in zip(T.codomain, X.uncoupled)
                ]
                dom_idcs = [slice(*l.slices[l.sector_decomposition_where(b)]) for l, b in zip(T.domain, Y.uncoupled)]
                expect[(*codom_idcs, *reversed(dom_idcs))] += contribution
            expect = numpy_block_backend.apply_basis_perm(expect, conventional_leg_order(T_res), inv=True)

            npt.assert_array_almost_equal(T_np, expect)

    # TODO test to_dense_block_trivial_sector
    # def OLD_test_Tensor_tofrom_dense_block_trivial_sector(make_compatible_tensor):
    #     # TODO move to SymmetricTensor test?
    #     tens = make_compatible_tensor(labels=['a'])
    #     leg, = tens.legs
    #     block_size = leg.sector_multiplicity(tens.symmetry.trivial_sector)
    #     #
    #     if isinstance(tens.backend, backends.FusionTreeBackend):
    #         with pytest.raises(NotImplementedError, match='to_dense_block_trivial_sector not implemented'):
    #             block = tens.to_dense_block_trivial_sector()
    #         return  # TODO
    #     #
    #     block = tens.to_dense_block_trivial_sector()
    #     assert tens.backend.block_shape(block) == (block_size,)
    #     tens2 = SymmetricTensor.from_dense_block_trivial_sector(leg=leg, block=block, backend=tens.backend, label='a')
    #     tens2.test_sanity()
    #     assert tensors.almost_equal(tens, tens2)
    #     block2 = tens2.to_dense_block_trivial_sector()
    #     npt.assert_array_almost_equal_nulp(tens.backend.block_to_numpy(block),
    #                                     tens.backend.block_to_numpy(block2),
    #                                     100)


def test_fixes_124(np_random):
    """Check if the bug discussed in PR #124 is fixed"""
    symm = SU2Symmetry()
    backend = get_backend(symm, 'numpy')
    a = ElementarySpace(symm, [[1]], [1])
    b = ElementarySpace(symm, [[1]], [1])
    c = ElementarySpace(symm, [[2]], [2])
    codomain = TensorProduct([a, b])
    domain = TensorProduct([c])

    trees = {}
    for coupled in codomain.sector_decomposition:
        for X, _, mults1, _ in codomain.iter_tree_blocks([coupled]):
            for Y, _, mults2, _ in domain.iter_tree_blocks([coupled]):
                shape = [*mults1, *reversed(mults2)]
                if len(trees) == 0 or np_random.choice([True, False]):
                    trees[X, Y] = np.ones(shape, float)

    T = SymmetricTensor.from_tree_pairs(trees, codomain, domain, backend=backend)
    T.test_sanity()

    T_np = T.to_numpy()
    expect = np.zeros_like(T_np)
    for (X, Y), block in trees.items():
        symmetry_data = np.tensordot(X.as_block().conj(), Y.as_block(), (-1, -1))
        # [a1...aJ,b1...bK] & [a1...aJ,bK...b1]
        symmetry_data = np.transpose(
            symmetry_data, [*range(T.num_codomain_legs), *reversed(range(T.num_codomain_legs, T.num_legs))]
        )
        contribution = np.kron(symmetry_data, block)
        codom_idcs = [slice(*l.slices[l.sector_decomposition_where(a)]) for l, a in zip(T.codomain, X.uncoupled)]
        dom_idcs = [slice(*l.slices[l.sector_decomposition_where(b)]) for l, b in zip(T.domain, Y.uncoupled)]
        expect[(*codom_idcs, *reversed(dom_idcs))] += contribution
    expect = backend.block_backend.apply_basis_perm(expect, conventional_leg_order(T), inv=True)
    npt.assert_array_almost_equal(T_np, expect)


def test_fixes_23():
    # See PR #23
    sym = SU2Symmetry()
    site = ElementarySpace(sym, sym.spin_half[None, :])
    block = np.zeros((2,) * 6, float)
    tens = SymmetricTensor.from_dense_block(block, codomain=[site] * 3, domain=[site] * 3)
    tens.test_sanity()
    npt.assert_allclose(tensors.norm(tens), 0)


def test_DiagonalTensor(make_compatible_tensor):
    T: DiagonalTensor = make_compatible_tensor(cls=DiagonalTensor)
    T.test_sanity()

    if not T.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    print('checking diagonal_as_numpy')
    np_diag = T.diagonal_as_numpy()

    print('checking from_diag_block')
    tens = DiagonalTensor.from_diag_block(np_diag, leg=T.leg, backend=T.backend)
    tens.test_sanity()
    res = tens.diagonal_as_numpy()
    npt.assert_array_almost_equal_nulp(res, np_diag, 100)

    print('checking to_numpy')
    np_full = T.to_numpy(understood_braiding=True)
    npt.assert_array_almost_equal_nulp(np_full, np.diag(np_diag), 100)

    print('checking from zero')
    zero_tens = DiagonalTensor.from_zero(T.leg, backend=T.backend)
    zero_tens.test_sanity()
    npt.assert_array_almost_equal_nulp(zero_tens.diagonal_as_numpy(), np.zeros_like(np_diag), 100)

    print('checking from eye')
    tens = DiagonalTensor.from_eye(T.leg, backend=T.backend)
    tens.test_sanity()
    npt.assert_array_almost_equal_nulp(tens.diagonal_as_numpy(), np.ones_like(np_diag), 100)

    # TODO from_random_*
    # TODO from_tensor

    print('checking repr and str')
    _ = str(T)
    _ = repr(T)
    _ = str(zero_tens)
    _ = repr(zero_tens)

    # TODO elementwise dunder methods. loop over operator.XXX functions?
    # TODO float(), complex(), bool()

    print('checking min / max')
    real_T = tensors.real(T)
    real_np = real_T.diagonal_as_numpy()
    npt.assert_almost_equal(real_T.max(), np.max(real_np))
    npt.assert_almost_equal(real_T.min(), np.min(real_np))


def test_Mask(make_compatible_tensor, compatible_symmetry_backend, np_random):
    M_projection: Mask = make_compatible_tensor(cls=Mask)
    backend = M_projection.backend
    symmetry = M_projection.symmetry
    large_leg = M_projection.domain[0]
    small_leg = M_projection.codomain[0]

    assert M_projection.is_projection is True
    M_projection.test_sanity()
    if symmetry.can_be_dropped:
        M_projection_np = M_projection.as_numpy_mask()

    print('checking inclusion Mask')
    M_inclusion: Mask = tensors.dagger(M_projection)
    assert M_inclusion.is_projection is False
    M_inclusion.test_sanity()

    print('checking properties')
    assert M_projection.large_leg == large_leg
    assert M_inclusion.large_leg == large_leg
    assert M_projection.small_leg == small_leg
    assert M_inclusion.small_leg == small_leg

    print('checking from_eye')
    for is_projection in [True, False]:
        M_eye = Mask.from_eye(large_leg, is_projection=is_projection, backend=backend)
        assert M_eye.is_projection is is_projection
        M_eye.test_sanity()

    if symmetry.can_be_dropped:
        # checks that rely on dense block representations
        print('checking from_block_mask / as_block_mask')
        block_mask = np_random.choice([True, False], large_leg.dim, replace=True)
        if not M_projection.symmetry.fusion_style.value == 0:
            # can only have True XOR False for all entries corresponding to the same multiplicity index
            for slc, sector in zip(large_leg.slices, large_leg.defining_sectors):
                dim = symmetry.sector_dim(sector)
                if dim > 1:
                    slc = slice(*slc)
                    stop = int(len(block_mask[slc]) // dim)
                    block_mask[slc] = np.tile(block_mask[slc][:stop], dim)
            block_mask = backend.block_backend.apply_basis_perm(block_mask, [large_leg], inv=True)
        M = Mask.from_block_mask(block_mask, large_leg=large_leg, backend=backend)
        M.test_sanity()
        assert M.large_leg == large_leg
        assert M.small_leg.dim == np.sum(block_mask)
        npt.assert_array_equal(M.as_numpy_mask(), block_mask)

        print('checking from_indices')
        indices = np.where(block_mask)[0]
        M = Mask.from_indices(indices, large_leg=large_leg, backend=backend)
        M.test_sanity()
        assert M.large_leg == large_leg
        assert M.small_leg.dim == np.sum(block_mask)
        npt.assert_array_equal(M.as_numpy_mask(), block_mask)

    print('checking from_DiagonalTensor / as_DiagonalTensor')
    diag = M_projection.as_DiagonalTensor(dtype=Dtype.float32)
    assert diag.leg == large_leg
    assert diag.dtype == Dtype.float32
    diag.test_sanity()
    #
    diag = M_projection.as_DiagonalTensor(dtype=Dtype.bool)
    assert diag.leg == large_leg
    assert diag.dtype == Dtype.bool
    diag.test_sanity()
    M = Mask.from_DiagonalTensor(diag)
    if symmetry.can_be_dropped:
        npt.assert_array_equal(M_projection_np, M.as_numpy_mask())
    assert (M == M_projection).all()
    #
    diag = M_inclusion.as_DiagonalTensor(dtype=Dtype.bool)
    assert diag.leg == large_leg
    assert diag.dtype == Dtype.bool
    diag.test_sanity()
    M = Mask.from_DiagonalTensor(diag)  # should reproduce the *projection* Mask.
    if symmetry.can_be_dropped:
        npt.assert_array_equal(M_projection_np, M.as_numpy_mask())
    assert (M == M_projection).all()

    print('checking from_random')
    M = Mask.from_random(large_leg, small_leg=None, backend=backend)
    M.test_sanity()
    # specifying small_leg is currently only possible if the legs have no permutation
    large_leg_no_perm = ElementarySpace(M.symmetry, large_leg.defining_sectors, large_leg.multiplicities)
    M2 = Mask.from_random(large_leg_no_perm, small_leg=None, backend=backend)
    M2.test_sanity()
    M3 = Mask.from_random(large_leg_no_perm, small_leg=M2.small_leg, backend=backend)
    M3.test_sanity()
    assert M2.small_leg == M3.small_leg

    print('checking from_zero')
    M_zero = Mask.from_zero(large_leg, backend=backend)
    M_zero.test_sanity()
    assert M_zero.small_leg.dim == 0

    print('checking bool()')
    with pytest.raises(TypeError, match='The truth value of a Mask is ambiguous.'):
        _ = bool(M_projection)

    print('checking .any() and .all()')
    if symmetry.can_be_dropped:
        assert M_projection.all() == np.all(M_projection_np)
        assert M_inclusion.all() == np.all(M_projection_np)
        assert M_projection.any() == np.any(M_projection_np)
        assert M_inclusion.any() == np.any(M_projection_np)
    assert M_eye.all()
    assert M_eye.any()
    assert not M_zero.all()
    assert not M_zero.any()

    print('checking to_numpy vs as_SymmetricTensor')
    M_SymmetricTensor = M_projection.as_SymmetricTensor()
    assert M_SymmetricTensor.shape == M_projection.shape
    M_SymmetricTensor.test_sanity()
    if symmetry.can_be_dropped:
        res_via_Symmetric = M_SymmetricTensor.to_numpy(understood_braiding=True)
        res_direct = M_projection.to_numpy(understood_braiding=True)
        npt.assert_allclose(res_via_Symmetric, res_direct)
    print('   also for inclusion Mask')
    M_SymmetricTensor = M_inclusion.as_SymmetricTensor()
    assert M_SymmetricTensor.shape == M_inclusion.shape
    M_SymmetricTensor.test_sanity()
    if symmetry.can_be_dropped:
        res_via_Symmetric = M_SymmetricTensor.to_numpy(understood_braiding=True)
        res_direct = M_inclusion.to_numpy(understood_braiding=True)
        npt.assert_allclose(res_via_Symmetric, res_direct)

    # TODO check binary operands: &, ==, !=, &, |, ^ :
    #   left and right
    #   with bool and with other mask
    #   with projection Masks and with inclusion Masks

    # TODO check orthogonal complement, also for inclusion Mask!

    # TODO check exactly one True per column and at most one True per row?

    # TODO check to_numpy() vs as_numpy_mask() (if not already done...)

    print('checking repr and str')
    _ = str(M_projection)
    _ = repr(M_projection)
    _ = str(M_inclusion)
    _ = repr(M_inclusion)
    _ = str(M_zero)
    _ = repr(M_zero)


@pytest.mark.deselect_invalid_ChargedTensor_cases(get_cls=lambda kw: ChargedTensor)
@pytest.mark.parametrize('leg_nums', [(1, 1), (2, 1), (3, 0), (0, 3)], ids=['1->1', '1->2', '0->3', '3->0'])
def test_ChargedTensor(make_compatible_tensor, make_compatible_sectors, compatible_symmetry, leg_nums):
    T: ChargedTensor = make_compatible_tensor(*leg_nums, cls=ChargedTensor)
    backend = T.backend

    T.test_sanity()
    assert T.num_codomain_legs == leg_nums[0]
    assert T.num_domain_legs == leg_nums[1]

    if not T.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    print('checking to_numpy')
    numpy_block = T.to_numpy(understood_braiding=True)
    assert T.shape == numpy_block.shape

    print('checking from_zero')
    zero_tens = ChargedTensor.from_zero(
        codomain=T.codomain, domain=T.domain, charge=T.charge_leg, charged_state=T.charged_state, backend=backend
    )
    zero_tens.test_sanity()
    zero_tens_np = zero_tens.to_numpy(understood_braiding=True)
    npt.assert_array_almost_equal_nulp(zero_tens_np, np.zeros(T.shape), 10)

    print('checking repr and str')
    _ = str(T)
    _ = repr(T)
    _ = str(zero_tens)
    _ = repr(zero_tens)

    print('checking to/from dense_block_single_sector')
    sector = make_compatible_sectors(1)[0]
    charge_leg = ElementarySpace(compatible_symmetry, sector[None, :])
    inv_part = make_compatible_tensor(codomain=1, domain=[charge_leg], labels=[None, '!'])
    tens = ChargedTensor(inv_part, charged_state=[1])
    leg = tens.codomain[0]

    if isinstance(backend, backends.FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='inv_part_to_dense_block_single_sector not implemented'):
            _ = tens.to_dense_block_single_sector()
        pytest.xfail('FTBackend does not support ChargedTensor.to_dense_block_single_sector yet')

    block = tens.to_dense_block_single_sector()
    if isinstance(leg, ElementarySpace):
        block_size = leg.sector_multiplicity(sector)
        assert backend.block_backend.get_shape(block) == (block_size,)
    else:
        # TODO for LegPipe, cant to .sector_multiplicity to get the expected size
        pass

    with pytest.raises(NotImplementedError):
        tens2 = ChargedTensor.from_dense_block_single_sector(vector=block, space=leg, sector=sector, backend=backend)
        tens2.test_sanity()
        assert tens2.charge_leg == tens.charge_leg
        assert tensors.almost_equal(tens, tens2)
        block2 = tens2.to_dense_block_single_sector()
        npt.assert_array_almost_equal_nulp(
            tens.backend.block_backend.to_numpy(block), tens.backend.block_backend.to_numpy(block2), 100
        )

    # TODO test to_dense_block_single_sector

    # def OLD_test_ChargedTensor_tofrom_dense_block_single_sector(compatible_symmetry, make_compatible_sectors,
    #                                                     make_compatible_tensor):
    #     pytest.xfail(reason='unclear')  # TODO
    #     # TODO revise this. purge the "dummy" language, its now "charged"
    #     # TODO move to ChargedTensor test?
    #     sector = make_compatible_sectors(1)[0]
    #     dummy_leg = Space(compatible_symmetry, sector[None, :]).dual
    #     inv_part = make_compatible_tensor(legs=[None, dummy_leg])
    #     tens = ChargedTensor(invariant_part=inv_part)
    #     leg = tens.legs[0]
    #     block_size = leg.sector_multiplicity(sector)
    #     #
    #     block = tens.to_flat_block_single_sector()
    #     assert tens.backend.block_shape(block) == (block_size,)
    #     tens2 = ChargedTensor.from_flat_block_single_sector(
    #         leg=leg, block=block, sector=sector, backend=tens.backend
    #     )
    #     tens2.test_sanity()
    #     assert tens2.dummy_leg == tens.dummy_leg
    #     assert tensors.almost_equal(tens, tens2)
    #     block2 = tens2.to_flat_block_single_sector()
    #     npt.assert_array_almost_equal_nulp(tens.backend.block_to_numpy(block),
    #                                     tens.backend.block_to_numpy(block2),
    #                                     100)
    #     # check detect_sectors_from_block while we are at it
    #     dense_block = tens.to_dense_block()
    #     detected, = tensors.detect_sectors_from_block(block=dense_block, legs=[leg], backend=tens.backend)
    #     npt.assert_array_equal(detected, sector)


@pytest.mark.parametrize('symmetry_backend', ['abelian', pytest.param('fusion_tree', marks=pytest.mark.FusionTree)])
def test_explicit_blocks(symmetry_backend, block_backend):
    """Do detailed tests with concrete examples.

    Convert a small dense block to a Tensor.
    Construct the expected data (blocks) manually and compare.

    This is useful e.g. for debugging from_dense_block and to check that the data format is
    what we expect.
    """
    backend = get_backend(symmetry_backend, block_backend)
    all_qi = z4_symmetry.all_sectors()
    q0, q1, q2, q3 = all_qi
    basis1 = [q3, q3, q2, q0, q3, q2]  # basis_perm [3, 2, 5, 0, 1, 4]
    basis2 = [q2, q0, q1, q2, q3, q0, q1]  # basis_perm [1, 5, 2, 6, 0, 3, 4]
    s1 = ElementarySpace.from_basis(z4_symmetry, basis1)  # sectors = [0, 2, 3]
    s2 = ElementarySpace.from_basis(z4_symmetry, basis2)  # sectors = [0, 1, 2, 3]

    print(f'\n\nBOTH LEGS IN CODOMAIN:\n')

    #             s2 : 2,  0,  1,  2,  3,  0,  1     s1
    data = np.array(
        [
            [0, 0, 1, 0, 0, 0, 2],  # 3
            [0, 0, 3, 0, 0, 0, 4],  # 3
            [5, 0, 0, 6, 0, 0, 0],  # 2
            [0, 7, 0, 0, 0, 8, 0],  # 0
            [0, 0, 9, 0, 0, 0, 10],  # 3
            [11, 0, 0, 12, 0, 0, 0],
        ],  # 2
        dtype=float,
    )

    print('after applying basis perm:')
    print(data[np.ix_(s1.basis_perm, s2.basis_perm)])
    # q: 0   0   1   1   2   2   3       q
    # [[ 7.  8.  0.  0.  0.  0.  0.]     0
    #  [ 0.  0.  0.  0.  5.  6.  0.]     2
    #  [ 0.  0.  0.  0. 11. 12.  0.]     2
    #  [ 0.  0.  1.  2.  0.  0.  0.]     3
    #  [ 0.  0.  3.  4.  0.  0.  0.]     3
    #  [ 0.  0.  9. 10.  0.  0.  0.]]    3
    block_00 = np.asarray([[7, 8]])
    block_31 = np.asarray([[1, 2], [3, 4], [9, 10]])
    block_22 = np.asarray([[5, 6], [11, 12]])

    # non-symmetric block:
    non_symmetric_data = data.copy()
    non_symmetric_data[0, 0] = 42
    with pytest.raises(ValueError, match='not symmetric'):
        t = SymmetricTensor.from_dense_block(non_symmetric_data, codomain=[s1, s2], backend=backend)
    # now continue with the symmetric block

    t = SymmetricTensor.from_dense_block(data, codomain=[s1, s2], backend=backend)
    t.test_sanity()

    # explicitly check the ``t.data`` vs what we expect
    if symmetry_backend == 'abelian':
        # listing this in an order such that the resulting block_inds are lexsorted:
        # blocks allowed for q:                [] -> [0, 0]  ;  [] -> [3, 1]  ;  [] -> [2, 2]
        # indices in .sector_decomposition:    [] -> [0, 0]  ;  [] -> [2, 1]  ;  [] -> [1, 2]
        expect_block_inds = np.array([[0, 0], [2, 1], [1, 2]])
        expect_blocks = [block_00, block_31, block_22]
        #
        valid_block_inds = backends.abelian._valid_block_inds(t.codomain, t.domain)
        npt.assert_array_equal(expect_block_inds, valid_block_inds)
        #
        assert len(expect_blocks) == len(t.data.blocks)
        for i, (actual, expect) in enumerate(zip(t.data.blocks, expect_blocks)):
            print(f'checking blocks[{i}]')
            npt.assert_array_almost_equal_nulp(t.backend.block_backend.to_numpy(actual), expect, 100)

    elif symmetry_backend == 'fusion_tree':
        assert np.all(t.data.block_inds == np.array([0, 0])[None, :])
        forest_block_00 = block_00.reshape((-1, 1))
        forest_block_22 = block_22.reshape((-1, 1))
        forest_block_31 = block_31.reshape((-1, 1))
        # forest blocks are sorted C-style, i.e. first by first row.
        expect_block = np.concatenate([forest_block_00, forest_block_22, forest_block_31], axis=0)
        assert len(t.data.blocks) == 1
        actual = t.backend.block_backend.to_numpy(t.data.blocks[0])
        npt.assert_array_almost_equal_nulp(actual, expect_block, 100)

    else:
        raise RuntimeError

    # check conversion back
    npt.assert_array_almost_equal_nulp(t.to_numpy(), data)

    # =======================================================
    # =======================================================
    # =======================================================
    print(f'\n\nONE LEG EACH IN DOMAIN AND CODOMAIN:\n')
    # note that this setup changes the charge rule! different entries are now allowed than before

    #             s2 : 2,  0,  1,  2,  3,  0,  1     s1
    data = np.array(
        [
            [0, 0, 0, 0, -1, 0, 0],  # 3
            [0, 0, 0, 0, -2, 0, 0],  # 3
            [5, 0, 0, 6, 0, 0, 0],  # 2
            [0, 7, 0, 0, 0, 8, 0],  # 0
            [0, 0, 0, 0, -3, 0, 0],  # 3
            [11, 0, 0, 12, 0, 0, 0],
        ],  # 2
        dtype=float,
    )

    print('after applying basis perm:')
    print(data[np.ix_(s1.basis_perm, s2.basis_perm)])
    # q: 0   0   1   1   2   2   3      q
    # [[ 7.  8.  0.  0.  0.  0.  0.]    0
    #  [ 0.  0.  0.  0.  5.  6.  0.]    2
    #  [ 0.  0.  0.  0. 11. 12.  0.]    2
    #  [ 0.  0.  0.  0.  0.  0. -1.]    3
    #  [ 0.  0.  0.  0.  0.  0. -2.]    3
    #  [ 0.  0.  0.  0.  0.  0. -3.]]   3
    block_00 = np.asarray([[7, 8]])
    block_22 = np.asarray([[5, 6], [11, 12]])
    block_33 = np.asarray([[-1], [-2], [-3]])

    # non-symmetric block:
    non_symmetric_data = data.copy()
    non_symmetric_data[0, 0] = 42
    with pytest.raises(ValueError, match='not symmetric'):
        t = SymmetricTensor.from_dense_block(non_symmetric_data, codomain=[s1], domain=[s2], backend=backend)
    # now continue with the symmetric block

    t = SymmetricTensor.from_dense_block(data, codomain=[s1], domain=[s2], backend=backend)
    t.test_sanity()

    # explicitly check the ``t.data`` vs what we expect
    if symmetry_backend == 'abelian':
        # listing this in an order such that the resulting block_inds are lexsorted:
        # blocks allowed for q:               [0] -> [0]  ;  [2] -> [2]  ;  [3] -> [3]
        # indices in .sector_decomposition:   [0] -> [0]  ;  [2] -> [1]  ;  [3] -> [2]
        # block_inds row:                     [0, 0]      ;  [1, 2]      ;  [2, 3]
        expect_block_inds = np.array([[0, 0], [1, 2], [2, 3]])
        expect_blocks = [block_00, block_22, block_33]
        #
        valid_block_inds = backends.abelian._valid_block_inds(t.codomain, t.domain)
        npt.assert_array_equal(expect_block_inds, valid_block_inds)
        #
        assert len(expect_blocks) == len(t.data.blocks)
        for i, (actual, expect) in enumerate(zip(t.data.blocks, expect_blocks)):
            print(f'checking blocks[{i}]')
            npt.assert_array_almost_equal_nulp(t.backend.block_backend.to_numpy(actual), expect, 100)

    elif symmetry_backend == 'fusion_tree':
        # expect coupled sectors q0, q2, q3.
        # codomain.sector_decomposition == [q0, q2, q3]
        # domain.sector_decomposition == [q0, q1, q2, q3]
        npt.assert_array_equal(t.data.block_inds, np.array([[0, 0], [1, 2], [2, 3]]))
        expect_blocks = [block_00, block_22, block_33]
        assert len(expect_blocks) == len(t.data.blocks)
        for i, (actual, expect) in enumerate(zip(t.data.blocks, expect_blocks)):
            print(f'checking blocks[{i}]')
            npt.assert_array_almost_equal_nulp(t.backend.block_backend.to_numpy(actual), expect, 100)

    else:
        raise RuntimeError

    # check conversion back
    npt.assert_array_almost_equal_nulp(t.to_numpy(), data)

    # =======================================================
    # =======================================================
    # =======================================================
    print(f'\n\nFOUR LEG EXAMPLE (2 -> 2):\n')
    s = ElementarySpace.from_basis(z4_symmetry, [q1, q0, q2])  # basis_perm [1, 0, 2]
    data = np.zeros((3, 3, 3, 3), float)
    # set the allowed elements manually
    # note the leg order for the dense array is [*codomain, *reversed(domain)] !!
    #                        SECTORS PER LEG  |  DOMAIN -> coupled -> CODOMAIN
    data[1, 1, 1, 1] = 1  # [0, 0, 0, 0]     |  [0, 0] -> 0 -> [0, 0]
    data[1, 1, 2, 2] = 2  # [0, 0, 2, 2]     |  [2, 2] -> 0 -> [0, 0]
    data[2, 2, 1, 1] = 3  # [2, 2, 0, 0]     |  [0, 0] -> 0 -> [2, 2]
    data[2, 2, 2, 2] = 4  # [2, 2, 2, 2]     |  [2, 2] -> 0 -> [2, 2]
    #
    data[0, 1, 0, 1] = 5  # [1, 0, 1, 0]     |  [0, 1] -> 1 -> [1, 0]
    data[0, 1, 1, 0] = 6  # [1, 0, 0, 1]     |  [1, 0] -> 1 -> [1, 0]
    #
    data[0, 0, 0, 0] = 7  # [1, 1, 1, 1]     |  [1, 1] -> 2 -> [1, 1]
    data[0, 0, 1, 2] = 8  # [1, 1, 0, 2]     |  [2, 0] -> 2 -> [1, 1]
    data[1, 2, 0, 0] = 9  # [0, 2, 1, 1]     |  [1, 1] -> 2 -> [0, 2]
    data[1, 2, 1, 2] = 10  # [0, 2, 0, 2]     |  [2, 0] -> 2 -> [0, 2]
    #
    data[0, 2, 0, 2] = 11  # [1, 2, 1, 2]     |  [2, 1] -> 3 -> [1, 2]
    data[0, 2, 2, 0] = 12  # [1, 2, 2, 1]     |  [1, 2] -> 3 -> [1, 2]
    data[2, 0, 0, 2] = 13  # [2, 1, 1, 2]     |  [2, 1] -> 3 -> [2, 1]
    data[2, 0, 2, 0] = 14  # [2, 1, 2, 1]     |  [1, 2] -> 3 -> [2, 1]

    # non-symmetric block:
    non_symmetric_data = data.copy()
    non_symmetric_data[0, 0, 1, 1] = 42
    with pytest.raises(ValueError, match='not symmetric'):
        t = SymmetricTensor.from_dense_block(non_symmetric_data, codomain=[s, s], domain=[s, s], backend=backend)
    # now continue with the symmetric block

    t = SymmetricTensor.from_dense_block(data, codomain=[s, s], domain=[s, s], backend=backend)
    t.test_sanity()

    # explicitly check the ``t.data`` vs what we expect
    if symmetry_backend == 'abelian':
        # all sectors appear only once, so each allowed entry is its own block.
        # In this case, the value of a sector is also its index in s.sector_decomposition
        # Thus the block inds are just the "SECTORS PER LEG" above.
        expect_block_inds = np.asarray(
            [
                [0, 0, 0, 0],
                [0, 0, 2, 2],
                [2, 2, 0, 0],
                [2, 2, 2, 2],
                [1, 0, 1, 0],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
                [1, 1, 0, 2],
                [0, 2, 1, 1],
                [0, 2, 0, 2],
                [1, 2, 1, 2],
                [1, 2, 2, 1],
                [2, 1, 1, 2],
                [2, 1, 2, 1],
            ],
            dtype=int,
        )
        expect_blocks = [np.asarray([[x]], dtype=float) for x in range(1, 15)]
        perm = np.lexsort(expect_block_inds.T)
        expect_block_inds = expect_block_inds[perm]
        expect_blocks = [expect_blocks[n] for n in perm]
        #
        # have not set all entries, so expect_block_inds should be a subset of _valid_block_inds
        valid_block_inds = backends.abelian._valid_block_inds(t.codomain, t.domain)
        for i, j in iter_common_noncommon_sorted_arrays(expect_block_inds, valid_block_inds):
            assert j is not None  # j=None would mean that the row of expect_block_inds is not in valid_block_inds
            actual_block = t.backend.block_backend.to_numpy(t.data.blocks[j])
            if i is None:
                expect_block = np.zeros_like(actual_block)
            else:
                expect_block = expect_blocks[i]
            npt.assert_array_almost_equal_nulp(actual_block, expect_block, 100)

    elif symmetry_backend == 'fusion_tree':
        # check block_inds. first make sure the (co)domain.sector_decomposition are what we expect
        assert np.all(t.codomain.sector_decomposition == np.stack([q0, q1, q2, q3]))
        assert np.all(t.domain.sector_decomposition == np.stack([q0, q1, q2, q3]))
        # expect coupled sectors [q0, q1, q2, q3]
        assert np.all(t.data.block_inds == np.repeat(np.arange(4)[:, None], 2, axis=1))
        #
        # build the blocks for fixed coupled sectors
        # note: when setting the data we listed the uncoupled sectors of the domain

        #      dom uncoupled:  (0, 0)  ;  (2, 2)  |  codom uncoupled:
        block_0 = np.asarray(
            [
                [1, 2],  # (0, 0)
                [3, 4],
            ],  # (2, 2)
            dtype=float,
        )
        #      dom uncoupled:  (0, 1)  ;  (1, 0)  |  codom uncoupled:
        block_1 = np.asarray(
            [
                [0, 0],  # (0, 1)
                [5, 6],
            ],  # (1, 0)
            dtype=float,
        )
        #      dom uncoupled:  (0, 2)  ;  (1, 1)  ;  (2, 0)  |  codom uncoupled:
        block_2 = np.asarray(
            [
                [0, 9, 10],  # (0, 2)
                [0, 7, 8],  # (1, 1)
                [0, 0, 0],
            ],  # (2, 0)
            dtype=float,
        )
        #      dom uncoupled:  (1, 2)  ;  (2, 1)  |  codom uncoupled:
        block_3 = np.asarray(
            [
                [12, 11],  # (1, 2)
                [14, 13],
            ],  # (2, 1)
            dtype=float,
        )
        expect_blocks = [block_0, block_1, block_2, block_3]
        assert len(expect_blocks) == len(t.data.blocks)
        for i, (actual, expect) in enumerate(zip(t.data.blocks, expect_blocks)):
            print(f'checking blocks[{i}]')
            npt.assert_array_almost_equal_nulp(t.backend.block_backend.to_numpy(actual), expect, 100)

    else:
        raise RuntimeError


@pytest.mark.parametrize('symmetry_backend', [pytest.param('fusion_tree', marks=pytest.mark.FusionTree)])
def test_from_block_su2_symm(symmetry_backend, block_backend):
    backend = get_backend(symmetry_backend, block_backend)
    sym = SU2Symmetry()
    spin_half = ElementarySpace(sym, [[1]])

    # basis order: [down, up]  ->  might look unusual
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = 0.5 * np.array([[0.0, 1.0j], [-1.0j, 0]], dtype=complex)
    sz = 0.5 * np.array([[-1.0, 0.0], [0.0, +1.0]], dtype=complex)
    heisenberg_4 = sum(si[:, :, None, None] * si[None, None, :, :] for si in [sx, sy, sz])  # [p1, p1*, p2, p2*]
    print(heisenberg_4.transpose([0, 2, 1, 3]).reshape((4, 4)))
    heisenberg_4 = np.transpose(heisenberg_4, [0, 2, 3, 1])  # [p1, p2, p2*, p1*]

    tens_4 = SymmetricTensor.from_dense_block(
        heisenberg_4,
        codomain=[spin_half, spin_half],
        domain=[spin_half, spin_half],
        backend=backend,
        labels=[['p1', 'p2'], ['p1*', 'p2*']],
    )
    tens_4.test_sanity()
    assert np.all(tens_4.data.block_inds == np.array([[0, 0], [1, 1]]))  # spin 0, spin 1
    # The blocks are the eigenvalue of the Heisenberg coupling in the fixed total spin sectors
    # For singlet states (coupled=spin-0), we have eigenvalue -3/4
    # For triplet states (coupled=spin-1), we have eigenvalue +1/4
    expect_spin_0 = -3 / 4
    expect_spin_1 = 1 / 4
    assert backend.block_backend.allclose(tens_4.data.blocks[0], expect_spin_0)
    assert backend.block_backend.allclose(tens_4.data.blocks[1], expect_spin_1)

    recovered_block = tens_4.to_dense_block()
    print()
    print('got:')
    print(recovered_block.reshape((4, 4)))
    print()
    print('expect:')
    print(heisenberg_4.reshape((4, 4)))

    assert backend.block_backend.allclose(recovered_block, heisenberg_4)


@pytest.mark.parametrize(
    'codomain_dims, domain_dims, labels, abelian_pipe',
    [
        ([42, 42], [42, 42], ['a', 'b', 'c', 'd'], None),
        ([42, 42], [42, 42], ['a', 'b', 'c', 'd'], None),
        ([42], [42, 42], ['a', 'b', 'c'], None),
        ([42, 42, 12345, 42], [1], ['a', 'b', 'c', 'lorem', 'ipsum'], None),
        ([], [42, 42], ['a', 'b'], None),
        ([42, 42, 42], [], ['a', 'b', 'c'], None),
        ([[3, 2], 42], [42, 42], ['a', 'b', 'c', 'd'], True),
        ([42, 42], [[4, 5], 42], ['a', 'b', 'c', 'd'], False),
        ([[3, 3], 42], [42, [4, 5]], ['a', 'b', 'c', 'd'], True),
    ],
)
def test_Tensor_ascii_diagram(codomain_dims, domain_dims, labels, abelian_pipe, np_random):
    """
    You may find useful (see comments in :func:`test_Tensor_str_repr`)::

        pytest -rP -k test_Tensor_ascii_diagram | less

    """
    codomain = []
    domain = []
    for co_domain, dims in zip([codomain, domain], [codomain_dims, domain_dims]):
        for d in dims:
            if isinstance(d, int):
                co_domain.append(ElementarySpace.from_trivial_sector(dim=d, is_dual=np_random.choice([True, False])))
            else:
                pipe_legs = []
                for _d in d:
                    pipe_legs.append(
                        ElementarySpace.from_trivial_sector(dim=_d, is_dual=np_random.choice([True, False]))
                    )
                if abelian_pipe:
                    pipe = AbelianLegPipe(pipe_legs, is_dual=np_random.choice([True, False]))
                else:
                    pipe = LegPipe(pipe_legs, is_dual=np_random.choice([True, False]))
                co_domain.append(pipe)
    T = DummyTensor(codomain, domain, backend=get_backend(), labels=labels, dtype=Dtype.complex128)
    print(f'domain dualities: {", ".join(str(getattr(l, "is_dual", None)) for l in T.domain)}')
    print(f'codomain dualities: {", ".join(str(getattr(l, "is_dual", None)) for l in T.codomain)}')
    print(T.ascii_diagram)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, codomain, domain',
    [
        pytest.param(SymmetricTensor, ['vL'], ['vR', 'p'], id='Sym-vL-p-vR'),
        pytest.param(SymmetricTensor, ['p'], ['p*'], id='Sym-p-p*'),
        pytest.param(DiagonalTensor, ['p'], ['p*'], id='Diag-p-p*'),
        pytest.param(Mask, ['vL'], ['vL*'], id='Mask-vL-vL*'),
        pytest.param(ChargedTensor, ['vL'], ['vR', 'p'], id='Charged-vL-p-vR'),
    ],
)
def test_Tensor_str_repr(cls, codomain, domain, make_compatible_tensor, str_max_lines=30, repr_max_lines=30):
    """Check if str and repr work.

    Automatically, we can only check if they run at all.
    To check if the output is sensible and useful, a human should look at it.
    Run e.g.::

        pytest -rP -k test_Tensor_str_repr

    to select only this test (``-k`` flag) and see the output (``-rP``), even if it passes.
    Since the output is rather long, it is convenient to write the output to file.
    To do that, and directly open that file in your favorite editor, run e.g. for VS Code::

        pytest -rP -k test_Tensor_str_repr > playground/test_Tensor_str_repr.txt && code playground/test_Tensor_str_repr.txt

    or for vim::

        pytest -rP -k test_Tensor_str_repr > playground/test_Tensor_str_repr.txt && vim playground/test_Tensor_str_repr.txt

    Assumes your cwd is the repository root, such that the file is generated in playground and therefore gitignored.

    Or without creating a file::

        pytest -rP -k test_Tensor_str_repr | less

    """
    terminal_width = 80
    T = make_compatible_tensor(codomain=codomain, domain=domain, cls=cls)

    print('repr(T):')
    res = repr(T)
    print(res)
    lines = res.split('\n')
    assert all(len(line) <= terminal_width for line in lines)
    assert len(lines) <= repr_max_lines
    #
    print()
    print('str(T):')
    res = str(T)
    print(res)
    lines = res.split('\n')
    assert all(len(line) <= terminal_width for line in lines)
    assert len(lines) <= str_max_lines


# TENSOR FUNCTIONS


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, domain, codomain, is_dual',
    [
        pytest.param(SymmetricTensor, 2, 2, True),
        pytest.param(SymmetricTensor, 2, 0, False),
        pytest.param(SymmetricTensor, 0, 2, True),
        pytest.param(SymmetricTensor, 1, 3, False),
        pytest.param(DiagonalTensor, 1, 1, True),
        pytest.param(Mask, 1, 1, False),
        pytest.param(ChargedTensor, 2, 2, False),
        pytest.param(ChargedTensor, 3, 0, True),
    ],
)
def test_add_trivial_leg(cls, domain, codomain, is_dual, make_compatible_tensor, np_random):
    tens: Tensor = make_compatible_tensor(domain, codomain, cls=cls)

    if not tens.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    if cls in [DiagonalTensor, Mask]:
        catch_warnings = pytest.warns(UserWarning, match='Converting to SymmetricTensor *')
    else:
        catch_warnings = nullcontext()

    tens_np = tens.to_numpy(understood_braiding=True)

    print('via positional arg')
    pos = np_random.choice(tens.num_legs + 1)
    with catch_warnings:
        res = tensors.add_trivial_leg(tens, pos, is_dual=is_dual)
    res_np = res.to_numpy(understood_braiding=True)
    expect = np.expand_dims(tens_np, pos)
    npt.assert_array_almost_equal_nulp(res_np, expect, 100)

    print('to_domain')
    pos = np_random.choice(tens.num_domain_legs + 1)
    with catch_warnings:
        res = tensors.add_trivial_leg(tens, domain_pos=pos, is_dual=is_dual)
    res_np = res.to_numpy(understood_braiding=True)
    expect = np.expand_dims(tens_np, -1 - pos)
    npt.assert_array_almost_equal_nulp(res_np, expect, 100)

    print('to_codomain')
    pos = np_random.choice(tens.num_codomain_legs + 1)
    with catch_warnings:
        res = tensors.add_trivial_leg(tens, codomain_pos=pos, is_dual=is_dual)
    res_np = res.to_numpy(understood_braiding=True)
    expect = np.expand_dims(tens_np, pos)
    npt.assert_array_almost_equal_nulp(res_np, expect, 100)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize('cls', [DiagonalTensor, SymmetricTensor, ChargedTensor])
def test_almost_equal(cls, make_compatible_tensor):
    T: Tensor = make_compatible_tensor(cls=cls)

    if cls is ChargedTensor:
        T: ChargedTensor
        T_diff_inv = make_compatible_tensor(
            domain=T.invariant_part.domain,
            codomain=T.invariant_part.codomain,
            cls=SymmetricTensor,
            labels=T.invariant_part.labels,
        )
        T_diff = ChargedTensor(T_diff_inv, T.charged_state)
    else:
        T_diff: Tensor = make_compatible_tensor(domain=T.domain, codomain=T.codomain, cls=cls)

    T2 = T + 1e-7 * T_diff
    assert tensors.almost_equal(T, T2, rtol=1e-5, atol=1e-5)
    assert not tensors.almost_equal(T, T2, rtol=1e-10, atol=1e-10)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, codomain, domain, which_leg',
    [
        pytest.param(SymmetricTensor, 2, 2, 1, id='Symm-2-2-codom'),
        pytest.param(SymmetricTensor, 2, 2, -1, id='Symm-2-2-dom'),
        pytest.param(ChargedTensor, 2, 2, 1, id='Charged-2-2-dom'),
        pytest.param(ChargedTensor, 2, 2, -1, id='Charged-2-2-dom'),
        pytest.param(DiagonalTensor, 1, 1, -1, id='Diag-dom'),
        pytest.param(Mask, 1, 1, 0, id='Mask-codom'),
        pytest.param(Mask, 1, 1, -1, id='Mask-dom'),
    ],
)
def test_apply_mask(cls, codomain, domain, which_leg, make_compatible_tensor, compatible_backend):
    num_legs = codomain + domain
    labels = list('abcdefghijkl')[:num_legs]

    kwargs = {}
    if isinstance(compatible_backend, backends.FusionTreeBackend):
        # TODO instead of disabling, can we generate pipes on the *other* legs, not to be masked?
        kwargs['use_pipes'] = False
    M: Mask = make_compatible_tensor(cls=Mask)
    num_legs = domain + codomain
    which_leg = to_valid_idx(which_leg, num_legs)
    if which_leg >= codomain:
        domain = [None] * domain
        domain[num_legs - which_leg - 1] = M.large_leg.dual
    else:
        codomain = [None] * codomain
        codomain[which_leg] = M.large_leg

    if cls is Mask and which_leg == 0:
        with pytest.raises(NotImplementedError, match='Mask generation broken'):
            _ = make_compatible_tensor(codomain=codomain, domain=domain, labels=labels, cls=cls, **kwargs)
        pytest.xfail(reason='Mask generation broken')
    T: tensors.Tensor = make_compatible_tensor(codomain=codomain, domain=domain, labels=labels, cls=cls, **kwargs)

    if cls is Mask:
        with pytest.raises(NotImplementedError, match='tensors._compose_with_Mask not implemented for Mask'):
            _ = tensors.apply_mask(T, M, which_leg)
        pytest.xfail('apply_mask(Mask, Mask) not yet supported')
    elif cls is DiagonalTensor:
        catch_warnings = pytest.warns(UserWarning, match='Converting to SymmetricTensor *')
    else:
        catch_warnings = nullcontext()

    with catch_warnings:
        res = tensors.apply_mask(T, M, which_leg)
    res.test_sanity()

    in_domain, co_domain_idx, leg_idx = T._parse_leg_idx(which_leg)
    expect_legs = T.legs
    expect_legs[leg_idx] = M.small_leg
    assert res.legs == expect_legs
    assert res.labels == T.labels

    if not T.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    T_np = T.to_numpy(understood_braiding=True)
    mask_np = M.as_numpy_mask()
    expect = T_np.compress(mask_np, leg_idx)
    npt.assert_almost_equal(res.to_numpy(understood_braiding=True), expect)


def test_apply_mask_DiagonalTensor(make_compatible_tensor):
    tensor: DiagonalTensor = make_compatible_tensor(cls=DiagonalTensor, labels=['a', 'b'])
    mask: Mask = make_compatible_tensor(domain=[tensor.leg], cls=Mask)

    res = tensors.apply_mask_DiagonalTensor(tensor, mask)
    res.test_sanity()
    assert isinstance(res, DiagonalTensor)
    assert res.leg == mask.small_leg
    assert res.labels == tensor.labels

    if not tensor.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    diag = tensor.diagonal_as_numpy()
    mask_np = mask.as_numpy_mask()
    expect = diag[mask_np]
    npt.assert_almost_equal(res.diagonal_as_numpy(), expect)


@pytest.mark.parametrize(
    'cls, codomain, domain, num_codomain_legs',
    [
        pytest.param(SymmetricTensor, 2, 2, 2),
        pytest.param(SymmetricTensor, 2, 2, 1),
        pytest.param(SymmetricTensor, 2, 2, 4),
    ],
)
def test_bend_legs(cls, codomain, domain, num_codomain_legs, make_compatible_tensor):
    tensor: Tensor = make_compatible_tensor(codomain, domain, cls=cls)

    res = tensors.bend_legs(tensor, num_codomain_legs)
    res.test_sanity()
    assert res.legs == tensor.legs

    if tensor.symmetry.can_be_dropped:
        tensor_np = tensor.to_numpy(understood_braiding=True)
        res_np = res.to_numpy(understood_braiding=True)
        npt.assert_array_almost_equal_nulp(res_np, tensor_np, 100)

    bent_back = tensors.bend_legs(tensor, num_codomain_legs=tensor.num_codomain_legs)
    bent_back.test_sanity()
    assert_tensors_almost_equal(bent_back, tensor)


@pytest.mark.parametrize(
    'use_pipes',
    [
        pytest.param(True, id='pipes'),
        pytest.param(False, id='no pipes'),
    ],
)
def test_combine_split(use_pipes, make_compatible_tensor):
    T: SymmetricTensor = make_compatible_tensor(['a', 'b'], ['d', 'c'], use_pipes=use_pipes)
    assert T.labels == ['a', 'b', 'c', 'd']

    # 1) combine in codomain
    combined1 = tensors.combine_legs(T, [0, 1])
    combined1.test_sanity()
    assert combined1.labels == ['(a.b)', 'c', 'd']
    assert len(combined1.codomain) == 1
    assert combined1.codomain[0].legs == [T.codomain[0], T.codomain[1]]
    assert combined1.domain == T.domain
    #
    split1 = tensors.split_legs(combined1, 0)
    split1.test_sanity()
    assert split1.labels == ['a', 'b', 'c', 'd']
    assert split1.codomain == T.codomain
    assert split1.domain == T.domain
    assert tensors.almost_equal(split1, T)

    # 2) combine in domain
    combined2 = tensors.combine_legs(T, [2, 3])
    combined2.test_sanity()
    assert combined2.labels == ['a', 'b', '(c.d)']
    assert combined2.codomain == T.codomain
    assert len(combined2.domain) == 1
    assert combined2.domain[0].legs == [T.domain[0], T.domain[1]]
    #
    split2 = tensors.split_legs(combined2, '(c.d)')
    split2.test_sanity()
    assert split2.labels == ['a', 'b', 'c', 'd']
    assert split2.codomain == T.codomain
    assert split2.domain == T.domain
    assert tensors.almost_equal(split2, T)

    # 3) combine in codomain with braid
    combined3 = tensors.combine_legs(T, [1, 0], levels={'a': 0, 'b': 1, 'c': 2, 'd': 3})
    combined3.test_sanity()
    assert combined3.labels == ['(b.a)', 'c', 'd']
    assert len(combined3.codomain) == 1
    assert combined3.codomain[0].legs == [T.codomain[1], T.codomain[0]]
    assert combined3.domain == T.domain
    #
    if use_pipes:
        split3 = tensors.split_legs(combined3, '(b.a)')
    else:
        # check that it can be called without specifying the leg
        split3 = tensors.split_legs(combined3)
    split3.test_sanity()
    assert split3.labels == ['b', 'a', 'c', 'd']
    assert split3.codomain.factors == [T.codomain[1], T.codomain[0]]
    assert split3.domain == T.domain
    expect3 = tensors.permute_legs(T, [1, 0], levels={'a': 0, 'b': 1, 'c': 2, 'd': 3})
    assert tensors.almost_equal(split3, expect3)

    # 4) combine, one from codomain, one from domain
    if T.symmetry.has_trivial_braid:
        combined4 = tensors.combine_legs(T, [1, 2])
    else:
        # need to do explicit permute_legs since we need the bend_right
        combined4 = tensors.combine_legs(tensors.permute_legs(T, [0, 1, 2], [3], bend_right=True), [1, 2])
    combined4.test_sanity()
    assert combined4.labels == ['a', '(b.c)', 'd']
    assert len(combined4.codomain) == 2
    assert combined4.codomain[0] == T.codomain[0]
    assert combined4.codomain[1].legs == [T.codomain[1], T.domain[1].dual]
    assert combined4.domain.factors == [T.domain[0]]
    #
    split4 = tensors.split_legs(combined4, 1)
    split4.test_sanity()
    assert split4.labels == ['a', 'b', 'c', 'd']
    assert split4.codomain.factors == [T.codomain[0], T.codomain[1], T.domain[1].dual]
    assert split4.domain.factors == [T.domain[0]]
    assert tensors.almost_equal(split4, tensors.permute_legs(T, [0, 1, 2], bend_right=True))

    if use_pipes:
        # check splitting a non-combined leg (that is a pipe) raises
        with pytest.raises(ValueError, match='Invalid format for a combined label'):
            _ = tensors.split_legs(combined4)
        with pytest.raises(ValueError, match='Invalid format for a combined label'):
            _ = tensors.split_legs(T, 0)
    else:
        # check splitting a non-combined leg raises
        with pytest.raises(ValueError, match='Not a LegPipe.'):
            _ = tensors.split_legs(combined4, 0)

    # 5) check compatibility with bending legs
    #  5a) combine in domain
    combined5 = tensors.combine_legs(T, [2, 3])
    combined5.test_sanity()
    assert combined5.labels == ['a', 'b', '(c.d)']
    assert combined5.codomain.factors == T.codomain.factors
    assert combined5.domain[0].legs == T.domain.factors
    #
    combined_then_bent = tensors.bend_legs(combined5, num_domain_legs=0)
    combined_then_bent.test_sanity()
    bent_individually = tensors.bend_legs(T, num_domain_legs=0)
    bent_then_combined = tensors.combine_legs(bent_individually, [2, 3])
    bent_then_combined.test_sanity()
    assert combined_then_bent.legs == bent_then_combined.legs
    assert tensors.almost_equal(combined_then_bent, bent_then_combined)

    #  5b) check split * bend_pipe * combine == bend_legs
    split5 = tensors.split_legs(combined_then_bent, 2)
    split5.test_sanity()
    assert split5.labels == ['a', 'b', 'c', 'd']
    assert split5.codomain.factors == T.legs
    assert split5.domain.factors == []
    expect5 = tensors.bend_legs(T, num_domain_legs=0)
    assert tensors.almost_equal(split5, expect5), 'bending does not commute through combine!'

    # 5c) combine in codomain
    assert combined1.labels == ['(a.b)', 'c', 'd']
    combined_then_bent = tensors.bend_legs(combined1, num_codomain_legs=0)
    combined_then_bent.test_sanity()
    bent_individually = tensors.bend_legs(T, num_codomain_legs=0)
    bent_then_combined = tensors.combine_legs(bent_individually, [0, 1])
    bent_then_combined.test_sanity()
    assert combined_then_bent.legs == bent_then_combined.legs
    assert tensors.almost_equal(combined_then_bent, bent_then_combined)

    # 5d) combine across codomain
    assert combined4.labels == ['a', '(b.c)', 'd']
    assert combined4.codomain.num_factors == 2
    combined_then_bent = tensors.bend_legs(combined4, num_codomain_legs=0)
    combined_then_bent.test_sanity()
    bent_individually = tensors.bend_legs(T, num_codomain_legs=0)
    bent_then_combined = tensors.combine_legs(bent_individually, [1, 2])
    bent_then_combined.test_sanity()
    assert combined_then_bent.legs == bent_then_combined.legs
    assert tensors.almost_equal(combined_then_bent, bent_then_combined)

    # check that combine_legs().to_numpy() is the same as to_numpy().reshape()
    if T.symmetry.can_be_dropped:
        T_np = T.to_numpy(understood_braiding=True)
        a, b, c, d = T_np.shape

        expect1 = np.reshape(T_np, (a * b, c, d))
        combined1_np = combined1.to_numpy(understood_braiding=True)
        assert np.allclose(combined1_np, expect1)

        expect2 = np.reshape(T_np, (a, b, c * d))
        combined2_np = combined2.to_numpy(understood_braiding=True)
        assert np.allclose(combined2_np, expect2)

        expect3 = np.reshape(np.transpose(T_np, [1, 0, 2, 3]), (a * b, c, d))
        combined3_np = combined3.to_numpy(understood_braiding=True)
        if T.symmetry.has_trivial_braid:
            assert np.allclose(combined3_np, expect3)
        else:
            # braids/twists would give -1 signs: lazy test is to expect that some signs are off
            assert np.all(np.isclose(combined3_np, expect3) | np.isclose(combined3_np, -expect3))

        expect4 = np.reshape(T_np, (a, b * c, d))
        combined4_np = combined4.to_numpy(understood_braiding=True)
        assert np.allclose(combined4_np, expect4)

    # 6) check contracting combined leg versus contracting the individual legs
    if T.symmetry.has_trivial_braid:
        # this example requires implicit bends
        T2: SymmetricTensor = make_compatible_tensor(
            [None, T.codomain[1].dual, T.domain[1]], [None], labels=['x', 'b*', 'c*', 'y']
        )
        contracted_individual = tensors.tdot(T, T2, ['b', 'c'], ['b*', 'c*'])
        contracted_individual.test_sanity()
        T2_combined = tensors.combine_legs(T2, ['c*', 'b*'], pipe_dualities=[True])
        contracted_via_pipes = tensors.tdot(combined4, T2_combined, '(b.c)', '(c*.b*)')
        contracted_via_pipes.test_sanity()
        assert tensors.almost_equal(contracted_individual, contracted_via_pipes)

    # 7) check contracting combined leg versus contracting the individual legs
    #    without implicit bends
    T3: SymmetricTensor = make_compatible_tensor(
        [T._as_domain_leg('d'), T._as_domain_leg('c')], [None, None], labels=['d*', 'c*', 'x', 'y']
    )
    contracted_individual = tensors.compose(T, T3)
    contracted_individual.test_sanity()
    T3_combined = tensors.combine_legs(T3, ['d*', 'c*'], pipe_dualities=[True])
    contracted_via_pipes = tensors.compose(combined5, T3_combined)
    contracted_via_pipes.test_sanity()
    assert tensors.almost_equal(contracted_individual, contracted_via_pipes)


def test_combine_split_pr_16():
    """Check if the bug addressed in PR :pull:`16` is fixed"""
    backend = get_backend('abelian', 'numpy')
    symmetry = u1_symmetry * z3_symmetry

    a = ElementarySpace(
        symmetry,
        defining_sectors=[[-2, 0], [-1, 0], [-2, 1], [-2, 2]],
        multiplicities=[1, 2, 4, 4],
        basis_perm=[8, 0, 7, 3, 6, 2, 4, 10, 1, 5, 9],
        is_dual=True,
    )
    b = ElementarySpace(
        symmetry,
        defining_sectors=[[-3, 0], [0, 0], [-3, 1], [-3, 2]],
        multiplicities=[1, 1, 1, 1],
        basis_perm=None,
        is_dual=False,
    )
    c = ElementarySpace(
        symmetry,
        defining_sectors=[[-4, 0], [-3, 0], [-7, 1], [-6, 1], [-3, 1]],
        multiplicities=[5, 5, 5, 5, 5],
        basis_perm=None,
        is_dual=False,
    )
    d = ElementarySpace(
        symmetry,
        defining_sectors=[[-2, 0], [1, 0], [2, 1]],
        multiplicities=[3, 3, 2],
        basis_perm=[6, 3, 4, 0, 7, 2, 5, 1],
        is_dual=True,
    )

    T = tensors.SymmetricTensor.from_random_normal([a, b], [d, c], backend=backend)
    combined5 = tensors.combine_legs(T, [2, 3])
    combined5.test_sanity()
    assert combined5.codomain.factors == T.codomain.factors
    assert combined5.domain[0].legs == T.domain.factors

    re_split = tensors.split_legs(combined5, 2)
    assert tensors.almost_equal(T, re_split)

    bent5 = tensors.bend_legs(combined5, num_domain_legs=0)
    split5 = tensors.split_legs(bent5, 2)
    split5.test_sanity()
    assert split5.codomain.factors == T.legs
    assert split5.domain.factors == []
    expect5 = tensors.bend_legs(T, num_domain_legs=0)

    assert tensors.almost_equal(split5, expect5)


@pytest.mark.parametrize(
    'cls_A, cls_B, cod_A, shared, dom_B',
    [
        pytest.param(SymmetricTensor, SymmetricTensor, 2, 2, 2, id='Sym@Sym-2-2-2'),
    ],
)
def test_compose(cls_A, cls_B, cod_A, shared, dom_B, make_compatible_tensor):
    labels_A = [list('abcd')[:cod_A], list('efgh')[:shared]]
    labels_B = [list('ijkl')[:shared], list('mnop')[:dom_B]]
    A: Tensor = make_compatible_tensor(codomain=cod_A, domain=shared, labels=labels_A, cls=cls_A)
    B: Tensor = make_compatible_tensor(codomain=A.domain, domain=dom_B, labels=labels_B, cls=cls_B)

    res = tensors.compose(A, B, relabel1={'a': 'x'}, relabel2={'m': 'y'})

    if not A.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    if cod_A == 0 == dom_B:  # scalar result
        assert isinstance(res, (float, complex))
        res_np = res
    else:
        res.test_sanity()
        assert res.codomain == A.codomain
        assert res.domain == B.domain
        expect_labels = []
        if A.num_codomain_legs > 0:
            expect_labels.append('x')
            expect_labels.extend(A.codomain_labels[1:])
        if B.num_domain_legs > 0:
            expect_labels.extend(reversed(B.domain_labels[1:]))
            expect_labels.append('y')
        assert res.labels == expect_labels

        res_np = res.to_numpy(understood_braiding=True)

    axes = [list(range(cod_A, cod_A + shared)), list(reversed(range(shared)))]
    expect = np.tensordot(A.to_numpy(understood_braiding=True), B.to_numpy(understood_braiding=True), axes)
    npt.assert_almost_equal(res_np, expect)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, cod, dom',
    [
        pytest.param(SymmetricTensor, 2, 2, id='Sym-2-2'),
        pytest.param(SymmetricTensor, 3, 0, id='Sym-3-0'),
        pytest.param(SymmetricTensor, 1, 1, id='Sym-1-1'),
        pytest.param(SymmetricTensor, 0, 3, id='Sym-3-0'),
        pytest.param(ChargedTensor, 2, 2, id='Charged-2-2'),
        pytest.param(ChargedTensor, 3, 0, id='Charged-3-0'),
        pytest.param(ChargedTensor, 1, 1, id='Charged-1-1'),
        pytest.param(ChargedTensor, 0, 3, id='Charged-3-0'),
        pytest.param(DiagonalTensor, 1, 1, id='Diag'),
        pytest.param(Mask, 1, 1, id='Mask'),
    ],
)
def test_dagger(cls, cod, dom, make_compatible_tensor, np_random):
    T_labels = list('abcdefghi')[: cod + dom]
    T: Tensor = make_compatible_tensor(cod, dom, cls=cls, labels=T_labels)

    if cls is ChargedTensor and not T.symmetry.has_symmetric_braid:
        # TODO : should be ok to just choose levels, as long as charge leg is very top or very bot
        with pytest.raises(BraidChiralityUnspecifiedError):
            _ = tensors.dagger(T)
        return

    how_to_call = np_random.choice(['dagger()', '.hc', '.dagger'])
    print(how_to_call)
    if how_to_call == 'dagger()':
        res = tensors.dagger(T)
    if how_to_call == '.hc':
        res = T.hc
    if how_to_call == '.dagger':
        res = T.dagger
    res.test_sanity()

    assert res.codomain == T.domain
    assert res.domain == T.codomain
    assert res.labels == [f'{l}*' for l in reversed(T_labels)]

    if T.symmetry.has_trivial_braid:
        expect = np.conj(np.transpose(T.to_numpy(), list(reversed(range(cod + dom)))))
        npt.assert_almost_equal(res.to_numpy(), expect)
    else:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc


@pytest.mark.parametrize(
    'cyten_func, numpy_func, dtype, kwargs',
    # dtype=None indicates that we need to special case the tensor creations to fulfill constraints.
    [
        pytest.param(tensors.angle, np.angle, Dtype.complex128, {}, id='angle()-complex'),
        pytest.param(tensors.angle, np.angle, Dtype.float64, {}, id='angle()-real'),
        pytest.param(tensors.imag, np.imag, Dtype.complex128, {}, id='imag()-complex'),
        pytest.param(tensors.imag, np.imag, Dtype.float64, {}, id='imag()-real'),
        pytest.param(tensors.real, np.real, Dtype.complex128, {}, id='real()-complex'),
        pytest.param(tensors.real, np.real, Dtype.float64, {}, id='real()-real'),
        pytest.param(tensors.real_if_close, np.real_if_close, Dtype.complex128, {}, id='real_if_close()'),
        pytest.param(tensors.real_if_close, np.real_if_close, Dtype.float64, dict(tol=100), id='real_if_close()'),
        pytest.param(tensors.real_if_close, np.real_if_close, None, {}, id='real_if_close()'),
        pytest.param(tensors.sqrt, np.sqrt, None, {}, id='sqrt()'),
        pytest.param(DiagonalTensor.__abs__, np.abs, Dtype.float64, {}, id='abs()-real'),
        pytest.param(DiagonalTensor.__abs__, np.abs, Dtype.complex128, {}, id='abs()-complex'),
        pytest.param(tensors.real, np.real, Dtype.float64, {}, id='real()-real'),
        pytest.param(tensors.complex_conj, np.conj, Dtype.float64, {}, id='conj()-real'),
        pytest.param(tensors.complex_conj, np.conj, Dtype.complex128, {}, id='conj()-complex'),
        pytest.param(tensors.cutoff_inverse, None, Dtype.float64, dict(cutoff=1e-16), id='cutoff_inverse())-real'),
        pytest.param(tensors.cutoff_inverse, None, Dtype.complex128, dict(cutoff=0.3), id='cutoff_inverse())-complex'),
    ],
    # TODO more functions? exp, log
)
def test_DiagonalTensor_elementwise_unary(cyten_func, numpy_func, dtype, kwargs, make_compatible_tensor):
    if dtype is not None:
        D: DiagonalTensor = make_compatible_tensor(cls=DiagonalTensor, dtype=dtype)
    elif cyten_func is tensors.sqrt:
        # need positive
        D: DiagonalTensor = abs(make_compatible_tensor(cls=DiagonalTensor, dtype=Dtype.float64))
    elif cyten_func is tensors.real_if_close:
        # want almost real
        rp = make_compatible_tensor(cls=DiagonalTensor, dtype=Dtype.float64)
        ip = make_compatible_tensor(domain=rp.domain, cls=DiagonalTensor, dtype=Dtype.float64)
        D = rp + 1 - 12j * ip
    else:
        raise ValueError

    res = cyten_func(D, **kwargs)
    res.test_sanity()

    if not D.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    if cyten_func is tensors.cutoff_inverse:
        numpy_func = partial(NumpyBlockBackend().cutoff_inverse, **kwargs)
        npt.assert_almost_equal(numpy_func(1), 1)
        npt.assert_almost_equal(numpy_func(0), 0)
        npt.assert_almost_equal(numpy_func(2 * kwargs['cutoff']), 0.5 / kwargs['cutoff'])
        npt.assert_almost_equal(numpy_func(0.5 * kwargs['cutoff']), 0)

    res_np = res.diagonal_as_numpy()
    expect = numpy_func(D.diagonal_as_numpy())
    npt.assert_almost_equal(res_np, expect)


@pytest.mark.parametrize(
    'cls, op, dtype',
    [
        pytest.param(DiagonalTensor, operator.add, Dtype.complex128, id='+'),
        pytest.param(DiagonalTensor, operator.ge, Dtype.bool, id='>='),
        pytest.param(DiagonalTensor, operator.gt, Dtype.bool, id='>'),
        pytest.param(DiagonalTensor, operator.le, Dtype.bool, id='<='),
        pytest.param(DiagonalTensor, operator.lt, Dtype.bool, id='<'),
        pytest.param(DiagonalTensor, operator.mul, Dtype.complex128, id='*'),
        pytest.param(DiagonalTensor, operator.pow, Dtype.complex128, id='**'),
        pytest.param(DiagonalTensor, operator.sub, Dtype.complex128, id='-'),
    ],
)
def test_DiagonalTensor_elementwise_binary(cls, op, dtype, make_compatible_tensor, np_random):
    t1: DiagonalTensor = make_compatible_tensor(cls=cls, dtype=dtype)
    t2: DiagonalTensor = make_compatible_tensor(domain=t1.domain, cls=cls, dtype=dtype)
    if dtype == Dtype.bool:
        scalar = bool(np_random.choice([True, False]))
    elif dtype.is_real:
        scalar = np_random.uniform()
    else:
        scalar = np_random.uniform() + 1.0j * np_random.uniform()

    if not t1.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    t1_np = t1.diagonal_as_numpy()
    t2_np = t2.diagonal_as_numpy()
    print('With other tensor')
    res = op(t1, t2)
    res.test_sanity()
    res_np = res.diagonal_as_numpy()
    expect = op(t1_np, t2_np)
    npt.assert_almost_equal(res_np, expect)

    print('With scalar')
    res = op(t1, scalar)
    res.test_sanity()
    res_np = res.diagonal_as_numpy()
    expect = op(t1_np, scalar)
    npt.assert_almost_equal(res_np, expect)


@pytest.mark.parametrize(
    'cls, dom, new_leg_dual',
    [
        pytest.param(SymmetricTensor, 1, False, id='Sym-1-False'),
        pytest.param(SymmetricTensor, 1, True, id='Sym-1-True'),
        pytest.param(SymmetricTensor, 2, False, id='Sym-2-False'),
        pytest.param(DiagonalTensor, 1, False, id='Diag-False'),
        pytest.param(DiagonalTensor, 1, True, id='Diag-True'),
    ],
)
def test_eigh(cls, dom, new_leg_dual, make_compatible_tensor):
    # prepare hermitian tensor
    T: Tensor = make_compatible_tensor(dom, dom, cls=cls)
    T: Tensor = make_compatible_tensor(T.domain, T.domain, cls=cls)
    T = T + T.hc
    T.set_labels(list('efghijk')[: 2 * dom])
    T.test_sanity()

    W, V = tensors.eigh(T, new_labels=['a', 'b', 'c'], new_leg_dual=new_leg_dual)
    W.test_sanity()
    V.test_sanity()
    assert W.labels == ['b', 'c']
    assert V.codomain_labels == T.codomain_labels
    assert V.domain_labels == ['a']

    assert tensors.almost_equal(V @ W @ V.hc, T, allow_different_types=True)  # is decomposition
    assert tensors.almost_equal(V @ V.hc, SymmetricTensor.from_eye(V.codomain, T.backend))  # unitary 1)
    assert tensors.almost_equal(V.hc @ V, SymmetricTensor.from_eye(V.domain, T.backend))  # unitary 2)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, codomain, domain, which_leg',
    [
        pytest.param(SymmetricTensor, 2, 2, 1, id='Symm-2-2-codom'),
        pytest.param(SymmetricTensor, 2, 2, 3, id='Symm-2-2-dom'),
        pytest.param(ChargedTensor, 2, 2, 1, id='Charged-2-2-codom'),
        pytest.param(ChargedTensor, 2, 2, 3, id='Charged-2-2-dom'),
        pytest.param(DiagonalTensor, 1, 1, 1, id='Diag-dom'),
        pytest.param(Mask, 1, 1, 0, id='Mask-codom'),
        pytest.param(Mask, 1, 1, 1, id='Mask-dom'),
    ],
)
def test_enlarge_leg(cls, codomain, domain, which_leg, make_compatible_tensor, make_compatible_space):
    num_legs = codomain + domain
    labels = list('abcdefghijkl')[:num_legs]

    large_leg = make_compatible_space()
    projection: Mask = make_compatible_tensor(domain=[large_leg], cls=Mask)
    M: Mask = tensors.dagger(projection)
    #
    T_codomain = [None] * codomain
    T_domain = [None] * domain
    if which_leg >= codomain:
        domain_idx = codomain + domain - 1 - which_leg
        T_domain[domain_idx] = M.small_leg.dual
    else:
        T_codomain[which_leg] = M.small_leg

    if cls is Mask and which_leg == 0:
        with pytest.raises(NotImplementedError, match='Mask generation broken'):
            _ = make_compatible_tensor(codomain=T_codomain, domain=T_domain, labels=labels, cls=cls)
        pytest.xfail('Mask generation broken')

    T: Tensor = make_compatible_tensor(codomain=T_codomain, domain=T_domain, labels=labels, cls=cls)

    if cls is Mask:
        with pytest.raises(NotImplementedError, match='tensors._compose_with_Mask not implemented for Mask'):
            _ = tensors.enlarge_leg(T, M, which_leg)
        pytest.xfail('apply_mask(Mask, Mask) not yet supported')
    if isinstance(T.backend, backends.FusionTreeBackend) and T.has_pipes:
        with pytest.raises(NotImplementedError, match='_mask_contract does not support pipes yet'):
            _ = tensors.enlarge_leg(T, M, which_leg)
        pytest.xfail('FTB mask application with pipes is not supported yet.')
    elif cls is DiagonalTensor:
        catch_warnings = pytest.warns(UserWarning, match='Converting to SymmetricTensor *')
    else:
        catch_warnings = nullcontext()

    with catch_warnings:
        res = tensors.enlarge_leg(T, M, which_leg)
    res.test_sanity()

    _, _, leg_idx = T._parse_leg_idx(which_leg)
    expect_legs = T.legs[:]
    expect_legs[leg_idx] = M.large_leg
    assert res.legs == expect_legs
    assert res.labels == T.labels

    if T.symmetry.can_be_dropped:
        T_np = T.to_numpy(understood_braiding=True)
        mask_np = M.as_numpy_mask()
        idcs = (slice(None, None, None),) * leg_idx + (mask_np,)
        expect_shape = list(T.shape)
        expect_shape[leg_idx] = M.shape[0]
        expect = np.zeros(expect_shape, dtype=T_np.dtype)
        expect[idcs] = T_np
        npt.assert_almost_equal(res.to_numpy(understood_braiding=True), expect)

    # TODO what can we check without resorting to numpy?


@pytest.mark.parametrize('n', [1, np.inf, 2, 3])
def test_entropy(n, make_compatible_tensor):
    D: DiagonalTensor = make_compatible_tensor(cls=DiagonalTensor)
    D = abs(D)  # non-negative
    D = D / tensors.norm(D)  # normalized
    p = D**2

    if not D.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    # make sure we have created a valid probability distribution
    D_np = D.diagonal_as_numpy()
    p_np = p.diagonal_as_numpy()
    assert np.all(D_np >= 0)
    npt.assert_almost_equal(np.sum(p_np), 1)

    ent = tensors.entropy(p, n)
    assert isinstance(ent, float)
    expect = tensors.entropy(p_np, n)
    npt.assert_almost_equal(ent, expect)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, cod, dom',
    [
        pytest.param(SymmetricTensor, 2, 2, id='Sym-2-2'),
        pytest.param(SymmetricTensor, 1, 3, id='Sym-1-3'),
        pytest.param(SymmetricTensor, 3, 0, id='Sym-3-0'),
        pytest.param(DiagonalTensor, 1, 1, id='Diag'),
        pytest.param(Mask, 1, 1, id='Mask-1-1'),
        pytest.param(ChargedTensor, 2, 2, id='Charged-2-2'),
        pytest.param(ChargedTensor, 0, 3, id='Charged-0-3'),
        pytest.param(ChargedTensor, 3, 1, id='Charged-3-1'),
    ],
)
def test_getitem(cls, cod, dom, make_compatible_tensor, np_random):
    T: Tensor = make_compatible_tensor(cod, dom, cls=cls)

    if not T.symmetry.can_be_dropped:
        with pytest.raises(SymmetryError, match='Can not access elements'):
            _ = T[(0,) * (cod + dom)]
        return

    if (cls is SymmetricTensor or cls is ChargedTensor) and isinstance(T.backend, backends.FusionTreeBackend):
        catch_warnings = pytest.warns(UserWarning, match='Accessing individual entries')
    else:
        catch_warnings = nullcontext()

    T_np = T.to_numpy(understood_braiding=True)
    random_idx = tuple(np_random.choice(d) for d in T.shape)

    def assert_same(a, b):
        if T.dtype == Dtype.bool:
            assert bool(a) == bool(b)
        else:
            npt.assert_almost_equal(a, b)

    with catch_warnings:
        entry = T[random_idx]
    assert isinstance(entry, (bool, float, complex))
    assert_same(entry, T_np[random_idx])

    # trying to set items raises
    with pytest.raises(TypeError, match='.* do.* not support item assignment.'):
        T[random_idx] = T.dtype.zero_scalar

    non_zero_idcs = np.where(np.abs(T_np) > 1e-2)
    which = np_random.choice(len(non_zero_idcs[0]))
    non_zero_idx = tuple(ax[which] for ax in non_zero_idcs)
    assert len(non_zero_idx) > 0
    with catch_warnings:
        entry = T[non_zero_idx]
    assert_same(entry, T_np[non_zero_idx])

    zero_idcs = np.where(np.abs(T_np) < 1e-8)
    if len(zero_idcs[0]) > 0:
        which = np_random.choice(len(zero_idcs[0]))
        zero_idx = tuple(ax[which] for ax in zero_idcs)
        with catch_warnings:
            entry = T[zero_idx]
        assert_same(entry, T_np[zero_idx])


@pytest.mark.parametrize('trunc', [None, 1e-10])
def test_horizontal_factorization(trunc, make_compatible_tensor):
    cod = 4
    cod_cut = 2
    dom = 3
    dom_cut = 1
    T_labels = list('efghijklmn')[: dom + cod]
    T: Tensor = make_compatible_tensor(
        cod, dom, labels=T_labels, cls=SymmetricTensor, use_pipes=False, max_blocks=3, max_block_size=3
    )

    A, B = tensors.horizontal_factorization(
        T, codomain_cut=cod_cut, domain_cut=dom_cut, new_labels=['a', 'b'], cutoff_singular_values=trunc
    )
    A.test_sanity()
    B.test_sanity()
    assert A.num_codomain_legs == cod_cut
    assert A.num_domain_legs == dom_cut + 1
    assert A.labels == T.labels[:cod_cut] + ['a'] + T.labels[-dom_cut:]
    assert B.num_codomain_legs == 1 + (cod - cod_cut)
    assert B.num_domain_legs == dom - dom_cut
    assert B.labels == ['b'] + T.labels[cod_cut:-dom_cut]

    if T.symmetry.can_be_dropped:
        A_np = A.to_numpy(understood_braiding=True)  # [*J1, a, *K1_rev]
        B_np = B.to_numpy(understood_braiding=True)  # [b, *J2, *K2_rev]
        T2_np = np.tensordot(A_np, B_np, (cod_cut, 0))  # [*J1, *K1_rev, *J2, *K2_rev]
        T2_np = np.transpose(
            T2_np, [*range(cod_cut), *range(cod_cut + dom_cut, cod + dom), *range(cod_cut, cod_cut + dom_cut)]
        )
        # [*J1, *J2, *J2_rev, *K1_rev] = [*J, *J_rev]
        T_np = T.to_numpy(understood_braiding=True)
        npt.assert_array_almost_equal(T2_np, T_np)

    A_bent = tensors.permute_legs(A, [*range(cod_cut + 1, A.num_legs), *range(cod_cut)], [cod_cut], bend_right=False)
    B_bent = tensors.permute_legs(B, [0], [*reversed(range(1, B.num_legs))], bend_right=True)
    T2 = tensors.compose(A_bent, B_bent)
    T2 = tensors.permute_legs(
        T2, T.codomain_labels, T.domain_labels, bend_right=[False] * T2.num_codomain_legs + [True] * T2.num_domain_legs
    )
    assert tensors.almost_equal(T2, T), 'input tensor not reproduced'


@pytest.mark.parametrize('trunc', [False, None, 1e-100, 1e-10])
def test_fixes_scale_axis_bug(trunc):
    # there was a bug in scale_axis that this test isolated
    s = u1_symmetry * z3_symmetry
    backend = backends.get_backend('fusion_tree')
    codomain = TensorProduct(
        [
            ElementarySpace.from_defining_sectors(s, [[-1, 0], [-2, 2]], [2, 3], is_dual=True),
            ElementarySpace.from_defining_sectors(s, [[1, 1], [-3, 2], [0, 2]], [3, 1, 1], is_dual=True),
            ElementarySpace.from_defining_sectors(s, [[-2, 0], [0, 0], [123, 1]], [3, 1, 2], is_dual=True),
        ]
    )
    domain = TensorProduct(
        [
            ElementarySpace.from_defining_sectors(s, [[-2, 0]], [3]),
            ElementarySpace.from_defining_sectors(s, [[-124, 0], [-1, 1], [1, 1]], [3, 3, 3]),
            ElementarySpace.from_defining_sectors(s, [[2, 0]], [3]),
            ElementarySpace.from_defining_sectors(s, [[-2, 0], [1, 1], [123, 1]], [1, 1, 3]),
        ]
    )
    T = SymmetricTensor.from_random_normal(codomain, domain, backend=backend)
    T.test_sanity()

    if trunc is False:
        U, S, Vh = tensors.svd(T)
    else:
        U, S, Vh, _, _ = tensors.truncated_svd(T, svd_min=trunc)
    U.test_sanity()
    S.test_sanity()
    Vh.test_sanity()

    S_Vh = tensors.compose(S, Vh)
    S_Vh.test_sanity()

    U_S = tensors.compose(U, S)
    U_S.test_sanity()

    T2 = tensors.compose(U, S_Vh)
    T2.test_sanity()
    assert tensors.almost_equal(T, T2)

    T3 = tensors.compose(U_S, Vh)
    T3.test_sanity()
    assert tensors.almost_equal(T, T3)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, cod, dom, do_dagger, allow_basis_perm',
    [
        pytest.param(SymmetricTensor, 2, 2, True, True, id='Sym-2-2-True'),
        pytest.param(SymmetricTensor, 2, 2, False, True, id='Sym-2-2-False'),
        pytest.param(SymmetricTensor, 3, 0, True, True, id='Sym-3-0-True'),
        pytest.param(SymmetricTensor, 0, 2, False, True, id='Sym-0-2-False'),
        pytest.param(ChargedTensor, 2, 2, True, True, id='Charged-2-2-True'),
        pytest.param(ChargedTensor, 2, 2, False, True, id='Charged-2-2-False'),
        pytest.param(ChargedTensor, 3, 0, True, True, id='Charged-3-0-True'),
        pytest.param(ChargedTensor, 0, 2, False, True, id='Charged-0-2-False'),
        pytest.param(DiagonalTensor, 1, 1, True, True, id='Diag-1-1-True'),
        pytest.param(DiagonalTensor, 1, 1, False, True, id='Diag-1-1-False'),
        pytest.param(Mask, 1, 1, True, False, id='Mask-1-1-True'),
        pytest.param(Mask, 1, 1, False, False, id='Mask-1-1-False'),
    ],
    # TODO also test mixed types
)
def test_inner(cls, cod, dom, do_dagger, allow_basis_perm, make_compatible_tensor):
    """
    Parameters
    ----------
    cod, dom, do_dagger : from parametrize
    allow_basis_perm : bool from parametrize
        Mask random generation does not allow basis perms right now. This flag allows generating
        only legs with trivial basis perm.
    """
    A: Tensor = make_compatible_tensor(cod, dom, cls=cls, allow_basis_perm=allow_basis_perm)
    if do_dagger:
        B: Tensor = make_compatible_tensor(codomain=A.codomain, domain=A.domain, cls=cls)
    else:
        B: Tensor = make_compatible_tensor(codomain=A.domain, domain=A.codomain, cls=cls)

    if cls is Mask:
        with pytest.raises(NotImplementedError, match='tensors._compose_with_Mask not implemented for Mask'):
            _ = tensors.inner(A, B, do_dagger=do_dagger)
        pytest.xfail('apply_mask(Mask, Mask) not yet supported')

    if not A.symmetry.can_be_dropped and cls is ChargedTensor:
        assert A.charged_state is None
        # for anyonic symmetry, can not specify charge state.
        # but the, we can not compute inner.
        return

    res = tensors.inner(A, B, do_dagger=do_dagger)
    assert isinstance(res, (float, complex))

    if A.symmetry.has_trivial_braid:
        if do_dagger:
            expect = np.sum(np.conj(A.to_numpy()) * B.to_numpy())
        else:
            expect = np.sum(np.transpose(A.to_numpy(), [*reversed(range(A.num_legs))]) * B.to_numpy())
        npt.assert_almost_equal(res, expect)
    else:
        pass  # TODO need to check some other way


def test_is_scalar():
    # python scalars
    assert tensors.is_scalar(42.0)
    assert tensors.is_scalar(1.0 + 3.0j)
    assert tensors.is_scalar(4e543j)
    assert tensors.is_scalar(3)
    assert tensors.is_scalar(True)
    # objects which are not scalar, just bc of their type
    assert not tensors.is_scalar('42')
    assert not tensors.is_scalar([3, 3, 1])
    assert not tensors.is_scalar([3])
    assert not tensors.is_scalar({'a': 3})
    assert not tensors.is_scalar(np.array(1))
    assert not tensors.is_scalar(np.array([True]))
    assert not tensors.is_scalar(np.array([1.0, 2]))
    # Tensors
    leg1 = ElementarySpace(z4_symmetry, [[2]])
    leg2 = ElementarySpace.from_defining_sectors(z4_symmetry, [[2], [3], [1]])
    backend = get_backend()
    scalar_tens1 = DummyTensor([leg1, leg1], [leg1, leg1], backend=backend, labels=None, dtype=Dtype.float64)
    scalar_tens2 = DummyTensor([], [leg1, leg1], backend=backend, labels=None, dtype=Dtype.float64)
    scalar_tens3 = DummyTensor([leg1], [leg1], backend=backend, labels=None, dtype=Dtype.float64)
    non_scalar_tens = DummyTensor([leg1, leg1], [leg2], backend=backend, labels=None, dtype=Dtype.float64)
    assert tensors.is_scalar(scalar_tens1)
    assert tensors.is_scalar(scalar_tens2)
    assert tensors.is_scalar(scalar_tens3)
    assert not tensors.is_scalar(non_scalar_tens)


def test_item(make_compatible_tensor, make_compatible_sectors, compatible_symmetry, np_random):
    sector_candidates = make_compatible_sectors(10)
    dims = compatible_symmetry.batch_qdim(sector_candidates)
    one_dim_sectors = sector_candidates[dims == 1, :]
    if len(one_dim_sectors) == 0:
        sector = compatible_symmetry.trivial_sector
    else:
        sector = np_random.choice(one_dim_sectors)
    leg = ElementarySpace(compatible_symmetry, [sector])
    T: SymmetricTensor = make_compatible_tensor([leg, leg], [leg, leg], cls=SymmetricTensor)

    res = tensors.item(T)

    if not T.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    expect = T.to_numpy(understood_braiding=True).item()
    npt.assert_almost_equal(res, expect)

    leg2 = ElementarySpace(compatible_symmetry, [sector], [3])
    non_scalar_T = make_compatible_tensor([leg, leg2], [leg, leg], cls=SymmetricTensor)
    with pytest.raises(ValueError, match='Not a scalar'):
        _ = tensors.item(non_scalar_T)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize('cls', [SymmetricTensor, DiagonalTensor, Mask, ChargedTensor])
def test_linear_combination(cls, make_compatible_tensor):
    if cls in [SymmetricTensor, ChargedTensor]:
        v = make_compatible_tensor(cls=cls, codomain=2, domain=2, max_block_size=3, max_blocks=3)
    elif cls is Mask:
        # Generating the second mask with the same legs can only be done right now if
        # permutation is trivial
        v = make_compatible_tensor(cls=cls, allow_basis_perm=False)
    else:
        v = make_compatible_tensor(cls=cls)
    w = make_compatible_tensor(like=v)

    if cls is Mask:
        catch_warnings = pytest.warns(UserWarning, match='Converting types')
    else:
        catch_warnings = nullcontext()

    if not w.symmetry.can_be_dropped:
        # TODO  Need to re-design checks, cant use .to_numpy() etc
        #       For now, just check if it runs at all.
        #       Could e.g. check versus inner product, if <x| av + bw> = a <x|v> + b <x|w>
        with catch_warnings:
            _ = tensors.linear_combination(42, v, 43j, w)
        return

    v_np = v.to_numpy(understood_braiding=True)
    w_np = w.to_numpy(understood_braiding=True)
    for valid_scalar in [0, 1.0, 2.0 + 3.0j, -42]:
        with catch_warnings:
            res = tensors.linear_combination(valid_scalar, v, 2 * valid_scalar, w)
        expect = valid_scalar * v_np + 2 * valid_scalar * w_np
        npt.assert_allclose(res.to_numpy(understood_braiding=True), expect)
    for invalid_scalar in [None, (1, 2), v, 'abc']:
        with pytest.raises(TypeError, match='unsupported scalar types'):
            _ = tensors.linear_combination(invalid_scalar, v, invalid_scalar, w)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, cod, dom, leg, codomain_pos, domain_pos, levels',
    [
        pytest.param(SymmetricTensor, 2, 2, 0, 1, None, None, id='Sym-a'),
        pytest.param(SymmetricTensor, 2, 2, 0, 0, None, None, id='Sym-b'),
        pytest.param(SymmetricTensor, 2, 2, 0, None, 0, None, id='Sym-c'),
        pytest.param(SymmetricTensor, 2, 2, 3, 1, None, None, id='Sym-d'),
        pytest.param(SymmetricTensor, 2, 2, 3, 0, None, None, id='Sym-e'),
        pytest.param(SymmetricTensor, 2, 2, 3, None, 0, None, id='Sym-f'),
        pytest.param(DiagonalTensor, 1, 1, 0, None, 1, None, id='Diag-a'),
        pytest.param(DiagonalTensor, 1, 1, 0, 0, None, None, id='Diag-b'),
        pytest.param(ChargedTensor, 2, 2, 0, 1, None, None, id='Charged-a'),
        pytest.param(ChargedTensor, 2, 2, 0, None, 1, None, id='Charged-b'),
        pytest.param(ChargedTensor, 2, 2, 3, 0, None, None, id='Charged-c'),
    ],
)
def test_move_leg(cls, cod, dom, leg, codomain_pos, domain_pos, levels, make_compatible_tensor, np_random):
    """Test `move_leg` for tensors without specifying any levels. In the case where the symmetry
    is anyonic, that is, `move_leg` needs levels as input, random levels are chosen and the
    resulting numerical values are not checked against another tensor.
    Such checks are however done within the tests provided in `test_backend_nonabelian`.
    """
    assert sum(x is None for x in [domain_pos, codomain_pos]) == 1

    T_labels = list('abcdefghi')[: cod + dom]
    T: Tensor = make_compatible_tensor(cod, dom, labels=T_labels, cls=cls)

    if cls in [DiagonalTensor, Mask]:
        if (leg, codomain_pos, domain_pos) == (0, 0, None):
            # that set of inputs means no actual legs move and no conversion happens
            catch_warnings = nullcontext()
        else:
            catch_warnings = pytest.warns(UserWarning, match='Converting to SymmetricTensor *')
    else:
        catch_warnings = nullcontext()

    if not T.symmetry.has_symmetric_braid:
        # need to specify levels for moving the leg
        levels = list(np_random.permutation(T.num_legs))

    codomain_perm = [n for n in range(cod) if n != leg]
    domain_perm = [n for n in reversed(range(cod, cod + dom)) if n != leg]
    if codomain_pos is not None:
        codomain_perm[codomain_pos:codomain_pos] = [leg]
    if domain_pos is not None:
        domain_perm[domain_pos:domain_pos] = [leg]
    perm = [*codomain_perm, *reversed(domain_perm)]
    bend_right = np_random.choice([True, False])

    with catch_warnings:
        res = tensors.move_leg(
            T, leg, codomain_pos=codomain_pos, domain_pos=domain_pos, levels=levels, bend_right=bend_right
        )
    res.test_sanity()

    assert res.labels == [T_labels[n] for n in perm]
    assert res.legs == [T.get_leg(n) for n in perm]
    assert res.num_codomain_legs == cod + int(codomain_pos is not None) - int(leg < cod)

    if T.symmetry.can_be_dropped:
        expect = T.to_numpy(understood_braiding=True).transpose(perm)
        res_np = res.to_numpy(understood_braiding=True)
        if T.symmetry.has_trivial_braid:
            npt.assert_allclose(res_np, expect, atol=1.0e-14)
        else:
            # would need explicit swap gates that introduce some -1 signs.
            # lazy version: just check if it matches with either sign
            assert np.all(np.isclose(res_np, expect) | np.isclose(res_np, -expect))


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, cod, dom',
    [
        pytest.param(SymmetricTensor, 2, 2, id='Sym-2-2'),
        pytest.param(SymmetricTensor, 3, 1, id='Sym-3-1'),
        pytest.param(SymmetricTensor, 3, 0, id='Sym-2-0'),
        pytest.param(SymmetricTensor, 0, 2, id='Sym-0-2'),
        pytest.param(ChargedTensor, 0, 2, id='Charged-0-2'),
        pytest.param(ChargedTensor, 2, 2, id='Charged-2-2'),
        pytest.param(DiagonalTensor, 1, 1, id='Diag'),
        pytest.param(Mask, 1, 1, id='Mask'),
    ],
)
def test_norm(cls, cod, dom, make_compatible_tensor):
    T: Tensor = make_compatible_tensor(cod, dom, cls=cls)

    if not T.symmetry.can_be_dropped and cls is ChargedTensor:
        assert T.charged_state is None  # can not have a charged state for anyons -> can not compute norm
        return

    res = tensors.norm(T)
    assert isinstance(res, (float, complex))

    if T.symmetry.can_be_dropped:
        expect = np.linalg.norm(T.to_numpy(understood_braiding=True))
        npt.assert_almost_equal(res, expect)


@pytest.mark.deselect_invalid_ChargedTensor_cases(
    get_cls=lambda kw: ChargedTensor if ChargedTensor in [kw['cls_A'], kw['cls_B']] else None
)
@pytest.mark.parametrize(
    'cls_A, cls_B, cA, dA, cB, dB',
    [
        pytest.param(SymmetricTensor, SymmetricTensor, 1, 2, 2, 1, id='Sym@Sym-1-2-2-1'),
        pytest.param(SymmetricTensor, SymmetricTensor, 2, 1, 1, 2, id='Sym@Sym-2-1-1-2'),
        pytest.param(SymmetricTensor, SymmetricTensor, 0, 3, 2, 0, id='Sym@Sym-0-3-2-0'),
        pytest.param(ChargedTensor, ChargedTensor, 1, 2, 2, 1, id='Charged@Charged-1-2-2-1'),
        pytest.param(ChargedTensor, ChargedTensor, 0, 3, 2, 0, id='Charged@Charged-0-3-2-0'),
        pytest.param(ChargedTensor, SymmetricTensor, 1, 2, 2, 1, id='Charged@Sym-1-2-2-1'),
        pytest.param(SymmetricTensor, ChargedTensor, 0, 3, 2, 0, id='Sym@Charged-0-3-2-0'),
    ],
)
def test_outer(cls_A, cls_B, cA, dA, cB, dB, make_compatible_tensor, compatible_symmetry):
    if isinstance(compatible_symmetry, SU2Symmetry):
        # need to make the test case smaller, so this does not need to many resources
        kwargs = dict(use_pipes=False, max_blocks=3)
        cA = dA = cB = dB = 1
    else:
        kwargs = {}
    A_labels = list('abcdefg')[: cA + dA]
    B_labels = list('hijklmn')[: cB + dB]
    A: Tensor = make_compatible_tensor(cA, dA, cls=cls_A, labels=A_labels, **kwargs)
    B: Tensor = make_compatible_tensor(cB, dB, cls=cls_B, labels=B_labels, **kwargs)

    if cls_A is ChargedTensor and cls_B is ChargedTensor:
        msg = reason = 'state_tensor_product not implemented'
        with pytest.raises(NotImplementedError, match=msg):
            _ = tensors.outer(A, B, relabel1={'a': 'x'}, relabel2={'h': 'y'})
        pytest.xfail(reason)

    res = tensors.outer(A, B, relabel1={'a': 'x'}, relabel2={'h': 'y'})

    res.test_sanity()
    A_relabelled = ['x', *A_labels[1:]]
    B_relabelled = ['y', *B_labels[1:]]
    assert res.labels == [*A_relabelled[:cA], *B_relabelled, *A_relabelled[cA:]]

    if A.symmetry.can_be_dropped:
        perm = [*range(cA), *range(cA + dA, cA + cB + dB + dA), *range(cA, cA + dA)]
        A_np = A.to_numpy(understood_braiding=True)
        B_np = B.to_numpy(understood_braiding=True)
        res_np = res.to_numpy(understood_braiding=True)
        expect = np.transpose(np.tensordot(A_np, B_np, [(), ()]), perm)
        npt.assert_almost_equal(res_np, expect)


@pytest.mark.deselect_invalid_ChargedTensor_cases(
    get_cls=lambda kw: ChargedTensor if ChargedTensor in [kw['cls_A'], kw['cls_B']] else None
)
@pytest.mark.parametrize(
    'cls_A, cls_B, legs_A, legs_B, A_contr_leg',
    [
        pytest.param(SymmetricTensor, SymmetricTensor, [2, 2], [1, 1], 1, id='Sym@Sym-2-2-1-1-codom'),
        pytest.param(SymmetricTensor, SymmetricTensor, [3, 1], [2, 2], 1, id='Sym@Sym-3-1-2-2-codom'),
        pytest.param(SymmetricTensor, SymmetricTensor, [3, 1], [1, 2], 1, id='Sym@Sym-3-1-1-2-codom'),
        pytest.param(SymmetricTensor, SymmetricTensor, [3, 0], [2, 2], 1, id='Sym@Sym-3-0-2-2-codom'),
        pytest.param(SymmetricTensor, SymmetricTensor, [2, 2], [1, 1], 2, id='Sym@Sym-2-2-1-1-dom'),
        pytest.param(SymmetricTensor, SymmetricTensor, [1, 3], [2, 2], 2, id='Sym@Sym-1-3-2-2-dom'),
        pytest.param(SymmetricTensor, SymmetricTensor, [1, 3], [2, 1], 2, id='Sym@Sym-1-3-2-1-dom'),
        pytest.param(SymmetricTensor, SymmetricTensor, [0, 3], [2, 2], 0, id='Sym@Sym-0-3-2-2-dom'),
        pytest.param(SymmetricTensor, Mask, [2, 2], [1, 1], 1, id='Sym@Mask-2-2-1-1-codom'),
        pytest.param(SymmetricTensor, Mask, [2, 2], [1, 1], 2, id='Sym@Mask-2-2-1-1-dom'),
        pytest.param(SymmetricTensor, DiagonalTensor, [2, 2], [1, 1], 1, id='Sym@Diag-2-2-1-1-codom'),
        pytest.param(SymmetricTensor, DiagonalTensor, [2, 2], [1, 1], 2, id='Sym@Diag-2-2-1-1-dom'),
        pytest.param(SymmetricTensor, ChargedTensor, [3, 1], [1, 2], 1, id='Sym@Charged-3-1-1-2-codom'),
        pytest.param(SymmetricTensor, ChargedTensor, [1, 3], [2, 1], 2, id='Sym@Charged-1-3-2-1-dom'),
        pytest.param(ChargedTensor, SymmetricTensor, [3, 1], [1, 2], 1, id='Charged@Sym-3-1-1-2-codom'),
        pytest.param(ChargedTensor, SymmetricTensor, [1, 3], [2, 1], 2, id='Charged@Sym-1-3-2-1-dom'),
        pytest.param(ChargedTensor, Mask, [2, 2], [1, 1], 1, id='Charged@Mask-2-2-1-1-codom'),
        pytest.param(ChargedTensor, Mask, [2, 2], [1, 1], 2, id='Charged@Mask-2-2-1-1-dom'),
        pytest.param(ChargedTensor, DiagonalTensor, [2, 2], [1, 1], 1, id='Charged@Diag-2-2-1-1-codom'),
        pytest.param(ChargedTensor, DiagonalTensor, [2, 2], [1, 1], 2, id='Charged@Diag-2-2-1-1-dom'),
        pytest.param(ChargedTensor, ChargedTensor, [3, 1], [1, 2], 1, id='Charged@Charged-3-1-1-2-codom'),
        pytest.param(ChargedTensor, ChargedTensor, [1, 3], [2, 1], 2, id='Charged@Charged-1-3-2-1-dom'),
    ],
)
def test_partial_compose(cls_A, cls_B, legs_A, legs_B, A_contr_leg, make_compatible_tensor):
    labels_A = [*list('abcde')[: legs_A[0]], *list('fghij')[: legs_A[1]][::-1]]
    labels_B = [*list('klmno')[: legs_B[0]], *list('pqrst')[: legs_B[1]][::-1]]
    A: Tensor = make_compatible_tensor(codomain=legs_A[0], domain=legs_A[1], labels=labels_A, cls=cls_A)

    if (A.has_pipes and cls_B is Mask) or (isinstance(A.get_leg(A_contr_leg), LegPipe) and cls_B is DiagonalTensor):
        # need to design such that the diagonal one doesnt need pipes
        pytest.xfail()

    if A_contr_leg < A.num_codomain_legs:
        codom_B = legs_B[0]
        dom_B = A.codomain[A_contr_leg : A_contr_leg + legs_B[1]]
        relabel1 = {'e': 'x'}
        relabel2 = {'i': 'y'}
        num_contr_legs = legs_B[1]
    else:
        domain_idx = A.num_legs - A_contr_leg - legs_B[0]
        codom_B = A.domain[domain_idx : domain_idx + legs_B[0]]
        dom_B = legs_B[1]
        relabel1 = {'a': 'x'}
        relabel2 = {'p': 'y'}
        num_contr_legs = legs_B[0]

    if cls_B is Mask and A_contr_leg >= A.num_codomain_legs:
        with pytest.raises(NotImplementedError, match='Mask generation broken'):
            _ = make_compatible_tensor(codomain=codom_B, domain=dom_B, labels=labels_B, cls=cls_B)
        pytest.xfail(reason='Mask generation broken')

    B: Tensor = make_compatible_tensor(codomain=codom_B, domain=dom_B, labels=labels_B, cls=cls_B)

    if isinstance(A.backend, backends.FusionTreeBackend) and A.has_pipes and cls_B is Mask:
        with pytest.raises(NotImplementedError, match='_mask_contract does not support pipes yet'):
            _ = tensors.partial_compose(A, B, A_contr_leg, relabel1=relabel1, relabel2=relabel2)
        pytest.xfail('_mask_contract does not support pipes yet')
    if isinstance(A, ChargedTensor) and isinstance(B, ChargedTensor):
        with pytest.raises(NotImplementedError, match='state_tensor_product not implemented'):
            _ = tensors.partial_compose(A, B, A_contr_leg, relabel1=relabel1, relabel2=relabel2)
        pytest.xfail(reason='state_tensor_product not implemented')

    res = tensors.partial_compose(A, B, A_contr_leg, relabel1=relabel1, relabel2=relabel2)
    res.test_sanity()

    if isinstance(B, Mask):
        expect = tensors.apply_mask(A, B, A_contr_leg)
        expect.relabel({A.labels[A_contr_leg]: B.labels[int(A_contr_leg >= A.num_codomain_legs)]})
        expect.relabel(relabel1)
        expect.relabel(relabel2)
    elif isinstance(B, DiagonalTensor):
        expect = tensors.scale_axis(A, B, A_contr_leg)
        expect.relabel({A.labels[A_contr_leg]: B.labels[int(A_contr_leg >= A.num_codomain_legs)]})
        expect.relabel(relabel1)
        expect.relabel(relabel2)
    else:
        # we can treat the invariant part of ChargedTensor in the same way, but need to take some extra care for the charge leg
        A_ = A.invariant_part if isinstance(A, ChargedTensor) else A
        B_ = B.invariant_part if isinstance(B, ChargedTensor) else B
        c = ChargedTensor._CHARGE_LEG_LABEL
        if isinstance(B, ChargedTensor):
            relabel2 = {**relabel2, c: f'{c}B'}

        # compare to compose (bend legs accordingly before and after composing)
        A_last_leg = A_contr_leg + num_contr_legs - 1
        if A_contr_leg < A_.num_codomain_legs:
            if isinstance(B, ChargedTensor):
                B_ = tensors.move_leg(B_, c, codomain_pos=B_.num_codomain_legs - 1, bend_right=True)
            inter_codom = list(range(A_contr_leg, A_last_leg + 1))
            inter_dom = list(reversed(range(A_contr_leg))) + list(reversed(range(A_last_leg + 1, A_.num_legs)))
            num_bend_right = A_.num_codomain_legs - 1 - A_last_leg
            inter_bend_right = [False] * A_contr_leg + [True] * (A_.num_legs - A_contr_leg)
            expect = tensors.compose(
                B_,
                tensors.permute_legs(A_, inter_codom, inter_dom, bend_right=inter_bend_right),
                relabel1=relabel2,
                relabel2=relabel1,
            )
            final_codom = list(range(expect.num_legs - A_contr_leg, expect.num_legs)) + list(
                range(expect.num_codomain_legs + num_bend_right)
            )
            final_dom = list(reversed(range(expect.num_codomain_legs + num_bend_right, expect.num_legs - A_contr_leg)))
            final_bend_right = [True] * (expect.num_legs - A_contr_leg) + [False] * A_contr_leg
            expect = tensors.permute_legs(expect, final_codom, final_dom, bend_right=final_bend_right)
        else:
            inter_codom = list(range(A_last_leg + 1, A_.num_legs)) + list(range(A_contr_leg))
            inter_dom = list(reversed(range(A_contr_leg, A_last_leg + 1)))
            num_bend_left = A_.num_legs - 1 - A_last_leg
            num_bend_right = A_contr_leg - A.num_codomain_legs
            inter_bend_right = [True] * A_contr_leg + [False] * (A_.num_legs - A_contr_leg)
            expect = tensors.compose(
                tensors.permute_legs(A_, inter_codom, inter_dom, bend_right=inter_bend_right),
                B_,
                relabel1=relabel1,
                relabel2=relabel2,
            )
            final_codom = list(range(num_bend_left, expect.num_codomain_legs - num_bend_right))
            final_dom = list(reversed(range(num_bend_left))) + list(
                reversed(range(expect.num_codomain_legs - num_bend_right, expect.num_legs))
            )
            final_bend_right = [False] * num_bend_left + [True] * (expect.num_legs - num_bend_left)
            expect = tensors.permute_legs(expect, final_codom, final_dom, bend_right=final_bend_right)

        if isinstance(A, ChargedTensor) and isinstance(B, ChargedTensor):
            expect = tensors.move_leg(expect, f'{c}B', domain_pos=1, bend_right=True)
            expect = ChargedTensor.from_two_charge_legs(expect, state1=A.charged_state, state2=B.charged_state)
        elif isinstance(A, ChargedTensor):
            expect = ChargedTensor.from_invariant_part(expect, A.charged_state)
        elif isinstance(B, ChargedTensor):
            expect = tensors.move_leg(expect, f'{c}B', domain_pos=0, bend_right=True).relabel({f'{c}B': c})
            expect = ChargedTensor.from_invariant_part(expect, B.charged_state)

    assert res.labels == expect.labels
    assert tensors.almost_equal(res, expect)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, codom, dom',
    [
        pytest.param(SymmetricTensor, ['a', 'b', 'a'], ['c', 'd'], id='Sym-aba-cd'),
        pytest.param(SymmetricTensor, ['a', 'b'], ['b', 'a'], id='Sym-ab-ba'),
        pytest.param(SymmetricTensor, ['a', 'c'], ['b', 'a'], id='Sym-ac-ba'),
        pytest.param(SymmetricTensor, ['a', 'b'], ['c', 'd'], id='Sym-ab-cd'),
        pytest.param(ChargedTensor, ['a', 'b'], ['b', 'a'], id='Charged-ab-ba'),
        pytest.param(ChargedTensor, ['a', 'b', 'a'], ['c', 'd'], id='Charged-aba-cd'),
        pytest.param(DiagonalTensor, ['a'], ['a'], id='Diag-a-a'),
    ],
)
def test_partial_trace(cls, codom, dom, make_compatible_space, make_compatible_tensor, np_random):
    #
    # 1) Prepare inputs
    #
    trace_legs = {l: make_compatible_space() for l in duplicate_entries([*codom, *dom])}
    # build compatible legs.
    # If we see a label for the second time, use opposite duality than the first time, and different label.
    # In the domain, use opposite duality than in the codomain.
    seen_labels = []
    codomain_spaces = []
    codomain_labels = []
    for l in codom:
        if l in seen_labels:
            codomain_spaces.append(trace_legs[l].dual)
            codomain_labels.append(f'{l}*')
        elif l in trace_legs:
            codomain_spaces.append(trace_legs[l])
            seen_labels.append(l)
            codomain_labels.append(l)
        else:
            codomain_spaces.append(make_compatible_space())
            codomain_labels.append(l)
    domain_spaces = []
    domain_labels = []
    for l in dom:
        if l in seen_labels:
            domain_spaces.append(trace_legs[l])
            domain_labels.append(f'{l}*')
        elif l in trace_legs:
            domain_spaces.append(trace_legs[l].dual)
            domain_labels.append(l)
            seen_labels.append(l)
        else:
            domain_spaces.append(make_compatible_space())
            domain_labels.append(l)
    #
    T: Tensor = make_compatible_tensor(
        codomain_spaces, domain_spaces, cls=cls, labels=[*codomain_labels, *reversed(domain_labels)]
    )
    #
    how_to_call = np_random.choice(['positions', 'labels'])
    labels = T.labels
    pairs_positions = [(labels.index(l), labels.index(f'{l}*')) for l in trace_legs]
    if how_to_call == 'positions':
        pairs = pairs_positions
    if how_to_call == 'labels':
        pairs = [(l, f'{l}*') for l in trace_legs]
    #
    # 2) Call the actual function
    #
    levels = None
    if isinstance(T.backend, backends.FusionTreeBackend) and T.symmetry.braiding_style.value >= 20:
        levels = list(np_random.permutation(T.num_legs - len(pairs)))
        idcs1 = [p[0] for p in pairs_positions]
        idcs2 = [p[1] for p in pairs_positions]
        for idx in np.argsort(idcs2):
            level = levels[idcs1[idx]]
            levels = [l + 1 if l > level else l for l in levels]
            levels.insert(idcs2[idx], level + 1)
    if cls is ChargedTensor and not T.symmetry.can_be_dropped and len(pairs) * 2 == T.num_legs:
        # trace over all legs except charge_leg -> charged_state != None must be fulfilled
        # -> symmetry.can_be_dropped == True must be fulfilled
        # with the current deselection of ChargedTensors in the tests, we catch FermionParity
        # with id='Charged-ab-ba' here
        return
    #
    res = tensors.partial_trace(T, *pairs, levels=levels)
    #
    # 3) Test the result
    if T.symmetry.has_trivial_braid:
        num_open = T.num_legs - 2 * len(pairs)
        if num_open == 0:
            assert isinstance(res, (float, complex))
            res_np = res
        else:
            assert isinstance(res, cls)
            res.test_sanity()
            assert res.labels == [l for l in T.labels if l[0] not in trace_legs]
            assert res.codomain.factors == [
                sp for sp, l in zip(T.codomain, T.codomain_labels) if l[0] not in trace_legs
            ]
            assert res.domain.factors == [sp for sp, l in zip(T.domain, T.domain_labels) if l[0] not in trace_legs]
            res_np = res.to_numpy()
        #
        idcs1 = [p[0] for p in pairs_positions]
        idcs2 = [p[1] for p in pairs_positions]
        remaining = [n for n in range(T.num_legs) if n not in idcs1 and n not in idcs2]
        expect = T.backend.block_backend.trace_partial(T.to_dense_block(), idcs1, idcs2, remaining)
        expect = T.backend.block_backend.to_numpy(expect)
        npt.assert_almost_equal(res_np, expect)
    elif T.symmetry.has_symmetric_braid:
        pass  # TODO need to adjust numpy code above to be more careful about leg order
    else:
        pass  # TODO need some other way to compare...


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, num_cod, num_dom, codomain, domain, levels, bend_right',
    [
        pytest.param(SymmetricTensor, 2, 2, [0, 1], [3, 2], None, True, id='Symmetric-2-2-trivial'),
        pytest.param(SymmetricTensor, 2, 2, [1, 0], [3, 2], [0, 1, 2, 3], True, id='Symmetric-2-2-braid_cod'),
        pytest.param(SymmetricTensor, 2, 2, [0, 1], [2, 3], [0, 1, 2, 3], True, id='Symmetric-2-2-braid_dom'),
        pytest.param(SymmetricTensor, 2, 2, [1, 0], [2, 3], [0, 1, 2, 3], True, id='Symmetric-2-2-braid_both'),
        pytest.param(SymmetricTensor, 2, 2, [0, 1, 2], [3], None, True, id='Symmetric-2-2-bend_right'),
        pytest.param(SymmetricTensor, 2, 2, [3, 0, 1], [2], None, False, id='Symmetric-2-2-bend_left'),
        pytest.param(SymmetricTensor, 2, 2, [0, 3], [1, 2], [0, 1, 2, 3], True, id='Symmetric-2-2-general'),
        pytest.param(SymmetricTensor, 2, 2, [0, 3], [1, 2], [0, 1, 2, 3], False, id='Symmetric-2-2-general2'),
        pytest.param(SymmetricTensor, 3, 3, [1, 4, 0], [2, 3, 5], [4, 1, 3, 5, 0, 2], True, id='Symmetric-3-3-general'),
        pytest.param(DiagonalTensor, 1, 1, [0], [1], [0, 1], True, id='Diagonal-trivial'),
        pytest.param(DiagonalTensor, 1, 1, [1], [0], [0, 1], True, id='Diagonal-swap'),
        pytest.param(DiagonalTensor, 1, 1, [1, 0], [], [0, 1], True, id='Diagonal-general'),
        pytest.param(Mask, 1, 1, [0], [1], [0, 1], True, id='Mask-trivial'),
        pytest.param(Mask, 1, 1, [1], [0], [0, 1], True, id='Mask-swap'),
        pytest.param(Mask, 1, 1, [1, 0], [], [0, 1], True, id='Mask-general'),
        pytest.param(ChargedTensor, 2, 2, [0, 1], [3, 2], None, True, id='Charged-2-2-trivial'),
        pytest.param(ChargedTensor, 2, 2, [1, 0], [2, 3], [0, 1, 2, 3], True, id='Charged-2-2-braid'),
        pytest.param(ChargedTensor, 2, 2, [0, 1, 2], [3], None, True, id='Charged-2-2-bend'),
        pytest.param(ChargedTensor, 2, 2, [0, 3], [1, 2], [0, 1, 2, 3], True, id='Charged-2-2-general'),
    ],
)
def test_permute_legs(
    cls, num_cod, num_dom, codomain, domain, levels, bend_right, make_compatible_tensor, compatible_symmetry, np_random
):
    if isinstance(compatible_symmetry, SU2Symmetry) and (num_cod + num_dom) > 4:
        # make sure we dont need symmetry data for too large sectors
        sectors = [[0], [1], [2]]
        legs = []
        for _ in range(num_cod + num_dom):
            mults = np_random.integers(1, 5, size=3)
            is_dual = np_random.choice([True, False])
            legs.append(ElementarySpace(compatible_symmetry, sectors, mults, is_dual))
        T_codomain = legs[:num_cod]
        T_domain = legs[num_cod:]
    else:
        T_codomain = num_cod
        T_domain = num_dom
    T = make_compatible_tensor(T_codomain, T_domain, max_block_size=3, cls=cls)

    if cls in [DiagonalTensor, Mask]:
        if len(codomain) == 1:
            # special case where legs are not actually permuted -> no warning expected
            catch_warnings = nullcontext()
        else:
            catch_warnings = pytest.warns(UserWarning, match='Converting to SymmetricTensor *')
    else:
        catch_warnings = nullcontext()

    with catch_warnings:
        res = tensors.permute_legs(T, codomain, domain, levels, bend_right=bend_right)
    res.test_sanity()

    for n, i in enumerate(codomain):
        assert res.codomain[n] == T._as_codomain_leg(i)
    for n, i in enumerate(domain):
        assert res.domain[n] == T._as_domain_leg(i)
    assert res.codomain_labels == [T.labels[n] for n in codomain]
    assert res.domain_labels == [T.labels[n] for n in domain]

    if T.symmetry.has_trivial_braid:
        # if the braid is trivial, we can compare to braiding the to_numpy() representations
        # for a symmetric braid, we can do to_numpy(), but the numpy rep loses the braiding information
        # for general braids, we cant even to to_numpy()
        # makes sense to compare with dense blocks
        expect = np.transpose(T.to_numpy(), [*codomain, *reversed(domain)])
        actual = res.to_numpy()
        npt.assert_allclose(actual, expect, atol=1.0e-14)
    else:
        # TODO (JU) is there anything we can do in that case to check?
        #           - we do a bunch of "concrete tests" in backends/test_fusion_tree_backend.py
        #           - we check if we can undo the permutation below.
        pass

    # construct the instructions needed to undo the original instructions
    leg_perm = [*codomain, *reversed(domain)]
    inv_leg_perm = inverse_permutation(leg_perm)
    codomain2 = inv_leg_perm[:num_cod]
    domain2 = inv_leg_perm[num_cod:][::-1]
    levels2 = None if levels is None else [levels[i] for i in leg_perm]

    res2 = tensors.permute_legs(res, codomain2, domain2, levels2, bend_right=bend_right)
    res2.test_sanity()
    assert tensors.almost_equal(res2, T, allow_different_types=True)


@pytest.mark.parametrize(
    'cls, dom, cod, new_leg_dual',
    [
        pytest.param(SymmetricTensor, 1, 1, False, id='Sym-1-1-False'),
        pytest.param(SymmetricTensor, 1, 1, True, id='Sym-1-1-True'),
        pytest.param(SymmetricTensor, 3, 1, False, id='Sym-3-1-False'),
        pytest.param(SymmetricTensor, 2, 2, False, id='Sym-2-2-False'),
        pytest.param(SymmetricTensor, 2, 2, True, id='Sym-2-2-True'),
        pytest.param(DiagonalTensor, 1, 1, False, id='Diag-False'),
        pytest.param(DiagonalTensor, 1, 1, True, id='Diag-True'),
        pytest.param(Mask, 1, 1, False, id='Mask-False'),
        pytest.param(Mask, 1, 1, True, id='Mask-True'),
    ],
)
def test_qr_lq(cls, dom, cod, new_leg_dual, make_compatible_tensor):
    T_labels = list('efghijk')[: dom + cod]
    T: Tensor = make_compatible_tensor(dom, cod, cls=cls, labels=T_labels)

    Q, R = tensors.qr(T, new_leg_dual=new_leg_dual)
    Q.test_sanity()
    R.test_sanity()
    assert tensors.almost_equal(Q @ R, T, allow_different_types=True)
    eye = tensors.SymmetricTensor.from_eye(Q.domain, backend=T.backend)
    assert tensors.almost_equal(Q.hc @ Q, eye, allow_different_types=True)

    L, Q2 = tensors.lq(T, new_leg_dual=new_leg_dual)
    L.test_sanity()
    Q2.test_sanity()
    assert tensors.almost_equal(L @ Q2, T, allow_different_types=True)
    eye = tensors.SymmetricTensor.from_eye(Q2.codomain, backend=T.backend)
    assert tensors.almost_equal(Q2 @ Q2.hc, eye, allow_different_types=True)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize('cls', [SymmetricTensor, DiagonalTensor, Mask, ChargedTensor])
def test_scalar_multiply(cls, make_compatible_tensor):
    if cls in [SymmetricTensor, ChargedTensor]:
        T = make_compatible_tensor(cls=cls, codomain=2, domain=2, max_block_size=3, max_blocks=3)
    else:
        T = make_compatible_tensor(cls=cls)

    if isinstance(T, Mask):
        catch_warnings = pytest.warns(UserWarning, match='Converting to SymmetricTensor *')
    else:
        catch_warnings = nullcontext()

    if T.symmetry.can_be_dropped:
        T_np = T.to_numpy(understood_braiding=True)

    for valid_scalar in [0, 1.0, 2.0 + 3.0j, -42]:
        with catch_warnings:
            res = tensors.scalar_multiply(valid_scalar, T)
        res.test_sanity()
        if T.symmetry.can_be_dropped:
            npt.assert_allclose(res.to_numpy(understood_braiding=True), valid_scalar * T_np)
    for invalid_scalar in [None, (1, 2), T, 'abc']:
        with pytest.raises(TypeError, match='unsupported scalar type'):
            _ = tensors.scalar_multiply(invalid_scalar, T)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, codom, dom, which_leg',
    [
        pytest.param(SymmetricTensor, 2, 2, 1, id='Sym-2-2-1'),
        pytest.param(SymmetricTensor, 2, 2, 3, id='Sym-2-2-3'),
        pytest.param(SymmetricTensor, 0, 2, 1, id='Sym-0-2-1'),
        pytest.param(SymmetricTensor, 3, 0, 1, id='Sym-3-0-1'),
        pytest.param(ChargedTensor, 2, 2, 1, id='Charged-2-2-1'),
        pytest.param(ChargedTensor, 2, 2, 3, id='Charged-2-2-3'),
        pytest.param(DiagonalTensor, 1, 1, 0, id='Diag-0'),
        pytest.param(DiagonalTensor, 1, 1, 1, id='Diag-1'),
        pytest.param(Mask, 1, 1, 0, id='Mask-0'),
        pytest.param(Mask, 1, 1, 1, id='Mask-1'),
    ],
)
def test_scale_axis(cls, codom, dom, which_leg, make_compatible_tensor, np_random):
    # 1) Prepare
    D = make_compatible_tensor(cls=DiagonalTensor, labels=['x', 'y'])
    T_labels = list('abcdefghi')[: codom + dom]
    if which_leg >= codom:
        num_legs = codom + dom
        dom = [None] * dom
        dom[num_legs - which_leg - 1] = D.leg if np_random.choice([True, False]) else D.leg.dual
    else:
        codom = [None] * codom
        codom[which_leg] = D.leg if np_random.choice([True, False]) else D.leg.dual
    if cls is Mask and which_leg == 0:
        with pytest.raises(NotImplementedError, match='Mask generation broken'):
            _ = make_compatible_tensor(codom, dom, cls=cls, labels=T_labels)
        pytest.xfail(reason='Mask generation broken')
    T: tensors.Tensor = make_compatible_tensor(codom, dom, cls=cls, labels=T_labels)

    if cls is Mask:
        catch_warnings = pytest.warns(UserWarning, match='Converting to SymmetricTensor *')
    else:
        catch_warnings = nullcontext()

    # 2) Call functions
    how_to_call = np_random.choice(['by_idx', 'by_label'])
    if how_to_call == 'by_idx':
        with catch_warnings:
            res = tensors.scale_axis(T, D, which_leg)
    if how_to_call == 'by_label':
        with catch_warnings:
            res = tensors.scale_axis(T, D, T_labels[which_leg])

    # 3) check tensor properties
    res.test_sanity()
    assert isinstance(res, SymmetricTensor if cls is Mask else cls)
    assert res.codomain == T.codomain
    assert res.domain == T.domain
    assert res.labels == T_labels

    # 4) compare to numpy
    if T.symmetry.can_be_dropped:
        T_np = T.to_numpy(understood_braiding=True)
        expect = np.swapaxes(T_np, which_leg, -1)  # swap axis to be scaled to the back
        expect = expect * D.diagonal_as_numpy()  # broadcasts to last axis of expect
        expect = np.swapaxes(expect, which_leg, -1)  # swap back
        npt.assert_allclose(res.to_numpy(understood_braiding=True), expect, atol=1.0e-14)


def test_squeeze_legs(make_compatible_tensor, compatible_symmetry):
    trivial_leg = ElementarySpace.from_trivial_sector(1, symmetry=compatible_symmetry)
    T = make_compatible_tensor(
        [None, trivial_leg, trivial_leg, None], [None, None, trivial_leg], labels=list('abcdefg')
    )

    res_all = tensors.squeeze_legs(T)
    res_all.test_sanity()
    res_1 = tensors.squeeze_legs(T, 1)
    res_1.test_sanity()
    res_2 = tensors.squeeze_legs(T, [1, 4])
    res_2.test_sanity()

    assert res_all.labels == ['a', 'd', 'f', 'g']
    assert res_1.labels == ['a', 'c', 'd', 'e', 'f', 'g']
    assert res_2.labels == ['a', 'c', 'd', 'f', 'g']

    if T.symmetry.can_be_dropped:
        T_np = T.to_numpy(understood_braiding=True)
        expect_all = T_np[:, 0, 0, :, 0, :, :]
        expect_1 = T_np[:, 0]
        expect_2 = T_np[:, 0, :, :, 0]
        npt.assert_allclose(res_all.to_numpy(understood_braiding=True), expect_all)
        npt.assert_allclose(res_1.to_numpy(understood_braiding=True), expect_1)
        npt.assert_allclose(res_2.to_numpy(understood_braiding=True), expect_2)
    else:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc


@pytest.mark.parametrize(
    'cls, dom, cod, new_leg_dual',
    [
        pytest.param(SymmetricTensor, 1, 1, False, id='Sym-1-1-False'),
        pytest.param(SymmetricTensor, 1, 3, False, id='Sym-1-3-False'),
        pytest.param(SymmetricTensor, 3, 1, False, id='Sym-3-1-False'),
        pytest.param(SymmetricTensor, 2, 2, False, id='Sym-2-2-False'),
        pytest.param(SymmetricTensor, 2, 2, True, id='Sym-2-2-True'),
        pytest.param(DiagonalTensor, 1, 1, False, id='Diag-False'),
        pytest.param(DiagonalTensor, 1, 1, True, id='Diag-True'),
        pytest.param(Mask, 1, 1, False, id='Mask-False'),
        pytest.param(Mask, 1, 1, True, id='Mask-True'),
    ],
)
def test_svd(cls, dom, cod, new_leg_dual, make_compatible_tensor):
    """Test svd and various related functions. Covers

    - svd
    - truncated_svd
        - truncate_singular_values
        - svd_apply_mask
    """
    T_labels = list('efghijklmn')[: dom + cod]
    T: Tensor = make_compatible_tensor(dom, cod, labels=T_labels, cls=cls)

    print('Normal (non-truncated) SVD')

    U, S, Vh = tensors.svd(T, new_labels=['a', 'b', 'c', 'd'], new_leg_dual=new_leg_dual)
    U.test_sanity()
    S.test_sanity()
    Vh.test_sanity()
    assert U.labels == [*T.codomain_labels, 'a']
    assert S.labels == ['b', 'c']
    assert Vh.labels == ['d', *reversed(T.domain_labels)]

    assert isinstance(S, DiagonalTensor)
    assert (S >= 0).all()
    npt.assert_almost_equal(tensors.norm(S), tensors.norm(T))

    assert tensors.almost_equal(U @ S @ Vh, T, allow_different_types=True)
    eye = tensors.SymmetricTensor.from_eye(S.domain, backend=T.backend)
    assert tensors.almost_equal(U.hc @ U, eye, allow_different_types=True)
    assert tensors.almost_equal(Vh @ Vh.hc, eye, allow_different_types=True)

    if isinstance(T.backend, backends.FusionTreeBackend) and T.has_pipes:
        with pytest.raises(NotImplementedError, match='_mask_contract does not support pipes yet'):
            _ = tensors.truncated_svd(T)
        pytest.xfail('_mask_contract does not support pipes yet')

    print('Truncated SVD')
    for svd_min, normalize_to in [(1e-14, None), (1e-4, None), (1e-4, 2.7)]:
        U, S, Vh, err, renormalize = tensors.truncated_svd(
            T, new_leg_dual=new_leg_dual, normalize_to=normalize_to, svd_min=svd_min
        )
        U.test_sanity()
        S.test_sanity()
        Vh.test_sanity()
        # check that U @ S @ Vd recovers the original tensor up to the error incurred
        T_approx = U @ S @ Vh / renormalize
        npt.assert_almost_equal(err, tensors.norm(T.as_SymmetricTensor() - T_approx))
        # check isometric properties
        eye = tensors.SymmetricTensor.from_eye(S.domain, backend=T.backend)
        assert tensors.almost_equal(U.hc @ U, eye, allow_different_types=True)
        assert tensors.almost_equal(Vh @ Vh.hc, eye, allow_different_types=True)


@pytest.mark.deselect_invalid_ChargedTensor_cases(
    get_cls=lambda kw: ChargedTensor if ChargedTensor in [kw['cls_A'], kw['cls_B']] else None
    # the fixture only listens to ``get_cls(kwargs) is ChargedTensor``, so we need to return
    # ChargedTensor if *either* of the two classes if ChargedTensor.
)
@pytest.mark.parametrize(
    'cls_A, cls_B, labels_A, labels_B, contr_A, contr_B',
    [
        pytest.param(
            SymmetricTensor,
            SymmetricTensor,
            [['a', 'b'], ['c', 'd']],
            [['c', 'e'], ['a', 'f']],
            [0, 3],
            [3, 0],
            id='Sym@Sym-4-2-4',
        ),
        pytest.param(
            SymmetricTensor, SymmetricTensor, [['a', 'b'], ['c']], [['d', 'e'], ['f']], [], [], id='Sym@Sym-3-0-3'
        ),
        pytest.param(
            SymmetricTensor,
            SymmetricTensor,
            [['a', 'b'], ['c', 'd']],
            [['c', 'a'], ['d', 'b']],
            [0, 1, 3, 2],
            [1, 2, 0, 3],
            id='Sym@Sym-4-4-4',
        ),
        pytest.param(
            SymmetricTensor, SymmetricTensor, [[], ['a', 'b']], [['b', 'c'], ['d']], [0], [0], id='Sym@Sym-2-1-3'
        ),
        #
        pytest.param(
            SymmetricTensor,
            ChargedTensor,
            [['a', 'b'], ['c', 'd']],
            [['c', 'e'], ['a', 'f']],
            [0, 3],
            [3, 0],
            id='Sym@Charged-4-2-4',
        ),
        pytest.param(
            ChargedTensor, SymmetricTensor, [['a', 'b'], ['c']], [['d', 'e'], ['f']], [], [], id='Charged@Sym-3-0-3'
        ),
        pytest.param(
            SymmetricTensor,
            ChargedTensor,
            [['a', 'b'], ['c', 'd']],
            [['c', 'a'], ['d', 'b']],
            [0, 1, 3, 2],
            [1, 2, 0, 3],
            id='Sym@Charged-4-4-4',
        ),
        pytest.param(
            SymmetricTensor, ChargedTensor, [[], ['a', 'b']], [['b', 'c'], ['d']], [0], [0], id='Sym@Charged-2-1-3'
        ),
        #
        # Note: need to put DiagonalTensor first to get correct legs. If SymmetricTensor is first,
        # it generates independent legs, which can not both be on a diagonalTensor.
        pytest.param(
            DiagonalTensor,
            SymmetricTensor,
            [['c'], ['b']],
            [['a', 'b'], ['c', 'd']],
            [1, 0],
            [1, 3],
            id='Diag@Sym-4-2-2',
        ),
        pytest.param(
            SymmetricTensor, DiagonalTensor, [['a', 'b'], ['c', 'd']], [['e'], ['b']], [1], [1], id='Sym@Diag-4-1-2'
        ),
        pytest.param(
            SymmetricTensor, DiagonalTensor, [['a', 'b'], ['c', 'd']], [['e'], ['f']], [], [], id='Sym@Diag-4-0-2'
        ),
        #
        # Note: If both legs of a mask are contracted, we should generate the mask first. otherwise its legs may be invalid.
        pytest.param(
            Mask, SymmetricTensor, [['c'], ['b']], [['a', 'b'], ['c', 'd']], [1, 0], [1, 3], id='Sym@Mask-4-2-2'
        ),
        pytest.param(SymmetricTensor, Mask, [['a', 'b'], ['c', 'd']], [['e'], ['b']], [1], [1], id='Sym@Mask-4-1-2'),
        pytest.param(SymmetricTensor, Mask, [['a', 'b'], ['c', 'd']], [['e'], ['f']], [], [], id='Sym@Mask-4-0-2'),
        #
        pytest.param(
            ChargedTensor, DiagonalTensor, [['a', 'b'], ['c', 'd']], [['e'], ['b']], [1], [1], id='Charged@Diag-4-1-2'
        ),
        pytest.param(ChargedTensor, Mask, [['a', 'b'], ['c', 'd']], [['e'], ['b']], [1], [1], id='Charged@Mask-4-1-2'),
        #
        pytest.param(DiagonalTensor, DiagonalTensor, [['a'], ['b']], [['c'], ['b']], [1], [1], id='Diag@Diag-2-1-2'),
        pytest.param(
            DiagonalTensor, DiagonalTensor, [['a'], ['b']], [['b'], ['a']], [1, 0], [0, 1], id='Diag@Diag-2-2-2'
        ),
        pytest.param(DiagonalTensor, DiagonalTensor, [['a'], ['b']], [['c'], ['d']], [], [], id='Diag@Diag-2-0-2'),
        #
        pytest.param(Mask, Mask, [['a'], ['b']], [['c'], ['b']], [1], [1], id='Mask@Mask-2-1-2'),
        #
        # TODO: having issues randomly generating the masks in this case...
        # pytest.param(Mask, Mask, [['a'], ['b']], [['a'], ['b']], [0, 1], [0, 1], id='Mask@Mask-2-2-2'),
        #
        pytest.param(Mask, Mask, [['a'], ['b']], [['c'], ['d']], [], [], id='Mask@Mask-2-0-2'),
    ],
)
@pytest.mark.parametrize('use_pipes', [False, 0.3])
def test_tdot(
    cls_A: type[tensors.Tensor],
    cls_B: type[tensors.Tensor],
    labels_A: list[list[str]],
    labels_B: list[list[str]],
    contr_A: list[int],
    contr_B: list[int],
    use_pipes: bool,
    make_compatible_tensor,
    np_random,
):
    if use_pipes and cls_A in [Mask, DiagonalTensor] or cls_B in [Mask, DiagonalTensor]:
        # need to design such that the diagonal one doesnt need pipes
        pytest.xfail()

    A: Tensor = make_compatible_tensor(
        codomain=len(labels_A[0]),
        domain=len(labels_A[1]),
        labels=[*labels_A[0], *reversed(labels_A[1])],
        max_block_size=3,
        max_blocks=3,
        cls=cls_A,
        use_pipes=use_pipes,
    )

    # create B such that legs with the same label can be contracted
    B: Tensor = make_compatible_tensor(
        codomain=[A._as_domain_leg(l) if A.has_label(l) else None for l in labels_B[0]],
        domain=[A._as_codomain_leg(l) if A.has_label(l) else None for l in labels_B[1]],
        labels=[*labels_B[0], *reversed(labels_B[1])],
        max_block_size=2,
        max_blocks=3,
        cls=cls_B,
        use_pipes=use_pipes,
    )

    num_contr = len(contr_A)
    num_open_A = A.num_legs - num_contr
    num_open_B = B.num_legs - num_contr
    num_open = num_open_A + num_open_B
    # make sure we defined compatible legs
    for ia, ib in zip(contr_A, contr_B):
        assert A._as_domain_leg(ia) == B._as_codomain_leg(ib), (
            f'{ia} / {A.labels[ia]} incompatible with {ib} / {B.labels[ib]}'
        )

    expect_codomain = [A._as_codomain_leg(n) for n in range(A.num_legs) if n not in contr_A]
    expect_domain = [B._as_domain_leg(n) for n in range(B.num_legs) if n not in contr_B][::-1]
    expect_legs = [A.get_leg(n) for n in range(A.num_legs) if n not in contr_A] + [
        B.get_leg(n) for n in range(B.num_legs) if n not in contr_B
    ]
    expect_labels = [A._labels[n] for n in range(A.num_legs) if n not in contr_A] + [
        B._labels[n] for n in range(B.num_legs) if n not in contr_B
    ]

    if (cls_A is Mask and cls_B is Mask) and num_contr > 0:
        with pytest.raises(NotImplementedError, match='tensors._compose_with_Mask not implemented for Mask'):
            _ = tensors.tdot(A, B, contr_A, contr_B)
        pytest.xfail('apply_mask(Mask, Mask) not yet supported')

    # expected errors:
    if not A.symmetry.has_symmetric_braid and (contr_A, contr_B) != ([1, 0], [0, 1]):
        # would need to specify levels and/or bend_right
        msg = 'Legs can not be permuted automatically. Explicitly use permute_legs()'
        with pytest.raises(SymmetryError, match=msg):
            _ = tensors.tdot(A, B, contr_A, contr_B)
        return
    if not A.symmetry.has_trivial_braid and (contr_A, contr_B) not in [([1, 0], [1, 3]), ([1, 0], [0, 1])]:
        # would need to specify bend_right
        msg = 'Legs can not be permuted automatically. Explicitly use permute_legs()'
        with pytest.raises(SymmetryError, match=msg):
            _ = tensors.tdot(A, B, contr_A, contr_B)
        return
    if cls_A is ChargedTensor and A.charged_state is None and A.num_legs + B.num_legs == 2 * num_contr:
        with pytest.raises(
            ValueError, match='Can not instantiate ChargedTensor with no legs and unspecified charged_states.'
        ):
            _ = tensors.tdot(A, B, contr_A, contr_B)
        return
    if cls_B is ChargedTensor and B.charged_state is None and A.num_legs + B.num_legs == 2 * num_contr:
        with pytest.raises(
            ValueError, match='Can not instantiate ChargedTensor with no legs and unspecified charged_states.'
        ):
            _ = tensors.tdot(A, B, contr_A, contr_B)
        return

    if not A.symmetry.has_symmetric_braid:
        if cls_A is not DiagonalTensor:
            levels_A = list(np_random.permutation(A.num_legs))
            codomain_A = [i for i in range(A.num_legs) if i not in contr_A]
            if any([isinstance(leg, LegPipe) for leg in A.legs]):
                with pytest.raises(RuntimeError, match='iter_tree_blocks can not deal with pipes'):
                    _ = tensors.permute_legs(A, codomain=codomain_A, domain=contr_A, levels=levels_A)
                pytest.xfail(reason='FTbackend cant deal with pipes yet')
            if cls_A is Mask and len(codomain_A) != 1:
                # otherwise, mask transpose is used
                catch_warnings = pytest.warns(UserWarning, match='Converting to SymmetricTensor *')
            else:
                catch_warnings = nullcontext()
            with catch_warnings:
                A = tensors.permute_legs(A, codomain=codomain_A, domain=contr_A, levels=levels_A)
            contr_A = [A.num_legs - 1 - i for i in range(num_contr)]
        if cls_B is not DiagonalTensor:
            levels_B = list(np_random.permutation(B.num_legs))
            domain_B = [i for i in range(B.num_legs) if i not in contr_B][::-1]
            if cls_B is Mask and len(domain_B) != 1:
                catch_warnings = pytest.warns(UserWarning, match='Converting to SymmetricTensor *')
            else:
                catch_warnings = nullcontext()
            with catch_warnings:
                B = tensors.permute_legs(B, codomain=contr_B, domain=domain_B, levels=levels_B)
            contr_B = list(range(num_contr))

    compare_numpy = A.symmetry.has_trivial_braid
    # if the braid is trivial, we can compare to braiding the to_numpy() representations
    # for a symmetric braid, we can do to_numpy(), but the numpy rep loses the braiding information
    # for general braids, we cant even do to_numpy()

    res = tensors.tdot(A, B, contr_A, contr_B)

    if num_open == 0:
        # scalar result
        assert isinstance(res, (float, complex))
        res_np = res
    else:
        # tensor result
        res.test_sanity()
        if compare_numpy:
            res_np = res.to_numpy()
        assert res.codomain.factors == expect_codomain
        assert res.domain.factors == expect_domain
        assert res.legs == expect_legs
        assert res.labels == expect_labels

    if compare_numpy:
        A_np = A.to_numpy()
        B_np = B.to_numpy()
        expect = np.tensordot(A_np, B_np, [contr_A, contr_B])
        npt.assert_allclose(res_np, expect, atol=1.0e-14)


@pytest.mark.parametrize(
    'cod, dom, row, col',
    [
        pytest.param(1, 1, 3, 3, id='Tens-1-1-Grid-3-3'),
        pytest.param(2, 1, 2, 3, id='Tens-2-1-Grid-2-3'),
        pytest.param(2, 2, 2, 2, id='Tens-2-2-Grid-2-2'),
        pytest.param(3, 1, 2, 2, id='Tens-3-1-Grid-2-2'),
        pytest.param(1, 3, 2, 2, id='Tens-1-3-Grid-2-2'),
    ],
)
def test_tensor_from_grid(cod, dom, row, col, make_compatible_tensor, make_compatible_space, np_random):
    codomain = [make_compatible_space()] + [None] * (cod - 1)
    domain = [None] * (dom - 1) + [make_compatible_space()]
    T: SymmetricTensor = make_compatible_tensor(codomain, domain, cls=SymmetricTensor)
    dual_codom = T.codomain[0].is_dual
    dual_dom = T.domain[-1].is_dual

    # build grid -> first finish first row and column, then fill in the rest
    grid = [[T]]
    for _ in range(col - 1):
        space = make_compatible_space(is_dual=dual_dom)
        grid[0].append(make_compatible_tensor(T.codomain, [*T.domain[:-1], space], cls=SymmetricTensor))
    for _ in range(row - 1):
        space = make_compatible_space(is_dual=dual_codom)
        grid.append([make_compatible_tensor([space, *T.codomain[1:]], T.domain, cls=SymmetricTensor)])
    for i in range(1, row):
        for j in range(1, col):
            grid[i].append(make_compatible_tensor(grid[i][0].codomain, grid[0][j].domain, cls=SymmetricTensor))

    # permuting legs should commute with building the full tensor
    # make sure codomain[0] and domain[-1] stay in their positions
    perm = np_random.permutation(T.num_legs)
    idx = np.where(perm == 0)[0][0]
    perm[[0, idx]] = perm[[idx, 0]]
    idx = np.where(perm == T.num_codomain_legs)[0][0]
    perm_codom = perm[:idx]
    perm_dom = perm[idx:][::-1]
    levels = np_random.permutation(T.num_legs)

    res1 = tensors.tensor_from_grid(grid)
    # check to_numpy
    if T.symmetry.can_be_dropped:
        res_np = [
            np.concatenate([op.to_numpy(understood_braiding=True) for op in row], axis=T.num_codomain_legs)
            for row in grid
        ]
        res_np = np.concatenate(res_np, axis=0)
        npt.assert_almost_equal(res1.to_numpy(understood_braiding=True), res_np)

    # check permute_legs commutes with tensor_from_grid
    bend_right = np_random.choice([True, False])
    res1 = tensors.permute_legs(res1, perm_codom, perm_dom, levels, bend_right=bend_right)
    res2 = [
        [tensors.permute_legs(op, perm_codom, perm_dom, levels, bend_right=bend_right) for op in row] for row in grid
    ]
    res2 = tensors.tensor_from_grid(res2)

    assert res1.codomain == res2.codomain
    assert res1.domain == res2.domain
    assert res1.backend.almost_equal(res1, res2, rtol=1e-12, atol=1e-12)
    if T.symmetry.can_be_dropped:
        npt.assert_almost_equal(res1.to_numpy(understood_braiding=True), res2.to_numpy(understood_braiding=True))


@pytest.mark.parametrize(
    'cls, legs',
    [
        pytest.param(SymmetricTensor, 2, id='Sym-2'),
        pytest.param(SymmetricTensor, 1, id='Sym-1'),
        pytest.param(ChargedTensor, 2, id='Charged-2'),
        pytest.param(ChargedTensor, 1, id='Charged-1'),
        pytest.param(DiagonalTensor, 1, id='Diag'),
    ],
)
def test_trace(cls, legs, make_compatible_tensor, compatible_symmetry, make_compatible_sectors, make_compatible_space):
    co_domain_spaces = [make_compatible_space() for _ in range(legs)]
    if cls is ChargedTensor:
        if not compatible_symmetry.can_be_dropped:
            # can not specify charged_state => can not do full trace.
            return
        # make a ChargedTensor that has the trivial sector, otherwise the trace is always 0
        other_sector = make_compatible_sectors(1)[0]
        charge_leg = ElementarySpace.from_defining_sectors(
            compatible_symmetry,
            [compatible_symmetry.trivial_sector, other_sector],
        )
        inv_part = make_compatible_tensor(co_domain_spaces, [charge_leg, *co_domain_spaces], cls=SymmetricTensor)
        charged_state = inv_part.backend.block_backend.as_block(list(range(charge_leg.dim)))
        tensor = ChargedTensor(inv_part.set_label(-1, '!'), charged_state)
    else:
        tensor: Tensor = make_compatible_tensor(co_domain_spaces, co_domain_spaces, cls=cls)

    res = tensors.trace(tensor)
    assert isinstance(res, (float, complex))

    if tensor.symmetry.can_be_dropped:
        expect = tensor.to_numpy(understood_braiding=True)
        while expect.ndim > 0:
            expect = np.trace(expect, axis1=0, axis2=-1)
        npt.assert_almost_equal(res, expect)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, cod, dom',
    [
        pytest.param(SymmetricTensor, 2, 2, id='Sym-2-2'),
        pytest.param(SymmetricTensor, 3, 0, id='Sym-3-0'),
        pytest.param(SymmetricTensor, 1, 1, id='Sym-1-1'),
        pytest.param(SymmetricTensor, 0, 3, id='Sym-3-0'),
        pytest.param(ChargedTensor, 2, 2, id='Charged-2-2'),
        pytest.param(ChargedTensor, 3, 0, id='Charged-3-0'),
        pytest.param(ChargedTensor, 1, 1, id='Charged-1-1'),
        pytest.param(ChargedTensor, 0, 3, id='Charged-3-0'),
        pytest.param(DiagonalTensor, 1, 1, id='Diag'),
        pytest.param(Mask, 1, 1, id='Mask'),
    ],
)
def test_transpose(cls, cod, dom, make_compatible_tensor, np_random):
    labels = list('abcdefghi')[: cod + dom]
    tensor: Tensor = make_compatible_tensor(cod, dom, cls=cls, labels=labels)

    if isinstance(tensor, ChargedTensor) and tensor.symmetry.braiding_style > BraidingStyle.bosonic:
        with pytest.raises(SymmetryError, match='not defined'):
            _ = tensor.T
        return

    how_to_call = np_random.choice(['transpose()', '.T'])
    print(how_to_call)
    if how_to_call == 'transpose()':
        res = tensors.transpose(tensor)
    if how_to_call == '.T':
        res = tensor.T
    res.test_sanity()

    assert res.codomain == tensor.domain.dual
    assert res.domain == tensor.codomain.dual
    assert res.labels == [*labels[cod:], *labels[:cod]]

    double_transpose = tensors.transpose(res)
    double_transpose.test_sanity()
    assert_tensors_almost_equal(double_transpose, tensor)

    # make sure the res agrees with both equivalent definitions of the transpose
    # note that for SymmetricTensors, the left_transpose is a copy of the implementation,
    #      but for DiagonalTensor and Mask checking vs left_transpose is not trivial.
    left_transpose = tensors.permute_legs(
        tensor, [*range(cod, cod + dom)], [*reversed(range(cod))], bend_right=[False] * cod + [True] * dom
    )
    right_transpose = tensors.permute_legs(
        tensor, [*range(cod, cod + dom)], [*reversed(range(cod))], bend_right=[True] * cod + [False] * dom
    )
    assert_tensors_almost_equal(res, left_transpose)
    assert_tensors_almost_equal(res, right_transpose)

    if tensor.symmetry.can_be_dropped:
        res_np = res.to_numpy(understood_braiding=True)
        tensor_np = tensor.to_numpy(understood_braiding=True)
        expect = np.transpose(tensor_np, [*range(cod, cod + dom), *range(cod)])
        if tensor.symmetry.has_trivial_braid or cls is Mask:
            npt.assert_almost_equal(res_np, expect)
        else:
            # would need to account for twists, which give +1/-1 factors
            # lazy version: just ignore the signs...
            assert np.all(np.isclose(res_np, expect) | np.isclose(res_np, -expect))
