"""A collection of tests for tenpy.linalg.tensors."""
# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations
import numpy as np
import numpy.testing as npt
from typing import Type
import pytest
import operator
from contextlib import nullcontext

from tenpy.linalg import backends, tensors
from tenpy.linalg.tensors import DiagonalTensor, SymmetricTensor, Mask, ChargedTensor
from tenpy.linalg.backends.backend_factory import get_backend
from tenpy.linalg.dtypes import Dtype
from tenpy.linalg.spaces import ElementarySpace, ProductSpace
from tenpy.linalg.symmetries import z4_symmetry, SU2Symmetry, SymmetryError
from tenpy.tools.misc import duplicate_entries, iter_common_noncommon_sorted_arrays


@pytest.fixture(params=[DiagonalTensor, SymmetricTensor, Mask, ChargedTensor])
def make_compatible_tensor_any_class(request, make_compatible_tensor, compatible_symmetry_backend):
    def make(num=None):
        cls = request.param

        if cls is Mask and compatible_symmetry_backend == 'fusion_tree':
            with pytest.raises(NotImplementedError, match='diagonal_to_mask not implemented'):
                _ = make_compatible_tensor(cls=cls)
            pytest.skip()
        
        if cls in [DiagonalTensor, Mask]:
            first = make_compatible_tensor(cls=cls)
            if num is None:
                return first
            more = [make_compatible_tensor(codomain=first.codomain, domain=first.domain)
                    for _ in range(num - 1)]
            return first, *more
        
        first = make_compatible_tensor(codomain=2, domain=2, max_block_size=3, max_blocks=3,
                                       cls=request.param)
        if num is None:
            return first
        if cls is ChargedTensor:
            more = []
            for _ in range(num - 1):
                inv_part = make_compatible_tensor(
                    codomain=first.codomain, domain=first.invariant_part.domain,
                    max_block_size=3, max_blocks=3, cls=SymmetricTensor,
                    labels=first.invariant_part._labels
                )
                more.append(ChargedTensor(inv_part, first.charged_state))
        else:
            more = [make_compatible_tensor(codomain=first.codomain, domain=first.domain,
                                           max_block_size=3, max_blocks=3, cls=cls)
                    for _ in range(num - 1)]
        return first, *more
    return make


# TENSOR CLASSES

class DummyTensor(tensors.Tensor):
    """Want to test the Tensor class directly.

    This overrides the abstractmethods, so we can actually make instances.
    """

    def copy(self, deep=True) -> tensors.Tensor:
        raise NotImplementedError

    def to_dense_block(self, leg_order: list[int | str] = None, dtype: Dtype = None):
        raise NotImplementedError

    def as_SymmetricTensor(self) -> SymmetricTensor:
        raise NotImplementedError

    def _get_item(self, idx: list[int]) -> bool | float | complex:
        raise NotImplementedError


def test_base_Tensor(make_compatible_space, compatible_backend):

    a, b, c, d, e = [make_compatible_space() for _ in range(5)]

    print('checking different labels input formats')
    tens1 = DummyTensor([a, b, c], [d, e], backend=compatible_backend, labels=None, dtype=Dtype.float64)
    tens1.test_sanity()
    assert tens1._labels == [None] * 5
    
    tens2 = DummyTensor([a, b, c], [d, e], backend=compatible_backend,
                        labels=['a', 'b', 'c', 'e*', 'd*'], dtype=Dtype.float64)
    tens2.test_sanity()
    assert tens2._labels == ['a', 'b', 'c', 'e*', 'd*']
    
    tens3 = DummyTensor([a, b, c], [d, e], backend=compatible_backend,
                        labels=[['a', None, None], ['d*', 'e*']], dtype=Dtype.float64)
    tens3.test_sanity()
    assert tens3._labels == ['a', None, None, 'e*', 'd*']
    
    tens4 = DummyTensor([a, b, c], [d, e], backend=compatible_backend,
                        labels=[['a', None, None], None], dtype=Dtype.float64)
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
        t.parent_space.test_sanity()

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


@pytest.mark.parametrize('leg_nums', [(1, 1), (2, 1), (3, 0), (0, 3)],
                         ids=['1->1', '1->2', '0->3', '3->0'])
def test_SymmetricTensor(make_compatible_tensor, leg_nums):
    T: SymmetricTensor = make_compatible_tensor(*leg_nums)
    backend = T.backend
    
    T.test_sanity()
    assert T.num_codomain_legs == leg_nums[0]
    assert T.num_domain_legs == leg_nums[1]

    if not T.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO

    print('checking to_numpy')
    numpy_block = T.to_numpy()
    dense_block = backend.block_backend.block_from_numpy(numpy_block)

    print('checking from_dense_block')
    tens = SymmetricTensor.from_dense_block(
        dense_block, codomain=T.codomain, domain=T.domain, backend=backend
    )
    tens.test_sanity()
    npt.assert_allclose(tens.to_numpy(), numpy_block)
    if T.num_parameters < T.size:  # otherwise all blocks are symmetric
        pert = tens.backend.block_backend.block_random_uniform(T.shape, dtype=T.dtype)
        non_symmetric_block = dense_block + pert
        with pytest.raises(ValueError, match='Block is not symmetric'):
            _ = SymmetricTensor.from_dense_block(
                non_symmetric_block, codomain=T.codomain, domain=T.domain, backend=backend
            )

    # TODO: missing coverage:
    # - from_block_func / from_sector_block_func
    # - random_uniform / random_normal
    # - diagonal

    print('checking from_zero')
    zero_tens = SymmetricTensor.from_zero(codomain=T.codomain, domain=T.domain, backend=backend)
    zero_tens.test_sanity()
    npt.assert_array_almost_equal_nulp(zero_tens.to_numpy(), np.zeros(T.shape), 10)
    
    print('checking from_eye')
    which = T.codomain if T.codomain.num_spaces > 0 else T.domain
    if which.num_spaces > 2:
        # otherwise it gets a bit expensive to compute
        which = ProductSpace(which.spaces[:2], backend=backend)
    labels=list('abcdefg')[:len(which)]
    tens = SymmetricTensor.from_eye(which, backend=T.backend, labels=labels)
    expect_from_backend = backend.block_backend.block_to_numpy(
        backend.block_backend.eye_block([leg.dim for leg in which.spaces], dtype=T.dtype)
    )
    res = tens.to_numpy()
    if which.num_spaces == 1:
        expect_explicit = np.eye(which.dim)
    elif which.num_spaces == 2:
        expect_explicit = (np.eye(which.spaces[0].dim)[:, None, None, :]
                           * np.eye(which.spaces[1].dim)[None, :, :, None])
    elif which.num_spaces == 3:
        expect_explicit = (np.eye(which.spaces[0].dim)[:, None, None, None, None, :]
                           * np.eye(which.spaces[1].dim)[None, :, None, None, :, None]
                           * np.eye(which.spaces[2].dim)[None, None, :, :, None, None])
    else:
        raise RuntimeError('Need to adjust test design')
    npt.assert_allclose(expect_from_backend, expect_explicit)
    npt.assert_allclose(res, expect_explicit, rtol=1e-7, atol=1e-10)

    print('checking repr and str')
    _ = str(T)
    _ = repr(T)
    _ = str(zero_tens)
    _ = repr(zero_tens)

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


def test_DiagonalTensor(make_compatible_tensor):
    T: DiagonalTensor = make_compatible_tensor(cls=DiagonalTensor)
    T.test_sanity()

    if not T.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO

    print('checking diagonal_as_numpy')
    np_diag = T.diagonal_as_numpy()

    print('checking from_diag_block')
    tens = DiagonalTensor.from_diag_block(np_diag, leg=T.leg, backend=T.backend)
    tens.test_sanity()
    res = tens.diagonal_as_numpy()
    npt.assert_array_almost_equal_nulp(res, np_diag, 100)

    print('checking to_numpy')
    np_full = T.to_numpy()
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

    if compatible_symmetry_backend == 'fusion_tree':
        # necessary functions to create Masks from fixture are not implemented yet
        with pytest.raises(NotImplementedError, match='diagonal_to_mask not implemented'):
            make_compatible_tensor(cls=Mask)
        pytest.xfail()

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
    M2 = Mask.from_random(large_leg, small_leg=M.small_leg, backend=backend)
    M2.test_sanity()
    assert M2.small_leg == M.small_leg

    print('checking from_zero')
    M_zero = Mask.from_zero(large_leg, backend=backend)
    M_zero.test_sanity()
    assert M_zero.small_leg.dim == 0

    print('checking bool()')
    with pytest.raises(TypeError, match='The truth value of a Mask is ambiguous.'):
        _ = bool(M_projection)

    print('checking .any() and .all()')
    assert M_projection.all() == np.all(M_projection_np)
    assert M_inclusion.all() == np.all(M_projection_np)
    assert M_projection.any() == np.any(M_projection_np)
    assert M_inclusion.any() == np.any(M_projection_np)
    assert M_eye.all()
    assert M_eye.any()
    assert not M_zero.all()
    assert not M_zero.any()

    print('checking to_numpy vs as_SymmetricTensor')
    res_direct = M_projection.to_numpy()
    M_SymmetricTensor = M_projection.as_SymmetricTensor(dtype=Dtype.float64)
    assert M_SymmetricTensor.shape == M_projection.shape
    M_SymmetricTensor.test_sanity()
    res_via_Symmetric = M_SymmetricTensor.to_numpy()
    npt.assert_allclose(res_via_Symmetric, res_direct)
    print('   also for inclusion Mask')
    res_direct = M_inclusion.to_numpy()
    M_SymmetricTensor = M_inclusion.as_SymmetricTensor(dtype=Dtype.float64)
    assert M_SymmetricTensor.shape == M_inclusion.shape
    M_SymmetricTensor.test_sanity()
    res_via_Symmetric = M_SymmetricTensor.to_numpy()
    npt.assert_allclose(res_via_Symmetric, res_direct)

    # TODO check binary operands: &, ==, !=, &, |, ^ :
    #   left and right
    #   with bool and with other mask
    #   with projection Masks and with inclusion Masks
    
    # TODO check orthogonal complement, also for inclusion Mask!

    print('checking repr and str')
    _ = str(M_projection)
    _ = repr(M_projection)
    _ = str(M_inclusion)
    _ = repr(M_inclusion)
    _ = str(M_zero)
    _ = repr(M_zero)


@pytest.mark.parametrize('leg_nums', [(1, 1), (2, 1), (3, 0), (0, 3)],
                         ids=['1->1', '1->2', '0->3', '3->0'])
def test_ChargedTensor(make_compatible_tensor, make_compatible_sectors, compatible_symmetry, leg_nums):
    T: ChargedTensor = make_compatible_tensor(*leg_nums, cls=ChargedTensor)
    backend = T.backend

    T.test_sanity()
    assert T.num_codomain_legs == leg_nums[0]
    assert T.num_domain_legs == leg_nums[1]

    if not T.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO

    print('checking to_numpy')
    numpy_block = T.to_numpy()
    assert T.shape == numpy_block.shape
    
    print('checking from_zero')
    zero_tens = ChargedTensor.from_zero(
        codomain=T.codomain, domain=T.domain, charge=T.charge_leg, charged_state=T.charged_state,
        backend=backend
    )
    zero_tens.test_sanity()
    npt.assert_array_almost_equal_nulp(zero_tens.to_numpy(), np.zeros(T.shape), 10)

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
    block_size = leg.sector_multiplicity(sector)

    if isinstance(backend, backends.FusionTreeBackend):
        with pytest.raises(NotImplementedError):
            _ = tens.to_dense_block_single_sector()
        pytest.xfail()
    
    block = tens.to_dense_block_single_sector()
    assert backend.block_backend.block_shape(block) == (block_size,)
    tens2 = ChargedTensor.from_dense_block_single_sector(
        vector=block, space=leg, sector=sector, backend=backend
    )
    tens2.test_sanity()
    assert tens2.charge_leg == tens.charge_leg
    assert tensors.almost_equal(tens, tens2)
    block2 = tens2.to_dense_block_single_sector()
    npt.assert_array_almost_equal_nulp(tens.backend.block_backend.block_to_numpy(block),
                                       tens.backend.block_backend.block_to_numpy(block2),
                                       100)

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
    data = np.array([[ 0,  0,  1,  0,  0,  0,  2],  # 3 
                     [ 0,  0,  3,  0,  0,  0,  4],  # 3 
                     [ 5,  0,  0,  6,  0,  0,  0],  # 2 
                     [ 0,  7,  0,  0,  0,  8,  0],  # 0 
                     [ 0,  0,  9,  0,  0,  0, 10],  # 3 
                     [11,  0,  0, 12,  0,  0,  0]], # 2
                    dtype=float)

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
        t = SymmetricTensor.from_dense_block(non_symmetric_data, codomain=[s1, s2],
                                                    backend=backend)
    # now continue with the symmetric block

    t = SymmetricTensor.from_dense_block(data, codomain=[s1, s2], backend=backend)
    t.test_sanity()
    
    # explicitly check the ``t.data`` vs what we expect
    if symmetry_backend == 'abelian':
        # listing this in an order such that the resulting block_inds are lexsorted:
        # blocks allowed for q:   [] -> [0, 0]  ;  [] -> [3, 1]  ;  [] -> [2, 2]
        # indices in .sectors:    [] -> [0, 0]  ;  [] -> [2, 1]  ;  [] -> [1, 2]
        expect_block_inds = np.array([[0, 0], [2, 1], [1, 2]])
        expect_blocks = [block_00, block_31, block_22]
        #
        valid_block_inds = backends.abelian._valid_block_inds(t.codomain, t.domain)
        npt.assert_array_equal(expect_block_inds, valid_block_inds)
        #
        assert len(expect_blocks) == len(t.data.blocks)
        for i, (actual, expect) in enumerate(zip(t.data.blocks, expect_blocks)):
            print(f'checking blocks[{i}]')
            npt.assert_array_almost_equal_nulp(t.backend.block_backend.block_to_numpy(actual), expect, 100)
    
    elif symmetry_backend == 'fusion_tree':
        assert np.all(t.data.block_inds == np.array([0, 0])[None, :])
        forest_block_00 = block_00.reshape((-1, 1))
        forest_block_22 = block_22.reshape((-1, 1))
        forest_block_31 = block_31.reshape((-1, 1))
        # forest blocks are sorted C-style, i.e. first by first row.
        expect_block = np.concatenate([forest_block_00, forest_block_22, forest_block_31], axis=0)
        assert len(t.data.blocks) == 1
        actual = t.backend.block_backend.block_to_numpy(t.data.blocks[0])
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
    data = np.array([[ 0,  0,  0,  0, -1,  0,  0],  # 3 
                     [ 0,  0,  0,  0, -2,  0,  0],  # 3 
                     [ 5,  0,  0,  6,  0,  0,  0],  # 2 
                     [ 0,  7,  0,  0,  0,  8,  0],  # 0 
                     [ 0,  0,  0,  0, -3,  0,  0],  # 3 
                     [11,  0,  0, 12,  0,  0,  0]], # 2
                    dtype=float)

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
        t = SymmetricTensor.from_dense_block(
            non_symmetric_data, codomain=[s1], domain=[s2], backend=backend
        )
    # now continue with the symmetric block

    t = SymmetricTensor.from_dense_block(
        data, codomain=[s1], domain=[s2], backend=backend
    )
    t.test_sanity()

    # explicitly check the ``t.data`` vs what we expect
    if symmetry_backend == 'abelian':
        # listing this in an order such that the resulting block_inds are lexsorted:
        # blocks allowed for q:   [0] -> [0]  ;  [2] -> [2]  ;  [3] -> [3]
        # indices in .sectors:    [0] -> [0]  ;  [2] -> [1]  ;  [3] -> [2]
        # block_inds row:         [0, 0]      ;  [1, 2]      ;  [2, 3]
        expect_block_inds = np.array([[0, 0], [1, 2], [2, 3]])
        expect_blocks = [block_00, block_22, block_33]
        #
        valid_block_inds = backends.abelian._valid_block_inds(t.codomain, t.domain)
        npt.assert_array_equal(expect_block_inds, valid_block_inds)
        #
        assert len(expect_blocks) == len(t.data.blocks)
        for i, (actual, expect) in enumerate(zip(t.data.blocks, expect_blocks)):
            print(f'checking blocks[{i}]')
            npt.assert_array_almost_equal_nulp(t.backend.block_backend.block_to_numpy(actual), expect, 100)
    
    elif symmetry_backend == 'fusion_tree':
        # expect coupled sectors q0, q2, q3.
        # codomain.sectors == [q0, q2, q3]
        # domain.sectors == [q0, q1, q2, q3]
        npt.assert_array_equal(t.data.block_inds, np.array([[0, 0], [1, 2], [2, 3]]))
        expect_blocks = [block_00, block_22, block_33]
        assert len(expect_blocks) == len(t.data.blocks)
        for i, (actual, expect) in enumerate(zip(t.data.blocks, expect_blocks)):
            print(f'checking blocks[{i}]')
            npt.assert_array_almost_equal_nulp(t.backend.block_backend.block_to_numpy(actual), expect, 100)

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
    data[1, 1, 1, 1] = 1   # [0, 0, 0, 0]     |  [0, 0] -> 0 -> [0, 0]
    data[1, 1, 2, 2] = 2   # [0, 0, 2, 2]     |  [2, 2] -> 0 -> [0, 0]
    data[2, 2, 1, 1] = 3   # [2, 2, 0, 0]     |  [0, 0] -> 0 -> [2, 2]
    data[2, 2, 2, 2] = 4   # [2, 2, 2, 2]     |  [2, 2] -> 0 -> [2, 2]
    #
    data[0, 1, 0, 1] = 5   # [1, 0, 1, 0]     |  [0, 1] -> 1 -> [1, 0]
    data[0, 1, 1, 0] = 6   # [1, 0, 0, 1]     |  [1, 0] -> 1 -> [1, 0]
    #
    data[0, 0, 0, 0] = 7   # [1, 1, 1, 1]     |  [1, 1] -> 2 -> [1, 1]
    data[0, 0, 1, 2] = 8   # [1, 1, 0, 2]     |  [2, 0] -> 2 -> [1, 1]
    data[1, 2, 0, 0] = 9   # [0, 2, 1, 1]     |  [1, 1] -> 2 -> [0, 2]
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
        t = SymmetricTensor.from_dense_block(
            non_symmetric_data, codomain=[s, s], domain=[s, s], backend=backend
        )
    # now continue with the symmetric block

    t = SymmetricTensor.from_dense_block(
        data, codomain=[s, s], domain=[s, s], backend=backend
    )
    t.test_sanity()
    
    # explicitly check the ``t.data`` vs what we expect
    if symmetry_backend == 'abelian':
        # all sectors appear only once, so each allowed entry is its own block.
        # In this case, the value of a sector is also its index in s.sectors
        # Thus the block inds are just the "SECTORS PER LEG" above.
        expect_block_inds = np.asarray([
            [0, 0, 0, 0], [0, 0, 2, 2], [2, 2, 0, 0], [2, 2, 2, 2],
            [1, 0, 1, 0], [1, 0, 0, 1],
            [1, 1, 1, 1], [1, 1, 0, 2], [0, 2, 1, 1], [0, 2, 0, 2],
            [1, 2, 1, 2], [1, 2, 2, 1], [2, 1, 1, 2], [2, 1, 2, 1]
        ], dtype=int)
        expect_blocks = [np.asarray([[x]], dtype=float) for x in range(1, 15)]
        perm = np.lexsort(expect_block_inds.T)
        expect_block_inds = expect_block_inds[perm]
        expect_blocks = [expect_blocks[n] for n in perm]
        #
        # have not set all entries, so expect_block_inds should be a subset of _valid_block_inds
        valid_block_inds = backends.abelian._valid_block_inds(t.codomain, t.domain)
        for i, j in iter_common_noncommon_sorted_arrays(expect_block_inds, valid_block_inds):
            assert j is not None  # j=None would mean that the row of expect_block_inds is not in valid_block_inds
            actual_block = t.backend.block_backend.block_to_numpy(t.data.blocks[j])
            if i is None:
                expect_block = np.zeros_like(actual_block)
            else:
                expect_block = expect_blocks[i]
            npt.assert_array_almost_equal_nulp(actual_block, expect_block, 100)

    elif symmetry_backend == 'fusion_tree':
        # check block_inds. first make sure the (co)domain.sectors are what we expect
        assert np.all(t.codomain.sectors == np.stack([q0, q1, q2, q3]))
        assert np.all(t.domain.sectors == np.stack([q0, q1, q2, q3]))
        # expect coupled sectors [q0, q1, q2, q3]
        assert np.all(t.data.block_inds == np.repeat(np.arange(4)[:, None], 2, axis=1))
        #
        # build the blocks for fixed coupled sectors
        # note: when setting the data we listed the uncoupled sectors of the domain

        #      dom uncoupled:  (0, 0)  ;  (2, 2)  |  codom uncoupled:
        block_0 = np.asarray([[    1,         2],   #  (0, 0)
                              [    3,         4]],  #  (2, 2)
                             dtype=float)
        #      dom uncoupled:  (0, 1)  ;  (1, 0)  |  codom uncoupled:
        block_1 = np.asarray([[    0,         0],   #  (0, 1)
                              [    5,         6]],  #  (1, 0)
                             dtype=float)
        #      dom uncoupled:  (0, 2)  ;  (1, 1)  ;  (2, 0)  |  codom uncoupled:
        block_2 = np.asarray([[    0,         9,        10],   #  (0, 2)
                              [    0,         7,         8],   #  (1, 1)
                              [    0,         0,         0]],  #  (2, 0)
                             dtype=float)
        #      dom uncoupled:  (1, 2)  ;  (2, 1)  |  codom uncoupled:
        block_3 = np.asarray([[   12,        11],   #  (1, 2)
                              [   14,        13]],  #  (2, 1)
                             dtype=float)
        expect_blocks = [block_0, block_1, block_2, block_3]
        assert len(expect_blocks) == len(t.data.blocks)
        for i, (actual, expect) in enumerate(zip(t.data.blocks, expect_blocks)):
            print(f'checking blocks[{i}]')
            npt.assert_array_almost_equal_nulp(t.backend.block_backend.block_to_numpy(actual), expect, 100)

    else:
        raise RuntimeError


@pytest.mark.parametrize('symmetry_backend', [pytest.param('fusion_tree', marks=pytest.mark.FusionTree)])
def test_from_block_su2_symm(symmetry_backend, block_backend):
    backend = get_backend(symmetry_backend, block_backend)
    sym = SU2Symmetry()
    spin_half = ElementarySpace(sym, [[1]])

    # basis order: [down, up]  ->  might look unusual
    sx = .5 * np.array([[0., 1.], [1., 0.]], dtype=complex)
    sy = .5 * np.array([[0., 1.j], [-1.j, 0]], dtype=complex)
    sz = .5 * np.array([[-1., 0.], [0., +1.]], dtype=complex)
    heisenberg_4 = sum(si[:, :, None, None] * si[None, None, :, :] for si in [sx, sy, sz])  # [p1, p1*, p2, p2*]
    print(heisenberg_4.transpose([0, 2, 1, 3]).reshape((4, 4)))
    heisenberg_4 = np.transpose(heisenberg_4, [0, 2, 3, 1])  # [p1, p2, p2*, p1*]

    tens_4 = SymmetricTensor.from_dense_block(
        heisenberg_4, codomain=[spin_half, spin_half], domain=[spin_half, spin_half],
        backend=backend, labels=[['p1', 'p2'], ['p1*', 'p2*']]
    )
    tens_4.test_sanity()
    assert np.all(tens_4.data.block_inds == np.array([[0, 0], [1, 1]]))  # spin 0, spin 1
    # The blocks are the eigenvalue of the Heisenberg coupling in the fixed total spin sectors
    # For singlet states (coupled=spin-0), we have eigenvalue -3/4
    # For triplet states (coupled=spin-1), we have eigenvalue +1/4
    expect_spin_0 = -3 / 4  
    expect_spin_1 = 1 / 4
    assert backend.block_backend.block_allclose(tens_4.data.blocks[0], expect_spin_0)
    assert backend.block_backend.block_allclose(tens_4.data.blocks[1], expect_spin_1)

    recovered_block = tens_4.to_dense_block()
    print()
    print('got:')
    print(recovered_block.reshape((4, 4)))
    print()
    print('expect:')
    print(heisenberg_4.reshape((4, 4)))

    assert backend.block_backend.block_allclose(recovered_block, heisenberg_4)


@pytest.mark.parametrize(
    'codomain_dims, domain_dims, labels',
    [
        ([42, 42], [42, 42], ['a', 'b', 'c', 'd']),
        ([42], [42, 42], ['a', 'b', 'c']),
        ([42, 42, 12345, 42], [1], ['a', 'b', 'c', 'lorem', 'ipsum']),
        ([], [42, 42], ['a', 'b']),
        ([42, 42, 42], [], ['a', 'b', 'c']),
    ]
)
def test_Tensor_ascii_diagram(codomain_dims, domain_dims, labels):
    """
    You may find useful (see comments in :func:`test_Tensor_str_repr`)::

        pytest -rP -k test_Tensor_ascii_diagram > playground/test_Tensor_ascii_diagram.txt && code playground/test_Tensor_ascii_diagram.txt

    or for vim::

        pytest -rP -k test_Tensor_ascii_diagram > playground/test_Tensor_ascii_diagram.txt && vim playground/test_Tensor_ascii_diagram.txt

    """
    codomain = [ElementarySpace.from_trivial_sector(dim=d) for d in codomain_dims]
    domain = [ElementarySpace.from_trivial_sector(dim=d) for d in domain_dims]
    T = DummyTensor(codomain, domain, backend=get_backend(), labels=labels, dtype=Dtype.complex128)
    print(T.ascii_diagram)


@pytest.mark.parametrize(
    'cls, codomain, domain',
    [
        pytest.param(SymmetricTensor, ['vL'], ['vR', 'p'], id='Sym-vL-p-vR'),
        pytest.param(SymmetricTensor, ['p'], ['p*'], id='Sym-p-p*'),
        pytest.param(DiagonalTensor, ['p'], ['p*'], id='Diag-p-p*'),
        pytest.param(Mask, ['vL'], ['vL*'], id='Mask-vL-vL*'),
        pytest.param(ChargedTensor, ['vL'], ['vR', 'p'], id='Charged-vL-p-vR')
    ]
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
    tens: cls = make_compatible_tensor(domain, codomain, cls=cls)

    if not tens.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO
        
    tens_np = tens.to_numpy()

    print('via positional arg')
    pos = np_random.choice(tens.num_legs + 1)
    res = tensors.add_trivial_leg(tens, pos, is_dual=is_dual)
    res_np = res.to_numpy()
    expect = np.expand_dims(tens_np, pos)
    npt.assert_array_almost_equal_nulp(res_np, expect, 100)

    print('to_domain')
    pos = np_random.choice(tens.num_domain_legs + 1)
    res = tensors.add_trivial_leg(tens, domain_pos=pos, is_dual=is_dual)
    res_np = res.to_numpy()
    expect = np.expand_dims(tens_np, -1-pos)
    npt.assert_array_almost_equal_nulp(res_np, expect, 100)

    print('to_codomain')
    pos = np_random.choice(tens.num_codomain_legs + 1)
    res = tensors.add_trivial_leg(tens, codomain_pos=pos, is_dual=is_dual)
    res_np = res.to_numpy()
    expect = np.expand_dims(tens_np, pos)
    npt.assert_array_almost_equal_nulp(res_np, expect, 100)


@pytest.mark.parametrize('cls', [DiagonalTensor, SymmetricTensor, ChargedTensor])
def test_almost_equal(cls, make_compatible_tensor):
    if cls is ChargedTensor:
        # TODO
        pytest.skip('Need to generate T_diff to have the same dummy leg!')
    
    T: cls = make_compatible_tensor(cls=cls)
    T_diff: cls = make_compatible_tensor(domain=T.domain, codomain=T.codomain, cls=cls)
    T2 = T + 1e-7 * T_diff
    assert tensors.almost_equal(T, T2, rtol=1e-5, atol=1e-5)
    assert not tensors.almost_equal(T, T2, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    'cls, codomain, domain, which_leg',
    [pytest.param(SymmetricTensor, 2, 2, 1, id='Symm-2-2-codom'),
     pytest.param(SymmetricTensor, 2, 2, -1, id='Symm-2-2-dom'),
     pytest.param(ChargedTensor, 2, 2, 1, id='Charged-2-2-dom'),
     pytest.param(ChargedTensor, 2, 2, -1, id='Charged-2-2-dom'),
     pytest.param(DiagonalTensor, 1, 1, -1, id='Diag-dom'),
     pytest.param(Mask, 1, 1, 0, id='Mask-codom'),
     pytest.param(Mask, 1, 1, -1, id='Mask-dom'),
    ]
)
def test_apply_mask(cls, codomain, domain, which_leg, make_compatible_tensor):
    num_legs = codomain + domain
    labels = list('abcdefghijkl')[:num_legs]
    T: cls = make_compatible_tensor(codomain=codomain, domain=domain, labels=labels, cls=cls)
    M: Mask = make_compatible_tensor(domain=[T.get_leg(which_leg)], cls=Mask)

    if cls is Mask:
        with pytest.raises(NotImplementedError):
            _ = tensors.apply_mask(T, M, which_leg)
        pytest.xfail()

    res = tensors.apply_mask(T, M, which_leg)
    res.test_sanity()

    in_domain, co_domain_idx, leg_idx = T._parse_leg_idx(which_leg)
    expect_legs = T.legs
    expect_legs[leg_idx] = M.small_leg
    assert res.legs == expect_legs
    assert res.labels == T.labels

    if not T.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO
        
    T_np = T.to_numpy()
    mask_np = M.as_numpy_mask()
    expect = T_np.compress(mask_np, leg_idx)
    npt.assert_almost_equal(res.to_numpy(), expect)


def test_apply_mask_DiagonalTensor(make_compatible_tensor):
    tensor: DiagonalTensor = make_compatible_tensor(cls=DiagonalTensor, labels=['a', 'b'])
    mask: Mask = make_compatible_tensor(domain=[tensor.leg], cls=Mask)

    res = tensors.apply_mask_DiagonalTensor(tensor, mask)
    res.test_sanity()
    assert isinstance(res, DiagonalTensor)
    assert res.leg == mask.small_leg
    assert res.labels == tensor.labels

    diag = tensor.diagonal_as_numpy()
    mask_np = mask.as_numpy_mask()
    expect = diag[mask_np]
    npt.assert_almost_equal(res.diagonal_as_numpy(), expect)


@pytest.mark.parametrize('cls, codomain, domain, num_codomain_legs',
                         [pytest.param(SymmetricTensor, 2, 2, 2),
                          pytest.param(SymmetricTensor, 2, 2, 1),
                          pytest.param(SymmetricTensor, 2, 2, 4),])
def test_bend_legs(cls, codomain, domain, num_codomain_legs, make_compatible_tensor):
    tensor: cls = make_compatible_tensor(codomain, domain, cls=cls)
    
    res = tensors.bend_legs(tensor, num_codomain_legs)
    res.test_sanity()
    assert res.legs == tensor.legs

    if not tensor.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO
        
    tensor_np = tensor.to_numpy()
    npt.assert_array_almost_equal_nulp(res.to_numpy(), tensor_np, 100)


# TODO fix combine. implement split.
def test_combine_split(make_compatible_tensor):
    T: SymmetricTensor = make_compatible_tensor(['a', 'b'], ['d', 'c'])
    assert T.labels == ['a', 'b', 'c', 'd']

    if isinstance(T.backend, backends.FusionTreeBackend):
        with pytest.raises(NotImplementedError):
            _ = tensors.combine_legs(T, [1, 2])
        pytest.xfail()

    if isinstance(T.backend, backends.AbelianBackend):
        with pytest.raises(NotImplementedError, match='currently bugged'):
            _ = tensors.combine_legs(T, [1, 2])
        pytest.xfail()
    
    # 1) combine in codomain
    combined1 = tensors.combine_legs(T, [0, 1])
    combined1.test_sanity()
    assert combined1.labels == ['(a.b)', 'c', 'd']
    assert len(combined1.codomain) == 1
    assert combined1.codomain[0].spaces == [T.codomain[0], T.codomain[1]]
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
    assert combined2.domain[0].spaces == [T.domain[0], T.domain[1]]
    #
    split2 = tensors.split_legs(combined2, '(c.d)')
    split2.test_sanity()
    assert split2.labels == ['a', 'b', 'c', 'd']
    assert split2.codomain == T.codomain
    assert split2.domain == T.domain
    assert tensors.almost_equal(split2, T)

    # 3) combine in codomain with braid
    combined3 = tensors.combine_legs(T, [1, 0])
    combined3.test_sanity()
    assert combined3.labels == ['(b.a)', 'c', 'd']
    assert len(combined3.codomain) == 1
    assert combined3.codomain[0].spaces == [T.codomain[1], T.codomain[0]]
    assert combined3.domain == T.domain
    #
    split3 = tensors.split_legs(combined3)
    split3.test_sanity()
    assert split3.labels == ['b', 'a', 'c', 'd']
    assert split3.codomain.spaces == [T.codomain[1], T.codomain[0]]
    assert split3.domain == T.domain
    assert tensors.almost_equal(split3, tensors.permute_legs(T, [1, 0]))

    # 4) combine, one from codomain, one from domain
    combined4 = tensors.combine_legs(T, [1, 2])
    combined4.test_sanity()
    assert combined4.labels == ['a', '(b.c)', 'd']
    assert len(combined4.codomain) == 2
    assert combined4.codomain[0] == T.codomain[0]
    assert combined4.codomain[1].spaces == [T.codomain[1], T.domain[1].dual]
    assert combined4.domain.spaces == [T.domain[0]]
    #
    split4 = tensors.split_legs(combined4, 1)
    split4.test_sanity()
    assert split4.labels == ['a', 'b', 'c', 'd']
    assert split4.codomain.spaces == [T.codomain[0], T.codomain[1], T.domain[1].dual]
    assert split4.domain.spaces == [T.domain[0]]
    assert tensors.almost_equal(split4, tensors.permute_legs(T, [0, 1, 2]))

    # check splitting a non-combined leg raises
    with pytest.raises(ValueError, match='Not a ProductSpace.'):
        _ = tensors.split_legs(combined4, 0)

    # TODO incorporate OLD_test_combine_legs_basis_trafo::
    # @pytest.mark.xfail  # TODO
    # def OLD_test_combine_legs_basis_trafo(make_compatible_tensor):
    #     tens = make_compatible_tensor(labels=['a', 'b', 'c'], max_blocks=5, max_block_size=5)
    #     a, b, c = tens.shape
    #     dense = tens.to_numpy()  # [a, b, c]
    #     combined = tensors.combine_legs(tens, ['a', 'b'])
    #     dense_combined = combined.to_dense_block()  # [(a.b), c]
    #     #
    #     print('check via perm')
    #     perm = combined.get_leg('(a.b)').get_basis_transformation_perm()
    #     assert all(0 <= p < len(perm) for p in perm)
    #     assert len(set(perm)) == len(perm)
    #     reconstruct_combined = np.reshape(dense, (a * b, c))[perm, :]
    #     #
    #     npt.assert_array_almost_equal_nulp(dense_combined, reconstruct_combined, 100)
    #     #
    #     print('check via trafo')
    #     trafo = combined.get_legs('(a.b)')[0].get_basis_transformation()  # [a, b, (a.b)]
    #     reconstruct_combined = np.tensordot(trafo, dense, [[0, 1], [0, 1]])  # [(a.b), c]
    #     npt.assert_array_almost_equal_nulp(dense_combined, reconstruct_combined, 100)


@pytest.mark.parametrize(
    'cls_A, cls_B, cod_A, shared, dom_B',
    [pytest.param(SymmetricTensor, SymmetricTensor, 2, 2, 2, id='Sym@Sym-2-2-2'),]
)
def test_compose(cls_A, cls_B, cod_A, shared, dom_B, make_compatible_tensor):
    labels_A = [list('abcd')[:cod_A], list('efgh')[:shared]]
    labels_B = [list('ijkl')[:shared], list('mnop')[:dom_B]]
    A: cls_A = make_compatible_tensor(
        codomain=cod_A, domain=shared, labels=labels_A, cls=cls_A
    )
    B: cls_B = make_compatible_tensor(
        codomain=A.domain, domain=dom_B, labels=labels_B, cls=cls_B
    )

    res = tensors.compose(A, B, relabel1={'a': 'x'}, relabel2={'m': 'y'})

    if not A.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO

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
        res_np = res.to_numpy()

    axes = [list(range(cod_A, cod_A + shared)), list(reversed(range(shared)))]
    expect = np.tensordot(A.to_numpy(), B.to_numpy(), axes)
    npt.assert_almost_equal(res_np, expect)


@pytest.mark.parametrize(
    'cls, cod, dom',
    [pytest.param(SymmetricTensor, 2, 2, id='Sym-2-2'),
     pytest.param(SymmetricTensor, 3, 0, id='Sym-3-0'),
     pytest.param(SymmetricTensor, 1, 1, id='Sym-1-1'),
     pytest.param(SymmetricTensor, 0, 3, id='Sym-3-0'),
     pytest.param(ChargedTensor, 2, 2, id='Charged-2-2'),
     pytest.param(ChargedTensor, 3, 0, id='Charged-3-0'),
     pytest.param(ChargedTensor, 1, 1, id='Charged-1-1'),
     pytest.param(ChargedTensor, 0, 3, id='Charged-3-0'),
     pytest.param(DiagonalTensor, 1, 1, id='Diag'),
     pytest.param(Mask, 1, 1, id='Mask')]
)
def test_dagger(cls, cod, dom, make_compatible_tensor, np_random):
    T_labels = list('abcdefghi')[:cod + dom]
    T: cls = make_compatible_tensor(cod, dom, cls=cls, labels=T_labels)

    if (isinstance(T.backend, backends.FusionTreeBackend) and cls is ChargedTensor and
        T.symmetry.braiding_style.value >= 20):
        pytest.skip('The concept of ChargedTensor is not sensical for anyonic symmetries')

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

    if not T.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO

    expect = np.conj(np.transpose(T.to_numpy(), list(reversed(range(cod + dom)))))
    npt.assert_almost_equal(res.to_numpy(), expect)


@pytest.mark.parametrize(
    'tenpy_func, numpy_func, dtype, kwargs',
    # dtype=None indicates that we need to special case the tensor creations to fulfill constraints.
    [pytest.param(tensors.angle, np.angle, Dtype.complex128, {}, id='angle()-complex'),
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
    ]
     # TODO more functions? exp, log
)
def test_DiagonalTensor_elementwise_unary(tenpy_func, numpy_func, dtype, kwargs, make_compatible_tensor):
    if dtype is not None:
        D: DiagonalTensor = make_compatible_tensor(cls=DiagonalTensor, dtype=dtype)
    elif tenpy_func is tensors.sqrt:
        # need positive
        D: DiagonalTensor = abs(make_compatible_tensor(cls=DiagonalTensor, dtype=Dtype.float64))
    elif tenpy_func is tensors.real_if_close:
        # want almost real
        rp = make_compatible_tensor(cls=DiagonalTensor, dtype=Dtype.float64)
        ip = make_compatible_tensor(domain=rp.domain, cls=DiagonalTensor, dtype=Dtype.float64)
        D = rp + 1-12j * ip
    else:
        raise ValueError

    res = tenpy_func(D, **kwargs)
    res.test_sanity()

    if not D.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO
    
    res_np = res.diagonal_as_numpy()
    expect = numpy_func(D.diagonal_as_numpy())
    npt.assert_almost_equal(res_np, expect)


@pytest.mark.parametrize(
    'cls, op, dtype',
    [pytest.param(DiagonalTensor, operator.add, Dtype.complex128, id='+'),
     pytest.param(DiagonalTensor, operator.ge, Dtype.bool, id='>='),
     pytest.param(DiagonalTensor, operator.gt, Dtype.bool, id='>'),
     pytest.param(DiagonalTensor, operator.le, Dtype.bool, id='<='),
     pytest.param(DiagonalTensor, operator.lt, Dtype.bool, id='<'),
     pytest.param(DiagonalTensor, operator.mul, Dtype.complex128, id='*'),
     pytest.param(DiagonalTensor, operator.pow, Dtype.complex128, id='**'),
     pytest.param(DiagonalTensor, operator.sub, Dtype.complex128, id='-'),
    ]
)
def test_DiagonalTensor_elementwise_binary(cls, op, dtype, make_compatible_tensor, np_random):
    t1: DiagonalTensor = make_compatible_tensor(cls=cls, dtype=dtype)
    t2: DiagonalTensor = make_compatible_tensor(domain=t1.domain, cls=cls, dtype=dtype)
    if dtype == Dtype.bool:
        scalar = bool(np_random.choice([True, False]))
    elif dtype.is_real:
        scalar = np_random.uniform()
    else:
        scalar = np_random.uniform() + 1.j * np_random.uniform()

    if not t1.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO

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
    [pytest.param(SymmetricTensor, 1, False, id='Sym-1-False'),
     pytest.param(SymmetricTensor, 1, True, id='Sym-1-True'),
     pytest.param(SymmetricTensor, 2, False, id='Sym-2-False'),
     pytest.param(DiagonalTensor, 1, False, id='Diag-False'),
     pytest.param(DiagonalTensor, 1, True, id='Diag-True'),]
)
def test_eigh(cls, dom, new_leg_dual, make_compatible_tensor):
    # prepare hermitian tensor
    T: cls = make_compatible_tensor(dom, dom, cls=cls)
    T: cls = make_compatible_tensor(T.domain, T.domain, cls=cls)
    T = T + T.hc
    T.set_labels(list('efghijk')[:2 * dom])
    T.test_sanity()

    if (not T.backend.can_decompose_tensors) and (dom > 1):
        # TODO combine_legs currently broken... when fixed, rm this clause and keep the one below
        with pytest.raises((NotImplementedError, ValueError, IndexError)):
            _ = tensors.eigh(T, new_labels=['a', 'b', 'c'], new_leg_dual=new_leg_dual)
        pytest.xfail()
        
        with pytest.raises(NotImplementedError, match='split_legs not implemented'):
            _ = tensors.eigh(T, new_labels=['a', 'b', 'c'], new_leg_dual=new_leg_dual)
        pytest.xfail()

    W, V = tensors.eigh(T, new_labels=['a', 'b', 'c'], new_leg_dual=new_leg_dual)
    W.test_sanity()
    V.test_sanity()
    assert W.labels == ['b', 'c']
    assert V.codomain_labels == T.codomain_labels
    assert V.domain_labels == ['a']

    assert tensors.almost_equal(V @ W @ V.hc, T, allow_different_types=True)  # is decomposition
    assert tensors.almost_equal(V @ V.hc, SymmetricTensor.from_eye(V.codomain, T.backend))  # unitary 1)
    assert tensors.almost_equal(V.hc @ V, SymmetricTensor.from_eye(V.domain, T.backend))  # unitary 2)


@pytest.mark.parametrize(
    'cls, codomain, domain, which_leg',
    [pytest.param(SymmetricTensor, 2, 2, 1, id='Symm-2-2-codom'),
     pytest.param(SymmetricTensor, 2, 2, 3, id='Symm-2-2-dom'),
     pytest.param(ChargedTensor, 2, 2, 1, id='Charged-2-2-dom'),
     pytest.param(ChargedTensor, 2, 2, 3, id='Charged-2-2-dom'),
     pytest.param(DiagonalTensor, 1, 1, 1, id='Diag-dom'),
     pytest.param(Mask, 1, 1, 0, id='Mask-codom'),
     pytest.param(Mask, 1, 1, 1, id='Mask-dom'),
    ]
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
        with pytest.raises(NotImplementedError):
            _ = make_compatible_tensor(codomain=T_codomain, domain=T_domain, labels=labels, cls=cls)
        pytest.xfail()
        
    T: cls = make_compatible_tensor(codomain=T_codomain, domain=T_domain, labels=labels, cls=cls)

    if cls is Mask:
        with pytest.raises(NotImplementedError):
            _ = tensors.enlarge_leg(T, M, which_leg)
        pytest.xfail()

    res = tensors.enlarge_leg(T, M, which_leg)
    res.test_sanity()

    _, _, leg_idx = T._parse_leg_idx(which_leg)
    expect_legs = T.legs[:]
    expect_legs[leg_idx] = M.large_leg
    assert res.legs == expect_legs
    assert res.labels == T.labels

    if not T.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO

    T_np = T.to_numpy()
    mask_np = M.as_numpy_mask()
    idcs = (slice(None, None, None),) * leg_idx + (mask_np,)
    expect_shape = list(T.shape)
    expect_shape[leg_idx] = M.shape[0]
    expect = np.zeros(expect_shape, dtype=T_np.dtype)
    expect[idcs] = T_np
    npt.assert_almost_equal(res.to_numpy(), expect)


@pytest.mark.parametrize('n', [1, np.inf, 2, 3])
def test_entropy(n, make_compatible_tensor):
    D: DiagonalTensor = make_compatible_tensor(cls=DiagonalTensor)
    D = abs(D)  # non-negative
    D = D / tensors.norm(D)  # normalized
    p = D ** 2

    if not D.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO

    # make sure we have created a valid probability distribution
    D_np = D.diagonal_as_numpy()
    p_np = p.diagonal_as_numpy()
    assert np.all(D_np >= 0)
    npt.assert_almost_equal(np.sum(p_np), 1)

    ent = tensors.entropy(p, n)
    expect = tensors.entropy(p_np, n)
    npt.assert_almost_equal(ent, expect)


@pytest.mark.parametrize(
    'cls, cod, dom',
    [pytest.param(SymmetricTensor, 2, 2, id='Sym-2-2'),
     pytest.param(SymmetricTensor, 1, 3, id='Sym-1-3'),
     pytest.param(SymmetricTensor, 3, 0, id='Sym-3-0'),
     pytest.param(DiagonalTensor, 1, 1, id='Diag'),
     pytest.param(Mask, 1, 1, id='Mask-1-1'),
     pytest.param(ChargedTensor, 2, 2, id='Charged-2-2'),
     pytest.param(ChargedTensor, 0, 3, id='Charged-0-3'),
     pytest.param(ChargedTensor, 3, 1, id='Charged-3-1'),
     ]
)
def test_getitem(cls, cod, dom, make_compatible_tensor, np_random):
    T: cls = make_compatible_tensor(cod, dom, cls=cls)

    if not T.symmetry.can_be_dropped:
        with pytest.raises(SymmetryError, match='Can not access elements'):
            _ = T[(0,) * (cod + dom)]
        return
    
    T_np = T.to_numpy()
    random_idx = tuple(np_random.choice(d) for d in T.shape)

    def assert_same(a, b):
        if T.dtype == Dtype.bool:
            assert bool(a) == bool(b)
        else:
            npt.assert_almost_equal(a, b)

    if isinstance(T.backend, backends.FusionTreeBackend):
        if cls is DiagonalTensor:
            # getting an off-diagonal element is handled without going to the backend function
            with pytest.raises(NotImplementedError, match='get_element.* not implemented'):
                _ = T[0, 0]
            pytest.xfail()
            
        with pytest.raises(NotImplementedError, match='get_element.* not implemented'):
            _ = T[random_idx]
        pytest.xfail()
    
    assert_same(T[random_idx], T_np[random_idx])

    # trying to set items raises
    with pytest.raises(TypeError, match='.* do.* not support item assignment.'):
        T[random_idx] = T.dtype.zero_scalar

    non_zero_idcs = np.where(np.abs(T_np) > 1e-2)
    which = np_random.choice(len(non_zero_idcs[0]))
    non_zero_idx = tuple(ax[which] for ax in non_zero_idcs)
    assert len(non_zero_idx) > 0
    assert_same(T[non_zero_idx], T_np[non_zero_idx])

    zero_idcs = np.where(np.abs(T_np) < 1e-8)
    if len(zero_idcs[0]) > 0:
        which = np_random.choice(len(zero_idcs[0]))
        zero_idx = tuple(ax[which] for ax in zero_idcs)
        assert_same(T[zero_idx], T_np[zero_idx])


@pytest.mark.parametrize(
    'cls, cod, dom, do_dagger',
    [pytest.param(SymmetricTensor, 2, 2, True, id='Sym-2-2-True'),
     pytest.param(SymmetricTensor, 2, 2, False, id='Sym-2-2-False'),
     pytest.param(SymmetricTensor, 3, 0, True, id='Sym-3-0-True'),
     pytest.param(SymmetricTensor, 0, 2, False, id='Sym-0-2-False'),
     pytest.param(ChargedTensor, 2, 2, True, id='Charged-2-2-True'),
     pytest.param(ChargedTensor, 2, 2, False, id='Charged-2-2-False'),
     pytest.param(ChargedTensor, 3, 0, True, id='Charged-3-0-True'),
     pytest.param(ChargedTensor, 0, 2, False, id='Charged-0-2-False'),
     pytest.param(DiagonalTensor, 1, 1, True, id='Diag-1-1-True'),
     pytest.param(DiagonalTensor, 1, 1, False, id='Diag-1-1-False'),
     pytest.param(Mask, 1, 1, True, id='Mask-1-1-True'),
     pytest.param(Mask, 1, 1, False, id='Mask-1-1-False'),]
    # TODO also test mixed types
)
def test_inner(cls, cod, dom, do_dagger, make_compatible_tensor):
    A: cls = make_compatible_tensor(cod, dom, cls=cls)
    if do_dagger:
        B: cls = make_compatible_tensor(codomain=A.codomain, domain=A.domain, cls=cls)
    else:
        B: cls = make_compatible_tensor(codomain=A.domain, domain=A.codomain, cls=cls)

    if (isinstance(A.backend, backends.FusionTreeBackend) and cls is ChargedTensor and
        A.symmetry.braiding_style.value >= 20):
        pytest.skip('The concept of ChargedTensor is not sensical for anyonic symmetries')
    if cls is Mask:
        with pytest.raises(NotImplementedError, match='tensors._compose_with_Mask not implemented for Mask'):
            _ = tensors.inner(A, B, do_dagger=do_dagger)
        pytest.xfail()

    if not A.symmetry.can_be_dropped and cls is ChargedTensor:
        assert A.charged_state is None
        # for anyonic symmetry, can not specify charge state.
        # but the, we can not compute inner.
        return

    res = tensors.inner(A, B, do_dagger=do_dagger)
    assert isinstance(res, (float, complex))

    if not A.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO

    if do_dagger:
        expect = np.sum(np.conj(A.to_numpy()) * B.to_numpy())
    else:
        expect = np.sum(np.transpose(A.to_numpy(), [*reversed(range(A.num_legs))]) * B.to_numpy())
    npt.assert_almost_equal(res, expect)


def test_is_scalar():
    # python scalars
    assert tensors.is_scalar(42.)
    assert tensors.is_scalar(1. + 3.j)
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
    assert not tensors.is_scalar(np.array([1., 2]))
    # Tensors
    leg1 = ElementarySpace(z4_symmetry, [[2]])
    leg2 = ElementarySpace.from_sectors(z4_symmetry, [[2], [3], [1]])
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
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO

    expect = T.to_numpy().item()
    npt.assert_almost_equal(res, expect)

    leg2 = ElementarySpace(compatible_symmetry, [sector], [3])
    non_scalar_T = make_compatible_tensor([leg, leg2], [leg, leg], cls=SymmetricTensor)
    with pytest.raises(ValueError, match='Not a scalar'):
        _ = tensors.item(non_scalar_T)


def test_linear_combination(make_compatible_tensor_any_class):
    v, w = make_compatible_tensor_any_class(2)
    assert v.domain == w.domain
    assert v.codomain == w.codomain

    if not w.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO
    
    v_np = v.to_numpy()
    w_np = w.to_numpy()
    for valid_scalar in [0, 1., 2. + 3.j, -42]:
        res = tensors.linear_combination(valid_scalar, v, 2 * valid_scalar, w)
        expect = valid_scalar * v_np + 2 * valid_scalar * w_np
        npt.assert_allclose(res.to_numpy(), expect)
    for invalid_scalar in [None, (1, 2), v, 'abc']:
        with pytest.raises(TypeError, match='unsupported scalar types'):
            _ = tensors.linear_combination(invalid_scalar, v, invalid_scalar, w)


@pytest.mark.parametrize(
    'cls, cod, dom, leg, codomain_pos, domain_pos, levels',
    [pytest.param(SymmetricTensor, 2, 2, 0, 1, None, None, id='Sym-a'),
     pytest.param(SymmetricTensor, 2, 2, 0, 0, None, None, id='Sym-b'),
     pytest.param(SymmetricTensor, 2, 2, 0, None, 0, None, id='Sym-c'),
     pytest.param(SymmetricTensor, 2, 2, 3, 1, None, None, id='Sym-d'),
     pytest.param(SymmetricTensor, 2, 2, 3, 0, None, None, id='Sym-e'),
     pytest.param(SymmetricTensor, 2, 2, 3, None, 0, None, id='Sym-f'),
     pytest.param(DiagonalTensor, 1, 1, 0, None, 1, None, id='Diag-a'),
     pytest.param(DiagonalTensor, 1, 1, 0, 0, None, None, id='Diag-b'),
     pytest.param(ChargedTensor, 2, 2, 0, 1, None, None, id='Charged-a'),
     pytest.param(ChargedTensor, 2, 2, 0, None, 1, None, id='Charged-b'),
     pytest.param(ChargedTensor, 2, 2, 3, 0, None, None, id='Charged-c'),]
)
def test_move_leg(cls, cod, dom, leg, codomain_pos, domain_pos, levels, make_compatible_tensor, np_random):
    """Test `move_leg` for tensors without specifying any levels. In the case where the symmetry
    is anyonic, that is, `move_leg` needs levels as input, random levels are chosen and the
    resulting numerical values are not checked against another tensor.
    Such checks are however done within the tests provided in `test_backend_nonabelian`.
    """
    assert sum(x is None for x in [domain_pos, codomain_pos]) == 1
    
    T_labels = list('abcdefghi')[:cod + dom]
    T: cls = make_compatible_tensor(cod, dom, labels=T_labels, cls=cls)

    if isinstance(T.backend, backends.FusionTreeBackend) and T.symmetry.braiding_style.value >= 20:
        if cls is ChargedTensor:
            pytest.skip('The concept of ChargedTensor is not sensical for anyonic symmetries')
        levels = list(np_random.permutation(T.num_legs))
    
    codomain_perm = [n for n in range(cod) if n != leg]
    domain_perm = [n for n in reversed(range(cod, cod + dom)) if n != leg]
    if codomain_pos is not None:
        codomain_perm[codomain_pos:codomain_pos] = [leg]
    if domain_pos is not None:
        domain_perm[domain_pos:domain_pos] = [leg]
    perm = [*codomain_perm, *reversed(domain_perm)]
    
    res = tensors.move_leg(T, leg, codomain_pos=codomain_pos, domain_pos=domain_pos, levels=levels)
    res.test_sanity()
    
    assert res.labels == [T_labels[n] for n in perm]
    assert res.legs == [T.get_leg(n) for n in perm]
    assert res.num_codomain_legs == cod + int(codomain_pos is not None) - int(leg < cod)

    if not T.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO

    expect = T.to_numpy().transpose(perm)
    npt.assert_allclose(res.to_numpy(), expect, atol=1.e-14)


@pytest.mark.parametrize(
    'cls, cod, dom',
    [pytest.param(SymmetricTensor, 2, 2, id='Sym-2-2'),
     pytest.param(SymmetricTensor, 3, 1, id='Sym-3-1'),
     pytest.param(SymmetricTensor, 3, 0, id='Sym-2-0'),
     pytest.param(SymmetricTensor, 0, 2, id='Sym-0-2'),
     pytest.param(ChargedTensor, 0, 2, id='Charged-0-2'),
     pytest.param(ChargedTensor, 2, 2, id='Charged-2-2'),
     pytest.param(DiagonalTensor, 1, 1, id='Diag'),
     pytest.param(Mask, 1, 1, id='Mask'),]
)
def test_norm(cls, cod, dom, make_compatible_tensor):
    T: cls = make_compatible_tensor(cod, dom, cls=cls)

    if not T.symmetry.can_be_dropped and cls is ChargedTensor:
        assert T.charged_state is None  # can not have a charged state for anyons -> can not compute norm
        return
    
    res = tensors.norm(T)

    if not T.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO
    
    expect = np.linalg.norm(T.to_numpy())
    npt.assert_almost_equal(res, expect)


@pytest.mark.parametrize(
    'cls_A, cls_B, cA, dA, cB, dB',
    [pytest.param(SymmetricTensor, SymmetricTensor, 1, 2, 2, 1, id='Sym@Sym-1-2-2-1'),
     pytest.param(SymmetricTensor, SymmetricTensor, 2, 1, 1, 2, id='Sym@Sym-2-1-1-2'),
     pytest.param(SymmetricTensor, SymmetricTensor, 0, 3, 2, 0, id='Sym@Sym-0-3-2-0'),
     pytest.param(ChargedTensor, ChargedTensor, 1, 2, 2, 1, id='Charged@Charged-1-2-2-1'),
     pytest.param(ChargedTensor, ChargedTensor, 0, 3, 2, 0, id='Charged@Charged-0-3-2-0'),
     pytest.param(ChargedTensor, SymmetricTensor, 1, 2, 2, 1, id='Charged@Sym-1-2-2-1'),
     pytest.param(SymmetricTensor, ChargedTensor, 0, 3, 2, 0, id='Sym@Charged-0-3-2-0'),
     ]
)
def test_outer(cls_A, cls_B, cA, dA, cB, dB, make_compatible_tensor):
    A_labels = list('abcdefg')[:cA + dA]
    B_labels = list('hijklmn')[:cB + dB]
    A: cls_A = make_compatible_tensor(cA, dA, cls=cls_A, labels=A_labels)
    B: cls_B = make_compatible_tensor(cB, dB, cls=cls_B, labels=B_labels)

    if isinstance(A.backend, backends.FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='outer not implemented'):
            _ = tensors.outer(A, B, relabel1={'a': 'x'}, relabel2={'h': 'y'})
        pytest.xfail()
    if cls_A is ChargedTensor and cls_B is ChargedTensor:
        if isinstance(A.backend, backends.AbelianBackend):
            match = 'currently bugged'
        else:
            match = 'state_tensor_product not implemented'
        with pytest.raises(NotImplementedError, match=match):
            _ = tensors.outer(A, B, relabel1={'a': 'x'}, relabel2={'h': 'y'})
        pytest.xfail()

    res = tensors.outer(A, B, relabel1={'a': 'x'}, relabel2={'h': 'y'})

    res.test_sanity()
    A_relabelled = ['x', *A_labels[1:]]
    B_relabelled = ['y', *B_labels[1:]]
    assert res.labels == [*A_relabelled[:cA], *B_relabelled, *A_relabelled[cA:]]

    perm = [*range(cA), *range(cA + dA, cA + cB + dB + dA), *range(cA, cA + dA)]
    expect = np.transpose(np.tensordot(A.to_numpy(), B.to_numpy(), [(), ()]), perm)
    npt.assert_almost_equal(res.to_numpy(), expect)


@pytest.mark.parametrize(
    'cls, codom, dom',
    [pytest.param(SymmetricTensor, ['a', 'b', 'a'], ['c', 'd'], id='Sym-aba-cd'),
     pytest.param(SymmetricTensor, ['a', 'b'], ['b', 'a'], id='Sym-ab-ba'),
     pytest.param(SymmetricTensor, ['a', 'c'], ['b', 'a'], id='Sym-ac-ba'),
     pytest.param(SymmetricTensor, ['a', 'b'], ['c', 'd'], id='Sym-ab-cd'),
     pytest.param(ChargedTensor, ['a', 'b'], ['b', 'a'], id='Charged-ab-ba'),
     pytest.param(ChargedTensor, ['a', 'b', 'a'], ['c', 'd'], id='Charged-aba-cd'),
     pytest.param(DiagonalTensor, ['a'], ['a'], id='Diag-a-a'),]
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
    T: cls = make_compatible_tensor(codomain_spaces, domain_spaces, cls=cls,
                                    labels=[*codomain_labels, *reversed(domain_labels)])
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
    if isinstance(T.backend, backends.FusionTreeBackend) and len(trace_legs) > 0 and cls is not DiagonalTensor:
        with pytest.raises(NotImplementedError, match='partial_trace not implemented'):
            _ = tensors.partial_trace(T, *pairs)
        pytest.xfail()
    #
    res = tensors.partial_trace(T, *pairs)
    #
    # 3) Test the result
    if not T.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO
    #
    num_open = T.num_legs - 2 * len(pairs)
    if num_open == 0:
        assert isinstance(res, (float, complex))
        res_np = res
    else:
        assert isinstance(res, cls)
        res.test_sanity()
        assert res.labels == [l for l in T.labels if l[0] not in trace_legs]
        assert res.codomain.spaces == [sp for sp, l in zip(T.codomain, T.codomain_labels)
                                       if l[0] not in trace_legs]
        assert res.domain.spaces == [sp for sp, l in zip(T.domain, T.domain_labels)
                                     if l[0] not in trace_legs]
        res_np = res.to_numpy()
    #
    idcs1 = [p[0] for p in pairs_positions]
    idcs2 = [p[1] for p in pairs_positions]
    remaining = [n for n in range(T.num_legs) if n not in idcs1 and n not in idcs2]
    expect = T.backend.block_backend.block_trace_partial(T.to_dense_block(), idcs1, idcs2, remaining)
    expect = T.backend.block_backend.block_to_numpy(expect)
    npt.assert_almost_equal(res_np, expect)


@pytest.mark.parametrize(
    'cls, num_cod, num_dom, codomain, domain, levels',
    [
        pytest.param(SymmetricTensor, 2, 2, [0, 1], [3, 2], None, id='Symmetric-2<2-trivial'),
        pytest.param(SymmetricTensor, 2, 2, [1, 0], [2, 3], [0, 1, 2, 3], id='Symmetric-2<2-braid'),
        pytest.param(SymmetricTensor, 2, 2, [0, 1, 2], [3], None, id='Symmetric-2<2-bend'),
        pytest.param(SymmetricTensor, 2, 2, [0, 3], [1, 2], [0, 1, 2, 3], id='Symmetric-2<2-general'),
        pytest.param(DiagonalTensor, 1, 1, [0], [1], [0, 1], id='Diagonal-trivial'),
        pytest.param(DiagonalTensor, 1, 1, [1], [0], [0, 1], id='Diagonal-swap'),
        pytest.param(DiagonalTensor, 1, 1, [1, 0], [], [0, 1], id='Diagonal-general'),
        pytest.param(Mask, 1, 1, [0], [1], [0, 1], id='Mask-trivial'),
        pytest.param(Mask, 1, 1, [1], [0], [0, 1], id='Mask-swap'),
        pytest.param(Mask, 1, 1, [1, 0], [], [0, 1], id='Mask-general'),
        pytest.param(ChargedTensor, 2, 2, [0, 1], [3, 2], None, id='Symmetric-2<2-trivial'),
        pytest.param(ChargedTensor, 2, 2, [1, 0], [2, 3], [0, 1, 2, 3], id='Symmetric-2<2-braid'),
        pytest.param(ChargedTensor, 2, 2, [0, 1, 2], [3], None, id='Symmetric-2<2-bend'),
        pytest.param(ChargedTensor, 2, 2, [0, 3], [1, 2], [0, 1, 2, 3], id='Symmetric-2<2-general'),
    ]
)
def test_permute_legs(cls, num_cod, num_dom, codomain, domain, levels, make_compatible_tensor):
    T = make_compatible_tensor(num_cod, num_dom, max_block_size=3, cls=cls)
    
    if isinstance(T.backend, backends.FusionTreeBackend) and cls is Mask and codomain == [1]:
        with pytest.raises(NotImplementedError, match='mask_transpose not implemented'):
            _ = tensors.permute_legs(T, codomain, domain, levels)
        pytest.xfail()
    elif (isinstance(T.backend, backends.FusionTreeBackend) and cls is ChargedTensor and
          T.symmetry.braiding_style.value >= 20):
        pytest.skip('The concept of ChargedTensor is not sensical for anyonic symmetries')

    res = tensors.permute_legs(T, codomain, domain, levels)
    res.test_sanity()

    for n, i in enumerate(codomain):
        assert res.codomain[n] == T._as_codomain_leg(i)
    for n, i in enumerate(domain):
        assert res.domain[n] == T._as_domain_leg(i)
    assert res.codomain_labels == [T.labels[n] for n in codomain]
    assert res.domain_labels == [T.labels[n] for n in domain]

    if T.symmetry.can_be_dropped:
        # makes sense to compare with dense blocks
        expect = np.transpose(T.to_numpy(), [*codomain, *reversed(domain)])
        actual = res.to_numpy()
        npt.assert_allclose(actual, expect, atol=1.e-14)
    else:
        # should we do a test like braiding two legs around each other with a single 
        # anyonic sector and checking if the result is equal up to the expected phase?
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO


@pytest.mark.parametrize(
    'cls, dom, cod, new_leg_dual',
    [pytest.param(SymmetricTensor, 1, 1, False, id='Sym-1-1-False'),
     pytest.param(SymmetricTensor, 1, 1, True, id='Sym-1-1-True'),
     pytest.param(SymmetricTensor, 3, 1, False, id='Sym-3-1-False'),
     pytest.param(SymmetricTensor, 2, 2, False, id='Sym-2-2-False'),
     pytest.param(SymmetricTensor, 2, 2, True, id='Sym-2-2-True'),
     pytest.param(DiagonalTensor, 1, 1, False, id='Diag-False'),
     pytest.param(DiagonalTensor, 1, 1, True, id='Diag-True'),
     pytest.param(Mask, 1, 1, False, id='Mask-False'),
     pytest.param(Mask, 1, 1, True, id='Mask-True'),]
)
def test_qr_lq(cls, dom, cod, new_leg_dual, make_compatible_tensor):
    T_labels = list('efghijk')[:dom + cod]
    T: cls = make_compatible_tensor(dom, cod, cls=cls, labels=T_labels)

    if (dom > 1 or cod > 1) and not T.backend.can_decompose_tensors:
        # TODO combine_legs currently broken... when fixed, rm this clause and keep the one below
        with pytest.raises((NotImplementedError, ValueError, IndexError)):
            _ = tensors.qr(T, new_leg_dual=new_leg_dual)
        pytest.xfail()
        
        with pytest.raises(NotImplementedError, match='split_legs not implemented'):
            _ = tensors.qr(T, new_leg_dual=new_leg_dual)
        pytest.xfail()

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
    

def test_scalar_multiply(make_compatible_tensor_any_class):
    T = make_compatible_tensor_any_class()

    if not T.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO
    
    T_np = T.to_numpy()
    for valid_scalar in [0, 1., 2. + 3.j, -42]:
        res = tensors.scalar_multiply(valid_scalar, T)
        npt.assert_allclose(res.to_numpy(), valid_scalar * T_np)
    for invalid_scalar in [None, (1, 2), T, 'abc']:
        with pytest.raises(TypeError, match='unsupported scalar type'):
            _ = tensors.scalar_multiply(invalid_scalar, T)


@pytest.mark.parametrize(
    'cls, codom, dom, which_leg',
    [pytest.param(SymmetricTensor, 2, 2, 1, id='Sym-2-2-1'),
     pytest.param(SymmetricTensor, 2, 2, 3, id='Sym-2-2-3'),
     pytest.param(SymmetricTensor, 0, 2, 1, id='Sym-0-2-1'),
     pytest.param(SymmetricTensor, 3, 0, 1, id='Sym-3-0-1'),
     pytest.param(ChargedTensor, 2, 2, 1, id='Charged-2-2-1'),
     pytest.param(ChargedTensor, 2, 2, 3, id='Charged-2-2-3'),
     pytest.param(DiagonalTensor, 1, 1, 0, id='Diag-0'),
     pytest.param(DiagonalTensor, 1, 1, 1, id='Diag-1'),
     pytest.param(Mask, 1, 1, 0, id='Mask-0'),
     pytest.param(Mask, 1, 1, 1, id='Mask-1'),]
)
def test_scale_axis(cls, codom, dom, which_leg, make_compatible_tensor, np_random):
    # 1) Prepare
    T_labels = list('abcdefghi')[:codom + dom]
    T: cls = make_compatible_tensor(codom, dom, cls=cls, labels=T_labels)
    leg = T.get_leg_co_domain(which_leg=which_leg)
    need_transpose = np_random.choice([True, False])
    if need_transpose:
        leg = leg.dual
    D: DiagonalTensor = make_compatible_tensor([leg], cls=DiagonalTensor, labels=['x', 'y'])

    # 2) Call functions    
    how_to_call = np_random.choice(['by_idx', 'by_label'])
    if how_to_call == 'by_idx':
        res = tensors.scale_axis(T, D, which_leg)
    if how_to_call == 'by_label':
        res = tensors.scale_axis(T, D, T_labels[which_leg])

    # 3) check tensor properties
    res.test_sanity()
    assert isinstance(res, SymmetricTensor if cls is Mask else cls)
    assert res.codomain == T.codomain
    assert res.domain == T.domain
    assert res.labels == T_labels

    if not T.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO

    # 4) compare to numpy
    expect = np.swapaxes(T.to_numpy(), which_leg, -1)  # swap axis to be scaled to the back
    expect = expect * D.diagonal_as_numpy()  # broadcasts to last axis of expect
    expect = np.swapaxes(expect, which_leg, -1)  # swap back
    npt.assert_allclose(res.to_numpy(), expect, atol=1.e-14)


def test_squeeze_legs(make_compatible_tensor, compatible_symmetry):
    trivial_leg = ElementarySpace.from_trivial_sector(1, symmetry=compatible_symmetry)
    T = make_compatible_tensor([None, trivial_leg, trivial_leg, None], [None, None, trivial_leg],
                               labels=list('abcdefg'))

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
        T_np = T.to_numpy()
        expect_all = T_np[:, 0, 0, :, 0, :, :]
        expect_1 = T_np [:, 0]
        expect_2 = T_np[:, 0, :, :, 0]
        npt.assert_allclose(res_all.to_numpy(), expect_all)
        npt.assert_allclose(res_1.to_numpy(), expect_1)
        npt.assert_allclose(res_2.to_numpy(), expect_2)
    else:
        pytest.skip('Need to design different checks')  # TODO


@pytest.mark.parametrize(
    'cls, dom, cod, new_leg_dual', 
    [pytest.param(SymmetricTensor, 1, 1, False, id='Sym-1-1-False'),
     pytest.param(SymmetricTensor, 1, 3, False, id='Sym-1-3-False'),
     pytest.param(SymmetricTensor, 3, 1, False, id='Sym-3-1-False'),
     pytest.param(SymmetricTensor, 2, 2, False, id='Sym-2-2-False'),
     pytest.param(SymmetricTensor, 2, 2, True, id='Sym-2-2-True'),
     pytest.param(DiagonalTensor, 1, 1, False, id='Diag-False'),
     pytest.param(DiagonalTensor, 1, 1, True, id='Diag-True'),
     pytest.param(Mask, 1, 1, False, id='Mask-False'),
     pytest.param(Mask, 1, 1, True, id='Mask-True'),
     ]
)
def test_svd(cls, dom, cod, new_leg_dual, make_compatible_tensor):
    """Test svd and various related functions. Covers

    - svd
    - truncated_svd
        - truncate_singular_values
        - svd_apply_mask
    """
    T_labels = list('efghijklmn')[:dom + cod]
    T: cls = make_compatible_tensor(dom, cod, labels=T_labels, cls=cls)

    if (dom > 1 or cod > 1) and not T.backend.can_decompose_tensors:
        # TODO combine_legs currently broken... when fixed, rm this clause and keep the one below
        with pytest.raises((NotImplementedError, ValueError, IndexError)):
            _ = tensors.svd(T, new_labels=['a', 'b', 'c', 'd'])
        pytest.xfail()
        
        with pytest.raises(NotImplementedError, match='split_legs not implemented'):
            _ = tensors.svd(T, new_labels=['a', 'b', 'c', 'd'])
        pytest.xfail()

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

    if isinstance(T.backend, backends.FusionTreeBackend):
        if not T.symmetry.can_be_dropped:
            # truncation uses diagonal_as_block, which can not be done
            with pytest.raises(SymmetryError):
                _ = tensors.truncated_svd(T, new_leg_dual=new_leg_dual)
            pytest.xfail()
        
        with pytest.raises(NotImplementedError, match='mask_from_block not implemented'):
            _ = tensors.truncated_svd(T, new_leg_dual=new_leg_dual)
        pytest.xfail()  # TODO

    print('Truncated SVD')
    for svd_min, normalize_to in [(1e-14, None), (1e-4, None), (1e-4, 2.7)]:
        options = dict(svd_min=svd_min)
        U, S, Vh, err, renormalize = tensors.truncated_svd(
            T, new_leg_dual=new_leg_dual, normalize_to=normalize_to, options=options
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


@pytest.mark.parametrize(
    'cls_A, cls_B, labels_A, labels_B, contr_A, contr_B',
    [pytest.param(SymmetricTensor, SymmetricTensor, [['a', 'b'], ['c', 'd']], [['c', 'e'], ['a', 'f']], [0, 3], [3, 0], id='Sym@Sym-4-2-4'),
     pytest.param(SymmetricTensor, SymmetricTensor, [['a', 'b'], ['c', 'd']], [['e', 'f'], ['g', 'h']], [], [], id='Sym@Sym-4-0-4'),
     pytest.param(SymmetricTensor, SymmetricTensor, [['a', 'b'], ['c', 'd']], [['c', 'a'], ['d', 'b']], [0, 1, 3, 2], [1, 2, 0, 3], id='Sym@Sym-4-4-4'),
     pytest.param(SymmetricTensor, SymmetricTensor, [[], ['a', 'b']], [['b', 'c'], ['d']], [0], [0], id='Sym@Sym-2-1-3'),
     #
     pytest.param(SymmetricTensor, ChargedTensor, [['a', 'b'], ['c', 'd']], [['c', 'e'], ['a', 'f']], [0, 3], [3, 0], id='Sym-Charged@4-2-4'),
     pytest.param(ChargedTensor, SymmetricTensor, [['a', 'b'], ['c', 'd']], [['e', 'f'], ['g', 'h']], [], [], id='Charged@Sym-4-0-4'),
     pytest.param(SymmetricTensor, ChargedTensor, [['a', 'b'], ['c', 'd']], [['c', 'a'], ['d', 'b']], [0, 1, 3, 2], [1, 2, 0, 3], id='Sym@Charged-4-4-4'),
     pytest.param(SymmetricTensor, ChargedTensor, [[], ['a', 'b']], [['b', 'c'], ['d']], [0], [0], id='Sym@Charged-2-1-3'),
     #
     # Note: need to put DiagonalTensor first to get correct legs. If SymmetricTensor is first,
     # it generates independent legs, which can not both be on a diagonalTensor.
     pytest.param(DiagonalTensor, SymmetricTensor, [['c'], ['b']], [['a', 'b'], ['c', 'd']], [1, 0], [1, 3], id='Diag@Sym-4-2-2'),
     pytest.param(SymmetricTensor, DiagonalTensor, [['a', 'b'], ['c', 'd']], [['e'], ['b']], [1], [1], id='Sym@Diag-4-1-2'),
     pytest.param(SymmetricTensor, DiagonalTensor, [['a', 'b'], ['c', 'd']], [['e'], ['f']], [], [], id='Sym-Diag-4-0-2'),
     #
     # Note: If both legs of a mask are contracted, we should generate the mask first. otherwise its legs may be invalid.
     pytest.param(Mask, SymmetricTensor, [['c'], ['b']], [['a', 'b'], ['c', 'd']], [1, 0], [1, 3], id='Sym@Mask-4-2-2'),
     pytest.param(SymmetricTensor, Mask, [['a', 'b'], ['c', 'd']], [['e'], ['b']], [1], [1], id='Sym@Mask-4-1-2'),
     pytest.param(SymmetricTensor, Mask, [['a', 'b'], ['c', 'd']], [['e'], ['f']], [], [], id='Sym@Mask-4-0-2'),
     #
     pytest.param(ChargedTensor, DiagonalTensor, [['a', 'b'], ['c', 'd']], [['e'], ['b']], [1], [1], id='Charged@Diag-4-1-2'),
     pytest.param(ChargedTensor, Mask, [['a', 'b'], ['c', 'd']], [['e'], ['b']], [1], [1], id='Charged@Mask-4-1-2'),
     #
     pytest.param(DiagonalTensor, DiagonalTensor, [['a'], ['b']], [['c'], ['b']], [1], [1], id='Diag@Diag-2-1-2'),
     pytest.param(DiagonalTensor, DiagonalTensor, [['a'], ['b']], [['b'], ['a']], [1, 0], [0, 1], id='Diag@Diag-2-2-2'),
     pytest.param(DiagonalTensor, DiagonalTensor, [['a'], ['b']], [['c'], ['d']], [], [], id='Diag@Diag-2-0-2'),
     #
     pytest.param(Mask, Mask, [['a'], ['b']], [['c'], ['b']], [1], [1], id='Mask@Mask-2-1-2'),
     pytest.param(Mask, Mask, [['a'], ['b']], [['a'], ['b']], [0, 1], [0, 1], id='Mask@Mask-2-2-2'),
     pytest.param(Mask, Mask, [['a'], ['b']], [['c'], ['d']], [], [], id='Mask@Mask-2-0-2'),
     
    ]
)
def test_tdot(cls_A: Type[tensors.Tensor], cls_B: Type[tensors.Tensor],
              labels_A: list[list[str]], labels_B: list[list[str]],
              contr_A: list[int], contr_B: list[int],
              make_compatible_tensor, np_random):
    A: cls_A = make_compatible_tensor(
        codomain=len(labels_A[0]), domain=len(labels_A[1]),
        labels=[*labels_A[0], *reversed(labels_A[1])], max_block_size=3, max_blocks=3, cls=cls_A
    )
    # create B such that legs with the same label can be contracted
    B: cls_B = make_compatible_tensor(
        codomain=[A._as_domain_leg(l) if A.has_label(l) else None for l in labels_B[0]],
        domain=[A._as_codomain_leg(l) if A.has_label(l) else None for l in labels_B[1]],
        labels=[*labels_B[0], *reversed(labels_B[1])], max_block_size=2, max_blocks=3, cls=cls_B
    )

    if (isinstance(A.backend, backends.FusionTreeBackend) and (cls_A is ChargedTensor or cls_B is ChargedTensor) and
        A.symmetry.braiding_style.value >= 20):
        pytest.skip('The concept of ChargedTensor is not sensical for anyonic symmetries')

    num_contr = len(contr_A)
    num_open_A = A.num_legs - num_contr
    num_open_B = B.num_legs - num_contr
    num_open = num_open_A + num_open_B
    # make sure we defined compatible legs
    for ia, ib in zip(contr_A, contr_B):
        assert A._as_domain_leg(ia) == B._as_codomain_leg(ib), f'{ia} / {A.labels[ia]} incompatible with {ib} / {B.labels[ib]}'

    expect_codomain = [A._as_codomain_leg(n) for n in range(A.num_legs) if n not in contr_A]
    expect_domain = [B._as_domain_leg(n) for n in range(B.num_legs) if not n in contr_B][::-1]
    expect_legs = [A.get_leg(n) for n in range(A.num_legs) if n not in contr_A] + [B.get_leg(n) for n in range(B.num_legs) if not n in contr_B]
    expect_labels = [A._labels[n] for n in range(A.num_legs) if n not in contr_A] + [B._labels[n] for n in range(B.num_legs) if not n in contr_B]

    # Context manager to catch expected errors
    catch_errors = nullcontext()
    
    if (cls_A is Mask and cls_B is Mask) and num_contr > 0:
        catch_errors = pytest.raises(NotImplementedError)
    if cls_A is ChargedTensor and A.charged_state == None and A.num_legs + B.num_legs == 2 * num_contr:
            catch_errors = pytest.raises(ValueError, match='Can not instantiate ChargedTensor with no legs and unspecified charged_states.')
    if cls_B is ChargedTensor and B.charged_state == None and A.num_legs + B.num_legs == 2 * num_contr:
            catch_errors = pytest.raises(ValueError, match='Can not instantiate ChargedTensor with no legs and unspecified charged_states.')

    if isinstance(A.backend, backends.FusionTreeBackend) and cls_A is DiagonalTensor and not cls_B is DiagonalTensor and num_contr == 2:
        catch_errors = pytest.raises(NotImplementedError)
    elif isinstance(A.backend, backends.FusionTreeBackend) and A.symmetry.braiding_style.value >= 20:
        if not cls_A is DiagonalTensor:
            levels_A = list(np_random.permutation(A.num_legs))
            codomain_A = [i for i in range(A.num_legs) if not i in contr_A]
            A = tensors.permute_legs(A, codomain=codomain_A, domain=contr_A, levels=levels_A)
            contr_A = [A.num_legs - 1 - i for i in range(num_contr)]
        if not cls_B is DiagonalTensor:
            levels_B = list(np_random.permutation(B.num_legs))
            domain_B = [i for i in range(B.num_legs) if not i in contr_B][::-1]
            B = tensors.permute_legs(B, codomain=contr_B, domain=domain_B, levels=levels_B)
            contr_B = list(range(num_contr))
    
    with catch_errors:
        res = tensors.tdot(A, B, contr_A, contr_B)
    if not isinstance(catch_errors, nullcontext):
        pytest.xfail()

    if num_open == 0:
        # scalar result
        assert isinstance(res, (float, complex))
        res_np = res
    else:
        # tensor result
        res.test_sanity()
        if A.symmetry.can_be_dropped:
            res_np = res.to_numpy()
        assert res.codomain.spaces == expect_codomain
        assert res.domain.spaces == expect_domain
        assert res.legs == expect_legs
        assert res.labels == expect_labels

    if not A.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO
    
    # compare with dense tensordot
    A_np = A.to_numpy()
    B_np = B.to_numpy()
    expect = np.tensordot(A_np, B_np, [contr_A, contr_B])
    npt.assert_allclose(res_np, expect, atol=1.e-14)


@pytest.mark.parametrize('cls, legs', [pytest.param(SymmetricTensor, 2, id='Sym-2'),
                                       pytest.param(SymmetricTensor, 1, id='Sym-1'),
                                       pytest.param(ChargedTensor, 2, id='Charged-2'),
                                       pytest.param(ChargedTensor, 1, id='Charged-1'),
                                       pytest.param(DiagonalTensor, 1, id='Diag'),])
def test_trace(cls, legs, make_compatible_tensor, compatible_symmetry, make_compatible_sectors,
                    make_compatible_space):
    co_domain_spaces = [make_compatible_space() for _ in range(legs)]
    if cls is ChargedTensor:
        if not compatible_symmetry.can_be_dropped:
            # can not specify charged_state => can not do full trace.
            return
        # make a ChargedTensor that has the trivial sector, otherwise the trace is always 0
        other_sector = make_compatible_sectors(1)[0]
        charge_leg = ElementarySpace.from_sectors(
            compatible_symmetry, [compatible_symmetry.trivial_sector, other_sector],
        )
        inv_part = make_compatible_tensor(co_domain_spaces, [charge_leg, *co_domain_spaces],
                                          cls=SymmetricTensor)
        charged_state = inv_part.backend.block_backend.as_block(list(range(charge_leg.dim)))
        tensor = ChargedTensor(inv_part.set_label(-1, '!'), charged_state)
    else:
        tensor: cls = make_compatible_tensor(co_domain_spaces, co_domain_spaces, cls=cls)

    if cls is ChargedTensor and isinstance(tensor.backend, backends.FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='partial_trace not implemented'):
            _ = tensors.trace(tensor)
        pytest.xfail()

    res = tensors.trace(tensor)
    assert isinstance(res, (float, complex))

    if not tensor.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO

    expect = tensor.to_numpy()
    while expect.ndim > 0:
        expect = np.trace(expect, axis1=0, axis2=-1)
    npt.assert_almost_equal(res, expect)


@pytest.mark.parametrize(
    'cls, cod, dom',
    [pytest.param(SymmetricTensor, 2, 2, id='Sym-2-2'),
     pytest.param(SymmetricTensor, 3, 0, id='Sym-3-0'),
     pytest.param(SymmetricTensor, 1, 1, id='Sym-1-1'),
     pytest.param(SymmetricTensor, 0, 3, id='Sym-3-0'),
     pytest.param(ChargedTensor, 2, 2, id='Charged-2-2'),
     pytest.param(ChargedTensor, 3, 0, id='Charged-3-0'),
     pytest.param(ChargedTensor, 1, 1, id='Charged-1-1'),
     pytest.param(ChargedTensor, 0, 3, id='Charged-3-0'),
     pytest.param(DiagonalTensor, 1, 1, id='Diag'),
     pytest.param(Mask, 1, 1, id='Mask')]
)
def test_transpose(cls, cod, dom, make_compatible_tensor, np_random):
    labels = list('abcdefghi')[:cod + dom]
    tensor: cls = make_compatible_tensor(cod, dom, cls=cls, labels=labels)

    if isinstance(tensor.backend, backends.FusionTreeBackend) and not cls is DiagonalTensor:
        with pytest.raises(NotImplementedError, match='transpose not implemented'):
            _ = tensors.transpose(tensor)
        pytest.xfail()

    how_to_call = np_random.choice(['dagger()', '.T'])
    print(how_to_call)
    if how_to_call == 'dagger()':
        res = tensors.transpose(tensor)
    if how_to_call == '.T':
        res = tensor.T
    res.test_sanity()

    assert res.codomain == tensor.domain.dual
    assert res.domain == tensor.codomain.dual
    assert res.labels == [*labels[cod:], *labels[:cod]]

    if not tensor.symmetry.can_be_dropped:
        pytest.skip('Need to re-design checks, cant use .to_numpy() etc')  # TODO

    expect = np.transpose(tensor.to_numpy(), [*range(cod, cod + dom), *range(cod)])
    npt.assert_almost_equal(res.to_numpy(), expect)
