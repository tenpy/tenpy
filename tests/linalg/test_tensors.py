"""A collection of tests for tenpy.linalg.tensors."""
# Copyright (C) TeNPy Developers, GNU GPLv3
import numpy as np
import numpy.testing as npt
import pytest
import operator

from tenpy.linalg import tensors
from tenpy.linalg.backends.no_symmetry import NoSymmetryBackend
from tenpy.linalg.backends.abelian import AbelianBackend
from tenpy.linalg.backends.fusion_tree_backend import FusionTreeBackend
from tenpy.linalg.backends.torch import TorchBlockBackend
from tenpy.linalg.backends.numpy import NumpyBlockBackend
from tenpy.linalg.backends.backend_factory import get_backend
from tenpy.linalg.dtypes import Dtype
from tenpy.linalg.spaces import Space, ElementarySpace, ProductSpace, _fuse_spaces
from tenpy.linalg.symmetries import ProductSymmetry, z4_symmetry, SU2Symmetry



def random_block(shape, backend):
    if isinstance(backend, NumpyBlockBackend):
        return np.random.random(shape)
    elif isinstance(backend, TorchBlockBackend):
        import torch
        return torch.randn(shape)


# TODO tests for ChargedTensor, also as input for tdot etc
# TODO DiagonalTensor


def check_shape(shape: tensors.Shape, dims: tuple[int, ...], labels: list[str]):
    shape.test_sanity()

    # check attributes
    assert shape.dims == list(dims)
    assert shape._labels == labels
    assert shape.labels == labels

    # check iter
    assert tuple(shape) == dims
    n = 0
    for d in shape:
        assert d == dims[n]
        n += 1

    # check __getitem__
    for n, label in enumerate(labels):
        # indexing by string label
        if label is not None:
            assert shape[label] == dims[n]
        # indexing by integer
        assert shape[n] == dims[n]
    assert shape[1:] == list(dims)[1:]
    with pytest.raises(IndexError):
        _ = shape['label_that_shape_does_not_have']

    assert shape.is_fully_labelled != (None in labels)


@pytest.mark.parametrize('num_domain_legs', [0, 1, 2, 3])
def test_Tensor_classmethods(make_compatible_tensor, num_domain_legs, tol=1e-6):
    # TODO why do we need such a loose tol?
    T: tensors.Tensor = make_compatible_tensor(num_legs=3, num_domain_legs=num_domain_legs)
    assert T.num_domain_legs == num_domain_legs
    backend = T.backend

    if (isinstance(backend, FusionTreeBackend)) and (isinstance(T.symmetry, ProductSymmetry)):
        with pytest.raises(NotImplementedError, match='should be implemented by subclass'):
            numpy_block = T.to_numpy()
        return
    
    legs = T.legs
    dims = tuple(T.shape)

    numpy_block = T.to_numpy()
    dense_block = backend.block_from_numpy(numpy_block)

    print('checking from_dense_block')
    tens = tensors.BlockDiagonalTensor.from_dense_block(dense_block, legs=legs, backend=backend, tol=tol)
    tens.test_sanity()
    data = backend.block_to_numpy(tens.to_dense_block())
    npt.assert_array_almost_equal_nulp(data, numpy_block, 100)
    #
    if T.num_parameters < T.parent_space.dim:  # otherwise all blocks are symmetric
        non_symmetric_block = dense_block + tens.backend.block_random_uniform(dims, dtype=T.dtype)
        with pytest.raises(ValueError, match='Block is not symmetric'):
            _ = tensors.BlockDiagonalTensor.from_dense_block(non_symmetric_block, legs=legs, backend=backend)

    print('checking from numpy')
    tens = tensors.BlockDiagonalTensor.from_dense_block(numpy_block, legs=legs, backend=backend, tol=tol)
    tens.test_sanity()
    data = tens.to_numpy()
    npt.assert_array_almost_equal_nulp(data, numpy_block, 100)

    # TODO from_block_func, from_numpy_func

    # TODO random_uniform, random_normal

    print('checking zero')
    tens = tensors.BlockDiagonalTensor.zero(legs, backend=backend)
    tens.test_sanity()
    npt.assert_array_almost_equal_nulp(tens.to_numpy(), np.zeros(dims), 10)

    print('checking eye (1 -> 1 legs)')
    tens = tensors.BlockDiagonalTensor.eye(legs[0], backend=backend)
    tens.test_sanity()
    npt.assert_array_equal(tens.to_numpy(), np.eye(legs[0].dim))
    
    print('checking eye (2 -> 2 legs)')
    tens = tensors.BlockDiagonalTensor.eye(legs[:2], backend=backend)
    tens.test_sanity()
    res = tens.to_numpy()
    expect = np.eye(dims[0])[:, None, None, :] * np.eye(dims[1])[None, :, :, None]
    
    npt.assert_almost_equal(res, expect)


def test_Tensor_methods(make_compatible_tensor, make_compatible_space):
    T = make_compatible_tensor(num_legs=3)  # compatible legs where we can have blocks
    backend = T.backend
    legs = T.legs
    dims = tuple(T.shape)

    data1 = T.data
    data2 = make_compatible_tensor(legs=legs).data

    print('checking __init__ with labels=None')
    tens1 = tensors.BlockDiagonalTensor(data1, legs=legs, num_domain_legs=0, backend=backend, labels=None)
    tens1.test_sanity()

    print('checking __init__, partially labelled')
    labels2 = [None, 'a', 'b']
    tens2 = tensors.BlockDiagonalTensor(data2, legs=legs, num_domain_legs=0, backend=backend, labels=labels2)
    tens2.test_sanity()

    print('checking __init__, fully labelled')
    labels3 = ['foo', 'a', 'b']
    tens3 = tensors.BlockDiagonalTensor(data1, legs=legs, num_domain_legs=0, backend=backend, labels=labels3)
    tens3.test_sanity()

    check_shape(tens1.shape, dims=dims, labels=[None, None, None])
    check_shape(tens2.shape, dims=dims, labels=labels2)
    check_shape(tens3.shape, dims=dims, labels=labels3)

    print('check size')
    assert tens3.size == np.prod(dims)

    print('check num_parameters')
    if isinstance(backend, NoSymmetryBackend):
        expect = np.prod(backend.block_shape(data1))
    elif isinstance(backend, AbelianBackend):
        tensor_with_all_blocks = tensors.BlockDiagonalTensor.from_block_func(
            func=backend.zero_block, legs=legs, backend=backend,
            func_kwargs=dict(dtype=Dtype.float64)
        )
        expect = sum(np.prod(backend.block_shape(block)) for block in tensor_with_all_blocks.data.blocks)
    else:
        pytest.xfail(f'Dont know how to construct expected num_parameters for {type(backend)}')
    assert tens3.num_parameters == expect

    print('check is_fully_labelled')
    assert not tens1.is_fully_labelled
    assert not tens2.is_fully_labelled
    assert tens3.is_fully_labelled

    print('check has_label')
    assert tens3.has_label('a')
    assert tens3.has_label('a', 'foo')
    assert not tens3.has_label('bar')
    assert not tens3.has_label('a', 'bar')

    print('check labels_are')
    assert tens3.labels_are('foo', 'a', 'b')
    assert tens3.labels_are('a', 'foo', 'b')
    assert not tens3.labels_are('a', 'foo', 'b', 'bar')

    print('check setting labels')
    tens3.set_labels(['i', 'j', 'k'])
    assert tens3.labels_are('i', 'j', 'k')
    tens3.labels = ['foo', 'a', 'b']
    assert tens3.labels_are('foo', 'a', 'b')

    print('check get_leg_idx')
    assert tens3.get_leg_idx(0) == 0
    assert tens3.get_leg_idx(-1) == 2
    with pytest.raises(ValueError):
        tens3.get_leg_idx(10)
    assert tens3.get_leg_idx('foo') == 0
    assert tens3.get_leg_idx('a') == 1
    with pytest.raises(ValueError):
        tens3.get_leg_idx('bar')
    with pytest.raises(TypeError):
        tens3.get_leg_idx(None)

    print('check get_leg_idcs')
    assert tens3.get_leg_idcs('foo') == [0]
    assert tens3.get_leg_idcs(['foo', 'b', 1]) == [0, 2, 1]

    print('check item')
    triv_legs = [make_compatible_space(1, 1) for _ in range(2)]
    for leg in triv_legs[:]:
        triv_legs.append(leg.dual)
    assert all(leg.dim == 1 for leg in triv_legs)
    data4 = make_compatible_tensor(legs=triv_legs, real=True).data
    tens4 = tensors.BlockDiagonalTensor(data4, num_domain_legs=0, backend=backend, legs=triv_legs)
    tens4_item = backend.data_item(data4)
    # not a good test, but the best we can do backend independent:
    assert tens4.item() == tens4_item

    print('check str and repr')
    str(tens1)
    str(tens3)
    repr(tens1)
    repr(tens3)

    print('convert to dense')
    dense1 = tens1.to_numpy()
    dense2 = tens2.to_numpy()
    dense3 = tens3.to_numpy()

    print('check addition + multiplication')
    neg_t3 = -tens3
    npt.assert_array_equal(neg_t3.to_numpy(), -dense3)
    a = 42
    b = 17
    with pytest.raises(ValueError) as err:
        res = a * tens1 - b * tens2
    assert "labelled tensors must be *fully* labelled" in err.value.args[0]  # TODO: check other config values?
    tens1.set_labels(['foo', 'a', 'b'])
    tens2.set_labels(['foo', 'a', 'b'])
    res = a * tens1 - b * tens2
    npt.assert_almost_equal(res.to_numpy(), a * dense1 - b * dense2)
    res = tens1 / a + tens2 / b
    npt.assert_almost_equal(res.to_numpy(), dense1 / a + dense2 / b)
    # TODO check strict label behavior!

    with pytest.raises(TypeError):
        tens1 == tens2

    print('check conversions, float, complex, array')
    tens4.set_labels(['i', 'j', 'i*', 'j*'])
    assert isinstance(float(tens4), float)
    npt.assert_equal(float(tens4), float(tens4_item))
    assert isinstance(complex(tens4 + 2.j * tens4), complex)
    npt.assert_equal(complex(tens4 + 2.j * tens4), complex(tens4_item + 2.j * tens4_item))

    with pytest.warns(UserWarning, match='converting complex to real, only return real part!'):
        r = float(tens4 + 2.j * tens4)
    assert r == tens4_item


def test_Tensor_tofrom_flat_block_trivial_sector(make_compatible_tensor):
    # TODO move to some other tests after restructuring
    tens = make_compatible_tensor(labels=['a'])
    leg, = tens.legs
    block_size = leg.sector_multiplicity(tens.symmetry.trivial_sector)

    if isinstance(tens.backend, FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='to_flat_block_trivial_sector not implemented'):
            block = tens.to_flat_block_trivial_sector()
        return  # TODO
    
    block = tens.to_flat_block_trivial_sector()
    assert tens.backend.block_shape(block) == (block_size,)
    tens2 = tensors.BlockDiagonalTensor.from_flat_block_trivial_sector(leg=leg, block=block, backend=tens.backend, label='a')
    tens2.test_sanity()
    assert tensors.almost_equal(tens, tens2)
    block2 = tens2.to_flat_block_trivial_sector()
    npt.assert_array_almost_equal_nulp(tens.backend.block_to_numpy(block),
                                       tens.backend.block_to_numpy(block2),
                                       100)


def test_ChargedTensor_tofrom_flat_block_single_sector(compatible_symmetry, make_compatible_sectors,
                                                       make_compatible_tensor):
    pytest.xfail(reason='unclear')  # TODO
    # TODO move to some other tests after restructuring
    sector = make_compatible_sectors(1)[0]
    dummy_leg = ElementarySpace(compatible_symmetry, sector[None, :]).dual
    inv_part = make_compatible_tensor(legs=[None, dummy_leg])
    tens = tensors.ChargedTensor(invariant_part=inv_part)
    leg = tens.legs[0]
    block_size = leg.sector_multiplicity(sector)

    block = tens.to_flat_block_single_sector()
    assert tens.backend.block_shape(block) == (block_size,)
    tens2 = tensors.ChargedTensor.from_flat_block_single_sector(
        leg=leg, block=block, sector=sector, backend=tens.backend
    )
    tens2.test_sanity()
    assert tens2.dummy_leg == tens.dummy_leg
    assert tensors.almost_equal(tens, tens2)
    block2 = tens2.to_flat_block_single_sector()
    npt.assert_array_almost_equal_nulp(tens.backend.block_to_numpy(block),
                                       tens.backend.block_to_numpy(block2),
                                       100)
    # check detect_sectors_from_block while we are at it
    dense_block = tens.to_dense_block()
    detected, = tensors.detect_sectors_from_block(block=dense_block, legs=[leg], backend=tens.backend)
    npt.assert_array_equal(detected, sector)


@pytest.mark.parametrize('symmetry_backend', ['abelian', pytest.param('fusion_tree', marks=pytest.mark.FusionTree)])
@pytest.mark.parametrize('num_domain_legs', [0, 1, 2])
def test_from_block_z4symm_2legs(symmetry_backend, num_domain_legs, block_backend):
    backend = get_backend(symmetry_backend, block_backend)
    all_qi = z4_symmetry.all_sectors()
    q0, q1, q2, q3 = all_qi
    basis1 = [q3, q3, q2, q0, q3, q2]  # basis_perm [3, 2, 5, 0, 1, 4]
    basis2 = [q2, q0, q1, q2, q3, q0, q1]  # basis_perm [1, 5, 2, 6, 0, 3, 4]
    s1 = ElementarySpace.from_basis(z4_symmetry, basis1)  # sectors = [0, 2, 3]
    s2 = ElementarySpace.from_basis(z4_symmetry, basis2)  # sectors = [0, 1, 2, 3]

    #      q: 2,  0,  1,  2,  3,  0,  1      q
    data = [[ 0,  0,  1,  0,  0,  0,  2],  # 3
            [ 0,  0,  3,  0,  0,  0,  4],  # 3
            [ 5,  0,  0,  6,  0,  0,  0],  # 2
            [ 0,  7,  0,  0,  0,  8,  0],  # 0
            [ 0,  0,  9,  0,  0,  0, 10],  # 3
            [11,  0,  0, 12,  0,  0,  0]]  # 2

    # after applying the basis perm
    # q: 0   0   1   1   2   2   3       q
    # [[ 7.  8.  0.  0.  0.  0.  0.]     0
    #  [ 0.  0.  0.  0.  5.  6.  0.]     2
    #  [ 0.  0.  0.  0. 11. 12.  0.]     2
    #  [ 0.  0.  1.  2.  0.  0.  0.]     3
    #  [ 0.  0.  3.  4.  0.  0.  0.]     3
    #  [ 0.  0.  9. 10.  0.  0.  0.]]    3
    
    block = backend.block_from_numpy(np.asarray(data, dtype=float))
    t = tensors.BlockDiagonalTensor.from_dense_block(
        block, [s1, s2], backend=backend, num_domain_legs=num_domain_legs
    )
    t.test_sanity()
    assert t.num_domain_legs == num_domain_legs
    assert t.num_codomain_legs == 2 - num_domain_legs
    # explicitly check the ``t.data`` vs what we expect
    block_0 = np.asarray([[7, 8]])
    block_1 = np.asarray([[1, 2], [3, 4], [9, 10]])
    block_2 = np.asarray([[5, 6], [11, 12]])

    if symmetry_backend == 'abelian':
        expect_block_inds = np.array([
            [0, 0],  # q=0, q=0
            [2, 1],  # q=2, q=2
            [1, 2],  # q=1, q=3
        ])
        assert np.all(t.data.block_inds == expect_block_inds)
        expect_blocks = [backend.block_from_numpy(b) for b in [block_0, block_1, block_2]]
        assert len(expect_blocks) == len(t.data.blocks)
        for i, (actual, expect) in enumerate(zip(t.data.blocks, expect_blocks)):
            print(f'checking blocks[{i}]')
            assert backend.block_allclose(actual, expect)
    
    elif symmetry_backend == 'fusion_tree' and num_domain_legs == 0:
        assert np.all(t.data.coupled_sectors == q0[None, :])
        forest_block_q0_q0 = block_0.reshape((-1, 1))
        forest_block_q1_q3 = block_2.reshape((-1, 1))
        forest_block_q2_q2 = block_1.reshape((-1, 1))
        expect_block = np.concatenate([forest_block_q0_q0, forest_block_q1_q3, forest_block_q2_q2],
                                      axis=0)
        expect_block = backend.block_from_numpy(expect_block)
        assert len(t.data.blocks) == 1
        assert backend.block_allclose(t.data.blocks[0], expect_block)
        # [array([[ 7.,  8.,  1.,  3.,  9.,  2.,  4., 10.,  5., 11.,  6., 12.]])]
    
    elif symmetry_backend == 'fusion_tree' and num_domain_legs == 1:
        # codomain: [s1]  , sector=[0, 2, 3]
        # domain: [s2.dual], sectors=[0, 3, 2, 1]
        # coupled sectors are sectors of the codomain.
        expect_sectors = [q0, q2, q3]
        expect_blocks = [block_0, block_2, block_1]  # block_1 appears with coupled sector [3].
        assert np.all(t.data.coupled_sectors == np.array(expect_sectors))
        for expect, actual in zip(expect_blocks, t.data.blocks):
            assert backend.block_allclose(actual, expect)

    elif symmetry_backend == 'fusion_tree' and num_domain_legs == 2:
        assert np.all(t.data.coupled_sectors == q0[None, :])
        expect_block = np.concatenate([block_0, block_1.T.reshape((1, -1)), block_2.T.reshape((1, -1))],
                                      axis=1)
        expect_block = backend.block_from_numpy(expect_block)
        assert len(t.data.blocks) == 1
        assert backend.block_allclose(t.data.blocks[0], expect_block)
        # [array([[ 7.,  8.,  1.,  3.,  9.,  2.,  4., 10.,  5., 11.,  6., 12.]])]
    
    else:
        raise RuntimeError('should have covered all cases above.')

    # check conversion back
    data_reconstructed = t.to_dense_block()
    assert backend.block_allclose(data_reconstructed, data)


@pytest.mark.parametrize('symmetry_backend', [pytest.param('fusion_tree', marks=pytest.mark.FusionTree)])
def test_from_block_su2_symm(symmetry_backend, block_backend):
    # TODO convert back when to_dense is implemented.
    backend = get_backend(symmetry_backend, block_backend)
    sym = SU2Symmetry()
    spin_half = ElementarySpace(sym, [[1]])

    # basis order: [down, up]
    sx = .5 * np.array([[0., 1.], [1., 0.]], dtype=complex)
    sy = .5 * np.array([[0., 1.j], [-1.j, 0]], dtype=complex)
    sz = .5 * np.array([[-1., 0.], [0., +1.]], dtype=complex)
    heisenberg_4 = sum(si[:, :, None, None] * si[None, None, :, :] for si in [sx, sy, sz])  # [p1, p1*, p2, p2*]
    print(heisenberg_4.transpose([0, 2, 1, 3]).reshape((4, 4)))
    # construct in a specific leg-order where we know what the blocks should be.
    heisenberg_4 = np.transpose(heisenberg_4, [0, 2, 3, 1])  # [p1, p2, p2*, p1*]

    tens_4 = tensors.BlockDiagonalTensor.from_dense_block(
        heisenberg_4, [spin_half, spin_half, spin_half.dual, spin_half.dual],
        backend, labels=['p1', 'p2', 'p2*', 'p1*'], num_domain_legs=2,
    )
    tens_4.test_sanity()
    assert np.all(tens_4.data.coupled_sectors == np.array([[0], [2]]))  # spin 0, spin 1
    # in this leg order, the blocks come from maps
    # [p1, p2] --X--> [coupled] --block--> [coupled] --Y--> [p1, p2].
    # Thus, the blocks are the eigenvalue of the Heisenberg coupling in the fixed total spin sectors
    # For singlet states (coupled=spin-0), we have eigenvalue -3/4
    # For triplet states (coupled=spin-1), we have eigenvalue +1/4
    expect_spin_0 = -3 / 4  
    expect_spin_1 = 1 / 4
    assert backend.block_allclose(tens_4.data.blocks[0], expect_spin_0)
    assert backend.block_allclose(tens_4.data.blocks[1], expect_spin_1)

    recovered_block = tens_4.to_dense_block()
    print()
    print('got:')
    print(recovered_block.reshape((4, 4)))
    print()
    print('expect:')
    print(heisenberg_4.reshape((4, 4)))

    assert backend.block_allclose(recovered_block, heisenberg_4)


def test_tdot(make_compatible_space, make_compatible_sectors, make_compatible_tensor):
    # define legs such that a tensor with the following combinations all allow non-zero num_parameters
    # [a, b] , [a, b, c*] , [a, b, d*]
    from conftest import find_compatible_leg
    a = make_compatible_space()
    b = find_compatible_leg([a], max_sectors=3, max_mult=3, extra_sectors=make_compatible_sectors(3))
    c = find_compatible_leg([a, b], max_sectors=3, max_mult=3, extra_sectors=make_compatible_sectors(3)).dual
    d = find_compatible_leg([a, b], max_sectors=3, max_mult=3, extra_sectors=make_compatible_sectors(3)).dual
    print([l.dim for l in [a, b, c, d]])
    
    legs_ = [[a, b, c.dual],
             [d, b.dual, a.dual],
             [b.dual, a.dual],
             [a, b]]
    labels_ = [['a', 'b', 'c*'],
               ['d', 'b*', 'a*'],
               ['b*', 'a*'],
               ['a', 'b']
               ]
    tensors_ = [
        make_compatible_tensor(legs=legs, labels=labels) for legs, labels in zip(legs_, labels_)
    ]
    for n, t in enumerate(tensors_):
        # make sure we are defining tensors which actually contain blocks and are not just zero by
        # charge conservation
        assert t.num_parameters > 0, f'tensor {n} has 0 free parameters'

    if isinstance(tensors_[0].backend, FusionTreeBackend) and isinstance(a.symmetry, ProductSymmetry):
        with pytest.raises(NotImplementedError, match='should be implemented by subclass'):
            dense_ = [t.to_numpy() for t in tensors_]
        return  # TODO
    
    dense_ = [t.to_numpy() for t in tensors_]

    checks = [("single leg", 0, 1, 1, 1, 'b', 'b*'),
              ("two legs", 0, 1, [0, 1], [2, 1], ['a', 'b'], ['a*', 'b*']),
              ("all legs of first tensor", 2, 0, [0, 1], [1, 0], ['a*', 'b*'], ['a', 'b']),
              ("all legs of second tensor", 1, 3, [1, 2], [1, 0], ['a*', 'b*'], ['a', 'b']),
              ("scalar result / inner()", 2, 3, [0, 1], [1, 0], ['a*', 'b*'], ['a', 'b']),
              ("no leg / outer()", 2, 3, [], [], [], []),
              ]
    for comment, i, j, ax_i, ax_j, lbl_i, lbl_j in checks:
        print('tdot: contract ', comment)
        expect = np.tensordot(dense_[i], dense_[j], (ax_i, ax_j))

        if isinstance(tensors_[0].backend, FusionTreeBackend):
            with pytest.raises(NotImplementedError, match='tdot not implemented'):        
                res1 = tensors.tdot(tensors_[i], tensors_[j], ax_i, ax_j)
            return  # TODO
        
        res1 = tensors.tdot(tensors_[i], tensors_[j], ax_i, ax_j)
        res2 = tensors.tdot(tensors_[i], tensors_[j], lbl_i, lbl_j)
        if len(expect.shape) > 0:
            res1.test_sanity()
            res2.test_sanity()
            res1_d = res1.to_numpy()
            res2_d = res2.to_numpy()
            npt.assert_array_almost_equal(res1_d, expect)
            npt.assert_array_almost_equal(res2_d, expect)
        else: # got scalar, but we can compare it to 0-dim ndarray
            npt.assert_almost_equal(res1, expect)
            npt.assert_almost_equal(res2, expect)

    # TODO check that trying to contract incompatible legs raises
    #  - opposite is_dual but different dim
    #  - opposite is_dual and dim but different sectors
    #  - same dim and sectors but same is_dual


def test_outer(make_compatible_tensor):
    tensors_ = [make_compatible_tensor(labels=labels) for labels in [['a'], ['b'], ['c', 'd']]]

    if isinstance(tensors_[0].backend, FusionTreeBackend) and isinstance(tensors_[0].symmetry, ProductSymmetry):
        with pytest.raises(NotImplementedError, match='should be implemented by subclass'):
            dense_ = [t.to_numpy() for t in tensors_]
        return  # TODO
    
    dense_ = [t.to_numpy() for t in tensors_]

    for i, j  in [(0, 1), (0, 2), (0, 0), (2, 2)]:
        print(i, j)
        expect = np.tensordot(dense_[i], dense_[j], axes=0)

        if isinstance(tensors_[0].backend, FusionTreeBackend):
            with pytest.raises(NotImplementedError, match='outer not implemented'):
                res = tensors.outer(tensors_[i], tensors_[j])
            return  # TODO
        
        res = tensors.outer(tensors_[i], tensors_[j])
        res.test_sanity()
        npt.assert_array_almost_equal(res.to_numpy(), expect)
        if i != j:
            assert res.labels_are(*(tensors_[i].labels + tensors_[j].labels))
        else:
            assert all(l is None for l in res.labels)


def test_permute_legs(make_compatible_tensor):
    labels = list('abcd')
    t = make_compatible_tensor(labels=labels)

    if isinstance(t.backend, FusionTreeBackend) and isinstance(t.symmetry, ProductSymmetry):
        with pytest.raises(NotImplementedError, match='should be implemented by subclass'):
            d = t.to_numpy()
        return  # TODO
    
    d = t.to_numpy()
    for perm in [[0, 2, 1, 3], [3, 2, 1, 0], [1, 0, 3, 2], [0, 1, 2, 3], [0, 3, 2, 1]]:
        expect = d.transpose(perm)

        if isinstance(t.backend, FusionTreeBackend):
            with pytest.raises(NotImplementedError, match='permute_legs not implemented'):
                res = t.permute_legs(perm)
            return  # TODO
        
        res = t.permute_legs(perm)
        res.test_sanity()
        npt.assert_array_equal(res.to_numpy(), expect)
        assert res.labels == [labels[i] for i in perm]


def test_inner(make_compatible_tensor):
    t0 = make_compatible_tensor(labels=['a'])
    t1 = make_compatible_tensor(legs=t0.legs, labels=t0.labels)
    t2 = make_compatible_tensor(labels=['a', 'b', 'c'])
    t3 = make_compatible_tensor(legs=t2.legs, labels=t2.labels)

    for t_i, t_j, perm in [(t0, t1, ['a']), (t2, t3, ['b', 'c', 'a'])]:
        d_i = t_i.to_numpy()
        d_j = t_j.to_numpy()

        expect = np.inner(d_i.flatten().conj(), d_j.flatten())
        if t_j.num_legs > 0:

            if isinstance(t0.backend, FusionTreeBackend):
                with pytest.raises(NotImplementedError, match='permute_legs not implemented'):
                    t_j = t_j.permute_legs(perm)
                return  # TODO
            
            t_j = t_j.permute_legs(perm)  # transpose should be reverted in inner()
        res = tensors.inner(t_i, t_j)
        npt.assert_allclose(res, expect)

        expect = np.linalg.norm(d_i) **2
        res = tensors.inner(t_i, t_i)
        npt.assert_allclose(res, expect)


def test_trace(make_compatible_space, make_compatible_tensor):
    a = make_compatible_space(3, 3)
    b = make_compatible_space(4, 3)
    t1 = make_compatible_tensor(legs=[a, a.dual], labels=['a', 'a*'])

    if isinstance(t1.backend, FusionTreeBackend) and isinstance(t1.symmetry, ProductSymmetry):
        with pytest.raises(NotImplementedError, match='should be implemented by subclass'):
            d1 = t1.to_numpy()
        return  # TODO
    
    d1 = t1.to_numpy()
    t2 = make_compatible_tensor(legs=[a, b, a.dual, b.dual], labels=['a', 'b', 'a*', 'b*'])
    d2 = t2.to_numpy()
    t3 = make_compatible_tensor(legs=[a, None, b, a.dual, b.dual], labels=['a', 'c', 'b', 'a*', 'b*'])
    d3 = t3.to_numpy()

    print('single legpair - full')
    expected = np.trace(d1, axis1=0, axis2=1)

    if isinstance(t1.backend, FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='trace_full not implemented'):
            res = tensors.trace(t1, 'a*', 'a')
        return  # TODO
    
    res = tensors.trace(t1, 'a*', 'a')
    npt.assert_array_almost_equal_nulp(res, expected, 100)

    print('single legpair - partial')
    expected = np.trace(d2, axis1=1, axis2=3)
    res = tensors.trace(t2, 'b*', 'b')
    res.test_sanity()
    assert res.labels_are('a', 'a*')
    npt.assert_array_almost_equal_nulp(res.to_numpy(), expected, 100)

    print('two legpairs - full')
    expected = np.trace(d2, axis1=1, axis2=3).trace(axis1=0, axis2=1)
    res = tensors.trace(t2, ['a', 'b*'], ['a*', 'b'])
    npt.assert_array_almost_equal_nulp(res, expected, 100)

    print('two legpairs - partial')
    expected = np.trace(d3, axis1=2, axis2=4).trace(axis1=0, axis2=2)
    res = tensors.trace(t3, ['a', 'b*'], ['a*', 'b'])
    res.test_sanity()
    assert res.labels_are('c')
    npt.assert_array_almost_equal_nulp(res.to_numpy(), expected, 100)


def test_conj_hconj(make_compatible_tensor):
    tens = make_compatible_tensor(labels=['a', 'b', None])

    if isinstance(tens.backend, FusionTreeBackend) and isinstance(tens.symmetry, ProductSymmetry):
        with pytest.raises(NotImplementedError, match='should be implemented by subclass'):
            expect = np.conj(tens.to_numpy())
        return  # TODO
    
    expect = np.conj(tens.to_numpy())
    assert np.linalg.norm(expect.imag) > 0 , "expect complex data!"

    if isinstance(tens.backend, FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='conj not implemented'):
            res = tensors.conj(tens)
        return  # TODO
    
    res = tensors.conj(tens)
    res.test_sanity()
    assert res.labels == ['a*', 'b*', None]
    assert [l1.can_contract_with(l2) for l1, l2 in zip(res.legs, tens.legs)]
    assert np.allclose(res.to_numpy(), expect)

    print('hconj 1-site operator')
    leg_a = tens.legs[0]
    op = make_compatible_tensor(legs=[leg_a, leg_a.dual], labels=['p', 'p*'])
    op_hc = tensors.hconj(op)
    op_hc.test_sanity()
    assert op_hc.labels == op.labels
    assert op_hc.legs == op.legs
    _ = op + op_hc  # just check if it runs
    npt.assert_array_equal(op_hc.to_numpy(), np.conj(op.to_numpy()).T)
    
    print('hconj 2-site op')
    leg_b = tens.legs[1]
    op2 = make_compatible_tensor(legs=[leg_a, leg_b.dual, leg_a.dual, leg_b],
                                 labels=['a', 'b*', 'a*', 'b'])
    op2_hc = op2.hconj(['a', 'b'], ['a*', 'b*'])
    op2_hc.test_sanity()
    assert op2_hc.labels == op2.labels
    assert op2_hc.legs == op2.legs
    _ = op2 + op2_hc  # just check if it runs
    expect = np.transpose(np.conj(op2.to_numpy()), [2, 3, 0, 1])
    npt.assert_array_equal(op2_hc.to_numpy(), expect)
    

def test_combine_split(make_compatible_tensor, compatible_symmetry_backend):
    tens = make_compatible_tensor(labels=['a', 'b', 'c', 'd'], max_blocks=5, max_block_size=5)

    if isinstance(tens.backend, FusionTreeBackend) and isinstance(tens.symmetry, ProductSymmetry):
        with pytest.raises(NotImplementedError, match='should be implemented by subclass'):
            dense = tens.to_numpy()
        return  # TODO
    
    dense = tens.to_numpy()
    d0, d1, d2, d3 = dims = tuple(tens.shape)

    print('check by idx')

    if isinstance(tens.backend, FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='combine_legs not implemented'):
            res = tensors.combine_legs(tens, [1, 2])
        return  # TODO
    
    res = tensors.combine_legs(tens, [1, 2])
    res.test_sanity()
    assert res.labels == ['a', '(b.c)', 'd']
    # note: dense reshape is not enough to check expect, since we have permutation in indices.
    # hence, we only check that we get back the same after split
    split = tensors.split_legs(res, 1)
    split.test_sanity()
    assert split.labels == ['a', 'b', 'c', 'd']
    npt.assert_equal(split.to_numpy(), dense)

    print('check by label')
    res = tensors.combine_legs(tens, ['b', 'd'])
    res.test_sanity()
    assert res.labels == ['a', '(b.d)', 'c']
    split = tensors.split_legs(res, '(b.d)')
    split.test_sanity()
    assert split.labels == ['a', 'b', 'd', 'c']
    assert np.allclose(split.to_numpy(), dense.transpose([0, 1, 3, 2]))

    print('check splitting a non-combined leg raises')
    with pytest.raises(ValueError):
        tensors.split_legs(res, 0)
    with pytest.raises(ValueError):
        tensors.split_legs(res, 'd')

    print('check combining multiple legs')
    res = tensors.combine_legs(tens, ['c', 'a'], ['b', 'd'], product_spaces_dual=[False, True])
    res.test_sanity()
    # leg order after combine:
    #
    #     replace by (b.d)
    #     |     omit
    #     |     |
    # [a, b, c, d]               ->   [(b.d), (c.a)]
    #  |     |
    #  omit  replace by (c.a)
    #
    assert res.labels == ['(b.d)', '(c.a)']
    assert res.legs[0].is_dual == True
    assert res.legs[1].is_dual == False
    split = tensors.split_legs(res)
    split.test_sanity()
    assert split.labels == ['b', 'd', 'c', 'a']
    npt.assert_equal(split.to_numpy(), dense.transpose([1, 3, 2, 0]))

    print('check _fuse_spaces')
    sectors1, mults1, metadata1 = _fuse_spaces(
        symmetry=tens.symmetry, spaces=tens.get_legs(['b', 'd'])
    )
    assert len(metadata1) == 1
    if compatible_symmetry_backend in ['abelian']:  # those backends that implement _fuse_spaces
        sectors2, mults2, metadata2 = tens.backend._fuse_spaces(
            symmetry=tens.symmetry, spaces=tens.get_legs(['b', 'd'])
        )
        npt.assert_array_equal(metadata1['fusion_outcomes_sort'], metadata2['fusion_outcomes_sort'])
        npt.assert_array_equal(sectors1, sectors2)
        npt.assert_array_equal(mults1, mults2)
        assert len(metadata2) == 4

    # FIXME redesign?
    for prod_space, comment in [
        (ProductSpace(tens.get_legs(['b', 'd']), backend=tens.backend), 'metadata via ProductSpace.__init__'),
        (tens.backend.add_leg_metadata(ProductSpace(tens.get_legs(['b', 'd']))), 'metadata via add_leg_metadata'),
        (ProductSpace(tens.get_legs(['b', 'd'])), 'no metadata'),
    ]:
        print(f'check combine_legs with ProductSpace. {comment}')
        res = tensors.combine_legs(tens, ['b', 'd'], product_spaces=[prod_space])
        res.test_sanity()
        assert res.labels == ['a', '(b.d)', 'c']
        split = tensors.split_legs(res, '(b.d)')
        split.test_sanity()
        assert split.labels == ['a', 'b', 'd', 'c']
        assert np.allclose(split.to_numpy(), dense.transpose([0, 1, 3, 2]))


@pytest.mark.xfail  # TODO
def test_combine_legs_basis_trafo(make_compatible_tensor):
    tens = make_compatible_tensor(labels=['a', 'b', 'c'], max_blocks=5, max_block_size=5)
    a, b, c = tens.shape
    dense = tens.to_numpy()  # [a, b, c]
    combined = tensors.combine_legs(tens, ['a', 'b'])
    dense_combined = combined.to_dense_block()  # [(a.b), c]

    print('check via perm')
    perm = combined.get_leg('(a.b)').get_basis_transformation_perm()
    assert all(0 <= p < len(perm) for p in perm)
    assert len(set(perm)) == len(perm)
    reconstruct_combined = np.reshape(dense, (a * b, c))[perm, :]
    
    npt.assert_array_almost_equal_nulp(dense_combined, reconstruct_combined, 100)

    print('check via trafo')
    trafo = combined.get_legs('(a.b)')[0].get_basis_transformation()  # [a, b, (a.b)]
    reconstruct_combined = np.tensordot(trafo, dense, [[0, 1], [0, 1]])  # [(a.b), c]
    npt.assert_array_almost_equal_nulp(dense_combined, reconstruct_combined, 100)


def test_is_scalar(make_compatible_tensor, make_compatible_space):
    for s in [1, 0., 1.+2.j, np.int64(123), np.float64(2.345), np.complex128(1.+3.j)]:
        assert tensors.is_scalar(s)
    triv_leg = make_compatible_space(1, 1)
    scalar_tens = make_compatible_tensor(legs=[triv_leg, triv_leg.dual])
    assert tensors.is_scalar(scalar_tens)
    # generate non-scalar tensor
    for i in range(20):
        non_scalar_tens = make_compatible_tensor(num_legs=3)
        if any(d > 1 for d in non_scalar_tens.shape):  # non-trivial
            assert not tensors.is_scalar(non_scalar_tens)
            break
    else:  # didn't break
        pytest.skip("can't generate non-scalar tensor")


@pytest.mark.parametrize('num_legs,', [1, 3])
def test_norm(make_compatible_tensor, num_legs):
    tens = make_compatible_tensor(num_legs=num_legs)

    if isinstance(tens.backend, FusionTreeBackend) and isinstance(tens.symmetry, ProductSymmetry):
        if tens.data.num_domain_legs >= 2 or tens.data.num_codomain_legs >= 2:  # otherwise fusion tensors are not needed
            with pytest.raises(NotImplementedError, match='should be implemented by subclass'):
                expect = np.linalg.norm(tens.to_numpy())
            return  # TODO
    
    expect = np.linalg.norm(tens.to_numpy())
    res = tensors.norm(tens)
    assert np.allclose(res, expect)


def test_almost_equal(make_compatible_tensor, np_random):
    for i in range(10):
        t1 = make_compatible_tensor(labels=['a', 'b', 'c'])
        t_diff = make_compatible_tensor(t1.legs, labels=['a', 'b', 'c'])
        if t_diff.norm() > 1.e-7:
            break
    else:
        pytest.skip("can't generate random nonzero tensor?")
    t2 = t1 + 1.e-7 * t_diff
    assert tensors.almost_equal(t1, t2), "default a_tol should be > 1e-7!"
    assert not tensors.almost_equal(t1, t2, atol=1.e-10, rtol=1.e-10), "tensors differ by 1e-7!"

    # TODO properly adapt test suite to the different tensor types
    print('test DiagonalTensor.almost_equal')
    leg = t1.legs[0]
    data1 = np_random.random(leg.dim)
    data2 = data1 + 1e-7 * np_random.random(leg.dim)

    if isinstance(t1.backend, FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='diagonal_from_block not implemented'):
            t1 = tensors.DiagonalTensor.from_diag(data1, leg, backend=t1.backend)
        return  # TODO
    
    t1 = tensors.DiagonalTensor.from_diag(data1, leg, backend=t1.backend)
    t2 = tensors.DiagonalTensor.from_diag(data2, leg, backend=t1.backend)
    assert tensors.almost_equal(t1, t2, atol=1e-5, rtol=1e-7)
    assert not tensors.almost_equal(t1, t2, atol=1e-10, rtol=1e-10)

    # TODO check all combinations of tensor types...


def test_squeeze_legs(make_compatible_tensor, compatible_symmetry):
    for i in range(10):
        triv_leg = ElementarySpace(compatible_symmetry,
                               compatible_symmetry.trivial_sector[np.newaxis, :])
        assert triv_leg.is_trivial
        tens = make_compatible_tensor(legs=[None, triv_leg, None, triv_leg.dual, triv_leg],
                                      labels=list('abcde'))
        if not tens.legs[0].is_trivial and not tens.legs[2].is_trivial:
            break
    else:
        pytest.skip("can't generate non-triv leg")

    if isinstance(tens.backend, FusionTreeBackend) and isinstance(tens.symmetry, ProductSymmetry):
        with pytest.raises(NotImplementedError, match='should be implemented by subclass'):
            dense = tens.to_numpy()
        return  # TODO
    dense = tens.to_numpy()

    print('squeezing all legs (default arg)')

    if isinstance(tens.backend, FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='squeeze_legs not implemented'):
            res = tensors.squeeze_legs(tens)
        return  # TODO
    
    res = tensors.squeeze_legs(tens)
    res.test_sanity()
    assert res.labels == ['a', 'c']
    npt.assert_array_equal(res.to_numpy(), dense[:, 0, :, 0, 0])

    print('squeeze specific leg by idx')
    res = tensors.squeeze_legs(tens, 1)
    res.test_sanity()
    assert res.labels == ['a', 'c', 'd', 'e']
    npt.assert_array_equal(res.to_numpy(), dense[:, 0, :, :, :])

    print('squeeze legs by labels')
    res = tensors.squeeze_legs(tens, ['b', 'e'])
    res.test_sanity()
    assert res.labels == ['a', 'c', 'd']
    npt.assert_array_equal(res.to_numpy(), dense[:, 0, :, :, 0])


def test_add_trivial_leg(make_compatible_tensor):
    A = make_compatible_tensor(labels=['a', 'b'])

    if isinstance(A.backend, FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='add_trivial_leg not implemented'):
            B = tensors.add_trivial_leg(A, 'c', is_dual=True)
        return  # TODO
    
    B = tensors.add_trivial_leg(A, 'c', is_dual=True)
    B.test_sanity()
    B = tensors.add_trivial_leg(B, 'xY', pos=1)
    B.test_sanity()
    assert B.labels == ['a', 'xY', 'b', 'c']
    assert [leg.is_dual for leg in B.legs] == [A.legs[0].is_dual, False, A.legs[1].is_dual, True]
    expect = A.to_numpy()[:, None, :, None]
    B_np = B.to_numpy()
    npt.assert_array_equal(B_np, expect)
    C = B.squeeze_legs()
    assert tensors.almost_equal(A, C)


def test_scale_axis(make_compatible_tensor):
    # TODO eventually this will be covered by tdot tests, when allowing combinations of Tensor and DiagonalTensor
    #  But I want to use it already now to debug backend.scale_axis()
    t = make_compatible_tensor(num_legs=3, max_blocks=4, max_block_size=4)

    if isinstance(t.backend, FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='diagonal_from_block_func not implemented'):
            d = tensors.DiagonalTensor.random_uniform(t.legs[0], second_leg_dual=True, backend=t.backend)
        return  # TODO
    
    d = tensors.DiagonalTensor.random_uniform(t.legs[0], second_leg_dual=True, backend=t.backend)
    expect = np.tensordot(t.to_numpy(), d.to_numpy(), (0, 1))
    res = tensors.tdot(t, d, 0, 1).to_numpy()
    npt.assert_almost_equal(expect, res)


def test_detect_sectors_from_block(compatible_backend, compatible_symmetry, make_compatible_sectors,
                                   np_random):
    num_sectors = int(min(4, compatible_symmetry.num_sectors))
    leg_dim = 5
    sectors = make_compatible_sectors(num_sectors, sort=True)

    if not compatible_symmetry.is_abelian:
        pytest.skip('need to design test with legal sectors_of_basis ')

    which_sectors_a = np_random.integers(num_sectors, size=(leg_dim,))  # indices of sectors
    which_sectors_b = np_random.integers(num_sectors, size=(leg_dim + 1,))
    sectors_of_basis_a = sectors[which_sectors_a]
    sectors_of_basis_b = sectors[which_sectors_b]
    a = ElementarySpace.from_basis(symmetry=compatible_symmetry, sectors_of_basis=sectors_of_basis_a)
    b = ElementarySpace.from_basis(symmetry=compatible_symmetry, sectors_of_basis=sectors_of_basis_b)

    target_sector_a = np_random.choice(which_sectors_a)
    target_sector_b = np_random.choice(which_sectors_b)
    
    data = 1e-9 * np_random.random(size=(a.dim, b.dim))
    for i in range(a.dim):
        if which_sectors_a[i] != target_sector_a:
            continue
        for j in range(b.dim):
            if which_sectors_b[j] != target_sector_b:
                continue
            data[i, j] = np_random.random()
    assert np.max(np.abs(data)) > 1e-9  # make sure that at least one entry above was actually set

    block = compatible_backend.block_from_numpy(data)
    detected = tensors.detect_sectors_from_block(block, legs=[a, b], backend=compatible_backend)
    npt.assert_array_equal(detected[0], sectors[target_sector_a])
    npt.assert_array_equal(detected[1], sectors[target_sector_b])

    print('check an explicit state')
    if num_sectors >= 4:
        #                         0  1  2  3  4  5  6  7  8  9
        which_sectors = np.array([0, 1, 3, 0, 2, 1, 1, 2, 0, 3])
        space = ElementarySpace(symmetry=compatible_symmetry,
                                sectors=sectors[:4],
                                multiplicities=[3, 3, 2, 2],
                                basis_perm=[0, 3, 8, 1, 5, 6, 4, 7, 2, 9])
        # make sure setup is correct
        space.test_sanity()
        assert np.all(sectors[which_sectors] == space.sectors_of_basis)

        for i, which in enumerate(which_sectors):
            data = np.zeros((len(which_sectors),), dtype=float)
            data[i] = 1.
            sector, = tensors.detect_sectors_from_block(
                compatible_backend.block_from_numpy(data), legs=[space], backend=compatible_backend
            )
            npt.assert_array_equal(sector, sectors[which])


def test_elementwise_function_decorator():
    assert tensors.sqrt.__doc__ == 'The square root of a number, elementwise.'


@pytest.mark.parametrize('function, data_imag', [('real', 0), ('real', 1),
                                                 ('imag', 0), ('imag', 1),
                                                 ('angle', 0), ('angle', 1.),
                                                 ('real_if_close', 0), ('real_if_close', 1e-16),
                                                 ('real_if_close', 1e-12), ('real_if_close', 1),
                                                 ('sqrt', 0),
                                                 ])
def test_elementwise_functions(make_compatible_space, compatible_backend, np_random, function, data_imag):
    leg = make_compatible_space()
    np_func = getattr(np, function)  # e.g. np.real
    tp_func = getattr(tensors, function)  # e.g. tenpy.linalg.tensors.real
    data = np_random.random((leg.dim,))
    if data_imag > 0:
        data = data + data_imag * np_random.random((leg.dim,))

    if isinstance(compatible_backend, FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='diagonal_from_block not implemented'):
            tens = tensors.DiagonalTensor.from_diag(diag=data, first_leg=leg, backend=compatible_backend)
        return  # TODO
    
    tens = tensors.DiagonalTensor.from_diag(diag=data, first_leg=leg, backend=compatible_backend)

    print('scalar input')
    res = tp_func(data[0])
    expect = np_func(data[0])
    npt.assert_array_almost_equal_nulp(res, expect)

    print('DiagonalTensor input')
    res = tp_func(tens).diag_numpy
    expect = np_func(data)
    npt.assert_array_almost_equal_nulp(res, expect)


@pytest.mark.parametrize('which_legs', [[0], [-1], ['b'], ['a', 'b', 'c', 'd'], ['b', -2]])
def test_flip_leg_duality(make_compatible_tensor, which_legs):
    T: tensors.BlockDiagonalTensor = make_compatible_tensor(labels=['a', 'b', 'c', 'd'])

    if isinstance(T.backend, FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='flip_leg_duality not implemented'):
            res = tensors.flip_leg_duality(T, *which_legs)
        return  # TODO
    
    res = tensors.flip_leg_duality(T, *which_legs)
    res.test_sanity()
    flipped = T.get_leg_idcs(which_legs)
    for i in range(T.num_legs):
        if i in flipped:
            assert np.all(res.legs[i].sectors_of_basis == T.legs[i].sectors_of_basis)
            assert res.legs[i].is_dual == (not T.legs[i].is_dual)
        else:
            assert res.legs[i] == T.legs[i]
    T_np = T.to_numpy()
    res_np = res.to_numpy()
    npt.assert_array_almost_equal_nulp(T_np, res_np, 100)


def test_str_repr(make_compatible_tensor, str_max_lines=30, repr_max_lines=30):
    """Check if str and repr work. Automatically, we can only check if they run at all.
    To check if the output is sensible and useful, a human should look at it.
    Run ``pytest -rP -k test_str_repr > output.txt`` to see the output.
    """
    terminal_width = 80
    str_max_len = terminal_width * str_max_lines
    repr_max_len = terminal_width * str_max_lines
    mps_tens = make_compatible_tensor(labels=['vL', 'p', 'vR'])
    local_state = make_compatible_tensor(labels=['p'])
    local_op = make_compatible_tensor(labels=['p', 'p*'])
    for t in [mps_tens, local_state, local_op]:
        print()
        print()
        print('----------------------')
        print('__repr__()')
        print('----------------------')
        res = repr(t)
        assert len(res) <= repr_max_len
        assert res.count('\n') <= repr_max_lines
        print(res)
        
        print()
        print()
        print('----------------------')
        print('__str__()')
        print('----------------------')
        res = str(t)
        assert len(res) <= str_max_len
        assert res.count('\n') <= str_max_lines
        print(res)


def test_Mask(np_random, make_compatible_space, compatible_backend):
    large_leg = make_compatible_space()
    blockmask = np_random.choice([True, False], size=large_leg.dim)
    num_kept = sum(blockmask)

    if not large_leg.symmetry.is_abelian:
        pytest.skip('Need to design a valid blockmask!')

    if isinstance(compatible_backend, FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='mask_from_block not implemented'):
            mask = tensors.Mask.from_blockmask(blockmask, large_leg=large_leg, backend=compatible_backend)
        return  # TODO
    
    mask = tensors.Mask.from_blockmask(blockmask, large_leg=large_leg, backend=compatible_backend)
    mask.test_sanity()

    npt.assert_array_equal(mask.numpymask, blockmask)
    assert mask.large_leg == large_leg
    assert mask.small_leg.dim == np.sum(blockmask)

    # mask2 : same mask, but build from indices
    indices = np.where(blockmask)[0]
    mask2 = tensors.Mask.from_indices(indices, large_leg=large_leg, backend=compatible_backend)
    mask2.test_sanity()
    npt.assert_array_equal(mask2.numpymask, blockmask)
    assert mask.same_mask(mask2)
    assert tensors.almost_equal(mask, mask2)

    # mask3 : different in exactly one entry
    print(f'{indices=}')
    indices3 = indices.copy()
    indices3[len(indices3) // 2] = not indices3[len(indices3) // 2]
    mask3 = tensors.Mask.from_indices(indices3, large_leg=large_leg, backend=compatible_backend)
    mask3.test_sanity()
    assert not mask.same_mask(mask3)
    assert not tensors.almost_equal(mask, mask3)

    # mask4: independent random mask
    blockmask4 = np_random.choice([True, False], size=large_leg.dim)
    mask4 = tensors.Mask.from_blockmask(blockmask4, large_leg=large_leg, backend=compatible_backend)
    mask4.test_sanity()

    mask_all = tensors.Mask.eye(large_leg=large_leg, backend=compatible_backend)
    mask_none = tensors.Mask.zero(large_leg=large_leg, backend=compatible_backend)
    assert mask_all.all()
    assert mask_all.any()
    assert not mask_none.all()
    assert not mask_none.any()
    assert mask.all() == np.all(blockmask)
    assert mask.any() == np.any(blockmask)

    as_tensor_arr = mask.as_Tensor().to_numpy()
    as_tensor_expect = np.zeros((len(blockmask), num_kept))
    as_tensor_expect[indices, np.arange(num_kept)] = 1.
    npt.assert_array_equal(as_tensor_arr, as_tensor_expect)

    npt.assert_array_equal(mask.logical_not().numpymask, np.logical_not(blockmask))
    for op in [operator.and_, operator.eq, operator.ne, operator.or_, operator.xor]:
        res = op(mask, mask4)
        res.test_sanity()
        npt.assert_array_equal(res.numpymask, op(blockmask, blockmask4))
    # illegal usages (those would cast bool(mask))
    with pytest.raises(ValueError):
        _ = mask and mask4
    with pytest.raises(ValueError):
        if mask == mask4:  # this is invalid
            pass
    # legal version:
    if tensors.Mask.all(mask == mask4):
        pass

    eye = tensors.Mask.eye(large_leg=large_leg, backend=compatible_backend)
    eye.test_sanity()
    assert eye.all()
    npt.assert_array_equal(eye.numpymask, np.ones(large_leg.dim, bool))

    diag = tensors.DiagonalTensor.from_diag(blockmask, first_leg=large_leg, backend=compatible_backend)
    diag.test_sanity()
    mask5 = tensors.Mask.from_DiagonalTensor(diag)
    npt.assert_array_equal(mask5.numpymask, mask.numpymask)
    assert tensors.almost_equal(mask5, mask)


@pytest.mark.parametrize('num_legs', [1, 3])
def test_apply_Mask_Tensor(make_compatible_tensor, compatible_backend, num_legs):
    T: tensors.BlockDiagonalTensor = make_compatible_tensor(num_legs=num_legs)

    if not T.symmetry.is_abelian:
        # TODO
        pytest.skip('Need to re-design make_compatible_tensor fixture to generate valid masks.')

    if isinstance(T.backend, FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='mask_from_block not implemented'):
            mask = make_compatible_tensor(legs=[T.legs[0], None], cls=tensors.Mask)
        return  # TODO
    
    mask = make_compatible_tensor(legs=[T.legs[0], None], cls=tensors.Mask)
    masked = T.apply_mask(mask, 0)
    masked.test_sanity()
    npt.assert_array_almost_equal_nulp(T.to_numpy()[mask.numpymask],
                                       masked.to_numpy(),
                                       10)


def test_apply_Mask_DiagonalTensor(make_compatible_tensor, compatible_backend):
    if isinstance(compatible_backend, FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='diagonal_from_block_func not implemented'):
            T: tensors.DiagonalTensor = make_compatible_tensor(cls=tensors.DiagonalTensor)
        return  # TODO

    
    T: tensors.DiagonalTensor = make_compatible_tensor(cls=tensors.DiagonalTensor)
    mask = make_compatible_tensor(legs=[T.legs[0], None], cls=tensors.Mask)
    # mask only one leg
    masked = T.apply_mask(mask, 0)
    assert isinstance(masked, tensors.BlockDiagonalTensor)
    masked.test_sanity()
    npt.assert_array_almost_equal_nulp(T.to_numpy()[mask.numpymask],
                                       masked.to_numpy(),
                                       10)
    # mask both legs
    masked = T._apply_mask_both_legs(mask)
    assert isinstance(masked, tensors.DiagonalTensor)
    masked.test_sanity()
    npt.assert_array_almost_equal_nulp(T.diag_numpy[mask.numpymask],
                                       masked.diag_numpy,
                                       10)


@pytest.mark.parametrize('num_legs', [1, 3])
def test_apply_Mask_ChargedTensor(make_compatible_tensor, num_legs):
    pytest.xfail('Fixture generates ChargedTensor with unspecified dummy_leg_state')
    
    T: tensors.ChargedTensor = make_compatible_tensor(num_legs=num_legs, cls=tensors.ChargedTensor)
    # first leg
    
    if not T.symmetry.is_abelian:
        # TODO
        pytest.skip('Need to re-design make_compatible_tensor fixture to generate valid masks.')

    if isinstance(T.backend, FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='mask_from_block not implemented'):
            mask = make_compatible_tensor(legs=[T.legs[0], None], cls=tensors.Mask)
        return  # TODO
    
    mask = make_compatible_tensor(legs=[T.legs[0], None], cls=tensors.Mask)
    masked = T.apply_mask(mask, 0)
    masked.test_sanity()
    npt.assert_array_almost_equal_nulp(T.to_numpy()[mask.numpymask],
                                       masked.to_numpy(),
                                       10)
    # last leg
    mask = make_compatible_tensor(legs=[T.legs[-1], None], cls=tensors.Mask)
    masked = T.apply_mask(mask, -1)
    masked.test_sanity()
    npt.assert_array_almost_equal_nulp(T.to_numpy()[..., mask.numpymask],
                                       masked.to_numpy(),
                                       10)
