"""A collection of tests for tenpy.linalg.tensors."""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
import numpy as np
import numpy.testing as npt
import pytest
import warnings
import operator

from tenpy.linalg import tensors
from tenpy.linalg.backends.no_symmetry import NoSymmetryBackend
from tenpy.linalg.backends.abelian import AbelianBackend
from tenpy.linalg.backends.nonabelian import NonabelianBackend
from tenpy.linalg.backends.torch import TorchBlockBackend
from tenpy.linalg.backends.numpy import NumpyBlockBackend, NoSymmetryNumpyBackend
from tenpy.linalg.dtypes import Dtype
from tenpy.linalg.spaces import VectorSpace, ProductSpace, _fuse_spaces
from tenpy.linalg.symmetries import ProductSymmetry



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


def test_Tensor_classmethods(backend, vector_space_rng, backend_data_rng, tensor_rng, np_random):
    T: tensors.Tensor = tensor_rng(num_legs=3)  # compatible legs where we can have blocks

    if (isinstance(backend, NonabelianBackend)) and (isinstance(T.symmetry, ProductSymmetry)):
        pytest.xfail(reason='Topo data for ProductSymmetry is missing')
    
    legs = T.legs
    dims = tuple(T.shape)

    numpy_block = T.to_numpy_ndarray()
    dense_block = backend.block_from_numpy(numpy_block)

    print('checking from_dense_block')
    tens = tensors.BlockDiagonalTensor.from_dense_block(dense_block, legs=legs, backend=backend)
    tens.test_sanity()
    data = backend.block_to_numpy(tens.to_dense_block())
    npt.assert_array_equal(data, numpy_block)
    #
    if T.num_parameters < T.parent_space.dim:  # otherwise all blocks are symmetric
        non_symmetric_block = dense_block + tens.backend.block_random_uniform(dims, dtype=T.dtype)
        with pytest.raises(ValueError, match='Block is not symmetric'):
            _ = tensors.BlockDiagonalTensor.from_dense_block(non_symmetric_block, legs=legs, backend=backend)

    print('checking from numpy')
    tens = tensors.BlockDiagonalTensor.from_dense_block(numpy_block, legs=legs, backend=backend)
    tens.test_sanity()
    data = tens.to_numpy_ndarray()
    npt.assert_array_equal(data, numpy_block)

    # TODO from_block_func, from_numpy_func

    # TODO random_uniform, random_normal

    print('checking zero')
    tens = tensors.BlockDiagonalTensor.zero(legs, backend=backend)
    tens.test_sanity()
    npt.assert_array_equal(tens.to_numpy_ndarray(), np.zeros(dims))

    print('checking eye')
    tens = tensors.BlockDiagonalTensor.eye(legs[0], backend=backend)
    tens.test_sanity()
    npt.assert_array_equal(tens.to_numpy_ndarray(), np.eye(legs[0].dim))
    tens = tensors.BlockDiagonalTensor.eye(legs[:2], backend=backend)
    tens.test_sanity()
    npt.assert_array_equal(tens.to_numpy_ndarray(), np.eye(np.prod(dims[:2])).reshape(dims[:2] + dims[:2]))


def test_Tensor_methods(backend, vector_space_rng, backend_data_rng, tensor_rng):
    T = tensor_rng(num_legs=3)  # compatible legs where we can have blocks
    legs = T.legs
    dims = tuple(T.shape)

    data1 = T.data
    data2 = backend_data_rng(legs)

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
    triv_legs = [vector_space_rng(1, 1) for i in range(2)]
    for leg in triv_legs[:]:
        triv_legs.append(leg.dual)
    assert all(leg.dim == 1 for leg in triv_legs)
    data4 = backend_data_rng(triv_legs)
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
    dense1 = tens1.to_numpy_ndarray()
    dense2 = tens2.to_numpy_ndarray()
    dense3 = tens3.to_numpy_ndarray()

    print('check addition + multiplication')
    neg_t3 = -tens3
    npt.assert_array_equal(neg_t3.to_numpy_ndarray(), -dense3)
    a = 42
    b = 17
    with pytest.raises(ValueError) as err:
        res = a * tens1 - b * tens2
    assert "labelled tensors must be *fully* labelled" in err.value.args[0]  # TODO: check other config values?
    tens1.set_labels(['foo', 'a', 'b'])
    tens2.set_labels(['foo', 'a', 'b'])
    res = a * tens1 - b * tens2
    npt.assert_almost_equal(res.to_numpy_ndarray(), a * dense1 - b * dense2)
    res = tens1 / a + tens2 / b
    npt.assert_almost_equal(res.to_numpy_ndarray(), dense1 / a + dense2 / b)
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


def test_Tensor_tofrom_flat_block_trivial_sector(vector_space_rng, tensor_rng):
    # TODO move to some other tests after restructuring
    for n in range(10):
        leg = vector_space_rng()
        block_size = leg.sector_multiplicity(leg.symmetry.trivial_sector)
        if block_size > 0:
            break
    else:
        pytest.xfail('Failed to generate a vector space that has the trivial sector')

    tens = tensor_rng(legs=[leg], labels=['a'])
    block = tens.to_flat_block_trivial_sector()
    assert tens.backend.block_shape(block) == (block_size,)
    tens2 = tensors.BlockDiagonalTensor.from_flat_block_trivial_sector(leg=leg, block=block, backend=tens.backend, label='a')
    tens2.test_sanity()
    assert tensors.almost_equal(tens, tens2)
    block2 = tens2.to_flat_block_trivial_sector()
    npt.assert_array_almost_equal_nulp(tens.backend.block_to_numpy(block),
                                       tens.backend.block_to_numpy(block2),
                                       100)


def test_ChargedTensor_tofrom_flat_block_single_sector(vector_space_rng, symmetry_sectors_rng, tensor_rng):
    # TODO move to some other tests after restructuring
    leg = vector_space_rng()
    sector = symmetry_sectors_rng(1)[0]

    block_size = leg.sector_multiplicity(sector)
    if block_size == 0:
        block_size = 4
        leg = VectorSpace.from_sectors(
            symmetry=leg.symmetry, sectors=np.concatenate([leg.sectors, sector[None, :]]),
            multiplicities=np.concatenate([leg.multiplicities, np.array([block_size])])
        )

    dummy_leg = VectorSpace(symmetry=leg.symmetry, sectors=[sector]).dual
    tens = tensors.ChargedTensor(invariant_part=tensor_rng(legs=[leg, dummy_leg]))

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


def test_from_block(block_backend):
    from tenpy.linalg.symmetries import z4_symmetry
    from tenpy.linalg.backends.backend_factory import get_backend
    backend = get_backend(symmetry=z4_symmetry, block_backend=block_backend)

    print('by constructing basis')
    q0, q1, q2, q3 = z4_symmetry.all_sectors()
    basis1 = [q3, q3, q2, q0, q3, q2]  # basis_perm [3, 2, 5, 0, 1, 4]
    basis2 = [q2, q0, q1, q2, q3, q0, q1]  # basis_perm [1, 5, 2, 6, 0, 3, 4]
    s1 = VectorSpace.from_basis(z4_symmetry, basis1)  # sectors = [0, 2, 3]
    s2 = VectorSpace.from_basis(z4_symmetry, basis2)  # sectors = [0, 1, 2, 3]

    #      q: 2,  0,  1,  2,  3,  0,  1      q
    data = [[ 0,  0,  1,  0,  0,  0,  2],  # 3
            [ 0,  0,  3,  0,  0,  0,  4],  # 3
            [ 5,  0,  0,  6,  0,  0,  0],  # 2
            [ 0,  7,  0,  0,  0,  8,  0],  # 0
            [ 0,  0,  9,  0,  0,  0, 10],  # 3
            [11,  0,  0, 12,  0,  0,  0]]  # 2
    block = backend.block_from_numpy(np.asarray(data, dtype=float))
    t = tensors.BlockDiagonalTensor.from_dense_block(block, [s1, s2])

    # block_i is the one with sector q_i on s1
    block_0 = [[7, 8]]
    block_1 = [[1, 2], [3, 4], [9, 10]]
    block_2 = [[5, 6], [11, 12]]
    block_3 = [[]]
    expect_blocks = [block_0, block_1, block_2, block_3]  # can be indexed by block_inds[1]
    expect_blocks = [backend.block_from_numpy(np.asarray(b, dtype=float)) for b in expect_blocks]

    for block, ind in zip(t.data.blocks, t.data.block_inds):
        print(block)
        print(expect_blocks[ind[1]])
        assert backend.block_allclose(block, expect_blocks[ind[1]], rtol=1e-5, atol=1e-8)


def test_tdot(backend, vector_space_rng, backend_data_rng, tensor_rng):
    # define legs such that a tensor with the following combinations all allow non-zero num_parameters
    # [a, b] , [a, b, c] , [a, b, d]
    a = vector_space_rng(3, 3)
    b = tensor_rng([a, None], 2, max_num_blocks=3, max_block_size=3).legs[-1]
    c = tensor_rng([a, b, None], 3, max_num_blocks=3, max_block_size=3).legs[-1]
    d = tensor_rng([a, b, None], 3, max_num_blocks=3, max_block_size=3).legs[-1]
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
    data_ = [backend_data_rng(l) for l in legs_]
    tensors_ = [tensors.BlockDiagonalTensor(data, legs, 0, backend, labels) for data, legs, labels in
                zip(data_, legs_, labels_)]
    for n, t in enumerate(tensors_):
        # make sure we are defining tensors which actually contain blocks and are not just zero by
        # charge conservation
        assert t.num_parameters > 0, f'tensor {n} has 0 free parameters'
    
    dense_ = [t.to_numpy_ndarray() for t in tensors_]

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
        res1 = tensors.tdot(tensors_[i], tensors_[j], ax_i, ax_j)
        res2 = tensors.tdot(tensors_[i], tensors_[j], lbl_i, lbl_j)
        if len(expect.shape) > 0:
            res1.test_sanity()
            res2.test_sanity()
            res1_d = res1.to_numpy_ndarray()
            res2_d = res2.to_numpy_ndarray()
            npt.assert_array_almost_equal(res1_d, expect)
            npt.assert_array_almost_equal(res2_d, expect)
        else: # got scalar, but we can compare it to 0-dim ndarray
            npt.assert_almost_equal(res1, expect)
            npt.assert_almost_equal(res2, expect)

    # TODO check that trying to contract incompatible legs raises
    #  - opposite is_dual but different dim
    #  - opposite is_dual and dim but different sectors
    #  - same dim and sectors but same is_dual


def test_outer(tensor_rng):
    tensors_ = [tensor_rng(labels=labels) for labels in [['a'], ['b'], ['c', 'd']]]
    dense_ = [t.to_numpy_ndarray() for t in tensors_]

    for i, j  in [(0, 1), (0, 2), (0, 0), (2, 2)]:
        print(i, j)
        expect = np.tensordot(dense_[i], dense_[j], axes=0)
        res = tensors.outer(tensors_[i], tensors_[j])
        res.test_sanity()
        npt.assert_array_almost_equal(res.to_numpy_ndarray(), expect)
        if i != j:
            assert res.labels_are(*(tensors_[i].labels + tensors_[j].labels))
        else:
            assert all(l is None for l in res.labels)


def test_permute_legs(tensor_rng):
    labels = list('abcd')
    t = tensor_rng(labels=labels)
    d = t.to_numpy_ndarray()
    for perm in [[0, 2, 1, 3], [3, 2, 1, 0], [1, 0, 3, 2], [0, 1, 2, 3], [0, 3, 2, 1]]:
        expect = d.transpose(perm)
        res = t.permute_legs(perm)
        res.test_sanity()
        npt.assert_array_equal(res.to_numpy_ndarray(), expect)
        assert res.labels == [labels[i] for i in perm]


def test_inner(tensor_rng):
    t0 = tensor_rng(labels=['a'], real=False)
    t1 = tensor_rng(legs=t0.legs, labels=t0.labels, real=False)
    t2 = tensor_rng(labels=['a', 'b', 'c'], real=False)
    t3 = tensor_rng(legs=t2.legs, labels=t2.labels, real=False)

    for t_i, t_j, perm in [(t0, t1, ['a']), (t2, t3, ['b', 'c', 'a'])]:
        d_i = t_i.to_numpy_ndarray()
        d_j = t_j.to_numpy_ndarray()

        expect = np.inner(d_i.flatten().conj(), d_j.flatten())
        if t_j.num_legs > 0:
            t_j = t_j.permute_legs(perm)  # transpose should be reverted in inner()
        res = tensors.inner(t_i, t_j)
        npt.assert_allclose(res, expect)

        expect = np.linalg.norm(d_i) **2
        res = tensors.inner(t_i, t_i)
        npt.assert_allclose(res, expect)


def test_trace(backend, vector_space_rng, tensor_rng):
    a = vector_space_rng(3, 3)
    b = vector_space_rng(4, 3)
    t1 = tensor_rng(legs=[a, a.dual], labels=['a', 'a*'])
    d1 = t1.to_numpy_ndarray()
    t2 = tensor_rng(legs=[a, b, a.dual, b.dual], labels=['a', 'b', 'a*', 'b*'])
    d2 = t2.to_numpy_ndarray()
    t3 = tensor_rng(legs=[a, None, b, a.dual, b.dual], labels=['a', 'c', 'b', 'a*', 'b*'])
    d3 = t3.to_numpy_ndarray()

    print('single legpair - full')
    expected = np.trace(d1, axis1=0, axis2=1)
    res = tensors.trace(t1, 'a*', 'a')
    npt.assert_array_almost_equal_nulp(res, expected, 100)

    print('single legpair - partial')
    expected = np.trace(d2, axis1=1, axis2=3)
    res = tensors.trace(t2, 'b*', 'b')
    res.test_sanity()
    assert res.labels_are('a', 'a*')
    npt.assert_array_almost_equal_nulp(res.to_numpy_ndarray(), expected, 100)

    print('two legpairs - full')
    expected = np.trace(d2, axis1=1, axis2=3).trace(axis1=0, axis2=1)
    res = tensors.trace(t2, ['a', 'b*'], ['a*', 'b'])
    npt.assert_array_almost_equal_nulp(res, expected, 100)

    print('two legpairs - partial')
    expected = np.trace(d3, axis1=2, axis2=4).trace(axis1=0, axis2=2)
    res = tensors.trace(t3, ['a', 'b*'], ['a*', 'b'])
    res.test_sanity()
    assert res.labels_are('c')
    npt.assert_array_almost_equal_nulp(res.to_numpy_ndarray(), expected, 100)


def test_conj_hconj(tensor_rng):
    tens = tensor_rng(labels=['a', 'b', None], real=False)
    expect = np.conj(tens.to_numpy_ndarray())
    assert np.linalg.norm(expect.imag) > 0 , "expect complex data!"
    res = tensors.conj(tens)
    res.test_sanity()
    assert res.labels == ['a*', 'b*', None]
    assert [l1.can_contract_with(l2) for l1, l2 in zip(res.legs, tens.legs)]
    assert np.allclose(res.to_numpy_ndarray(), expect)

    print('hconj 1-site operator')
    leg_a = tens.legs[0]
    op = tensor_rng(legs=[leg_a, leg_a.dual], labels=['p', 'p*'], real=False)
    op_hc = tensors.hconj(op)
    op_hc.test_sanity()
    assert op_hc.labels == op.labels
    assert op_hc.legs == op.legs
    _ = op + op_hc  # just check if it runs
    npt.assert_array_equal(op_hc.to_numpy_ndarray(), np.conj(op.to_numpy_ndarray()).T)
    
    print('hconj 2-site op')
    leg_b = tens.legs[1]
    op2 = tensor_rng(legs=[leg_a, leg_b.dual, leg_a.dual, leg_b], labels=['a', 'b*', 'a*', 'b'],
                     real=False)
    op2_hc = op2.hconj(['a', 'b'], ['a*', 'b*'])
    op2_hc.test_sanity()
    assert op2_hc.labels == op2.labels
    assert op2_hc.legs == op2.legs
    _ = op2 + op2_hc  # just check if it runs
    expect = np.transpose(np.conj(op2.to_numpy_ndarray()), [2, 3, 0, 1])
    npt.assert_array_equal(op2_hc.to_numpy_ndarray(), expect)
    

def test_combine_split(tensor_rng):
    tens = tensor_rng(labels=['a', 'b', 'c', 'd'], max_num_blocks=5, max_block_size=5)
    dense = tens.to_numpy_ndarray()
    d0, d1, d2, d3 = dims = tuple(tens.shape)

    print('check by idx')
    res = tensors.combine_legs(tens, [1, 2])
    res.test_sanity()
    assert res.labels == ['a', '(b.c)', 'd']
    # note: dense reshape is not enough to check expect, since we have permutation in indices.
    # hence, we only check that we get back the same after split
    split = tensors.split_legs(res, 1)
    split.test_sanity()
    assert split.labels == ['a', 'b', 'c', 'd']
    npt.assert_equal(split.to_numpy_ndarray(), dense)

    print('check by label')
    res = tensors.combine_legs(tens, ['b', 'd'])
    res.test_sanity()
    assert res.labels == ['a', '(b.d)', 'c']
    split = tensors.split_legs(res, '(b.d)')
    split.test_sanity()
    assert split.labels == ['a', 'b', 'd', 'c']
    assert np.allclose(split.to_numpy_ndarray(), dense.transpose([0, 1, 3, 2]))

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
    npt.assert_equal(split.to_numpy_ndarray(), dense.transpose([1, 3, 2, 0]))

    print('check _fuse_spaces')
    sectors1, mults1, fusion_outcomes_sort1, metadata1 = _fuse_spaces(
        symmetry=tens.symmetry, spaces=tens.get_legs(['b', 'd']), _is_dual=False
    )
    sectors2, mults2, fusion_outcomes_sort2, metadata2 = tens.backend._fuse_spaces(
        symmetry=tens.symmetry, spaces=tens.get_legs(['b', 'd']), _is_dual=False
    )
    npt.assert_array_equal(fusion_outcomes_sort1, fusion_outcomes_sort2)
    npt.assert_array_equal(sectors1, sectors2)
    npt.assert_array_equal(mults1, mults2)
    assert len(metadata1) == 0
    assert len(metadata2) == (3 if isinstance(tens.backend, AbelianBackend) else 0)

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
        assert np.allclose(split.to_numpy_ndarray(), dense.transpose([0, 1, 3, 2]))


@pytest.mark.xfail  # TODO
def test_combine_legs_basis_trafo(tensor_rng):
    tens = tensor_rng(labels=['a', 'b', 'c'], max_num_blocks=5, max_block_size=5)
    a, b, c = tens.shape
    dense = tens.to_numpy_ndarray()  # [a, b, c]
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


def test_is_scalar(backend, tensor_rng, vector_space_rng):
    for s in [1, 0., 1.+2.j, np.int64(123), np.float64(2.345), np.complex128(1.+3.j)]:
        assert tensors.is_scalar(s)
    triv_leg = vector_space_rng(1, 1)
    scalar_tens = tensor_rng(legs=[triv_leg, triv_leg.dual])
    assert tensors.is_scalar(scalar_tens)
    # generate non-scalar tensor
    for i in range(20):
        non_scalar_tens = tensor_rng(num_legs=3)
        if any(d > 1 for d in non_scalar_tens.shape):  # non-trivial
            assert not tensors.is_scalar(non_scalar_tens)
            break
    else:  # didn't break
        pytest.skip("can't generate non-scalar tensor")


def test_norm(tensor_rng):
    tens = tensor_rng(real=False)
    expect = np.linalg.norm(tens.to_numpy_ndarray())
    res = tensors.norm(tens)
    assert np.allclose(res, expect)


def test_almost_equal(tensor_rng, np_random):
    for i in range(10):
        t1 = tensor_rng(labels=['a', 'b', 'c'], real=False)
        t_diff = tensor_rng(t1.legs, labels=['a', 'b', 'c'])
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
    t1 = tensors.DiagonalTensor.from_diag(data1, leg)
    t2 = tensors.DiagonalTensor.from_diag(data2, leg)
    assert tensors.almost_equal(t1, t2)
    assert not tensors.almost_equal(t1, t2, atol=1e-10, rtol=1e-10)

    # TODO check all combinations of tensor types...


def test_squeeze_legs(tensor_rng, symmetry):
    for i in range(10):
        triv_leg = VectorSpace(symmetry, symmetry.trivial_sector[np.newaxis, :], np.ones((1,)))
        assert triv_leg.is_trivial
        tens = tensor_rng([None, triv_leg, None, triv_leg.dual, triv_leg], labels=list('abcde'))
        if not tens.legs[0].is_trivial and not tens.legs[2].is_trivial:
            break
    else:
        pytest.skip("can't generate non-triv leg")
    dense = tens.to_numpy_ndarray()

    print('squeezing all legs (default arg)')
    res = tensors.squeeze_legs(tens)
    res.test_sanity()
    assert res.labels == ['a', 'c']
    npt.assert_array_equal(res.to_numpy_ndarray(), dense[:, 0, :, 0, 0])

    print('squeeze specific leg by idx')
    res = tensors.squeeze_legs(tens, 1)
    res.test_sanity()
    assert res.labels == ['a', 'c', 'd', 'e']
    npt.assert_array_equal(res.to_numpy_ndarray(), dense[:, 0, :, :, :])

    print('squeeze legs by labels')
    res = tensors.squeeze_legs(tens, ['b', 'e'])
    res.test_sanity()
    assert res.labels == ['a', 'c', 'd']
    npt.assert_array_equal(res.to_numpy_ndarray(), dense[:, 0, :, :, 0])


def test_add_trivial_leg(tensor_rng):
    A = tensor_rng(num_legs=2, labels=['a', 'b'])
    B = tensors.add_trivial_leg(A, 'c', is_dual=True)
    B.test_sanity()
    B = tensors.add_trivial_leg(B, 'xY', pos=1)
    B.test_sanity()
    assert B.labels == ['a', 'xY', 'b', 'c']
    assert [leg.is_dual for leg in B.legs] == [A.legs[0].is_dual, False, A.legs[1].is_dual, True]
    expect = A.to_numpy_ndarray()[:, None, :, None]
    B_np = B.to_numpy_ndarray()
    npt.assert_array_equal(B_np, expect)
    C = B.squeeze_legs()
    assert tensors.almost_equal(A, C)


def test_scale_axis(backend, vector_space_rng, backend_data_rng, tensor_rng):
    # TODO eventually this will be covered by tdot tests, when allowing combinations of Tensor and DiagonalTensor
    #  But I want to use it already now to debug backend.scale_axis()
    a = vector_space_rng(max_num_blocks=4, max_block_size=4)
    b = vector_space_rng(max_num_blocks=4, max_block_size=4)
    t = tensor_rng([a, b, None], num_legs=3, max_num_blocks=4, max_block_size=4)
    d = tensors.DiagonalTensor.random_uniform(a, second_leg_dual=True, backend=backend)
    expect = np.tensordot(t.to_numpy_ndarray(), d.to_numpy_ndarray(), (0, 1))
    res = tensors.tdot(t, d, 0, 1).to_numpy_ndarray()
    npt.assert_almost_equal(expect, res)


def test_detect_sectors_from_block(backend, symmetry, symmetry_sectors_rng, np_random):
    num_sectors = int(min(4, symmetry.num_sectors))
    leg_dim = 5
    sectors = symmetry_sectors_rng(num_sectors, sort=True)

    which_sectors_a = np_random.integers(num_sectors, size=(leg_dim,))  # indices of sectors
    which_sectors_b = np_random.integers(num_sectors, size=(leg_dim + 1,))
    sectors_of_basis_a = sectors[which_sectors_a]
    sectors_of_basis_b = sectors[which_sectors_b]
    a = VectorSpace.from_basis(symmetry=symmetry, sectors_of_basis=sectors_of_basis_a)
    b = VectorSpace.from_basis(symmetry=symmetry, sectors_of_basis=sectors_of_basis_b)

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

    block = backend.block_from_numpy(data)
    detected = tensors.detect_sectors_from_block(block, legs=[a, b], backend=backend)
    npt.assert_array_equal(detected[0], sectors[target_sector_a])
    npt.assert_array_equal(detected[1], sectors[target_sector_b])

    print('check an explicit state')
    if num_sectors >= 4:
        #                         0  1  2  3  4  5  6  7  8  9
        which_sectors = np.array([0, 1, 3, 0, 2, 1, 1, 2, 0, 3])
        space = VectorSpace(symmetry=symmetry,
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
                backend.block_from_numpy(data), legs=[space], backend=backend
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
def test_elementwise_functions(vector_space_rng, np_random, function, data_imag):
    leg = vector_space_rng()
    np_func = getattr(np, function)  # e.g. np.real
    tp_func = getattr(tensors, function)  # e.g. tenpy.linalg.tensors.real
    data = np_random.random((leg.dim,))
    if data_imag > 0:
        data = data + data_imag * np_random.random((leg.dim,))
    tens = tensors.DiagonalTensor.from_diag(diag=data, first_leg=leg)

    print('scalar input')
    res = tp_func(data[0])
    expect = np_func(data[0])
    npt.assert_array_almost_equal_nulp(res, expect)

    print('DiagonalTensor input')
    res = tp_func(tens).diag_numpy
    expect = np_func(data)
    npt.assert_array_almost_equal_nulp(res, expect)


@pytest.mark.parametrize('which_legs', [[0], [-1], ['b'], ['a', 'b', 'c', 'd'], ['b', -2]])
def test_flip_leg_duality(tensor_rng, which_legs):
    T: tensors.BlockDiagonalTensor = tensor_rng(labels=['a', 'b', 'c', 'd'])
    res = tensors.flip_leg_duality(T, *which_legs)
    res.test_sanity()
    flipped = T.get_leg_idcs(which_legs)
    for i in range(T.num_legs):
        if i in flipped:
            assert np.all(res.legs[i].sectors_of_basis == T.legs[i].sectors_of_basis)
            assert res.legs[i].is_dual == (not T.legs[i].is_dual)
        else:
            assert res.legs[i] == T.legs[i]
    T_np = T.to_numpy_ndarray()
    res_np = res.to_numpy_ndarray()
    npt.assert_array_almost_equal_nulp(T_np, res_np, 100)


def demo_repr():
    # this is intended to generate a bunch of demo reprs
    # can not really make this an automated test, the point is for a human to have a look
    # and decide if the output is useful, concise, correct, etc.
    #
    # run e.g. via the following command
    # python -c "from test_tensors import demo_repr; demo_repr()"
    from tests import conftest
    from tenpy.linalg.backends.backend_factory import get_backend

    labels = ['vL', 'p', 'vR*', 'a', 'q', 'x', 'y', 'z', 'i', 'o']
    
    for symmetry in conftest.symmetry._pytestfixturefunction.params:
        backend = get_backend(symmetry)
        for num_legs in [2, 4, 10]:
            legs = [conftest.random_vector_space(symmetry, max_num_blocks=3, max_block_size=2) for _ in range(num_legs)]
            tens = tensors.BlockDiagonalTensor.random_uniform(legs, backend=backend, labels=labels[:num_legs])
            print()
            print('=' * 70)
            print()
            print(str(tens))


def test_Mask(np_random, vector_space_rng, backend):
    large_leg = vector_space_rng()
    blockmask = np_random.choice([True, False], size=large_leg.dim)
    num_kept = sum(blockmask)
    mask = tensors.Mask.from_blockmask(blockmask, large_leg=large_leg, backend=backend)
    mask.test_sanity()

    npt.assert_array_equal(mask.numpymask, blockmask)
    assert mask.large_leg == large_leg
    assert mask.small_leg.dim == np.sum(blockmask)

    # mask2 : same mask, but build from indices
    indices = np.where(blockmask)[0]
    mask2 = tensors.Mask.from_indices(indices, large_leg=large_leg, backend=backend)
    mask2.test_sanity()
    npt.assert_array_equal(mask2.numpymask, blockmask)
    assert mask.same_mask(mask2)
    assert tensors.almost_equal(mask, mask2)

    # mask3 : different in exactly one entry
    print(f'{indices=}')
    indices3 = indices.copy()
    indices3[len(indices3) // 2] = not indices3[len(indices3) // 2]
    mask3 = tensors.Mask.from_indices(indices3, large_leg=large_leg, backend=backend)
    mask3.test_sanity()
    assert not mask.same_mask(mask3)
    assert not tensors.almost_equal(mask, mask3)

    # mask4: independent random mask
    blockmask4 = np_random.choice([True, False], size=large_leg.dim)
    mask4 = tensors.Mask.from_blockmask(blockmask4, large_leg=large_leg, backend=backend)
    mask4.test_sanity()

    mask_all = tensors.Mask.eye(large_leg=large_leg, backend=backend)
    mask_none = tensors.Mask.zero(large_leg=large_leg, backend=backend)
    assert mask_all.all()
    assert mask_all.any()
    assert not mask_none.all()
    assert not mask_none.any()
    assert mask.all() == np.all(blockmask)
    assert mask.any() == np.any(blockmask)

    as_tensor_arr = mask.as_Tensor().to_numpy_ndarray()
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

    eye = tensors.Mask.eye(large_leg=large_leg, backend=backend)
    eye.test_sanity()
    assert eye.all()
    npt.assert_array_equal(eye.numpymask, np.ones(large_leg.dim, bool))

    diag = tensors.DiagonalTensor.from_diag(blockmask, first_leg=large_leg, backend=backend)
    diag.test_sanity()
    mask5 = tensors.Mask.from_DiagonalTensor(diag)
    npt.assert_array_equal(mask5.numpymask, mask.numpymask)
    assert tensors.almost_equal(mask5, mask)


@pytest.mark.parametrize('num_legs', [1, 3])
def test_apply_Mask_Tensor(tensor_rng, num_legs):
    T: tensors.BlockDiagonalTensor = tensor_rng(num_legs=num_legs)
    mask = tensor_rng(legs=[T.legs[0]], cls=tensors.Mask)
    masked = T.apply_mask(mask, 0)
    masked.test_sanity()
    npt.assert_array_equal(T.to_numpy_ndarray()[mask.numpymask], masked.to_numpy_ndarray())


def test_apply_Mask_DiagonalTensor(tensor_rng):
    T: tensors.DiagonalTensor = tensor_rng(cls=tensors.DiagonalTensor)
    mask = tensor_rng(legs=[T.legs[0]], cls=tensors.Mask)
    # mask only one leg
    masked = T.apply_mask(mask, 0)
    assert isinstance(masked, tensors.BlockDiagonalTensor)
    masked.test_sanity()
    npt.assert_array_equal(T.to_numpy_ndarray()[mask.numpymask], masked.to_numpy_ndarray())
    # mask both legs
    masked = T._apply_mask_both_legs(mask)
    assert isinstance(masked, tensors.DiagonalTensor)
    masked.test_sanity()
    npt.assert_array_equal(T.diag_numpy[mask.numpymask], masked.diag_numpy)


@pytest.mark.parametrize('num_legs', [1, 3])
def test_apply_Mask_ChargedTensor(tensor_rng, num_legs):
    T: tensors.ChargedTensor = tensor_rng(num_legs=num_legs, cls=tensors.ChargedTensor)
    # first leg
    mask = tensor_rng(legs=[T.legs[0]], cls=tensors.Mask)
    masked = T.apply_mask(mask, 0)
    masked.test_sanity()
    npt.assert_array_equal(T.to_numpy_ndarray()[mask.numpymask], masked.to_numpy_ndarray())
    # last leg
    mask = tensor_rng(legs=[T.legs[-1]], cls=tensors.Mask)
    masked = T.apply_mask(mask, -1)
    masked.test_sanity()
    npt.assert_array_equal(T.to_numpy_ndarray()[..., mask.numpymask], masked.to_numpy_ndarray())
