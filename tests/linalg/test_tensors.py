"""A collection of tests for tenpy.linalg.tensors."""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
import numpy as np
import pytest

from tenpy.linalg import tensors
from tenpy.linalg.backends.torch import TorchBlockBackend
from tenpy.linalg.backends.numpy import NumpyBlockBackend
from tenpy.linalg.symmetries.spaces import VectorSpace



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


def test_Tensor_classmethods(backend, vector_space_rng, backend_data_rng, np_random):
    legs = [vector_space_rng(d, 4, backend.VectorSpaceCls) for d in (3, 1, 7)]
    dims = tuple(leg.dim for leg in legs)

    numpy_block = np_random.normal(dims)
    dense_block = backend.block_from_numpy(numpy_block)

    print('checking from_dense_block')
    tens = tensors.Tensor.from_dense_block(dense_block, backend=backend)
    data = backend.block_to_numpy(tens.to_dense_block())
    assert np.allclose(data, numpy_block)

    print('checking from_numpy')
    tens = tensors.Tensor.from_numpy(numpy_block, backend=backend)
    data = tens.to_numpy_ndarray()
    assert np.allclose(data, numpy_block)

    # TODO from_block_func, from_numpy_func

    # TODO random_uniform, random_normal

    print('checking zero')
    tens = tensors.Tensor.zero(dims, backend=backend)
    assert np.allclose(tens.to_numpy_ndarray(), np.zeros(dims))
    tens = tensors.Tensor.zero(legs, backend=backend)
    assert np.allclose(tens.to_numpy_ndarray(), np.zeros(dims))

    print('checking eye')
    tens = tensors.Tensor.eye(legs[0], backend=backend)
    assert np.allclose(tens.to_numpy_ndarray(), np.eye(legs[0].dim))
    tens = tensors.Tensor.eye(legs[:2], backend=backend)
    assert np.allclose(tens.to_numpy_ndarray(), np.eye(np.prod(dims[:2])).reshape(dims[:2] + dims[:2]))


def test_Tensor_methods(backend, vector_space_rng, backend_data_rng):
    legs = [vector_space_rng(d, 4, backend.VectorSpaceCls) for d in (3, 1, 7)]
    dims = tuple(leg.dim for leg in legs)

    data1 = backend_data_rng(legs)
    data2 = backend_data_rng(legs)


    print('checking __init__ with labels=None')
    tens1 = tensors.Tensor(data1, legs=legs, backend=backend, labels=None)
    tens1.test_sanity()

    print('checking __init__, partially labelled')
    labels2 = [None, 'a', 'b']
    tens2 = tensors.Tensor(data2, legs=legs, backend=backend, labels=labels2)
    tens2.test_sanity()

    print('checking __init__, fully labelled')
    labels3 = ['foo', 'a', 'b']
    tens3 = tensors.Tensor(data1, legs=legs, backend=backend, labels=labels3)
    tens3.test_sanity()

    check_shape(tens1.shape, dims=dims, labels=[None, None, None])
    check_shape(tens2.shape, dims=dims, labels=labels2)
    check_shape(tens3.shape, dims=dims, labels=labels3)

    print('check size')
    assert tens3.size == np.prod(dims)

    # TODO reintroduce when implemented
    # print('check num_parameters')
    # assert tens3.num_parameters == prod(data.shape)

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
    triv_legs = [vector_space_rng(1, 1, backend.VectorSpaceCls) for i in range(2)]
    for leg in triv_legs[:]:
        triv_legs.append(leg.dual)
    assert all(leg.dim == 1 for leg in triv_legs)
    data4 = backend_data_rng(triv_legs)
    tens4 = tensors.Tensor(data4, backend=backend, legs=triv_legs)
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
    assert np.allclose(neg_t3.to_numpy_ndarray(), -dense3)
    a = 42
    b = 17
    res = a * tens1 - b * tens2
    assert np.allclose(res.to_numpy_ndarray(), a * dense1 - b * dense2)
    res = tens1 / a + tens2 / b
    assert np.allclose(res.to_numpy_ndarray(), dense1 / a + dense2 / b)
    # TODO check strict label behavior!

    with pytest.raises(TypeError):
        tens1 == tens2

    print('check converisions, float, complex, array')
    assert isinstance(float(tens4), float)
    assert np.allclose(float(tens4), float(tens4_item))
    assert isinstance(complex(tens4 + 2.j * tens4), complex)
    assert np.allclose(complex(tens4 + 2.j * tens4), complex(tens4_item + 2.j * tens4_item))
    # TODO check that float of a complex tensor raises a warning
    t1_np = np.asarray(tens1)
    assert np.allclose(t1_np, dense1)


def test_tdot(backend, vector_space_rng, backend_data_rng):
    a, b, c, d = [vector_space_rng(d, 3, backend.VectorSpaceCls) for d in [3, 7, 5, 9]]
    legs_ = [[a, b, c],
             [b.dual, c.dual, d.dual],
             [a.dual, b.dual],
             [c.dual, b.dual],
             [c.dual, a.dual, b.dual]]
    labels_ = [['a', 'b', 'c'],
               ['b*', 'c*', 'd*'],
               ['a*', 'b*'],
               ['c*', 'b*'],
               ['c*', 'a*', 'b*']]
    data_ = [backend_data_rng(l) for l in legs_]
    tensors_ = [tensors.Tensor(data, legs, backend, labels) for data, legs, labels in
                zip(data_, legs_, labels_)]
    dense_ = [t.to_numpy_ndarray() for t in tensors_]

    checks = [("single leg", 0, 1, 1, 0, 'b', 'b*'),
              ("two legs", 0, 1, [1, 2], [0, 1], ['b', 'c'], ['b*', 'c*']),
              ("all legs of first tensor", 2, 0, [1, 0], [1, 0], ['a*', 'b*'], ['a', 'b']),
              ("all legs of second tensor", 0, 3, [1, 2], [1, 0], ['b', 'c'], ['b*', 'c*']),
              ("scalar result / inner()", 0, 4, [0, 1, 2], [1, 2, 0], ['a', 'b', 'c'], ['a*', 'b*', 'c*']),
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
            res1 = res1.to_numpy_ndarray()
            res2 = res2.to_numpy_ndarray()
        # else: got scalar, but we can compare it to 0-dim ndarray
        assert np.allclose(res1, expect)
        assert np.allclose(res2, expect)

    # TODO check that trying to contract incompatible legs raises
    #  - opposite is_dual but different dim
    #  - opposite is_dual and dim but different sectors
    #  - same dim and sectors but same is_dual


# TODO (JH): continue to fix tests below to work with new fixtures for any backend
def test_outer(some_backend):
    backend = some_backend
    data1 = np.random.random([3, 5])
    data2 = np.random.random([4, 8])
    t1 = tensors.Tensor.from_numpy(data1, backend=backend, labels=['a', 'f'])
    t2 = tensors.Tensor.from_numpy(data2, backend=backend, labels=['g', 'b'])
    expect = data1[:, :, None, None] * data2[None, None, :, :]
    res = tensors.outer(t1, t2)
    assert np.allclose(expect, res.data)
    assert res.labels_are('a', 'f', 'g', 'b')


def test_inner(some_backend):
    backend = some_backend
    data1 = np.random.random([3, 5]) + 1.j * np.random.random([3, 5])
    data2 = np.random.random([3, 5]) + 1.j * np.random.random([3, 5])
    data3 = np.random.random([5, 3]) + 1.j * np.random.random([5, 3])
    t1 = tensors.Tensor.from_numpy(data1, backend=backend)
    t2 = tensors.Tensor.from_numpy(data2, backend=backend, labels=['a', 'b'])
    t3 = tensors.Tensor.from_numpy(data3, backend=backend, labels=['b', 'a'])
    expect1 = np.tensordot(np.conj(data1), data2, ([0, 1], [0, 1]))
    res1 = tensors.inner(t1, t2)
    assert np.allclose(expect1, res1)
    expect2 = np.tensordot(np.conj(data2), data3, ([0, 1], [1, 0]))
    res2 = tensors.inner(t2, t3)
    assert np.allclose(expect2, res2)


def test_transpose(some_backend):
    backend = some_backend
    shape = [3, 5, 7, 10]
    data = np.random.random(shape) + 1.j * np.random.random(shape)
    t = tensors.Tensor.from_numpy(data, backend=backend, labels=['a', 'b', 'c', 'd'])
    res = tensors.transpose(t, [2, 0, 3, 1])
    assert res.labels == ['c', 'a', 'd', 'b']
    assert np.allclose(res.data, np.transpose(data, [2, 0, 3, 1]))


def test_trace(some_backend):
    backend = some_backend

    print('single legpair - default legs* args')
    data = np.random.random([7, 7, 7]) + 1.j * np.random.random([7, 7, 7])
    legs = [VectorSpace.non_symmetric(7), VectorSpace.non_symmetric(7), VectorSpace.non_symmetric(7).dual]
    tens = tensors.Tensor.from_numpy(data, legs=legs, backend=backend, labels=['a', 'b', 'b*'])
    expect = np.trace(data, axis1=-2, axis2=-1)
    res = tensors.trace(tens)
    assert res.labels_are('a')
    assert np.allclose(expect, res.data)

    print('single legpair - via idx or label')
    expect = np.trace(data, axis1=0, axis2=2)
    res_idx = tensors.trace(tens, 0, 2)
    res_label = tensors.trace(tens, 'a', 'b*')
    assert res_idx.labels_are('b')
    assert np.allclose(res_idx.data, expect)
    assert res_label.labels_are('b')
    assert np.allclose(res_label.data, expect)

    print('two legpairs')
    data = np.random.random([11, 13, 11, 7, 13]) + 1.j * np.random.random([11, 13, 11, 7, 13])
    expect = np.trace(np.trace(data, axis1=1, axis2=4), axis1=0, axis2=1)
    a = VectorSpace.non_symmetric(11)
    b = VectorSpace.non_symmetric(13)
    c = VectorSpace.non_symmetric(7)
    tens = tensors.Tensor.from_numpy(data, legs=[a, b.dual, a.dual, c, b], backend=backend,
                                     labels=['a', 'b*', 'a*', 'c', 'b'])
    res_idx = tensors.trace(tens, [0, 1], [2, 4])
    res_label = tensors.trace(tens, ['a', 'b*'], ['a*', 'b'])
    assert res_idx.labels_are('c')
    assert np.allclose(res_idx.data, expect)
    assert res_label.labels_are('c')
    assert np.allclose(res_label.data, expect)

    print('scalar result')
    data = np.random.random([11, 13, 11, 13]) + 1.j * np.random.random([11, 13, 11, 13])
    expect = np.trace(np.trace(data, axis1=1, axis2=3), axis1=0, axis2=1)
    a = VectorSpace.non_symmetric(11)
    b = VectorSpace.non_symmetric(13)
    tens = tensors.Tensor.from_numpy(data, legs=[a, b.dual, a.dual, b], backend=backend,
                                     labels=['a', 'b*', 'a*', 'b'])
    res_idx = tensors.trace(tens, [0, 1], [2, 3])
    res_label = tensors.trace(tens, ['a', 'b*'], ['a*', 'b'])
    assert isinstance(res_idx, complex)
    assert np.allclose(res_idx, expect)
    assert isinstance(res_label, complex)
    assert np.allclose(res_label, expect)


def test_conj(some_backend):
    backend = some_backend
    data = np.random.random([2, 4, 5]) + 1.j * np.random.random([2, 4, 5])
    tens = tensors.Tensor.from_numpy(data, backend=backend, labels=['a', 'b', None])
    res = tensors.conj(tens)
    if isinstance(backend, TorchBlockBackend):
        res_data = res.data.resolve_conj().numpy()
    else:
        res_data = res.data
    assert np.allclose(res_data, np.conj(data))
    assert res.labels == ['a*', 'b*', None]
    assert [l1.is_dual_of(l2) for l1, l2 in zip(res.legs, tens.legs)]


def test_combine_split(some_backend):
    backend = some_backend
    data = np.random.random([2, 4, 7, 5]) + 1.j * np.random.random([2, 4, 7, 5])
    tens = tensors.Tensor.from_numpy(data, backend=backend, labels=['a', 'b', 'c', 'd'])

    print('check by idx')
    res = tensors.combine_legs(tens, [1, 2])
    assert np.allclose(res.data, np.reshape(data, [2, 28, 5]))
    assert res.labels == ['a', '(b.c)', 'd']
    split = tensors.split_leg(res, 1)
    assert np.allclose(split.data, data)
    assert split.labels == ['a', 'b', 'c', 'd']

    print('check by label')
    res = tensors.combine_legs(tens, ['b', 'd'])
    expect = np.reshape(np.transpose(data, [0, 1, 3, 2]), [2, 20, 7])
    assert np.allclose(res.data, expect)
    assert res.labels == ['a', '(b.d)', 'c']
    split = tensors.split_leg(res, '(b.d)')
    assert np.allclose(split.data, np.transpose(data, [0, 1, 3, 2]))
    assert split.labels == ['a', 'b', 'd', 'c']

    print('check splitting a non-combined leg raises')
    with pytest.raises(ValueError):
        tensors.split_leg(res, 0)
    with pytest.raises(ValueError):
        tensors.split_leg(res, 'a')


def test_is_scalar(some_backend):
    backend = some_backend
    for s in [1, 0., 1.+2.j, np.int64(123), np.float64(2.345), np.complex128(1.+3.j)]:
        assert tensors.is_scalar(s)
    scalar_tens = tensors.Tensor.from_numpy([[1.]], backend=backend)
    assert tensors.is_scalar(scalar_tens)
    non_scalar_tens = tensors.Tensor.from_numpy([[1., 2.], [3., 4.]], backend=backend)
    assert not tensors.is_scalar(non_scalar_tens)


def test_almost_equal(some_backend):
    backend = some_backend
    data1  = np.random.random([2, 4, 3, 5])
    data2 = data1 + 1e-7 * np.random.random([2, 4, 3, 5])
    t1 = tensors.Tensor.from_numpy(data1, backend=backend)
    t2 = tensors.Tensor.from_numpy(data2, backend=backend)
    assert tensors.almost_equal(t1, t2)
    assert not tensors.almost_equal(t1, t2, atol=1e-10, rtol=1e-10)


def test_squeeze_legs(some_backend):
    backend=some_backend
    data = np.random.random([2, 1, 7, 1, 1]) + 1.j * np.random.random([2, 1, 7, 1, 1])
    tens = tensors.Tensor.from_numpy(data, backend=backend, labels=['a', 'b', 'c', 'd', 'e'])

    print('squeezing all legs (default arg)')
    res = tensors.squeeze_legs(tens)
    assert np.allclose(res.data, data[:, 0, :, 0, 0])
    assert res.labels == ['a', 'c']

    print('squeeze specific leg by idx')
    res = tensors.squeeze_legs(tens, 1)
    assert np.allclose(res.data, data[:, 0, :, :, :])
    assert res.labels == ['a', 'c', 'd', 'e']

    print('squeeze legs by labels')
    res = tensors.squeeze_legs(tens, ['b', 'e'])
    assert np.allclose(res.data, data[:, 0, :, :, 0])
    assert res.labels == ['a', 'c', 'd']


def test_norm(some_backend):
    data = np.random.random([2, 3, 7]) + 1.j * np.random.random([2, 3, 7])
    tens = tensors.Tensor.from_numpy(data, backend=some_backend)
    res = tensors.norm(tens)
    expect = np.linalg.norm(data)
    assert np.allclose(res, expect)


def demo_repr():
    # this is intended to generate a bunch of demo reprs
    # can not really make this an automated test, the point is for a human to have a look
    # and decide if the output is useful, concise, correct, etc.
    #
    # run e.g. via the following command
    # python -c "from tenpy.linalg.test_tensor import demo_repr; demo_repr()"

    print()
    separator = '=' * 80

    backend = NoSymmetryNumpyBackend()
    dims = (5, 2, 5)
    data = random_block(dims, backend)
    legs = [VectorSpace.non_symmetric(d) for d in dims]
    tens1 = tensors.Tensor(data, legs=legs, backend=backend, labels=['vL', 'p', 'vR'])
    tens2 = tensors.combine_legs(tens1, ['p', 'vR'])

    for command in ['repr(tens1.legs[0])',
                    'str(tens1.legs[0])',
                    'repr(tens1)',
                    'repr(tens2.legs[1])',
                    'str(tens2.legs[1])',
                    'repr(tens2)']:
        output = eval(command)
        print()
        print(separator)
        print(command)
        print(separator)
        print(output)
        print(separator)
        print()
