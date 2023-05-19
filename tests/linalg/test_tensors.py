"""A collection of tests for tenpy.linalg.tensors."""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
import numpy as np
import numpy.testing as npt
import pytest
import warnings

from tenpy.linalg import tensors
from tenpy.linalg.backends.abelian import AbstractAbelianBackend
from tenpy.linalg.backends.torch import TorchBlockBackend
from tenpy.linalg.backends.numpy import NumpyBlockBackend, NoSymmetryNumpyBackend
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

    numpy_block = np_random.normal(size=dims)
    dense_block = backend.block_from_numpy(numpy_block)

    if isinstance(backend, AbstractAbelianBackend):
        # There are two problems:
        #  - We need to generate numpy_block and dense_block such that they are symmetric,
        #    i.e. only non-zero (up to tolerance) within the allowed blocks
        #  - Randomly generating the legs seems to be a bad idea.
        #    Here, I get a combination of legs that allows no valid blocks...
        #    I.e. `backends.abelian._valid_block_indices(legs)` is empty.
        pytest.xfail('Need to redesign tests')

    print('checking from_dense_block')
    tens = tensors.Tensor.from_dense_block(dense_block, legs=legs, backend=backend)
    tens.test_sanity()
    data = backend.block_to_numpy(tens.to_dense_block())
    npt.assert_array_equal(data, numpy_block)

    print('checking from_numpy')
    tens = tensors.Tensor.from_numpy(numpy_block, legs=legs, backend=backend)
    tens.test_sanity()
    data = tens.to_numpy_ndarray()
    npt.assert_array_equal(data, numpy_block)

    # TODO from_block_func, from_numpy_func

    # TODO random_uniform, random_normal

    print('checking zero')
    tens = tensors.Tensor.zero(legs, backend=backend)
    tens.test_sanity()
    npt.assert_array_equal(tens.to_numpy_ndarray(), np.zeros(dims))

    print('checking eye')
    tens = tensors.Tensor.eye(legs[0], backend=backend)
    tens.test_sanity()
    npt.assert_array_equal(tens.to_numpy_ndarray(), np.eye(legs[0].dim))
    tens = tensors.Tensor.eye(legs[:2], backend=backend)
    tens.test_sanity()
    npt.assert_array_equal(tens.to_numpy_ndarray(), np.eye(np.prod(dims[:2])).reshape(dims[:2] + dims[:2]))


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
    npt.assert_array_equal(neg_t3.to_numpy_ndarray(), -dense3)
    a = 42
    b = 17
    with pytest.raises(ValueError) as err:
        res = a * tens1 - b * tens2
    assert "required in strict label mode" in err.value.args[0]  # TODO: check other config values?
    tens1.set_labels(['foo', 'a', 'b'])
    tens2.set_labels(['foo', 'a', 'b'])
    res = a * tens1 - b * tens2
    npt.assert_almost_equal(res.to_numpy_ndarray(), a * dense1 - b * dense2)
    res = tens1 / a + tens2 / b
    npt.assert_almost_equal(res.to_numpy_ndarray(), dense1 / a + dense2 / b)
    # TODO check strict label behavior!

    with pytest.raises(TypeError):
        tens1 == tens2

    print('check converisions, float, complex, array')
    tens4.set_labels(['i', 'j', 'i*', 'j*'])
    assert isinstance(float(tens4), float)
    npt.assert_equal(float(tens4), float(tens4_item))
    assert isinstance(complex(tens4 + 2.j * tens4), complex)
    npt.assert_equal(complex(tens4 + 2.j * tens4), complex(tens4_item + 2.j * tens4_item))

    with warnings.catch_warnings(record=True) as caught:
        r = float(tens4 + 2.j * tens4)
    if abs(tens4_item) > 0.:
        assert len(caught) == 1
        assert "converting complex to real" in str(caught[0].message)
    else:
        assert len(caught) == 0
    assert r == tens4_item

    # TODO check that float of a complex tensor raises a warning


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


# TODO (JH): continue to fix tests below to work with new fixtures for any backend
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
    t2 = tensor_rng(labels=['a', 'b'], real=False)
    t3 = tensor_rng(legs=t2.legs, labels=t2.labels, real=False)

    for t_i, t_j in [(t0, t1), (t2, t3)]:
        d_i = t_i.to_numpy_ndarray()
        d_j = t_j.to_numpy_ndarray()

        expect = np.inner(d_i.flatten().conj(), d_j.flatten())
        if t_j.num_legs > 0:
            t_j = t_j.permute_legs(t_j.labels[::-1])  # transpose should be reverted in inner()
        res = tensors.inner(t_i, t_j)
        npt.assert_allclose(res, expect)

        expect = np.linalg.norm(d_i) **2
        res = tensors.inner(t_i, t_i)
        npt.assert_allclose(res, expect)


def test_trace(backend, vector_space_rng, tensor_rng):
    a = vector_space_rng(3, 3, backend.VectorSpaceCls)
    b = vector_space_rng(4, 3, backend.VectorSpaceCls)
    c = vector_space_rng(2, 2, backend.VectorSpaceCls)
    t1 = tensor_rng(legs=[a, a.dual], labels=['a', 'a*'])
    d1 = t1.to_numpy_ndarray()
    t2 = tensor_rng(legs=[a, b, a.dual, b.dual], labels=['a', 'b', 'a*', 'b*'])
    d2 = t2.to_numpy_ndarray()
    t3 = tensor_rng(legs=[a, c, b, a.dual, b.dual], labels=['a', 'c', 'b', 'a*', 'b*'])
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


def test_conj(tensor_rng):
    tens = tensor_rng(labels=['a', 'b', None], real=False)
    expect = np.conj(tens.to_numpy_ndarray())
    assert np.linalg.norm(expect.imag) > 0 , "expect complex data!"
    res = tensors.conj(tens)
    res.test_sanity()
    assert res.labels == ['a*', 'b*', None]
    assert [l1.can_contract_with(l2) for l1, l2 in zip(res.legs, tens.legs)]
    assert np.allclose(res.to_numpy_ndarray(), expect)


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


def test_almost_equal(tensor_rng):
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


def demo_repr():
    # this is intended to generate a bunch of demo reprs
    # can not really make this an automated test, the point is for a human to have a look
    # and decide if the output is useful, concise, correct, etc.
    #
    # run e.g. via the following command
    # python -c "from test_tensors import demo_repr; demo_repr()"

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
