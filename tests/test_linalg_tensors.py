"""A collection of tests for tenpy.linalg.tensors."""
from math import prod
from multiprocessing.sharedctypes import Value
import numpy as np
from tenpy.linalg import tensors
from tenpy.linalg.backends import NoSymmetryNumpyBackend
from tenpy.linalg.symmetries import VectorSpace
import pytest



def test_Tensor_methods():

    backend = NoSymmetryNumpyBackend()
    data = np.ones((2, 3, 10))  # FIXME random instead
    data2 = np.ones((2, 3, 10))  # FIXME random instead

    # FIXME dummy for debugging
    data = np.ones((1, 1, 1))
    data2 = np.ones((1, 1, 1))
    
    legs = [VectorSpace.non_symmetric(d) for d in data.shape]
    
    print('checking __init__ with labels=None')
    tens1 = tensors.Tensor(data, backend, legs, labels=None)
    tens1.check_sanity()

    print('checking __init__, partially labelled')
    tens2 = tensors.Tensor(data2, backend, legs, labels=[None, 'a', 'b'])
    tens2.check_sanity()

    print('checking __init__, fully labelled')
    tens3 = tensors.Tensor(data, backend, legs, labels=['foo', 'a', 'b'])
    tens3.check_sanity()

    print('check size')
    assert tens3.size == prod(data.shape)

    # TODO reintroduced when implemented
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

    
    tens3.set_labels(['i', 'j', 'k'])
    assert tens3.labels_are('i', 'j', 'k')

    print('check get_leg_idx')
    assert tens3.get_leg_idx(0) == 0
    assert tens3.get_leg_idx(-1) == 2
    with pytest.raises(KeyError):
        tens3.get_leg_idx(10)
    assert tens3.get_leg_idx('i') == 0
    assert tens3.get_leg_idx('j') == 1
    with pytest.raises(KeyError):
        tens3.get_leg_idx('bar')
    with pytest.raises(TypeError):
        tens3.get_leg_idx(None)

    print('check get_leg_idcs')
    assert tens3.get_leg_idcs('i') == [0]
    assert tens3.get_leg_idcs(['i', 'k', 1]) == [0, 2, 1]

    print('check item')
    tens4 = tensors.Tensor(np.ones((1,)), backend, legs=[VectorSpace.non_symmetric(1)])
    assert np.allclose(tens4.item(), 1)
    
    print('check str and repr')
    str(tens1)
    str(tens3)
    repr(tens1)
    repr(tens3)

    print('check addition + multiplication')
    neg_t3 = -tens3
    assert np.allclose(neg_t3.data, -data)
    a = 42
    b = 17
    res = a * tens1 - b * tens2
    assert np.allclose(res.data, a * data - b * data2)
    res = tens1 / a + tens2 / b
    assert np.allclose(res.data, data / a + data2 / b)
    # TODO check strict label behavior!

    with pytest.raises(TypeError):
        tens1 == tens2

    print('check converisions, float, complex, array')
    assert isinstance(float(tens4), float)
    assert np.allclose(float(tens4), 1)
    assert isinstance(complex(tens4 + 2.j * tens4), complex)
    assert np.allclose(complex(tens4 + 2.j * tens4), 1 + 2.j)
    # TODO check that float of a complex tensor raises a warning
    t1_np = np.asarray(tens1)
    assert np.allclose(t1_np, data)



def test_Tensor_classmethods():
    pass  # TODO


    

