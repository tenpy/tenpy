"""A collection of tests for tenpy.linalg.sparse"""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
import numpy as np
import numpy.testing as npt
import pytest
import warnings

from tenpy.linalg import sparse
from tenpy.linalg.tensors import AbstractTensor, Tensor, Shape, Dtype, almost_equal


# define a few simple operators to test the wrappers:


class ScalingDummyOperator(sparse.TenpyLinearOperator):
    def __init__(self, factor, vector_shape: Shape):
        super().__init__(vector_shape=vector_shape, dtype=Dtype.complex128)
        self.factor = factor
        self.some_weird_attribute = 'arbitrary value'

    def some_unrelated_function(self, x):
        return 2 * x

    def matvec(self, vec: AbstractTensor) -> AbstractTensor:
        return self.factor * vec

    def adjoint(self):
        return ScalingDummyOperator(np.conj(self.factor), self.vector_shape)


class TensorDummyOperator(sparse.TenpyLinearOperator):
    def __init__(self, tensor: Tensor):
        assert tensor.labels == ['a', 'b*', 'a*', 'b']
        acts_on = ['a', 'b']
        vector_shape = Shape(legs=tensor.get_legs(acts_on), labels=acts_on)
        super().__init__(vector_shape=vector_shape, dtype=tensor.dtype)
        self.tensor = tensor
        self.some_weird_attribute = 42

    def some_unrelated_function(self, x):
        return 'buzz'

    def matvec(self, vec: AbstractTensor) -> AbstractTensor:
        return self.tensor.tdot(vec, ['a*', 'b*'], ['a', 'b'])

    def adjoint(self):
        return TensorDummyOperator(self.tensor.conj())


def test_SumTenpyLinearOperator(tensor_rng, vector_space_rng):
    a = vector_space_rng()
    b = vector_space_rng()
    T = tensor_rng(legs=[a, b.dual, a.dual, b], real=False, labels=['a', 'b*', 'a*', 'b'])
    vec = tensor_rng(legs=[a, b], labels=['a', 'b'])

    factor1 = 2.4
    factor3 = 3.1 - 42.j
    op1 = ScalingDummyOperator(factor1, vec.shape)
    op2 = TensorDummyOperator(T)
    op3 = ScalingDummyOperator(factor3, vec.shape)

    print('single operator')
    op = sparse.SumTenpyLinearOperator(op1)
    # check matvec correct
    assert almost_equal(op.matvec(vec), factor1 *  vec)
    # check access to attributes of original_operator
    assert op.some_weird_attribute == 'arbitrary value'
    assert op.some_unrelated_function(2) == 4

    print('two operators')
    print(op1.vector_shape)
    print(op2.vector_shape)
    op = sparse.SumTenpyLinearOperator(op2, op1)
    assert almost_equal(op.matvec(vec), factor1 *  vec + T.tdot(vec, ['a*', 'b*'], ['a', 'b']))
    assert op.some_weird_attribute == 42
    assert op.some_unrelated_function(2) == 'buzz'

    print('three operators')
    op = sparse.SumTenpyLinearOperator(op1, op2, op3)
    assert almost_equal(op.matvec(vec), (factor1 + factor3) * vec + T.tdot(vec, ['a*', 'b*'], ['a', 'b']))
    assert op.some_weird_attribute == 'arbitrary value'
    assert op.some_unrelated_function(2) == 4


def test_ShiftedTenpyLinearOperator(tensor_rng, vector_space_rng):
    a = vector_space_rng()
    b = vector_space_rng()
    vec = tensor_rng(legs=[a, b], labels=['a', 'b'])
    factor = 3.2
    op1 = ScalingDummyOperator(factor=factor, vector_shape=vec.shape)
    shift = 5.j
    
    op = sparse.ShiftedTenpyLinearOperator(op1, shift)
    assert almost_equal(op.matvec(vec), (factor + shift) * vec)
    assert op.some_weird_attribute == 'arbitrary value'
    assert op.some_unrelated_function(2) == 4


@pytest.mark.parametrize('penalty', [None, 2.-.3j])
def test_ProjectedTenpyLinearOperator(tensor_rng, vector_space_rng, penalty):
    a = vector_space_rng()
    b = vector_space_rng()
    o1 = tensor_rng(legs=[a, b], labels=['a', 'b'])
    o2 = tensor_rng(legs=[a, b], labels=['a', 'b'])
    factor = 3.2
    op1 = ScalingDummyOperator(factor=factor, vector_shape=o1.shape)

    pytest.xfail('Need to port gram_schmidt first')  # TODO
    op = sparse.ProjectedTenpyLinearOperator(op1, [o1, o2], penalty=penalty)

    print('check vector in ortho_vecs subspace')
    expect = 0. * o1 if penalty is None else penalty * o1
    assert almost_equal(op.matvec(o1), expect)

    print('check vector orthogonal to ortho_vecs')
    vec = vec1 - o1.inner(vec1) * o1 - o2.inner(vec) * o2
    assert almost_equal(op.matvec(vec), op1.matvec(vec))

    assert op.some_weird_attribute == 'arbitrary value'
    assert op.some_unrelated_function(2) == 4


def test_FlatLinearOperator(n=30, k=5, tol=1.e-14):
    pytest.xfail('Need to port the implementation first')
    # TODO blindly pasted old code below:
    # leg = gen_random_legcharge(ch, n)
    # H = npc.Array.from_func_square(rmat.GUE, leg)
    # H_flat = H.to_ndarray()
    # E_flat, psi_flat = np.linalg.eigh(H_flat)
    # E0_flat, psi0_flat = E_flat[0], psi_flat[:, 0]
    # qtotal = npc.detect_qtotal(psi0_flat, [leg])

    # H_sparse = OLD_sparse.FlatLinearOperator.from_NpcArray(H, charge_sector=qtotal)
    # psi_init = npc.Array.from_func(np.random.random, [leg], qtotal=qtotal)
    # psi_init /= npc.norm(psi_init)
    # psi_init_flat = H_sparse.npc_to_flat(psi_init)

    # # check diagonalization
    # E, psi = scipy.sparse.linalg.eigsh(H_sparse, k, v0=psi_init_flat, which='SA')
    # E0, psi0 = E[0], psi[:, 0]
    # print("full spectrum:", E_flat)
    # print("E0 = {E0:.14f} vs exact {E0_flat:.14f}".format(E0=E0, E0_flat=E0_flat))
    # print("|E0-E0_flat| / |E0_flat| =", abs((E0 - E0_flat) / E0_flat))
    # assert (abs((E0 - E0_flat) / E0_flat) < tol)
    # psi0_H_psi0 = np.inner(psi0.conj(), H_sparse.matvec(psi0)).item()
    # print("<psi0|H|psi0> / E0 = 1. + ", psi0_H_psi0 / E0 - 1.)
    # assert (abs(psi0_H_psi0 / E0 - 1.) < tol)


def test_FlatHermitianOperator(n=30, k=5, tol=1.e-14):
    pytest.xfail('Need to port the implementation first')
    # TODO blindly pasted old code below:
    # leg = gen_random_legcharge(ch, n // 2)
    # leg2 = gen_random_legcharge(ch, 2)
    # pipe = npc.LegPipe([leg, leg2], qconj=+1)
    # H = npc.Array.from_func_square(rmat.GUE, pipe, labels=["(a.b)", "(a*.b*)"])
    # H_flat = H.to_ndarray()
    # E_flat, psi_flat = np.linalg.eigh(H_flat)
    # E0_flat, psi0_flat = E_flat[0], psi_flat[:, 0]
    # qtotal = npc.detect_qtotal(psi0_flat, [pipe])

    # H_sparse = OLD_sparse.FlatHermitianOperator.from_NpcArray(H, charge_sector=qtotal)
    # psi_init = npc.Array.from_func(np.random.random, [pipe], qtotal=qtotal, labels=["(a.b)"])
    # psi_init /= npc.norm(psi_init)
    # psi_init_flat = H_sparse.npc_to_flat(psi_init)

    # # check diagonalization
    # E, psi = scipy.sparse.linalg.eigsh(H_sparse, k, v0=psi_init_flat, which='SA')
    # E0, psi0 = E[0], psi[:, 0]
    # print("full spectrum:", E_flat)
    # print("E0 = {E0:.14f} vs exact {E0_flat:.14f}".format(E0=E0, E0_flat=E0_flat))
    # print("|E0-E0_flat| / |E0_flat| =", abs((E0 - E0_flat) / E0_flat))
    # assert (abs((E0 - E0_flat) / E0_flat) < tol)
    # psi0_H_psi0 = np.inner(psi0.conj(), H_sparse.matvec(psi0)).item()
    # print("<psi0|H|psi0> / E0 = 1. + ", psi0_H_psi0 / E0 - 1.)
    # assert (abs(psi0_H_psi0 / E0 - 1.) < tol)

    # # split H to check `FlatHermitianOperator.from_guess_with_pipe`.
    # print("=========")
    # print("split legs and define separate matvec")
    # assert psi_init.legs[0] is pipe
    # psi_init_split = psi_init.split_legs([0])
    # H_split = H.split_legs()

    # def H_split_matvec(vec):
    #     vec = npc.tensordot(H_split, vec, [["a*", "b*"], ["a", "b"]])
    #     # TODO as additional challenge, transpose the resulting vector
    #     return vec

    # H_sparse_split, psi_init_split_flat = OLD_sparse.FlatLinearOperator.from_guess_with_pipe(
    #     H_split_matvec, psi_init_split, dtype=H_split.dtype)

    # # diagonalize
    # E, psi = scipy.sparse.linalg.eigsh(H_sparse_split, k, v0=psi_init_split_flat, which='SA')
    # E0, psi0 = E[0], psi[:, 0]
    # print("full spectrum:", E_flat)
    # print("E0 = {E0:.14f} vs exact {E0_flat:.14f}".format(E0=E0, E0_flat=E0_flat))
    # print("|E0-E0_flat| / |E0_flat| =", abs((E0 - E0_flat) / E0_flat))
    # assert (abs((E0 - E0_flat) / E0_flat) < tol)
    # psi0_H_psi0 = np.inner(psi0.conj(), H_sparse.matvec(psi0)).item()
    # print("<psi0|H|psi0> / E0 = 1. + ", psi0_H_psi0 / E0 - 1.)
    # assert (abs(psi0_H_psi0 / E0 - 1.) < tol)
