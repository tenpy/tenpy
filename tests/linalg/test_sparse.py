"""A collection of tests for tenpy.linalg.sparse"""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
import numpy as np
import numpy.testing as npt
import pytest
import scipy

from tenpy.linalg import sparse, tensors
from tenpy.linalg.symmetries import spaces
from tenpy.linalg.backends.backend_factory import get_backend
from tenpy.linalg.tensors import AbstractTensor, Tensor, Shape, Dtype, almost_equal


# define a few simple operators to test the wrappers:


class ScalingDummyOperator(sparse.LinearOperator):
    def __init__(self, factor, vector_shape: Shape):
        super().__init__(vector_shape=vector_shape, dtype=Dtype.complex128)
        self.factor = factor
        self.some_weird_attribute = 'arbitrary value'

    def some_unrelated_function(self, x):
        return 2 * x

    def matvec(self, vec: AbstractTensor) -> AbstractTensor:
        return self.factor * vec

    def to_tensor(self, backend=None) -> AbstractTensor:
        assert backend is not None, 'backend kwarg is needed for ScalingDummyOperator.to_tensor'
        return self.factor * Tensor.eye(
            legs=self.vector_shape.legs, backend=backend, labels=self.vector_shape.labels
        )

    def adjoint(self):
        return ScalingDummyOperator(np.conj(self.factor), self.vector_shape)


class TensorDummyOperator(sparse.LinearOperator):
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

    def to_tensor(self, **kw) -> AbstractTensor:
        return self.tensor.permute_legs(['a', 'b', 'a*', 'b*'])

    def adjoint(self):
        return TensorDummyOperator(self.tensor.conj())


def check_to_tensor(op: sparse.LinearOperator, vec: AbstractTensor, backend):
    """perform common checks of the LinearOperator.to_tensor method"""
    res_matvec = op.matvec(vec)
    tensor = op.to_tensor(backend=backend)
    _ = op.to_matrix(backend=backend)  # just check if it runs...
    res_tensor = tensor.tdot(vec, range(vec.num_legs, 2 * vec.num_legs), range(vec.num_legs))
    assert almost_equal(res_matvec, res_tensor)


def test_SumLinearOperator(backend, tensor_rng, vector_space_rng):
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
    op = sparse.SumLinearOperator(op1)
    # check matvec correct
    assert almost_equal(op.matvec(vec), factor1 *  vec)
    # check access to attributes of original_operator
    assert op.some_weird_attribute == 'arbitrary value'
    assert op.some_unrelated_function(2) == 4
    check_to_tensor(op, vec, backend)

    print('two operators')
    op = sparse.SumLinearOperator(op2, op1)
    assert almost_equal(op.matvec(vec), factor1 *  vec + T.tdot(vec, ['a*', 'b*'], ['a', 'b']))
    assert op.some_weird_attribute == 42
    assert op.some_unrelated_function(2) == 'buzz'
    check_to_tensor(op, vec, backend)

    print('three operators')
    op = sparse.SumLinearOperator(op1, op2, op3)
    assert almost_equal(op.matvec(vec), (factor1 + factor3) * vec + T.tdot(vec, ['a*', 'b*'], ['a', 'b']))
    assert op.some_weird_attribute == 'arbitrary value'
    assert op.some_unrelated_function(2) == 4
    check_to_tensor(op, vec, backend)


def test_ShiftedLinearOperator(backend, tensor_rng, vector_space_rng):
    a = vector_space_rng()
    b = vector_space_rng()
    vec = tensor_rng(legs=[a, b], labels=['a', 'b'])
    factor = 3.2
    op1 = ScalingDummyOperator(factor=factor, vector_shape=vec.shape)
    shift = 5.j
    
    op = sparse.ShiftedLinearOperator(op1, shift)
    assert almost_equal(op.matvec(vec), (factor + shift) * vec)
    assert op.some_weird_attribute == 'arbitrary value'
    assert op.some_unrelated_function(2) == 4
    check_to_tensor(op, vec, backend)


@pytest.mark.parametrize('penalty', [None, 2.-.3j])
def test_ProjectedLinearOperator(tensor_rng, vector_space_rng, penalty):
    a = vector_space_rng()
    b = vector_space_rng()
    o1 = tensor_rng(legs=[a, b], labels=['a', 'b'])
    o2 = tensor_rng(legs=[a, b], labels=['a', 'b'])
    factor = 3.2
    op1 = ScalingDummyOperator(factor=factor, vector_shape=o1.shape)

    pytest.xfail('Need to port gram_schmidt first')  # TODO
    op = sparse.ProjectedLinearOperator(op1, [o1, o2], penalty=penalty)

    print('check vector in ortho_vecs subspace')
    expect = 0. * o1 if penalty is None else penalty * o1
    assert almost_equal(op.matvec(o1), expect)

    print('check vector orthogonal to ortho_vecs')
    vec = vec1 - o1.inner(vec1) * o1 - o2.inner(vec) * o2
    assert almost_equal(op.matvec(vec), op1.matvec(vec))

    assert op.some_weird_attribute == 'arbitrary value'
    assert op.some_unrelated_function(2) == 4


@pytest.mark.parametrize('use_hermitian', [True, False])
def test_NumpyArrayLinearOperator_sector(vector_space_rng, tensor_rng, use_hermitian, k=5, tol=1e-14):
    a = vector_space_rng()
    b = vector_space_rng()
    H = tensor_rng(legs=[a, b.dual, a.dual, b], labels=['a', 'b*', 'a*', 'b'])
    H = H + H.conj()
    #
    H_np = H.to_numpy_ndarray(leg_order=['a', 'b*', 'a*', 'b'])
    H_mat = np.transpose(H_np, [0, 3, 2, 1]).reshape([a.dim * b.dim, a.dim * b.dim])
    E_np, psi_np = np.linalg.eigh(H_mat)
    E0_np, psi0_np = E_np[0], psi_np[:, 0]
    print(f'full spectrum: {E_np}')
    #
    sectors = tensors.detect_sectors_from_block(
        psi0_np.reshape([a.dim, b.dim]), legs=[a, b],
        backend=get_backend(symmetry=a.symmetry, block_backend='numpy')
    )
    sector = a.symmetry.fusion_outcomes(*sectors)[0, :]
    #
    if use_hermitian:
        H_op = sparse.HermitianNumpyArrayLinearOperator.from_Tensor(
            H, legs1=['a*', 'b*'], legs2=['a', 'b'], charge_sector=sector
        )
    else:
        H_op = sparse.NumpyArrayLinearOperator.from_Tensor(
            H, legs1=['a*', 'b*'], legs2=['a', 'b'], charge_sector=sector
        )
    psi_init = tensors.ChargedTensor.random_uniform(legs=[a, b], charge=sector, labels=['a', 'b'])
    psi_init_np = H_op.tensor_to_flat_array(psi_init)
    #
    E, psi = scipy.sparse.linalg.eigsh(H_op, k, v0=psi_init_np, which='SA')
    E0, psi0 = E[0], psi[:, 0]
    assert abs((E0 - E0_np) / E0_np) < tol
    psi0_H_psi0 = np.inner(psi0.conj(), H_op.matvec(psi0)).item()
    assert abs(psi0_H_psi0 / E0 - 1.) < tol


@pytest.mark.parametrize('num_legs', [1, 2])
def test_gram_schmidt(tensor_rng, vector_space_rng, num_legs, num_vecs=5, tol=1e-15):
    legs = [vector_space_rng() for _ in range(num_legs)]
    n = np.prod([l.dim for l in legs])
    vecs_old = [tensor_rng(legs) for _ in range(num_vecs)]
    vecs_new = sparse.gram_schmidt(vecs_old, rcond=tol)
    ovs = np.zeros((len(vecs_new), len(vecs_new)), dtype=np.complex128)
    vecs = [v.to_numpy_ndarray().flatten() for v in vecs_new]
    for i, v in enumerate(vecs):
        for j, w in enumerate(vecs):
            ovs[i, j] = np.inner(v.conj(), w)
    npt.assert_allclose(ovs, np.eye(len(vecs_new)), atol=2 * n * (num_vecs) ** 2 * tol)
