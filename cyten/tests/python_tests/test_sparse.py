"""A collection of tests for cyten.sparse"""

# Copyright (C) TeNPy Developers, Apache license
import numpy as np
import numpy.testing as npt
import pytest
import scipy

from cyten import Dtype, backends, get_backend, sparse, tensors
from cyten.tensors import SymmetricTensor, Tensor, almost_equal

pytest.skip('sparse not yet revised', allow_module_level=True)  # TODO


# define a few simple operators to test the wrappers:


class ScalingDummyOperator(sparse.LinearOperator):
    def __init__(self, factor, vector_shape):
        super().__init__(vector_shape=vector_shape, dtype=Dtype.complex128)
        self.factor = factor
        self.some_weird_attribute = 'arbitrary value'

    def some_unrelated_function(self, x):
        return 2 * x

    def matvec(self, vec: Tensor) -> Tensor:
        return self.factor * vec

    def to_tensor(self, backend=None) -> Tensor:
        assert backend is not None, 'backend kwarg is needed for ScalingDummyOperator.to_tensor'
        return self.factor * SymmetricTensor.eye(
            legs=self.vector_shape.legs, backend=backend, labels=self.vector_shape.labels
        )

    def adjoint(self):
        return ScalingDummyOperator(np.conj(self.factor), self.vector_shape)


# class TensorDummyOperator(sparse.LinearOperator):
#     def __init__(self, tensor: SymmetricTensor):
#         assert tensor.labels == ['a', 'b*', 'a*', 'b']
#         acts_on = ['a', 'b']
#         # TODO should we be strict about num_domain_legs here?
#         vector_shape = Shape(legs=tensor.get_legs(acts_on), num_domain_legs=0, labels=acts_on)
#         super().__init__(vector_shape=vector_shape, dtype=tensor.dtype)
#         self.tensor = tensor
#         self.some_weird_attribute = 42

#     def some_unrelated_function(self, x):
#         return 'buzz'

#     def matvec(self, vec: Tensor) -> Tensor:
#         return self.tensor.tdot(vec, ['a*', 'b*'], ['a', 'b'])

#     def to_tensor(self, **kw) -> Tensor:
#         return self.tensor.permute_legs(['a', 'b', 'b*', 'a*'])

#     def adjoint(self):
#         return TensorDummyOperator(self.tensor.conj())


def check_to_tensor(op: sparse.LinearOperator, vec: Tensor):
    """perform common checks of the LinearOperator.to_tensor method"""
    res_matvec = op.matvec(vec)
    tensor = op.to_tensor(backend=vec.backend)
    _ = op.to_matrix(backend=vec.backend)  # just check if it runs...
    res_tensor = tensor.tdot(vec, range(vec.num_legs, 2 * vec.num_legs), reversed(range(vec.num_legs)))
    assert almost_equal(res_matvec, res_tensor)


def test_SumLinearOperator(make_compatible_tensor):
    vec = make_compatible_tensor(labels=['a', 'b'])
    # a, b = vec.legs
    # T = make_compatible_tensor(legs=[a, b.dual, a.dual, b], labels=['a', 'b*', 'a*', 'b'])

    # factor1 = 2.4
    # factor3 = 3.1 - 42.j
    # op1 = ScalingDummyOperator(factor1, vec.shape)
    # op2 = TensorDummyOperator(T)
    # op3 = ScalingDummyOperator(factor3, vec.shape)

    # print('single operator')
    # op = sparse.SumLinearOperator(op1)
    # # check matvec correct

    # assert almost_equal(op.matvec(vec), factor1 *  vec)
    # # check access to attributes of original_operator
    # assert op.some_weird_attribute == 'arbitrary value'
    # assert op.some_unrelated_function(2) == 4

    # if isinstance(T.backend, backends.FusionTreeBackend):
    #     with pytest.raises(NotImplementedError, match='permute_legs not implemented|combine_legs not implemented'):
    #         check_to_tensor(op, vec)
    #     return  # TODO

    # check_to_tensor(op, vec)

    # print('two operators')
    # op = sparse.SumLinearOperator(op2, op1)
    # assert almost_equal(op.matvec(vec), factor1 *  vec + T.tdot(vec, ['a*', 'b*'], ['a', 'b']))
    # assert op.some_weird_attribute == 42
    # assert op.some_unrelated_function(2) == 'buzz'
    # check_to_tensor(op, vec)

    # print('three operators')
    # op = sparse.SumLinearOperator(op1, op2, op3)
    # assert almost_equal(op.matvec(vec), (factor1 + factor3) * vec + T.tdot(vec, ['a*', 'b*'], ['a', 'b']))
    # assert op.some_weird_attribute == 'arbitrary value'
    # assert op.some_unrelated_function(2) == 4
    # check_to_tensor(op, vec)


def test_ShiftedLinearOperator(make_compatible_tensor):
    vec = make_compatible_tensor(labels=['a', 'b'])
    factor = 3.2
    op1 = ScalingDummyOperator(factor=factor, vector_shape=vec.shape)
    shift = 5.0j

    op = sparse.ShiftedLinearOperator(op1, shift)
    assert almost_equal(op.matvec(vec), (factor + shift) * vec)
    assert op.some_weird_attribute == 'arbitrary value'
    assert op.some_unrelated_function(2) == 4

    if isinstance(vec.backend, backends.FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='combine_legs not implemented'):
            check_to_tensor(op, vec)
        return  # TODO

    check_to_tensor(op, vec)


@pytest.mark.parametrize(['penalty', 'project_operator'], [(None, True), (2.0 - 0.3j, True), (-4, False)])
def test_ProjectedLinearOperator(make_compatible_tensor, penalty, project_operator):
    vec = make_compatible_tensor(labels=['a', 'b'])
    a, b = vec.legs
    o1 = make_compatible_tensor(legs=[a, b], labels=['a', 'b'])
    assert (o1_norm := o1.norm()) > 0
    o1 = o1 / o1_norm
    o2 = make_compatible_tensor(legs=[a, b], labels=['a', 'b'])

    if isinstance(vec.backend, backends.FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='inner not implemented'):
            o2 = o2 - o1.inner(o2) * o1
        return  # TODO

    o2 = o2 - o1.inner(o2) * o1
    assert (o2_norm := o2.norm()) > 0
    o2 = o2 / o2_norm
    factor = 3.2
    original_op = ScalingDummyOperator(factor=factor, vector_shape=o1.shape)

    projected_op = sparse.ProjectedLinearOperator(
        original_op, [o1, o2], project_operator=project_operator, penalty=penalty
    )

    print('check vector in ortho_vecs subspace')
    if project_operator:
        expect = 0.0 * o1
    else:
        expect = original_op.matvec(o1)
    if penalty is not None:
        expect += penalty * o1
    assert almost_equal(projected_op.matvec(o1), expect)

    print('check vector orthogonal to ortho_vecs')
    vec1 = vec - o1.inner(vec) * o1 - o2.inner(vec) * o2
    expect = original_op.matvec(vec1)
    res = projected_op.matvec(vec1)
    assert almost_equal(res, expect)

    assert projected_op.some_weird_attribute == 'arbitrary value'
    assert projected_op.some_unrelated_function(2) == 4


@pytest.mark.parametrize('use_hermitian', [True, False])
def test_NumpyArrayLinearOperator_sector(make_compatible_space, make_compatible_tensor, use_hermitian, k=5, tol=1e-14):
    a = make_compatible_space()
    b = make_compatible_space()
    H = make_compatible_tensor(legs=[a, b.dual, a.dual, b], labels=['a', 'b*', 'a*', 'b'])

    if isinstance(H.backend, backends.FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='conj not implemented'):
            H = H + H.conj()
        return  # TODO

    H = H + H.conj()
    #
    H_np = H.to_numpy(leg_order=['a', 'b*', 'a*', 'b'])
    H_mat = np.transpose(H_np, [0, 3, 2, 1]).reshape([a.dim * b.dim, a.dim * b.dim])
    E_np, psi_np = np.linalg.eigh(H_mat)
    E0_np, psi0_np = E_np[0], psi_np[:, 0]
    print(f'full spectrum: {E_np}')
    #
    sectors = tensors.detect_sectors_from_block(
        psi0_np.reshape([a.dim, b.dim]), legs=[a, b], backend=get_backend(symmetry=a.symmetry, block_backend='numpy')
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
    psi_init = tensors.ChargedTensor.random_uniform(legs=[a, b], charge=sector, labels=['a', 'b'], charged_state=[1])
    psi_init_np = H_op.tensor_to_flat_array(psi_init)
    #
    E, psi = scipy.sparse.linalg.eigsh(H_op, k, v0=psi_init_np, which='SA')
    E0, psi0 = E[0], psi[:, 0]
    assert abs((E0 - E0_np) / E0_np) < tol
    psi0_H_psi0 = np.inner(psi0.conj(), H_op.matvec(psi0)).item()
    assert abs(psi0_H_psi0 / E0 - 1.0) < tol


@pytest.mark.parametrize('num_legs', [1, 2])
def test_gram_schmidt(make_compatible_tensor, num_legs, num_vecs=5, tol=1e-15):
    first = make_compatible_tensor(num_legs=num_legs)
    vecs_old = [first] + [make_compatible_tensor(first.legs) for _ in range(num_vecs - 1)]
    # note: depending on the dimension of `legs` (which is random),
    # some of those can be linearly dependent!

    if isinstance(first.backend, backends.FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='inner not implemented'):
            vecs_new = sparse.gram_schmidt(vecs_old)
        return  # TODO

    vecs_new = sparse.gram_schmidt(vecs_old)  # rtol=tol is too small for some random spaces
    assert len(vecs_new) <= len(vecs_old)
    ovs = np.zeros((len(vecs_new), len(vecs_new)), dtype=np.complex128)
    vecs = [v.to_numpy().flatten() for v in vecs_new]
    for i, v in enumerate(vecs):
        for j, w in enumerate(vecs):
            ovs[i, j] = np.inner(v.conj(), w)
    atol = 2 * first.num_parameters * (num_vecs) ** 2 * tol
    npt.assert_allclose(ovs, np.eye(len(vecs_new)), atol=atol)
