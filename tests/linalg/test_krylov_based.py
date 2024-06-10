"""A collection of tests for tenpy.linalg.krylov_based."""
# Copyright (C) TeNPy Developers, GNU GPLv3
import pytest
from numpy import testing as npt
import numpy as np
from scipy.linalg import expm

import tenpy as tp
from tenpy.linalg import krylov_based, sparse, tensors, random_matrix


pytest.skip("krylov_based not yet revised", allow_module_level=True)  # TODO


@pytest.mark.parametrize(['N_cache', 'tol'], [(10, 5.e-13), (20, 5.e-14)])
def test_lanczos_gs(compatible_backend, make_compatible_space, N_cache, tol):
    # TODO revise this. purge the "dummy" language, its now "charged"
    
    # generate hermitian test array
    leg = make_compatible_space()
    backend = compatible_backend
    
    if isinstance(compatible_backend, tp.linalg.backends.FusionTreeBackend):
        # TODO need to be more careful with from func.
        # shapes of the blocks depend on num_domain_legs!
        # and GUE((1, 9)) generates blocks with shape (9, 9) without error!!
        # should GUE etc be Tensor classmethods?
        with pytest.raises(AssertionError, match='not a square matrix shape'):
            H = tensors.SymmetricTensor.from_numpy_func(random_matrix.GUE, legs=[leg, leg.dual], backend=backend)
        return
    
    H = tensors.SymmetricTensor.from_numpy_func(random_matrix.GUE, legs=[leg, leg.dual], backend=backend)

    if isinstance(H.backend, tp.linalg.backends.FusionTreeBackend) and isinstance(leg.symmetry, tp.ProductSymmetry):
        # TODO
        with pytest.raises(NotImplementedError, match='fusion_tensor is not implemented'):
            _ = H.to_numpy()
        return

    H_np = H.to_numpy()
    H_op = sparse.TensorLinearOperator(H, which_leg=1)
    npt.assert_allclose(H_np, H_np.conj().transpose())  # make sure we generated a hermitian operator
    E_np, psi_np = np.linalg.eigh(H_np)
    E0_np, psi0_np = E_np[0], psi_np[:, 0]

    # detect in which charge sector the groundstate lives
    sector, = tensors.detect_sectors_from_block(backend.block_from_numpy(psi0_np), legs=[leg], backend=backend)
    # TODO having to take the dual here is pretty unintuitive...
    psi_init = tensors.ChargedTensor.random_uniform(legs=[leg], charge=sector, backend=backend, dummy_leg_state=[1.])

    E0, psi0, N = krylov_based.lanczos(H_op, psi_init, {'N_cache': N_cache})
    assert abs(psi0.norm() - 1.) < tol
    print(f'full spectrum: {E_np}')
    print(f'E0 = {E0:.14f} vs exact {E0_np:.14f}')
    print(f'{abs((E0 - E0_np) / E0_np)=}')
    assert abs((E0 - E0_np) / E0_np) < tol
    psi0_H_psi0 = psi0.inner(H.tdot(psi0))
    assert abs(psi0.norm() - 1.) < tol
    print(f'<psi0|H|psi0> / E0 = 1. + {psi0_H_psi0 / E0 - 1.}')
    assert (abs(psi0_H_psi0 / E0 - 1.) < tol)
    print(f'<psi0_np|H_np|psi0_np> / E0_np = {np.inner(psi0_np.conj(), np.dot(H_np, psi0_np)) / E0_np}')
    ov = np.inner(psi0.to_numpy().conj(), psi0_np)
    print(f'|<psi0|psi0_np>| = {abs(ov)}')
    assert abs(1. - abs(ov)) < tol

    print('Now look for second eigenvector in the *same* charge sector')
    orthogonal_to = [psi0]
    for i in range(1, len(E_np)):
        E1_np, psi1_np = E_np[i], psi_np[:, i]
        _sector, = tensors.detect_sectors_from_block(backend.block_from_numpy(psi1_np), legs=[leg], backend=backend)
        if np.any(_sector != sector):
            continue  # psi1_np is in different sector
        print("--- excited state #", len(orthogonal_to))
        lanczos_params = {'reortho': True}
        if E1_np > -0.01:
            lanczos_params['E_shift'] = -2. * E1_np - 0.2
        H_proj = sparse.ProjectedLinearOperator(H_op, ortho_vecs=orthogonal_to[:])
        E1, psi1, N = krylov_based.lanczos(H_proj, psi_init, lanczos_params)
        print(f'E1 = {E1:.14f} vs exact {E1_np:.14f}')
        print(f'{abs((E1 - E1_np) / E1_np)=}')
        psi1_H_psi1 = psi1.inner(H.tdot(psi1))
        print(f'<psi1|H|psi1> / E1 = 1. + {psi1_H_psi1 / E1 - 1.}')
        assert (abs(psi1_H_psi1 / E1 - 1.) < 100 * tol)  # TODO why does this need such large tolerance?
        print(f'<psi1_np|H_np|psi1_np> / E1_np = {np.inner(psi1_np.conj(), np.dot(H_np, psi1_np)) / E1_np}')
        ov = np.inner(psi1.to_numpy().conj(), psi1_np)
        print(f'|<psi1|psi1_np>| = {abs(ov)}')
        assert (abs(1. - abs(ov)) < tol)
        for psi_prev in orthogonal_to:
            ov = psi_prev.inner(psi1)
            assert abs(ov) < tol
        orthogonal_to.append(psi1)
    if len(orthogonal_to) == 1:
        print("warning: test didn't find a second eigenvector in the same charge sector!")


def test_lanczos_arpack():
    pytest.xfail('Not implemented yet (the operator in linalg.sparse is missing)')
    # TODO old below
    # print("version with arpack")
    # E0a, psi0a = lanczos.lanczos_arpack(H_Op, psi_init, {})
    # print("E0a = {E0a:.14f} vs exact {E0_flat:.14f}".format(E0a=E0a, E0_flat=E0_flat))
    # print("|E0a-E0_flat| / |E0_flat| =", abs((E0a - E0_flat) / E0_flat))
    # psi0a_H_psi0a = npc.inner(psi0a, npc.tensordot(H, psi0a, axes=[1, 0]), 'range', do_conj=True)
    # print("<psi0a|H|psi0a> / E0a = 1. + ", psi0a_H_psi0a / E0a - 1.)
    # assert (abs(psi0a_H_psi0a / E0a - 1.) < tol)
    # ov = np.inner(psi0a.to_ndarray().conj(), psi0_flat)
    # print("|<psi0a|psi0_flat>|=", abs(ov))
    # assert (abs(1. - abs(ov)) < tol)


@pytest.mark.parametrize(['N_cache', 'tol'], [(10, 5.e-13), (20, 5.e-14)])
def test_lanczos_evolve(compatible_backend, make_compatible_space, N_cache, tol):
    backend = compatible_backend
    leg = make_compatible_space()
    
    if isinstance(compatible_backend, tp.linalg.backends.FusionTreeBackend):
        # TODO need to be more careful with from func.
        # shapes of the blocks depend on num_domain_legs!
        # and GUE((1, 9)) generates blocks with shape (9, 9) without error!!
        # should GUE etc be Tensor classmethods?
        with pytest.raises(AssertionError, match='not a square matrix shape'):
            H = tensors.SymmetricTensor.from_numpy_func(random_matrix.GUE, legs=[leg, leg.dual], backend=backend)
        return
    
    H = tensors.SymmetricTensor.from_numpy_func(random_matrix.GUE, legs=[leg, leg.dual], backend=backend)
    H_op = sparse.TensorLinearOperator(H, which_leg=1)

    if isinstance(H.backend, tp.linalg.backends.FusionTreeBackend) and isinstance(leg.symmetry, tp.ProductSymmetry):
        # TODO
        with pytest.raises(NotImplementedError, match='fusion_tensor is not implemented'):
            _ = H.to_numpy()
        return
    
    if isinstance(H.backend, tp.linalg.backends.FusionTreeBackend):
        # TODO
        with pytest.raises(AssertionError, match='norm not preserved'):
            _ = H.to_numpy()
        return
    
    H_np = H.to_numpy()
    npt.assert_allclose(H_np, H_np.conj().transpose())  # make sure we generated a hermitian operator

    sector = leg.sectors[0]
    psi_init = tensors.ChargedTensor.random_uniform(legs=[leg], charge=sector, backend=backend, dummy_leg_state=[1])

    psi_init_np = psi_init.to_numpy()

    lanc = krylov_based.LanczosEvolution(H_op, psi_init, {'N_cache': N_cache})
    for delta in [-0.1j, 0.1j, 1.j, 0.1, 1.]:
        psi_final_np = expm(H_np * delta).dot(psi_init_np)
        norm = np.linalg.norm(psi_final_np)
        psi_final, N = lanc.run(delta, normalize=False)
        diff = np.linalg.norm(psi_final.to_numpy() - psi_final_np)
        print("norm(|psi_final> - |psi_final_flat>)/norm = ", diff / norm)  # should be 1.
        assert diff / norm < tol
        psi_final2, N = lanc.run(delta, normalize=True)
        assert tensors.norm(psi_final / norm - psi_final2) < tol


@pytest.mark.parametrize('which', ['LM', 'SR', 'LR'])
def test_arnoldi(compatible_backend, make_compatible_space, which, N_max=20):
    backend = compatible_backend
    leg = make_compatible_space()
    tol = 1.e-13 if leg.dim <= N_max else 1.e-10
    # if looking for small/large real part, ensure hermitian H
    func = random_matrix.GUE if which[-1] == 'R' else random_matrix.standard_normal_complex

    if which[-1] == 'R' and isinstance(compatible_backend, tp.linalg.backends.FusionTreeBackend):
        # TODO need to be more careful with from func.
        # shapes of the blocks depend on num_domain_legs!
        # and GUE((1, 9)) generates blocks with shape (9, 9) without error!!
        # should GUE etc be Tensor classmethods?
        with pytest.raises(AssertionError, match='not a square matrix shape'):
            H = tensors.SymmetricTensor.from_numpy_func(random_matrix.GUE, legs=[leg, leg.dual], backend=backend)
        return
    
    H = tensors.SymmetricTensor.from_numpy_func(func, legs=[leg, leg.dual], backend=backend)
    H_op = sparse.TensorLinearOperator(H, which_leg=1)

    if isinstance(H.backend, tp.linalg.backends.FusionTreeBackend) and isinstance(leg.symmetry, tp.ProductSymmetry):
        # TODO
        with pytest.raises(NotImplementedError, match='should be implemented by subclass'):
            _ = H.to_numpy()
        return
    
    H_np = H.to_numpy()
    E_np, psi_np = np.linalg.eig(H_np)
    if which == 'LM':
        i = np.argmax(np.abs(E_np))
    elif which == 'LR':
        i = np.argmax(np.real(E_np))
    elif which == 'SR':
        i = np.argmin(np.real(E_np))
    E0_np, psi0_np = E_np[i], psi_np[:, i]

    sector, = tensors.detect_sectors_from_block(backend.block_from_numpy(psi0_np), legs=[leg], backend=backend)
    psi_init = tensors.ChargedTensor.random_uniform(legs=[leg], charge=sector, backend=backend, dummy_leg_state=[1])

    engine = krylov_based.Arnoldi(H_op, psi_init, {'which': which, 'num_ev': 1, 'N_max': N_max})

    if isinstance(compatible_backend, tp.linalg.backends.FusionTreeBackend):
        # TODO
        with pytest.raises(NotImplementedError, match='tdot not implemented'):
            _ = engine.run()
        return
    
    (E0,), (psi0,), N = engine.run()
    print("full spectrum:", E_np)
    print("E0 = {E0:.14f} vs exact {E0_flat:.14f}".format(E0=E0, E0_flat=E0_np))
    print("|E0-E0_np| / |E0_np| =", abs((E0 - E0_np) / E0_np))
    assert abs((E0 - E0_np) / E0_np) < tol
    psi0_H_psi0 = psi0.inner(H.tdot(psi0))
    print("<psi0|H|psi0> / E0 = 1. + ", psi0_H_psi0 / E0 - 1.)
    assert (abs(psi0_H_psi0 / E0 - 1.) < tol)
    print("<psi0_flat|H_flat|psi0_flat> / E0_flat = ", end=' ')
    print(np.inner(psi0_np.conj(), np.dot(H_np, psi0_np)) / E0_np)
    ov = np.inner(psi0.to_numpy().conj(), psi0_np)
    print("|<psi0|psi0_flat>|=", abs(ov))
    assert (abs(1. - abs(ov)) < tol)
