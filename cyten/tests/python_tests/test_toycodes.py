"""Testing if the toycodes can run."""

# Copyright (C) TeNPy Developers, Apache license
import cyten as ct
from toycodes.tenpy_toycodes.a_mps import init_Fib_anyon_MPS, init_FM_MPS, init_Neel_MPS, init_SU2_sym_MPS
from toycodes.tenpy_toycodes.b_model import (
    GoldenChainModel,
    HeisenbergModel,
    TFIModel,
    heisenberg_finite_gs_energy,
    tfi_finite_gs_energy,
)
from toycodes.tenpy_toycodes.d_dmrg import DMRGEngine


def test_toy_MPS():
    _ = init_FM_MPS(L=10, d=2, bc='finite')
    _ = init_FM_MPS(L=10, d=2, bc='infinite', conserve='Z2')
    _ = init_FM_MPS(L=10, d=2, bc='finite', backend='fusion_tree', conserve='Z2')
    _ = init_FM_MPS(L=10, d=2, bc='finite', backend='no_symmetry')
    _ = init_Neel_MPS(L=10)
    _ = init_Neel_MPS(L=10, conserve='Z2')
    _ = init_SU2_sym_MPS(L=10, d=2, bc='finite')
    _ = init_Fib_anyon_MPS(L=10, bc='finite')


def test_toy_models():
    _ = TFIModel(L=10, J=1, g=0.8, bc='finite')
    _ = TFIModel(L=10, J=1, g=0.8, bc='infinite', conserve='Z2')
    _ = HeisenbergModel(L=10, J=1, bc='finite', conserve='none')
    _ = HeisenbergModel(L=10, J=1, bc='infinite', conserve='Z2')
    _ = HeisenbergModel(L=10, J=1, bc='finite', conserve='SU2')
    _ = GoldenChainModel(L=10, J=1, bc='finite')


def test_dmrg_golden_chain():
    # energies from MPSKit.jl with DMRG
    GC_energies = {6: -4.02595560765756, 8: -5.54888659415890, 10: -7.0735949995638}
    L = 8
    psi = init_Fib_anyon_MPS(L)
    model = GoldenChainModel(L, J=1)
    dmrg = DMRGEngine(psi, model)
    e = dmrg.run()
    assert abs(e - GC_energies[L]) < 1e-9


def test_dmrg_heisenberg():
    backend = ct.get_backend('fusion_tree', 'numpy')
    L = 8
    e_exact = heisenberg_finite_gs_energy(L, J=1)
    for conserve in ['none', 'Z2', 'SU2']:
        if conserve == 'SU2':
            psi = init_SU2_sym_MPS(L, backend=backend)
        else:
            psi = init_Neel_MPS(L, backend=backend, conserve=conserve)
        model = HeisenbergModel(L, J=1, backend=backend, conserve=conserve)
        dmrg = DMRGEngine(psi, model)
        e = dmrg.run()
        assert abs(e - e_exact) < 1e-9


def test_dmrg_tfi(np_random):
    backend = ct.get_backend('abelian', 'numpy')
    L = 16
    J, g = np_random.random(2)
    e_exact = tfi_finite_gs_energy(L, J, g)

    for conserve in ['none', 'Z2']:
        psi = init_FM_MPS(L, backend=backend, conserve=conserve)
        model = TFIModel(L, J, g, backend=backend, conserve=conserve)
        dmrg = DMRGEngine(psi, model)
        e = dmrg.run()
        assert abs(e - e_exact) < 1e-9
