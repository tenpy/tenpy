from tenpy.models.mixed_xk import (MixedXKLattice, MixedXKModel, SpinlessMixedXKSquare,
                                   HubbardMixedXKSquare)
from test_model import check_general_model
from tenpy.models.fermions_spinless import FermionModel
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import pytest
import numpy as np


@pytest.mark.parametrize('model_class', [SpinlessMixedXKSquare, HubbardMixedXKSquare])
def test_MixedXKModel_general(model_class):
    check_general_model(model_class, dict(Lx=2, Ly=3, bc_MPS='infinite'))


def test_mixed_spinless():
    """
    compare a small system of spinless fermions in real and mixed space
    """
    Lx, Ly = 2, 4
    J, V = 1., 10
    chimax = 100
    #real space
    model_params = dict(J=J,
                        V=V,
                        lattice='Square',
                        Lx=Lx,
                        Ly=Ly,
                        bc_x='open',
                        bc_y='cylinder',
                        bc_MPS='finite')
    M = FermionModel(model_params)

    dmrg_params = {
        'mixer': True,
        'max_E_err': 1.e-12,
        'max_S_err': 1.e-8,
        'trunc_params': {
            'chi_max': chimax,
            'svd_min': 1.e-10
        },
    }
    product_state = [[['full'], ['empty'], ['full'], ['empty']],
                     [['full'], ['empty'], ['full'], ['empty']]]  # half filling
    psi = MPS.from_lat_product_state(M.lat, product_state)
    info = dmrg.run(psi, M, dmrg_params)
    E_real = info['E']

    #measure particle number and Cd C correlators
    N_real = M.lat.mps2lat_values(psi.expectation_value('N'))
    CdC_real = M.lat.mps2lat_values(psi.correlation_function('Cd', 'C')[0, :])

    #mixed space
    model_params = dict(t=J, V=V, Lx=Lx, Ly=Ly, bc_MPS='finite', conserve_k=True)
    M = SpinlessMixedXKSquare(model_params)

    #initial product state with momentum 0
    product_xk = [['full', 'empty', 'full', 'empty'], ['full', 'empty', 'full', 'empty']]
    psi_xk = MPS.from_lat_product_state(M.lat, product_xk)
    info = dmrg.run(psi_xk, M, dmrg_params)  # the main work...
    E_mixed = info['E']

    #measure particle number and Cd C correlators
    N_mixed = np.zeros((Lx, Ly), dtype='complex')
    CdC_mixed = np.zeros((Lx, Ly), dtype='complex')
    for i in range(Lx):
        for j in range(Ly):
            terms_N = M.real_to_mixed_onsite([[1]], (i, j))
            N_mixed[i, j] = psi_xk.expectation_value_terms_sum(terms_N)[0]
            terms_CdC = M.real_to_mixed_correlations_any(['Cd', 'C'], [(1., [0, 0])], [(0, 0),
                                                                                       (i, j)])
            CdC_mixed[i, j] = psi_xk.expectation_value_terms_sum(terms_CdC)[0]

    assert np.abs(E_real - E_mixed) < 1e-10
    assert np.all(np.abs(N_real - N_mixed) < 1e-10)
    assert np.all(np.abs(CdC_real - CdC_mixed) < 1e-10)


@pytest.mark.slow
def test_mixed_hubbard():
    """
    compare the Hubbard model on a small square lattice in in real and mixed space
    """
    Lx, Ly = 2, 3
    t, U = 1., 10
    chimax = 100
    #real space
    model_params = dict(t=t,
                        U=U,
                        lattice='Square',
                        Lx=Lx,
                        Ly=Ly,
                        bc_x='open',
                        bc_y='cylinder',
                        bc_MPS='finite')
    M = FermiHubbardModel(model_params)

    dmrg_params = {
        'mixer': True,
        'max_E_err': 1.e-12,
        'max_S_err': 1.e-10,
        'trunc_params': {
            'chi_max': chimax,
            'svd_min': 1.e-10
        },
    }
    product_state = [[['full'], ['empty'], ['full']],
                     [['full'], ['empty'], ['empty']]]  #yapf: disable
    psi = MPS.from_lat_product_state(M.lat, product_state)
    info = dmrg.run(psi, M, dmrg_params)
    E_real = info['E']

    #measure Sz onsite and SpSm correlators
    Sz_real = M.lat.mps2lat_values(psi.expectation_value('Sz'))
    SpSm_real = M.lat.mps2lat_values(psi.correlation_function('Sp', 'Sm')[0, :])

    #mixed space
    #implemented with two spinless fermion sites for up/down !
    model_params = dict(t=t, U=U, Lx=Lx, Ly=Ly, bc_MPS='finite', conserve_k=True)
    M2 = HubbardMixedXKSquare(model_params)

    #initial product state with momentum 0
    product_xk = [['full', 'full', 'full', 'empty', 'empty', 'empty'],
                  ['full', 'full', 'empty', 'empty', 'empty', 'full']]
    psi_xk = MPS.from_lat_product_state(M2.lat, product_xk)
    info = dmrg.run(psi_xk, M2, dmrg_params)  # the main work...
    E_mixed = info['E']

    #measure Sz onsite and SpSm correlators
    Sz_mixed = np.zeros((Lx, Ly), dtype='complex')
    SpSm_mixed = np.zeros((Lx, Ly), dtype='complex')
    Sp = np.array([[0, 1], [0, 0]])
    Sm = np.array([[0, 0], [1, 0]])
    Sz = np.array([[1, 0], [0, -1]]) * 0.5
    for i in range(Lx):
        for j in range(Ly):
            terms_Sz = M2.real_to_mixed_onsite(Sz, (i, j))
            Sz_mixed[i, j] = psi_xk.expectation_value_terms_sum(terms_Sz)[0]
            terms_SpSm = M2.real_to_mixed_two_site(Sp, (0, 0), Sm, (i, j))
            SpSm_mixed[i, j] = psi_xk.expectation_value_terms_sum(terms_SpSm)[0]

    assert np.abs(E_real - E_mixed) < 1e-7
    assert np.all(np.abs(Sz_real - Sz_mixed) < 1e-7)
    assert np.all(np.abs(SpSm_real - SpSm_mixed) < 1e-7)
