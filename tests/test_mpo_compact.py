"""
NOTE: THIS IS A MAKESHIFT TEST IMPLEMENTATION FOR MPOCompact.

EXPECT LOTS OF CHANGES

Execute with
>>> pytest tests/test_mpo_compact.py -s
to show intermediate results
"""

import lasyten as lasy
import pytest

from tenpy.models.aklt import AKLTChain
from tenpy.models.clock import ClockChain
from tenpy.models.fermions_spinless import FermionChain, FermionModel
from tenpy.models.haldane import BosonicHaldaneModel, FermionicHaldaneModel
from tenpy.models.hofstadter import HofstadterBosons, HofstadterFermions
from tenpy.models.hubbard import BoseHubbardChain, BoseHubbardModel, FermiHubbardChain, FermiHubbardModel
from tenpy.models.pxp import PXPChain
from tenpy.models.spins import *
from tenpy.models.spins_nnn import SpinChainNNN2
from tenpy.models.tf_ising import TFIChain
from tenpy.models.toric_code import ToricCode
from tenpy.models.xxz_chain import XXZChain2
from tenpy.networks.mpo import MPO, MPOCompact, MPOGraph

L = 6
CASES = [
    # --- 1D spin chains (nearest neighbour; the ones you verified already) ---
    (TFIChain, dict(L=L, conserve='parity')),
    (XXZChain2, dict(L=L, conserve='Sz')),
    (SpinChain, dict(L=L, conserve='Sz')),
    (SpinChain, dict(L=L, conserve='parity')),
    (SpinChain, dict(L=4, conserve='SU2')),  # SU(2) data limit: keep L <= 6
    (AKLTChain, dict(L=L, conserve='Sz')),
    (ClockChain, dict(L=L, q=3)),
    # --- longer range / multi-site (shared prefixes) ---
    (SpinChainNNN2, dict(L=L, conserve='Sz')),  # next-nearest neighbour
    (DipolarSpinChain, dict(L=L, conserve='Sz')),  # 4-site dipolar term
    (DipolarSpinChain, dict(L=L, conserve=None, J4=1.0)),
    (PXPChain, dict(L=L)),  # 3-site
    (ToricCode, dict(Lx=2, Ly=2, conserve='parity')),  # 4-site, 2D, 8 sites
    (ToricCode, dict(Lx=2, Ly=2, conserve=None)),
    # --- bosons / fermions (fermions need Jordan-Wigner handling) ---
    (BoseHubbardChain, dict(L=L, conserve='N')),
    (FermionChain, dict(L=L, conserve='N')),
    (FermiHubbardChain, dict(L=L)),
    # --- 2D models mapped onto an MPS ---
    (SpinModel, dict(lattice='Square', Lx=3, Ly=3)),
    (FermionModel, dict(lattice='Square', Lx=2, Ly=3)),
    (BoseHubbardModel, dict(lattice='Square', Lx=2, Ly=3)),
    (FermiHubbardModel, dict(lattice='Square', Lx=2, Ly=3)),
    (BosonicHaldaneModel, dict(Lx=2, Ly=2)),
    (FermionicHaldaneModel, dict(Lx=2, Ly=2)),
    (HofstadterBosons, dict(Lx=3, Ly=3, phi=(1, 3), conserve='N', Nmax=1)),
    (HofstadterFermions, dict(Lx=3, Ly=3, phi=(1, 3), conserve='N')),
]


# @pytest.mark.parametrize('length', range(2, 10))
def _test_pauli_strings_left_right_spin(length):
    model_params = {'L': length, 'bc_MPS': 'finite', 'conserve': 'SU2'}
    model = SpinChain(model_params)

    mpo_builder = MPOCompact.from_terms([model.coupling_terms], model, bc='finite')

    PL = mpo_builder._gen_left_pauli_strings()
    PR = mpo_builder._gen_right_pauli_strings()

    print(f'PL = {PL}')
    print(f'PR = {PR}')
    print(f'V(puq) = {mpo_builder._coeff_V_map}')
    print(f'opstr_per_site = {mpo_builder._opstr_per_site}')
    print(f'op_basis = {mpo_builder._op_basis}')

    Ws: list[lasy.Tensor] = []
    for n in range(0, length - 1):
        Ws.append(mpo_builder._sparse_MPO(n, PL[n], PL[n + 1]))

    for W in Ws:
        print(W.to_numpy())


@pytest.mark.parametrize('Model, params', CASES, ids=[f'{M.__name__}-{i}' for i, (M, _) in enumerate(CASES)])
def test_compare_sparse_MPO_and_MPOGraph(Model, params, test_equal=True):
    print(Model.__name__)
    # model_params = {
    #     'L': length,
    #     'bc_MPS': 'finite',
    #     'conserve': 'best'
    # }
    model_params = params
    model_params['bc_MPS'] = 'finite'

    model = Model(model_params)

    length = model.lattice.N_sites  # model_params['L']

    mpo_compact = MPOCompact.from_terms(
        model.get_nonzero_terms(),  # model.get_nonzero_terms(),
        model,
        bc='finite',
        unit_cell_width=model.lattice.mps_unit_cell_width,
    )
    # for positions, name, strength in model.iter_terms():
    #     mpo_compact.add_multi_coupling(model.couplings[name], positions, strength)

    PL = mpo_compact._gen_left_pauli_strings()
    PR = mpo_compact._gen_right_pauli_strings()

    Ws: list[lasy.Tensor] = []
    for n in range(0, length):
        Ws.append(mpo_compact._sparse_MPO(n, PL[n], PL[n + 1]))

    def keep(W, leg, name):
        return lasy.apply_mask(W, W.get_leg(leg).projection_onto_summand(name), leg)

    Ws[0] = lasy.apply_mask(Ws[0], Ws[0].get_leg('wL').projection_onto_summand('IdL'), 'wL')
    Ws[-1] = lasy.apply_mask(Ws[-1], Ws[-1].get_leg('wR').projection_onto_summand('IdR'), 'wR')

    mpo1 = MPO(
        mpo_compact.sites,
        Ws,
        bc=mpo_compact.bc,
        max_range=None,
        explicit_plus_hc=False,
        mps_unit_cell_width=mpo_compact.unit_cell_width,
    )

    mpo_graph = MPOGraph.from_terms(
        model.get_nonzero_terms(),
        model,
        bc='finite',
        unit_cell_width=model.lattice.mps_unit_cell_width,
    )

    mpo2 = mpo_graph.build_MPO(explicit_plus_hc=False)

    print(f'mpo1.chi = {mpo1.chi}')
    print(f'mpo2.chi = {mpo2.chi}')
    if test_equal:
        dist = mpo_distance(mpo1, mpo2)
        print(f'relative distance = {dist:.2e}')
        assert dist < 1e-6, f'MPOs differ: relative distance {dist:.2e}'
        assert all(c1 <= c2 for c1, c2 in zip(mpo1.chi, mpo2.chi)), (
            f'compressed MPO is larger: {mpo1.chi} vs {mpo2.chi}'
        )
        print('Passed equivalence check!')


@pytest.mark.parametrize('Model, params', CASES, ids=[f'{M.__name__}-{i}' for i, (M, _) in enumerate(CASES)])
def test_pauli_string_MPO(Model, params, test_equal=True):
    print(Model.__name__)
    model_params = params
    model_params['bc_MPS'] = 'finite'

    model = Model(model_params)

    length = model.lattice.N_sites  # model_params['L']

    mpo_compact = MPOCompact.from_terms(
        model.get_nonzero_terms(),  # model.get_nonzero_terms(),
        model,
        bc='finite',
        unit_cell_width=model.lattice.mps_unit_cell_width,
    )

    PL = mpo_compact._gen_left_pauli_strings()
    PR = mpo_compact._gen_right_pauli_strings()

    Ws: list[lasy.Tensor] = []
    Wsc: list[lasy.Tensor] = []
    # for n in range(0, length):
    #     Ws.append(mpo_compact._sparse_MPO(n, PL[n], PL[n+1]))

    Ws, SVs = mpo_compact.pauli_string_MPO()

    def keep(W, leg, name):
        return lasy.apply_mask(W, W.get_leg(leg).projection_onto_summand(name), leg)

    Ws[0] = lasy.apply_mask(Ws[0], Ws[0].get_leg('wL').projection_onto_summand('IdL'), 'wL')
    Ws[-1] = lasy.apply_mask(Ws[-1], Ws[-1].get_leg('wR').projection_onto_summand('IdR'), 'wR')

    mpo1 = MPO(
        mpo_compact.sites,
        Ws,
        bc=mpo_compact.bc,
        max_range=None,
        explicit_plus_hc=False,
        mps_unit_cell_width=mpo_compact.unit_cell_width,
    )

    mpo_graph = MPOGraph.from_terms(
        model.get_nonzero_terms(),
        model,
        bc='finite',
        unit_cell_width=model.lattice.mps_unit_cell_width,
    )

    mpo2 = mpo_graph.build_MPO()

    print(f'mpo1.chi = {mpo1.chi}')
    print(f'mpo2.chi = {mpo2.chi}')
    # if test_equal:
    #     tensor1 = mpo_to_tensor(mpo1)
    #     tensor2 = mpo_to_tensor(mpo2)
    #     assert lasy.almost_equal(tensor1, tensor2)
    #     print('Passed equivalence check!')

    if test_equal:
        dist = mpo_distance(mpo1, mpo2)
        print(f'relative distance = {dist:.2e}')
        assert dist < 1e-6, f'MPOs differ: relative distance {dist:.2e}'
        assert all(c1 <= c2 for c1, c2 in zip(mpo1.chi, mpo2.chi)), (
            f'compressed MPO is larger: {mpo1.chi} vs {mpo2.chi}'
        )
        print('Passed equivalence check!')


def mpo_to_tensor(mpo):
    T = mpo.get_W(0)
    for i in range(1, mpo.L):
        r1 = {'p': 'p0', 'p*': 'p0*'} if i == 1 else {}
        T = lasy.planar_contraction(T, mpo.get_W(i), 'wR', 'wL', relabel1=r1, relabel2={'p': f'p{i}', 'p*': f'p{i}*'})
    return lasy.squeeze_legs(T, ['wL', 'wR'])


def mpo_inner(A, B):
    """<A, B> = Tr(A† B) for two finite MPOs with trivial boundary legs."""
    wA = lasy.squeeze_legs(A.get_W(0), ['wL'])  # drop the trivial left boundary leg
    wB = lasy.squeeze_legs(B.get_W(0), ['wL'])
    res = lasy.planar_contraction(wA.hc, wB, ['p', 'p*'], ['p*', 'p'], relabel1={'wR*': 'wRa'}, relabel2={'wR': 'wRb'})
    for i in range(1, A.L):
        res = lasy.planar_contraction(res, A.get_W(i).hc, 'wRa', 'wL*', relabel2={'wR*': 'wRa'})
        res = lasy.planar_contraction(res, B.get_W(i), ['wRb', 'p*', 'p'], ['wL', 'p', 'p*'], relabel2={'wR': 'wRb'})
    return complex(lasy.item(lasy.squeeze_legs(res, ['wRa', 'wRb'])))


def mpo_distance(A, B):
    """Relative Frobenius distance ||A - B|| / ||B||."""
    aa, bb, ab = mpo_inner(A, A), mpo_inner(B, B), mpo_inner(A, B)
    d2 = (aa + bb - 2 * ab.real).real
    return max(d2, 0) ** 0.5 / bb.real**0.5


# if __name__ == '__main__':

#     # for n in range(2, 10):
#     #     print(n)
#     #     test_pauli_strings_left_right_spin(n)


#     params = (5, SpinChain)

#     L = 8

#     # CASES = [
#     #     (L, SpinChain, dict(conserve='Sz', Jx=1.0, Jy=1.0, Jz=0.5)),   # abelian, three strings per bond
#     #     (L, SpinChain, dict(conserve='parity', Jx=1.0, Jy=0.5, Jz=0.7)),  # Z2, anisotropic
#     #     (3, SpinChain, dict(conserve='SU2', S=1)),                      # non-abelian, spin 1
#     #     (L, TFIChain, dict(conserve='parity')),
#     #     (L, FourSiteChain, dict(conserve='Sz')),
#     #     (L, SharedPrefixChain, dict(conserve='parity'))
#     # ]

#     L = 6
#     CASES = [
#         # --- 1D spin chains (nearest neighbour; the ones you verified already) ---
#         (TFIChain,          dict(L=L, conserve='parity')),
#         (XXZChain2,         dict(L=L, conserve='Sz')),
#         (SpinChain,         dict(L=L, conserve='Sz')),
#         (SpinChain,         dict(L=L, conserve='parity')),
#         (SpinChain,         dict(L=4, conserve='SU2')),              # SU(2) data limit: keep L <= 6
#         (AKLTChain,         dict(L=L, conserve='Sz')),
#         (ClockChain,        dict(L=L, q=3)),
#         # --- longer range / multi-site (shared prefixes) ---
#         (SpinChainNNN2,     dict(L=L, conserve='Sz')),               # next-nearest neighbour
#         (DipolarSpinChain,  dict(L=L, conserve='Sz')),               # 4-site dipolar term
#         (DipolarSpinChain,  dict(L=L, conserve=None, J4=1.0)),
#         (PXPChain,          dict(L=L)),                              # 3-site
#         (ToricCode,         dict(Lx=2, Ly=2, conserve='parity')),    # 4-site, 2D, 8 sites
#         (ToricCode,         dict(Lx=2, Ly=2, conserve=None)),
#         # --- bosons / fermions (fermions need Jordan-Wigner handling) ---
#         (BoseHubbardChain,  dict(L=L, conserve='N')),
#         (FermionChain,      dict(L=L, conserve='N')),
#         (FermiHubbardChain, dict(L=L)),
#         # --- 2D models mapped onto an MPS ---
#         (SpinModel,             dict(lattice='Square', Lx=3, Ly=3)),
#         (FermionModel,          dict(lattice='Square', Lx=2, Ly=3)),
#         (BoseHubbardModel,      dict(lattice='Square', Lx=2, Ly=3)),
#         (FermiHubbardModel,     dict(lattice='Square', Lx=2, Ly=3)),
#         (BosonicHaldaneModel,   dict(Lx=2, Ly=2)),
#         (FermionicHaldaneModel, dict(Lx=2, Ly=2)),
#         (HofstadterBosons,      dict(Lx=3, Ly=3, phi=(1, 3), conserve='N', Nmax=1)),
#         (HofstadterFermions,    dict(Lx=3, Ly=3, phi=(1, 3), conserve='N')),
#     ]

#     test_func = test_pauli_string_MPO

# for case in CASES:
#     test_pauli_string_MPO(*case, test_equal=True)
