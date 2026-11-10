"""A collection of tests for (classes in) :mod:`tenpy.models.model`."""

# Copyright (C) TeNPy Developers, Apache license
import itertools

import lasyten as lasy
import numpy as np
import numpy.testing as npt
import pytest
from lasyten.models.sites import SpinSite
from lasyten.tensors import almost_equal

from tenpy.algorithms import ExactDiag
from tenpy.models import lattice, model
from tenpy.models.spins import Coupling_from_numpy_onsite, DipolarSpinChain, SpinChain
from tenpy.models.xxz_chain import XXZChain2
from tenpy.tools import asConfig

__all__ = ['check_model_sanity', 'check_general_model']


def check_model_sanity(M, hermitian=True):
    """call M.test_sanity() for all different subclasses of M."""
    M.test_sanity()
    if hermitian:
        with pytest.raises(NotImplementedError, match='MPO.dagger not working well, see #682'):
            assert M.get_H_MPO().is_hermitian()
        # skipping the check while it does not work -> reactivate it!

        try:
            H_bond = M.get_H_bond()
        except Exception:
            pass
        else:
            for H in H_bond:
                if H is None:
                    continue
                H = H.to_tensor()
                assert almost_equal(H, H.hc)


def check_general_model(ModelClass, model_pars={}, check_pars={}, hermitian=True):
    """Create a model for different sets of parameters and check its sanity.

    Parameters
    ----------
    ModelClass :
        We generate models of this class
    model_pars : dict
        Model parameters used.
    check_pars : dict
        pairs (`key`, `list of values`); we update ``model_pars[key]`` with any values of
        ``check_pars[key]`` (in each possible combination!) and create a model for it.
    hermitian : bool
        If True, check that the Hamiltonian is Hermitian.
    """
    if len(check_pars) == 0:
        M = ModelClass(model_pars.copy())
        check_model_sanity(M, hermitian)
        return
    for vals in itertools.product(*list(check_pars.values())):
        print('-' * 40)
        params = model_pars.copy()
        for k, v in zip(list(check_pars.keys()), vals):
            params[k] = v
        print('check_model_sanity with following parameters:')
        print(params)
        M = ModelClass(params)
        check_model_sanity(M, hermitian)


@pytest.mark.parametrize('bc', ['open', 'periodic'])
@pytest.mark.parametrize('bc_MPS', ['finite', 'infinite'])
@pytest.mark.parametrize('tensor_backend', ['fusion_tree', 'abelian', 'no_symmetry'])
@pytest.mark.parametrize('conserve', ['SU2', 'Sz', 'parity', None])
def test_CouplingModel_basic(bc, bc_MPS, tensor_backend, conserve):
    backend = lasy.get_backend(tensor_backend)

    if (tensor_backend == 'abelian' and conserve == 'SU2') or (
        tensor_backend == 'no_symmetry' and conserve is not None
    ):
        with pytest.raises(ValueError, match='backend does not support this symmetry'):
            site = SpinSite(conserve=conserve, backend=backend)
        # error is expected behavior
        return

    site = SpinSite(conserve=conserve, backend=backend)

    if bc == 'open' and bc_MPS == 'infinite':
        with pytest.raises(ValueError, match='Need periodic boundary conditions'):
            spin_half_lat = lattice.Chain(5, site, bc=bc, bc_MPS=bc_MPS)
        # error is expected behavior
        return

    spin_half_lat = lattice.Chain(5, site, bc=bc, bc_MPS=bc_MPS)

    M = model.CouplingModel(spin_half_lat)
    M.define_coupling(
        lasy.models.spin_spin_coupling([site, site], Jx=1, Jy=1, Jz=1, name='S.S'),
        hc='S.S',
        permute='invariant',
    )
    if conserve not in ['SU2', 'Sz']:
        M.define_coupling(
            lasy.models.spin_spin_coupling([site, site], Jx=1, name='Sx.Sx'),
            hc='Sx.Sx',
            permute='invariant',
        )

    # 1) Only NN terms
    M.add_two_site_coupling(1.2, 'S.S', dx=1)
    if conserve == 'SU2':
        with pytest.raises(lasy.SymmetryError):
            M.add_onsite_operator(0.3, 'Sz')
    else:
        M.add_onsite_operator(0.3, 'Sz')
    if conserve not in ['SU2', 'Sz']:
        M.add_two_site_coupling(0.7, 'Sx.Sx', dx=1)
    if conserve is None:
        M.add_onsite_operator(0.8, 'Sx')
    with pytest.raises(ValueError):
        M.add_onsite_operator(0.2, 'unknownOp')
    with pytest.raises(ValueError):
        M.add_two_site_coupling(0.7, 'unknownCoupling', dx=1)

    M.test_sanity()

    H_MPO = M.get_H_MPO()
    H_MPO.test_sanity()

    if bc == 'periodic' and bc_MPS == 'finite':
        with pytest.raises(ValueError, match='not nearest neighbor'):
            # periodic bc but finite bc_MPS leads to a long-range coupling
            M.get_H_bond()
        # error is expected behavior
    else:
        H_bond = M.get_H_bond()

        with pytest.raises(ValueError, match='Incompatible legs'):
            H_bond_from_MPO = model.calc_H_bond_from_MPO(H_MPO)
            for h1, h2 in zip(H_bond, H_bond_from_MPO):
                assert lasy.almost_equal(h1, h2)
        pytest.xfail('calc_H_bond_from_MPO currently bugged')


def test_ext_flux():
    Lx, Ly = 3, 4
    fermion_site = lasy.sites.SpinlessFermionSite()
    lat = lattice.Square(Lx, Ly, fermion_site, bc=['periodic', 'periodic'], bc_MPS='infinite')
    M = model.CouplingModel(lat)
    strength = 1.23
    strength_array = np.ones((Lx, Ly)) * strength
    for phi in [0, 2 * np.pi]:  # flux shouldn't do anything
        print('phi = ', phi)
        for dx in [1, 0], [0, 1], [0, 2], [1, -1], [-2, 2]:
            print('dx = ', dx)
            strength_flux = M.coupling_strength_add_ext_flux(strength, [1, 0], [0, phi])
            npt.assert_array_almost_equal_nulp(strength_flux, strength_array, 10)
    for phi in [np.pi / 2, 0.123]:
        print('phi = ', phi)
        strength_hop_x = M.coupling_strength_add_ext_flux(strength, [1, 0], [0, phi])
        npt.assert_array_almost_equal_nulp(strength_hop_x, strength_array, 10)
        expect_y_1 = np.array(strength_array, dtype=np.complex128)
        expect_y_1[:, -1:] = strength * np.exp(1.0j * phi)
        for dx in [[0, 1], [0, -1], [1, -1], [1, 1]]:
            print('dx = ', dx)
            strength_hop_y_1 = M.coupling_strength_add_ext_flux(strength, dx, [0, phi])
            if dx[1] < 0:
                npt.assert_array_almost_equal_nulp(strength_hop_y_1, expect_y_1, 10)
            else:
                npt.assert_array_almost_equal_nulp(strength_hop_y_1, np.conj(expect_y_1), 10)
        expect_y_2 = np.array(strength_array, dtype=np.complex128)
        expect_y_2[:, -2:] = strength * np.exp(1.0j * phi)
        for dx in [[0, 2], [0, -2], [1, 2], [3, 2]]:
            print('dx = ', dx)
            strength_hop_y_2 = M.coupling_strength_add_ext_flux(strength, dx, [0, phi])
            if dx[1] < 0:
                npt.assert_array_almost_equal_nulp(strength_hop_y_2, expect_y_2, 10)
            else:
                npt.assert_array_almost_equal_nulp(strength_hop_y_2, np.conj(expect_y_2), 10)


def test_CouplingModel_shift(Lx=3, Ly=3, shift=1):
    bc = ['periodic', shift]
    s = lasy.sites.SpinSite()
    spin_half_square = lattice.Square(Lx, Ly, s, bc=bc, bc_MPS='infinite')
    M = model.CouplingModel(spin_half_square)
    Sz = s.get_op('Sz').as_SymmetricTensor()
    Sz_Sz = lasy.outer(Sz, Sz, relabel1={'p': 'p0', 'p*': 'p0*'}, relabel2={'p': 'p1', 'p*': 'p1*'})
    Sz_Sz_Sz = lasy.outer(Sz_Sz, Sz, relabel2={'p': 'p2', 'p*': 'p2*'})
    M.define_coupling(lasy.Coupling.from_tensor(Sz_Sz, [s, s], 'Sz.Sz'), hc='Sz.Sz', permute='invariant')
    M.define_coupling(lasy.Coupling.from_tensor(Sz_Sz_Sz, [s, s, s], 'Sz.Sz.Sz'), hc='Sz.Sz.Sz', permute='invariant')
    M.add_two_site_coupling(1.2, 'Sz.Sz', dx=[1, 0])
    M.add_multi_site_coupling(0.8, 'Sz.Sz.Sz', dx=[[0, 0], [0, 1], [1, 0]])
    M.test_sanity()
    H = M.get_H_MPO()

    # check translation invariance of the MPO
    for y in range(Ly):
        W = H._W[y]
        wL_labels = W.get_leg('wL').summand_labels
        assert 'IdL' in wL_labels
        assert 'IdR' in wL_labels
        assert sum(l.startswith('Sz.Sz#') for l in wL_labels) == 3
        assert sum(l.startswith('Sz.Sz.Sz#') for l in wL_labels) == (4 if y == 0 else 5)
        for x in range(1, Lx):
            W_shifted = H._W[y + x * Ly]
            wL_labels = W_shifted.get_leg('wL').summand_labels
            assert 'IdL' in wL_labels
            assert 'IdR' in wL_labels
            assert sum(l.startswith('Sz.Sz#') for l in wL_labels) == 3
            assert sum(l.startswith('Sz.Sz.Sz#') for l in wL_labels) == (4 if y == 0 else 5)
            # comparing tensors is too annoying, since they dont have matching order of cells...
            # -> at least compare the legs
            assert W.get_leg('wL').is_isomorphic_to(W_shifted.get_leg('wL'))


@pytest.mark.parametrize('bc, bc_MPS', [['open', 'finite'], ['periodic', 'infinite']])
def test_CouplingModel_fermions(bc, bc_MPS):
    s = lasy.sites.SpinlessFermionSite()
    fermion_lat = lattice.Chain(5, s, bc=bc, bc_MPS=bc_MPS)
    M = model.CouplingModel(fermion_lat)
    M.define_coupling(lasy.models.hopping([s, s], t_right=0, name='Cd.C'), hc='C.Cd', permute='swap_hc')
    M.add_two_site_coupling(1.2, 'Cd.C', dx=1)
    M.add_two_site_coupling(1.2, 'Cd.C', dx=-1)
    M.test_sanity()
    M.get_H_MPO()
    M.get_H_bond()


def test_CouplingModel_explicit():
    s = lasy.sites.SpinlessFermionSite()
    fermion_lat_cyl = lattice.Square(1, 2, s, bc='periodic', bc_MPS='infinite')

    M = model.CouplingModel(fermion_lat_cyl)
    M.define_coupling(lasy.models.hopping([s, s], t_right=0, name='Cd.C'), hc='C.Cd', permute='swap_hc')
    M.define_coupling(lasy.models.density_density_interaction([s, s], name='N.N'), hc='N.N', permute='invariant')

    M.add_onsite_operator(0.125, 'N')
    M.add_two_site_coupling(0.25, 'Cd.C', dx=[0, 1])
    M.add_two_site_coupling(0.25, 'Cd.C', dx=[0, -1])
    M.add_two_site_coupling(1.5, 'Cd.C', dx=[1, 0])
    M.add_two_site_coupling(1.5, 'Cd.C', dx=[-1, 0])
    M.add_two_site_coupling(4.0, 'N.N', dx=[-2, -1])  # a full unit cell inbetween!

    H_mpo = M.get_H_MPO()

    W0_new = H_mpo.get_W(0)
    W1_new = H_mpo.get_W(1)

    w_triv = lasy.ElementarySpace.from_trivial_sector(symmetry=M.symmetry)
    w_odd = lasy.ElementarySpace(M.symmetry, M.symmetry.factors[0].odd)
    Id_0 = s.get_onsite_W_block('Id', w=w_triv)
    Id_1 = s.get_onsite_W_block('Id', w=w_odd)
    N_block = s.get_onsite_W_block('N', w=w_triv)
    Cd_L, C_R = lasy.models.hopping([s, s], t_right=0).factorization
    C_L, Cd_R = lasy.models.hopping([s, s], t_left=0).factorization
    N_L, N_R = lasy.models.density_density_interaction([s, s]).factorization

    print()
    print(H_mpo.show_W_grid(0))
    print()
    print(H_mpo.show_W_grid(1))

    # fmt: off
    # NOTE: we currently do not re-use edges in the MPOGraph
    #       commented out is what we may expect once we do
    #       The first two terms (dx=[0, 1] and dx=[0, -1]) act on the exact same sites,
    #       so those *are* combined

    # W0_ex = [[Id,   Cd_L, C_L,  N_L,  None, None, None, None, None, 0.125 * N_block],
    #          [None, None, None, None, None, None, None, None, None, 1.5 * C_R      ],
    #          [None, None, None, None, None, None, None, None, None, 1.5 * Cd_R     ],
    #          [None, None, None, None, None, None, None, Id,   None, None           ],
    #          [None, None, None, None, Id,   None, None, None, None, None           ],
    #          [None, None, None, None, None, Id,   None, None, None, None           ],
    #          [None, None, None, None, None, None, Id,   None, None, None           ],
    #          [None, None, None, None, None, None, None, None, Id,   None           ],
    #          [None, None, None, None, None, None, None, None, None, 4.0 * N_R      ],
    #          [None, None, None, None, None, None, None, None, None, Id             ]]

    # W1_ex = [[Id,   None, None, None, Cd_L, C_L,  N_L,  None, None, 0.125 * N_block],
    #          [None, Id,   None, None, None, None, None, None, None, 0.5 * C_R      ],
    #          [None, None, Id,   None, None, None, None, None, None, .5 * Cd_R      ],
    #          [None, None, None, Id,   None, None, None, None, None, None           ],
    #          [None, None, None, None, None, None, None, None, None, 1.5 * C_R      ],
    #          [None, None, None, None, None, None, None, None, None, 1.5 * Cd_R     ],
    #          [None, None, None, None, None, None, None, None, Id,   None           ],
    #          [None, None, None, None, None, None, None, Id,   None, None           ],
    #          [None, None, None, None, None, None, None, None, None, 4.0 * N_R      ],
    #          [None, None, None, None, None, None, None, None, None, Id             ]]

    W0_ex = [[Id_0, C_L,  C_L,  None, Cd_L, Cd_L, None, N_L,  None, None, None, 0.125 * N_block],
             [None, None, None, None, None, None, None, None, None, None, None, 1.5 * Cd_R     ],
             [None, None, None, Id_1, None, None, None, None, None, None, None, None           ],
             [None, None, None, None, None, None, None, None, None, None, None, 1.5 * C_R      ],
             [None, None, None, None, None, None, Id_1, None, None, None, None, None           ],
             [None, None, None, None, None, None, None, None, Id_0, None, None, None           ],
             [None, None, None, None, None, None, None, None, None, Id_0, None, None           ],
             [None, None, None, None, None, None, None, None, None, None, Id_0, None           ],
             [None, None, None, None, None, None, None, None, None, None, None, 4.0 * N_R      ],
             [None, None, None, None, None, None, None, None, None, None, None, Id_0           ]]

    W1_ex = [[Id_0, None, C_L,  None, Cd_L, None, None, N_L,  None, 0.125 * N_block],
             [None, None, None, None, None, None, None, None, None, 2 * 0.25 * Cd_R],
             [None, Id_1, None, None, None, None, None, None, None, None           ],
             [None, None, None, None, None, None, None, None, None, 1.5 * Cd_R     ],
             [None, None, None, None, None, None, None, None, None, 2 * 0.25 * C_R ],
             [None, None, None, Id_1, None, None, None, None, None, None           ],
             [None, None, None, None, None, None, None, None, None, 1.5 * C_R      ],
             [None, None, None, None, None, Id_0, None, None, None, None           ],
             [None, None, None, None, None, None, Id_0, None, None, None           ],
             [None, None, None, None, None, None, None, None, None, 4.0 * N_R      ],
             [None, None, None, None, None, None, None, None, Id_0, None           ],
             [None, None, None, None, None, None, None, None, None, Id_0           ]]
    # fmt: on

    # compare cells individually: better for debugging
    _compare_grids(W0_new, W0_ex, 'W0')
    _compare_grids(W1_new, W1_ex, 'W1')

    # compare reconstructed tensor: easier to trust
    W0_ex = lasy.tensor_from_grid(W0_ex, ['wL', 'p', 'wR', 'p*'])
    W1_ex = lasy.tensor_from_grid(W1_ex, ['wL', 'p', 'wR', 'p*'])
    assert lasy.almost_equal(W0_new, W0_ex)
    assert lasy.almost_equal(W1_new, W1_ex)


@pytest.mark.parametrize('use_plus_hc', [True, False])
def test_CouplingModel_multi_couplings_explicit(use_plus_hc):
    s = lasy.sites.SpinlessFermionSite()
    fermion_lat_cyl = lattice.Square(1, 2, s, bc='periodic', bc_MPS='infinite')

    # create a weird fermionic model with 3-body interactions
    M = model.CouplingModel(fermion_lat_cyl)
    M.define_coupling(lasy.models.hopping([s, s], t_right=0, name='Cd.C'), hc='C.Cd', permute='swap_hc')
    M.define_coupling(lasy.models.density_density_interaction([s, s], name='N.N'), hc='N.N', permute='invariant')
    N_Cd_C = lasy.outer(
        s.get_op('N').as_SymmetricTensor(),
        M.couplings['Cd.C'].to_tensor(),
        relabel1={'p': 'p0', 'p*': 'p0*'},
        relabel2={'p0': 'p1', 'p0*': 'p1*', 'p1': 'p2', 'p1*': 'p2*'},
    )
    Cd_N_C = -1 * lasy.permute_legs(N_Cd_C, codomain=['p1', 'p0', 'p2'], domain=['p1*', 'p0*', 'p2*']).relabel(
        {'p0': 'p1', 'p0*': 'p1*', 'p1': 'p0', 'p1*': 'p0*'}
    )
    Cd_C_N = -1 * lasy.permute_legs(N_Cd_C, codomain=['p0', 'p2', 'p1'], domain=['p0*', 'p2*', 'p1*']).relabel(
        {'p2': 'p1', 'p2*': 'p1*', 'p1': 'p2', 'p1*': 'p2*'}
    )

    def N_Cd_C_permute(perm):
        if np.array_equal(perm, [0, 1, 2]):
            return {'N.Cd.C': 1.0}
        if np.array_equal(perm, [0, 2, 1]):
            return {'N.C.Cd': -1.0}
        if np.array_equal(perm, [1, 0, 2]):
            return {'Cd.N.C': -1.0}
        if np.array_equal(perm, [1, 2, 0]):
            return {'Cd.C.N': 1.0}
        if np.array_equal(perm, [2, 1, 0]):
            return {'C.Cd.N': -1.0}
        if np.array_equal(perm, [2, 0, 1]):
            return {'Cd.C.N': 1.0}
        raise RuntimeError

    M.define_coupling(lasy.Coupling.from_tensor(N_Cd_C, [s, s, s], name='N.Cd.C'), hc='N.C.Cd', permute=N_Cd_C_permute)
    M.define_coupling(lasy.Coupling.from_tensor(Cd_N_C, [s, s, s], name='Cd.N.C'), hc='C.N.Cd', permute=None)
    M.define_coupling(lasy.Coupling.from_tensor(Cd_C_N, [s, s, s], name='Cd.C.N'), hc='C.Cd.N', permute=None)

    M.add_onsite_operator(0.125, 'N')
    M.add_two_site_coupling(0.25, 'Cd.C', dx=[0, 1], plus_hc=use_plus_hc)
    M.add_two_site_coupling(1.5, 'Cd.C', dx=[1, 0], plus_hc=use_plus_hc)
    if not use_plus_hc:
        M.add_two_site_coupling(0.25, 'Cd.C', dx=[0, -1], plus_hc=use_plus_hc)
        M.add_two_site_coupling(1.5, 'Cd.C', dx=[-1, 0], plus_hc=use_plus_hc)
    M.add_multi_site_coupling(4.0, 'N.N', dx=[[0, 0], [-2, -1]])
    M.add_multi_site_coupling(1.125, 'N.Cd.C', dx=[[0, 0], [0, 1], [1, 0]])

    H_mpo = M.get_H_MPO()

    W0_new = H_mpo.get_W(0)
    W1_new = H_mpo.get_W(1)

    w_triv = lasy.ElementarySpace.from_trivial_sector(symmetry=M.symmetry)
    w_odd = lasy.ElementarySpace(M.symmetry, M.symmetry.factors[0].odd)
    Id_0 = s.get_onsite_W_block('Id', w=w_triv)
    Id_1 = s.get_onsite_W_block('Id', w=w_odd)
    N_0 = s.get_onsite_W_block('N', w=w_triv)
    N_1 = s.get_onsite_W_block('N', w=w_odd)
    N_block = s.get_onsite_W_block('N', w=w_triv)
    Cd_L, C_R = lasy.models.hopping([s, s], t_right=0).factorization
    C_L, Cd_R = lasy.models.hopping([s, s], t_left=0).factorization

    print()
    print(H_mpo.show_W_grid(0))
    print()
    print(H_mpo.show_W_grid(1))

    # fmt: off
    # NOTE: we currently do not re-use edges in the MPOGraph
    #       commented out is what we may expect once we do
    #       The first two terms (dx=[0, 1] and dx=[0, -1]) act on the exact same sites,
    #       so those *are* combined
    W0_ex = [[Id_0, C_L,  C_L,  None, Cd_L, Cd_L, None, Cd_L, None, N_0,  N_0,  None, None, None, 0.125 * N_block],
             [None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1.5 * Cd_R     ],
             [None, None, None, Id_1, None, None, None, None, None, None, None, None, None, None, None           ],
             [None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1.5 * C_R      ],
             [None, None, None, None, None, None, Id_1, None, None, None, None, None, None, None, None           ],
             [None, None, None, None, None, None, None, None, Id_1, None, None, None, None, None, None           ],
             [None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1.125 * C_R    ],
             [None, None, None, None, None, None, None, None, None, None, None, Id_0, None, None, None           ],
             [None, None, None, None, None, None, None, None, None, None, None, None, Id_0, None, None           ],
             [None, None, None, None, None, None, None, None, None, None, None, None, None, Id_0, None           ],
             [None, None, None, None, None, None, None, None, None, None, None, None, None, None, 4.0 * N_0      ],
             [None, None, None, None, None, None, None, None, None, None, None, None, None, None, Id_0           ]]

    # Note: the signs come fermionic anti-commutation and were carefully checked by hand.
    W1_ex = [[Id_0, None, C_L,  None, Cd_L, None, None, None, None, N_0,  None, 0.125 * N_block],
             [None, None, None, None, None, None, None, None, None, None, None, 2 * 0.25 * Cd_R],
             [None, Id_1, None, None, None, None, None, None, None, None, None, None           ],
             [None, None, None, None, None, None, None, None, None, None, None, 1.5 * Cd_R     ],
             [None, None, None, None, None, None, None, None, None, None, None, 2 * 0.25 * C_R ],
             [None, None, None, Id_1, None, None, None, None, None, None, None, None           ],
             [None, None, None, None, None, None, None, None, None, None, None, 1.5 * C_R      ],
             [None, None, None, None, None, -N_1, None, None, None, None, None, None           ],
             [None, None, None, None, None, None, None, None, None, None, None, -1.125 * C_R   ],
             [None, None, None, None, None, None, Cd_L, None, None, None, None, None           ],
             [None, None, None, None, None, None, None, Id_0, None, None, None, None           ],
             [None, None, None, None, None, None, None, None, Id_0, None, None, None           ],
             [None, None, None, None, None, None, None, None, None, None, None, 4.0 * N_0      ],
             [None, None, None, None, None, None, None, None, None, None, Id_0, None           ],
             [None, None, None, None, None, None, None, None, None, None, None, Id_0           ]]
    # fmt: on

    # compare cells individually: better for debugging
    _compare_grids(W0_new, W0_ex, 'W0')
    _compare_grids(W1_new, W1_ex, 'W1')

    # compare reconstructed tensor: easier to trust
    W0_ex = lasy.tensor_from_grid(W0_ex, ['wL', 'p', 'wR', 'p*'])
    W1_ex = lasy.tensor_from_grid(W1_ex, ['wL', 'p', 'wR', 'p*'])
    assert lasy.almost_equal(W0_new, W0_ex)
    assert lasy.almost_equal(W1_new, W1_ex)


@pytest.mark.parametrize('use_fermions', [False, True], ids=['spins', 'fermions'])
@pytest.mark.parametrize('add_hc', [False, 'manually', 'flag'], ids=['no_hc', 'manual_hc', 'plus_hc'])
@pytest.mark.parametrize('bc', ['finite', 'infinite'])
def test_CouplingModel_exponentially_decaying_coupling(use_fermions, add_hc, bc, L=6):
    """test add_exponentially_decaying coupling by comparing with manual couplings"""

    if use_fermions:
        s = lasy.sites.SpinlessFermionSite()
        coupling = 'Cd.C'
        hc_coupling = 'C.Cd'
    else:
        s = lasy.sites.SpinSite()
        coupling = 'Sp.Sm'
        hc_coupling = 'Sm.Sp'

    lat = lattice.Chain(L=L, site=s, bc=['open' if bc == 'finite' else 'periodic'], bc_MPS=bc)

    def init_model():
        M = model.CouplingModel(lat)
        if use_fermions:
            M.define_coupling(lasy.models.hopping([s, s], t_right=0, name=coupling), hc=hc_coupling, permute='swap_hc')
        else:
            M.define_coupling(
                lasy.models.spin_spin_coupling([s, s], Jpm=1, name=coupling), hc=hc_coupling, permute='swap_hc'
            )
        return M

    print('standard case: no subsites')
    m_exp = init_model()
    m_manual = init_model()
    a = 3.0 + 0.42j
    l = 0.2

    with pytest.raises(NotImplementedError, match='exp_decaying not ported'):
        m_exp.add_exponentially_decaying_coupling(a, l, coupling, plus_hc=(add_hc == 'flag'))
    pytest.xfail('exp_decaying not ported')

    m_exp.add_exponentially_decaying_coupling(a, l, coupling, plus_hc=(add_hc == 'flag'))
    if add_hc == 'manually':
        m_exp.add_exponentially_decaying_coupling(np.conj(a), np.conj(l), hc_coupling)
    max_range = L if bc == 'finite' else int(np.ceil(np.log(1e-10) / np.log(l)))
    for k in range(1, max_range):
        m_manual.add_two_site_coupling(a * (l**k), coupling, dx=k, plus_hc=add_hc is not False)
    assert m_exp.get_H_MPO().is_equal(m_manual.get_H_MPO())

    print('non-uniform decay')
    m_exp = init_model()
    m_manual = init_model()
    a = 3.0 + 0.42j
    l = np.random.uniform(0.01, 0.2, size=L)
    m_exp.add_exponentially_decaying_coupling(a, l, coupling, plus_hc=(add_hc == 'flag'))
    if add_hc == 'manually':
        m_exp.add_exponentially_decaying_coupling(np.conj(a), np.conj(l), hc_coupling)
    # use the max_range from before. since l <= .2, this can only decay faster
    for k in range(1, max_range):
        strength = a
        for j in range(k):
            if bc == 'finite':
                strength = strength * l[j : -k + j]
            else:
                strength = strength * np.roll(l, -j)
        if np.all(strength < 1e-10):
            continue
        m_manual.add_two_site_coupling(strength, coupling, dx=k, plus_hc=add_hc is not False)
    assert m_exp.get_H_MPO().is_equal(m_manual.get_H_MPO())

    print('with subsites')
    m_exp = init_model()
    m_manual = init_model()
    a = 3.0 + 0.42j
    l = np.random.uniform(0.01, 0.2, size=L)
    subsites = [1, 3, 5]
    m_exp.add_exponentially_decaying_coupling(a, l, coupling, subsites, plus_hc=(add_hc == 'flag'))
    if add_hc == 'manually':
        m_exp.add_exponentially_decaying_coupling(np.conj(a), np.conj(l), hc_coupling, subsites)
    # use the max_range from before. since l <= .2, this can only decay faster
    # except now we interpret it as a distance *within* subsites
    for n_i, i in enumerate(subsites):
        for k in range(1, max_range):
            n_j = n_i + k
            if bc == 'finite' and n_j >= len(subsites):
                continue
            j = subsites[n_j % len(subsites)] + L * (n_j // len(subsites))
            strength = a
            for m in range(n_i, n_j):
                strength *= l[subsites[m % len(subsites)]]
            m_manual._add_two_site_coupling_mps(strength, i, j, coupling, plus_hc=add_hc is not False)
    assert m_exp.get_H_MPO().is_equal(m_manual.get_H_MPO())

    print('with subsites and subsites_start')
    m_exp = init_model()
    m_manual = init_model()
    a = 3.0 + 0.42j
    l = np.random.uniform(0.01, 0.2, size=L)
    subsites_start = [0, 2]
    subsites = [1, 3, 5]
    m_exp.add_exponentially_decaying_coupling(
        a, l, coupling, subsites, subsites_start=subsites_start, plus_hc=(add_hc == 'flag')
    )
    if add_hc == 'manually':
        m_exp.add_exponentially_decaying_coupling(
            np.conj(a),
            np.conj(l),
            hc_coupling,
            subsites,
            subsites_start=subsites_start,
        )
    # use the max_range from before. since l <= .2, this can only decay faster
    # except now we interpret it as a distance *within* subsites
    for n_i, i in enumerate(subsites_start):
        min_n_j = min(n_j for n_j, j in enumerate(subsites) if j > i)
        for n_j in range(min_n_j, min_n_j + max_range):
            if bc == 'finite' and n_j >= len(subsites):
                continue
            j = subsites[n_j % len(subsites)] + L * (n_j // len(subsites))
            strength = a * l[i]
            for m in range(min_n_j, n_j):
                strength *= l[subsites[m % len(subsites)]]
            m_manual._add_two_site_coupling_mps(strength, i, j, coupling, plus_hc=add_hc is not False)
    assert m_exp.get_H_MPO().is_equal(m_manual.get_H_MPO())


class MyMod(model.ModelBuilder):
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'parity')
        return lasy.sites.SpinSite(S=0.5, conserve=conserve, backend=self.backend)

    def init_couplings(self, model_params):
        site, *_ = self.lattice.unit_cell
        self.define_coupling(
            lasy.models.spin_spin_coupling([site, site], Jx=1, name='Sx.Sx'), hc='Sx.Sx', permute='invariant'
        )
        self.define_coupling(
            lasy.models.spin_spin_coupling([site, site], Jy=1, name='Sy.Sy'), hc='Sy.Sy', permute='invariant'
        )

    def init_terms(self, model_params):
        x = model_params.get('x', 1.0)
        y = model_params.get('y', 0.25)
        self.add_onsite_operator(y, 'Sz')
        self.add_onsite_operator(y, 'Sz', x=4, u=0)
        self._add_two_site_coupling_mps(x, 0, 1, 'Sx.Sx')
        self.add_two_site_coupling(3 * x, 'Sy.Sy', dx=1, x1=3)


@pytest.mark.parametrize('symmetry_backend', ['no_symmetry', 'abelian', 'fusion_tree', None])
@pytest.mark.parametrize('block_backend', ['numpy', 'torch', None])
@pytest.mark.parametrize('conserve', ['SU2', 'Sz', None])
def test_ModelBuilder_backend(symmetry_backend, block_backend, conserve):
    model_params = dict(conserve=conserve, Jx=1, Jy=1, Jz=1)
    if symmetry_backend is not None:
        model_params['symmetry_backend'] = symmetry_backend
    if block_backend is not None:
        model_params['block_backend'] = block_backend

    conserve_symmetry = lasy.sites.SpinSite(conserve=conserve).symmetry
    expect_backend: lasy.TensorBackend = lasy.get_backend(
        conserve_symmetry if symmetry_backend is None else symmetry_backend, block_backend
    )

    if not expect_backend.supports_symmetry(conserve_symmetry):
        model_params = asConfig(model_params, 'SpinChain')
        model_params.touch(*model_params.keys())  # to suppress warnings on __del__
        with pytest.raises(ValueError, match='backend does not support this symmetry'):
            M = SpinChain(model_params)
        return  # error is expected behavior

    M = SpinChain(model_params)
    M.test_sanity()
    assert M.backend == expect_backend
    assert M.lattice.backend == expect_backend
    assert all(site.backend == expect_backend for site in M.lattice.unit_cell)


@pytest.mark.parametrize('which_model', ['MyMod', 'XXZ'])
def test_ModelBuilder_group_sites(which_model):
    if which_model == 'MyMod':
        m = MyMod(dict(x=0.5, L=5, bc_MPS='finite'))
    elif which_model == 'XXZ':
        model_params = {'L': 6, 'hz': np.random.random([6]), 'bc_MPS': 'finite'}
        m = XXZChain2(model_params)

    assert m.H_MPO.max_range == 1
    # test grouping sites
    ED = ExactDiag.from_model(m)
    #  ED.build_full_H_from_mpo()
    ED.build_full_H_from_bonds()

    with pytest.raises(NotImplementedError, match='site grouping not implemented'):
        m.group_sites(n=2)
    pytest.xfail('site grouping not implemented')

    m.group_sites(n=2)
    assert m.H_MPO.max_range == 1
    ED_gr = ExactDiag.from_model(m)
    ED_gr.build_full_H_from_mpo()
    H = ED.full_H.split_legs().to_ndarray()
    Hgr = ED_gr.full_H.split_legs()
    Hgr.idrop_labels()
    Hgr = Hgr.split_legs().to_ndarray()
    assert np.linalg.norm(H - Hgr) < 1.0e-14
    ED_gr.full_H = None
    ED_gr.build_full_H_from_bonds()
    Hgr = ED_gr.full_H.split_legs()
    Hgr.idrop_labels()
    Hgr = Hgr.split_legs().to_ndarray()
    assert np.linalg.norm(H - Hgr) < 1.0e-14


@pytest.mark.parametrize('which_model', ['XXZ', 'dipolar'])
def test_model_H_MPO_vs_H_bond(which_model, L=6):
    bc = 'finite'
    if which_model == 'XXZ':
        model_params = {'L': L, 'hz': np.random.random([L]), 'bc_MPS': bc}
        m = XXZChain2(model_params)
    elif which_model == 'dipolar':
        model_params = dict(L=L, S=1, J3=1.0, J4=0.5, bc_MPS='finite')
        m = DipolarSpinChain(model_params)

        with pytest.raises(NotImplementedError, match='site grouping not implemented'):
            m.group_sites(3)
        pytest.xfail('site grouping not implemented')

        m.group_sites(3)

    ED = ExactDiag.from_model(m)
    ED.build_full_H_from_bonds()
    full_H_bond = ED.full_H  # this should be correct

    ED.full_H = None
    ED.build_full_H_from_mpo()
    full_H_mpo = ED.full_H

    assert lasy.almost_equal(full_H_bond, full_H_mpo)


def compare_models_plus_hc(
    m_manual: model.CouplingModel,
    m_plus_hc: model.CouplingModel,
    m_explicit: model.CouplingModel,
    expect_non_hermitian_mpo: bool = False,  # if MPO without hc is non-hermitian
):
    """helper for test_model_plus_hc; check if the models are equivalent"""

    with pytest.raises(NotImplementedError, match='MPO.dagger not working well, see #682'):
        assert m_manual.get_H_MPO().is_hermitian(), 'm_manual not hermitian'
        assert m_plus_hc.get_H_MPO().is_hermitian(), 'm_plus_hc not hermitian'
        assert m_explicit.get_H_MPO().is_hermitian(), 'm_explicit not hermitian'
    # skipping the checks while it does not work, once it does, simply remove the context manager

    # for m_explicit, look at the "bare" MPO, without its hc
    m_explicit_bare_MPO = m_explicit.get_H_MPO().copy()
    m_explicit_bare_MPO.explicit_plus_hc = False  # now this is an MPO without the +hc part
    if expect_non_hermitian_mpo:
        with pytest.raises(NotImplementedError, match='MPO.dagger not working well, see #682'):
            assert not m_explicit_bare_MPO.is_hermitian()
        # skipping the checks while it does not work, once it does, simply remove the context manager

    if expect_non_hermitian_mpo and m_explicit.get_H_MPO().chi[3] > 2:
        # check for smaller MPO bond dimension
        assert m_explicit.get_H_MPO().chi[3] < m_plus_hc.get_H_MPO().chi[3]

    assert m_plus_hc.get_H_MPO().is_equal(m_manual.get_H_MPO())
    assert m_explicit.get_H_MPO().is_equal(m_manual.get_H_MPO())


@pytest.mark.parametrize(
    'which_site, which_ops',
    [
        ('spin', 'Sp-Sm'),
        ('fermion', 'Cd-C'),
        ('spin-fermion', 'Sp-Sm'),
        ('spin-fermion', 'Cd-C'),
    ],
)
def test_model_plus_hc(which_site, which_ops, L=6):
    """Same as `test_model_plus_hc`, but uses fermions and default JW behavior"""
    if which_site == 'spin':
        s = lasy.sites.SpinSite()
    elif which_site == 'fermion':
        s = lasy.sites.SpinlessFermionSite(filling=0.5)
    elif which_site == 'spin-fermion':
        s = lasy.sites.SpinHalfFermionSite(filling=1)
    else:
        raise ValueError
    lat = lattice.Chain(L=L, site=s)

    def init_model(explicit_plus_hc=False):
        m = model.CouplingModel(lat, explicit_plus_hc=explicit_plus_hc)
        if which_site in ['spin', 'spin-fermion']:
            Sx = s.spin_vector[:, :, 0]
            Sy = s.spin_vector[:, :, 1]
            Sz = s.spin_vector[:, :, 2]
            Sp = Sx + 1j * Sy
            Sm = Sx - 1j * Sy
            m.define_coupling(
                lasy.models.spin_spin_coupling([s, s], Jpm=1, name='Sp.Sm'), hc='Sm.Sp', permute='swap_hc'
            )
            m.define_coupling(
                Coupling_from_numpy_onsite([Sp, Sz, Sm], [s, s, s], name='Sp.Sz.Sm'), hc='Sm.Sz.Sp', permute=None
            )
        if which_site in ['fermion', 'spin-fermion']:
            Cd, C = s.creation_annihilation_ops(s.num_species)
            Cd, C = Cd[:, :, 0], C[:, :, 0]
            N = Cd @ C
            m.define_coupling(lasy.models.hopping([s, s], t_right=0, name='Cd.C'), hc='C.Cd', permute='swap_hc')
            m.define_coupling(
                Coupling_from_numpy_onsite([Cd, N, C], [s, s, s], name='Cd.N.C'), hc='C.N.Cd', permute=None
            )
        return m

    if which_ops == 'Sp-Sm':
        onsite_op = 'Sp'
        Sp_Sm = 'Sp.Sm'
        Sm_Sp = 'Sm.Sp'
        op3 = 'Sp.Sz.Sm'
        op3hc = 'Sm.Sz.Sp'
    elif which_ops == 'Cd-C':
        onsite_op = 'dN'
        Sp_Sm = 'Cd.C'
        Sm_Sp = 'C.Cd'
        op3 = 'Cd.N.C'
        op3hc = 'Cd.N.C'
    else:
        raise ValueError

    m_manual = init_model()
    m_plus_hc = init_model()
    m_explicit = init_model(explicit_plus_hc=True)

    print('onsite')
    hx = np.random.random(L)
    m_manual.add_onsite_operator(hx, onsite_op)
    m_manual.add_onsite_operator(hx, 'Sm' if onsite_op == 'Sp' else 'dN')
    m_plus_hc.add_onsite_operator(hx, onsite_op, plus_hc=True)
    m_explicit.add_onsite_operator(hx, onsite_op, plus_hc=True)
    compare_models_plus_hc(m_manual, m_plus_hc, m_explicit, expect_non_hermitian_mpo=(which_ops != 'Cd-C'))

    print('verify assumption about what the manual dagger should be')
    assert lasy.almost_equal(m_manual.couplings[Sp_Sm].to_tensor().dagger, m_manual.couplings[Sm_Sp].to_tensor())

    print('coupling')
    t = np.random.random(L - 1) + 1.0j * np.random.random(L - 1)
    m_manual.add_two_site_coupling(t, Sp_Sm, dx=1)
    m_manual.add_two_site_coupling(np.conj(t), Sm_Sp, dx=1)
    m_plus_hc.add_two_site_coupling(t, Sp_Sm, dx=1, plus_hc=True)
    m_explicit.add_two_site_coupling(t, Sp_Sm, dx=1, plus_hc=True)
    compare_models_plus_hc(m_manual, m_plus_hc, m_explicit)

    print('multi coupling (2-site)')
    t2 = np.random.random(L - 1)
    m_manual.add_multi_site_coupling(t2, Sp_Sm, dx=[+1, 0])
    m_manual.add_multi_site_coupling(t2, Sp_Sm, dx=[-1, 0])
    m_plus_hc.add_multi_site_coupling(t2, Sp_Sm, dx=[+1, 0], plus_hc=True)
    m_explicit.add_multi_site_coupling(t2, Sp_Sm, dx=[+1, 0], plus_hc=True)
    compare_models_plus_hc(m_manual, m_plus_hc, m_explicit)

    print('multi coupling (3-site)')
    t3 = np.random.random(L - 2)
    m_manual.add_multi_site_coupling(t3, op3, dx=[0, 1, 2])
    m_manual.add_multi_site_coupling(t3, op3hc, dx=[0, 1, 2])
    m_plus_hc.add_multi_site_coupling(t3, op3, dx=[0, 1, 2], plus_hc=True)
    m_explicit.add_multi_site_coupling(t3, op3, dx=[0, 1, 2], plus_hc=True)
    compare_models_plus_hc(m_manual, m_plus_hc, m_explicit)

    with pytest.raises(NotImplementedError, match='exp_decaying not ported'):
        m_manual.add_exponentially_decaying_coupling(0.25, 0.2, Sp_Sm)
    return  # once the NotImplementedError goes away, just remove these three lines.

    print('exponentially decaying coupling')
    d = 0.25
    l = 0.2
    m_manual.add_exponentially_decaying_coupling(d, l, Sp_Sm)
    m_manual.add_exponentially_decaying_coupling(np.conj(d), np.conj(l), Sm_Sp)
    m_plus_hc.add_exponentially_decaying_coupling(d, l, Sp_Sm, plus_hc=True)
    m_explicit.add_exponentially_decaying_coupling(d, l, Sp_Sm, plus_hc=True)
    compare_models_plus_hc(m_manual, m_plus_hc, m_explicit)

    print('exponentially decaying coupling with subsites')
    subsite_kwargs = dict(subsites=[1, 3, 5], subsites_start=[0, 2])
    e = 3 + 0.42j
    m_manual.add_exponentially_decaying_coupling(e, l, Sp_Sm, **subsite_kwargs)
    m_manual.add_exponentially_decaying_coupling(np.conj(e), np.conj(l), Sm_Sp, **subsite_kwargs)
    m_plus_hc.add_exponentially_decaying_coupling(e, l, Sp_Sm, **subsite_kwargs, plus_hc=True)
    m_explicit.add_exponentially_decaying_coupling(e, l, Sp_Sm, **subsite_kwargs, plus_hc=True)
    compare_models_plus_hc(m_manual, m_plus_hc, m_explicit)

    print('exponentially decaying centered terms')
    subsites = [1, 3, 5]
    f = 0.42j
    if which_ops == 'Cd-C':
        with pytest.raises(NotImplementedError):
            m_manual.add_exponentially_decaying_centered_terms(f, l, Sp_Sm, 3, subsites=subsites)
    else:
        m_manual.add_exponentially_decaying_centered_terms(f, l, Sp_Sm, 3, subsites=subsites)
        m_manual.add_exponentially_decaying_centered_terms(np.conj(f), np.conj(l), Sm_Sp, 3, subsites=subsites)
        m_plus_hc.add_exponentially_decaying_centered_terms(f, l, Sp_Sm, 3, subsites=subsites, plus_hc=True)
        m_explicit.add_exponentially_decaying_centered_terms(f, l, Sp_Sm, 3, subsites=subsites, plus_hc=True)
        compare_models_plus_hc(m_manual, m_plus_hc, m_explicit)


class DisorderedLatticeModel(model.ModelBuilder):
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'parity')
        return lasy.sites.SpinSite(conserve=conserve, backend=self.backend)

    def init_lattice(self, model_params):
        lat = super().init_lattice(model_params)
        sigma = model_params.get('disorder_sigma', 0.1)
        shape = lat.shape + (lat.basis.shape[-1],)
        lat.position_disorder = np.random.normal(scale=sigma, size=shape)
        return lat

    def init_couplings(self, model_params):
        site, *_ = self.lattice.unit_cell
        self.define_coupling(
            lasy.models.spin_spin_coupling([site, site], Jz=1, name='Sz.Sz'),
            hc='Sz.Sz',
            permute='invariant',
        )
        self.define_coupling(
            lasy.models.spin_spin_coupling([site, site], Jx=1, name='Sx.Sx'),
            hc='Sx.Sx',
            permute='invariant',
        )

    def init_terms(self, model_params):
        J = model_params.get('J', 1.0)
        for u1, u2, dx in self.lattice.pairs['nearest_neighbors']:
            dist = self.lattice.distance(u1, u2, dx)
            self.add_two_site_coupling(J / dist, 'Sz.Sz', dx, u1, u2)
        for u1, u2, dx in self.lattice.pairs['next_nearest_neighbors']:
            dist = self.lattice.distance(u1, u2, dx)
            self.add_two_site_coupling(J / dist, 'Sx.Sx', dx, u1, u2)


@pytest.mark.parametrize('bc', ['open', 'periodic'])
def test_disordered_lattice_model(bc, J=2.0):
    model_params = {
        'lattice': 'Kagome',
        'Lx': 2,
        'Ly': 3,
        'bc_y': bc,
        'bc_x': bc,
        'bc_MPS': 'finite' if bc == 'open' else 'infinite',
        'disorder_sigma': 0.1,
        'J': J,
    }
    M = DisorderedLatticeModel(model_params)
    for i, j, op, need_pbc in [
        ([0, 0, 0], [0, 0, 1], 'Sz', False),
        ([1, 0, 0], [0, 0, 1], 'Sz', False),
        ([1, 0, 2], [0, 1, 1], 'Sz', False),
        ([0, 0, 1], [0, 1, 0], 'Sx', False),
        ([1, 1, 2], [0, 2, 0], 'Sx', False),
        ([0, 2, 2], [1, 2, 0], 'Sx', False),
        ([0, 2, 2], [1, 2, 0], 'Sx', False),
        ([1, 0, 1], [2, 0, 0], 'Sz', True),
        ([1, 1, 1], [2, 0, 2], 'Sz', True),
        ([1, 2, 2], [1, 3, 0], 'Sz', True),
    ]:
        if need_pbc and bc == 'open':
            continue
        ij = np.array([i, j])
        mps_i, mps_j = M.lattice.lat2mps_idx(ij)
        try:
            strength = M.coupling_terms.coupling_terms[mps_i][f'{op}.{op}'][mps_j]
        except KeyError:
            strength = M.coupling_terms.coupling_terms[mps_j][f'{op}.{op}'][mps_i]
        pos_i, pos_j = M.lattice.position(ij)
        dist = np.linalg.norm(pos_i - pos_j)
        if need_pbc:
            dist = min(dist, np.linalg.norm(pos_i - pos_j + M.lattice.basis[1] * M.lattice.Ls[1]))
        assert abs(strength - J / dist) < 1.0e-14


def test_fixes_511(L=5, t=1.234, tp=2.54):
    # https://github.com/tenpy/tenpy/issues/511

    class TTprimeSpinfulChain(model.ModelBuilder):
        """Spin-1/2 fermions on a 1D chain with NN hopping t and NNN hopping t'."""

        def init_sites(self, p):
            return lasy.sites.SpinHalfFermionSite(backend=self.backend)

        def init_couplings(self, model_params):
            site, *_ = self.lattice.unit_cell
            self.define_coupling(
                lasy.models.hopping([site, site], t_left=1, t_right=0, species=None, name='sum_Cd.C'),
                hc='sum_C.Cd',
                permute='swap_hc',
            )  # summed over all species, i.e. Cud.Cu + Cdd.Cd

        def init_terms(self, p):
            t = float(p.get('t', 1.0))  # NN hopping
            tp = float(p.get('tp', 0.0))  # NNN hopping
            mu = float(p.get('mu', 0.0))  # chemical potential
            U = float(p.get('U', 0.0))  # onsite Hubbard U (optional)

            # onsite: -mu * (n_up + n_down) + U * n_up n_down
            self.add_onsite_operator(-mu, 'Ntot')
            self.add_onsite_operator(U, 'NuNd')

            # NN hopping: -t * (c†_{iσ} c_{i+1,σ} + h.c.)
            for u1, u2, dx in self.lattice.pairs['nearest_neighbors']:
                self.add_two_site_coupling(-t, 'sum_C.Cd', dx, u1, u2, plus_hc=True)

            # NNN hopping: -t' * (c†_{iσ} c_{i+2,σ} + h.c.)
            for u1, u2, dx in self.lattice.pairs['next_nearest_neighbors']:
                self.add_two_site_coupling(-tp, 'sum_C.Cd', dx, u1, u2, plus_hc=True)

    H_full = {}
    for key, _t, _tp in zip(['full', 'NN', 'NNN'], [t, t, 0], [tp, 0, tp]):
        m = TTprimeSpinfulChain(dict(L=L, t=_t, tp=_tp, mu=0, bc_MPS='finite'))
        ED = ExactDiag.from_model(m, max_size=2e7)
        ED.build_full_H_from_mpo()
        H_full[key] = ED.full_H

    assert lasy.almost_equal(H_full['full'], H_full['NN'] + H_full['NNN'])


def _dense_Heisenberg(site, L, J):
    """Reference dense Heisenberg chain Hamiltonian, built directly from the `site` operators."""
    Sx = site.get_op('Sx').to_numpy()
    Sy = site.get_op('Sy').to_numpy()
    Sz = site.get_op('Sz').to_numpy()
    Id = np.eye(site.dim)

    def kron_list(ops):
        out = np.eye(1)
        for op in ops:
            out = np.kron(out, op)
        return out

    H = np.zeros((site.dim**L, site.dim**L), dtype=complex)
    for i in range(L - 1):
        for S in (Sx, Sy, Sz):
            ops_i = [Id] * L
            ops_i[i] = S
            ops_j = [Id] * L
            ops_j[i + 1] = S
            H += J * kron_list(ops_i) @ kron_list(ops_j)
    return H


def _dense_TFIM(L, J, g):
    r"""Hard-coded reference transverse field Ising Hamiltonian, built purely with numpy
        to independently verify CouplingModel.add_coupling.

        H = -J \sum_i Sx_i Sx_{i+1} - g \sum_i Sz_i

    Uses the same spin-1/2 basis convention as ``lasyten.models.sites.SpinSite(S=0.5, conserve=None)``:
    state 0 = down (Sz=-1/2), state 1 = up (Sz=+1/2).
    """
    Sx = np.array([[0, 0.5], [0.5, 0]])
    Sz = np.array([[-0.5, 0], [0, 0.5]])
    Id = np.eye(2)

    def kron_list(ops):
        out = np.eye(1)
        for op in ops:
            out = np.kron(out, op)
        return out

    H = np.zeros((2**L, 2**L))
    for i in range(L - 1):
        ops_i = [Id] * L
        ops_i[i] = Sx
        ops_j = [Id] * L
        ops_j[i + 1] = Sx
        H -= J * kron_list(ops_i) @ kron_list(ops_j)
    for i in range(L):
        ops_i = [Id] * L
        ops_i[i] = Sz
        H -= g * kron_list(ops_i)
    return H


@pytest.mark.skip
def test_fixes_consistency_check_IrregularLattice():
    # tests if the bug reported at https://tenpy.johannes-hauschild.de/viewtopic.php?t=757 is fixed.

    class SpinModel_triangular_finite(tenpy.models.CouplingMPOModel):
        def init_sites(self, model_params):
            return tenpy.SpinHalfSite(conserve=None, backend=self.backend)

        def init_terms(self, model_params):
            self.add_onsite(-1, 0, 'Sz')

        def init_lattice(self, model_params):
            regular_lat = tenpy.models.lattice.Triangular(5, 5, self.init_sites(model_params))
            return tenpy.IrregularLattice(regular_lat, remove=[(0, 0, 0)])

    model = SpinModel_triangular_finite({})

    ad_hoc_fix = False
    if ad_hoc_fix:
        model.lattice.N_sites_per_ring = 1

    psi = tenpy.MPS.from_lat_product_state(model.lattice, [[['up']]])
    dmrg_params = dict(max_sweeps=2)
    _ = tenpy.algorithms.dmrg.run(psi, model, dmrg_params)


def _compare_grids(W, grid, name):
    for i, row in enumerate(grid):
        for j, grid_cell in enumerate(row):
            W_cell = lasy.tensor_grid_cell(W, i, j, 'wL', 'wR')
            if grid_cell is None:
                assert lasy.norm(W_cell) < 1e-16, f'{name}:  not zero: grid[{i}][{j}]'
            else:
                assert W_cell.get_leg('wL') == grid_cell.get_leg('wL'), f'{name}:  wL mismatch [{i}][{j}]'
                assert W_cell.get_leg('wR') == grid_cell.get_leg('wR'), f'{name}:  wR mismatch [{i}][{j}]'
                assert lasy.almost_equal(W_cell, grid_cell), f'{name}:  Tensor mismatch: [{i}][{j}]'
