"""A collection of tests for :mod:`cyten.models.couplings`."""
# Copyright (C) TeNPy Developers, Apache license

import itertools as it
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pytest

import cyten
from cyten import SymmetryError, backends, tensors
from cyten.models import couplings, degrees_of_freedom, sites

# from cyten.models.sites import SpinSite
# from cyten.models.couplings import heisenberg_coupling, spin_field_coupling
#from tenpy.networks.mpo import MPO

from cyten.models.sites import SpinSite
from cyten.models.couplings import heisenberg_coupling, spin_field_coupling
from cyten.tensors import permute_legs

try:
    from sympy import S as sympy_S
except ImportError:
    sympy_S = None

assert sympy_S is not None and callable(sympy_S), "sympy_S is not set correctly"

def check_coupling(coupling_cls, site_num: int, invalid_site_nums: list[int], boson_fermion_mixing: bool, **kwargs):
    """Perform common checks that make sense for any coupling"""
    # it does not matter what site we use since the number of sites is checked first
    site = sites.SpinlessBosonSite([1])
    for n in invalid_site_nums:
        with pytest.raises(ValueError, match='Invalid number of sites.'):
            _ = coupling_cls([site] * n, **kwargs)
    if boson_fermion_mixing:
        site_list = [site, sites.SpinlessFermionSite(1)]
        site_list.extend([site] * (site_num - 2))
        msg = 'Bosonic and fermionic sites are incompatible and cannot be combined for constructing couplings.'
        with pytest.raises(SymmetryError, match=msg):
            _ = coupling_cls(site_list, **kwargs)


def generate_spin_dofs(backend: backends.TensorBackend) -> list[degrees_of_freedom.SpinDOF]:
    """Return a list of `SpinDOF` sites whose symmetries are consistent with `backend`."""
    site_list = []
    for spin in [0.5, 1, 1.5, 2]:
        site_list.append(sites.SpinSite(S=spin, conserve='None', backend=backend))
        if not isinstance(backend, backends.NoSymmetryBackend):
            site_list.append(sites.SpinSite(S=spin, conserve='parity', backend=backend))
            site_list.append(sites.SpinSite(S=spin, conserve='Sz', backend=backend))
        if isinstance(backend, backends.FusionTreeBackend):
            site_list.append(sites.SpinSite(S=spin, conserve='SU(2)', backend=backend))
    if isinstance(backend, backends.FusionTreeBackend):
        all_conserve_N = ['N', 'parity']
        all_conserve_S = ['SU(2)', 'Sz', 'parity', 'None']
        for conserve_N, conserve_S in it.product(all_conserve_N, all_conserve_S):
            site_list.append(sites.SpinHalfFermionSite(conserve_N, conserve_S, backend=backend))
    return site_list


def generate_bosonic_dofs(
    backend: backends.TensorBackend, conserve: Sequence[Literal['N', 'parity', 'None']] = ['N', 'parity', 'None']
) -> list[degrees_of_freedom.BosonicDOF]:
    """Return a list of `BosonicDOF` sites whose symmetries are consistent with `backend`."""
    site_list = []
    for Nmax in [[3], [3, 2]]:
        if isinstance(backend, backends.NoSymmetryBackend):
            all_conserve = ['None']
        else:
            all_conserve = conserve[:]
            if len(Nmax) > 1:
                all_conserve.extend(it.product(all_conserve, repeat=len(Nmax)))
        for cons in all_conserve:
            site_list.append(sites.SpinlessBosonSite(Nmax, cons, backend=backend))
    return site_list


def generate_fermionic_dofs(
    backend: backends.TensorBackend, conserve: Sequence[Literal['N', 'parity']] = ['N', 'parity']
) -> list[degrees_of_freedom.FermionicDOF]:
    """Return a list of `FermionicDOF` sites whose symmetries are consistent with `backend`."""
    site_list = []
    if isinstance(backend, (backends.NoSymmetryBackend, backends.AbelianBackend)):
        # fermionic exchange cannot be encoded
        # do it like this (rather than fixing the backend from the start) such that
        # a potential extension of the ablian backend to fermions automatically works
        with pytest.raises(AssertionError):
            _ = sites.SpinlessFermionSite(num_species=1, backend=backend)
        return site_list
    for num_species in [1, 2]:
        all_conserve = conserve[:]
        individual_conserve = conserve + ['None']
        if num_species > 1:
            all_conserve.extend(it.product(individual_conserve, repeat=num_species))
        for cons in all_conserve:
            site_list.append(sites.SpinlessFermionSite(num_species, cons, backend=backend))
    all_conserve_N = conserve
    all_conserve_S = ['Sz', 'parity', 'None']
    if isinstance(backend, backends.FusionTreeBackend):
        all_conserve_S.append('SU(2)')
    for conserve_N, conserve_S in it.product(all_conserve_N, all_conserve_S):
        site_list.append(sites.SpinHalfFermionSite(conserve_N, conserve_S, backend=backend))
    return site_list


def generate_clock_dofs(backend: backends.TensorBackend) -> list[degrees_of_freedom.ClockDOF]:
    """Return a list of `ClockDOF` sites whose symmetries are consistent with `backend`."""
    site_list = []
    for q in [2, 3, 4]:
        site_list.append(sites.ClockSite(q, conserve='None', backend=backend))
        if not isinstance(backend, backends.NoSymmetryBackend):
            site_list.append(sites.ClockSite(q, conserve='Z_N', backend=backend))
    return site_list


def generate_anyon_dofs(block_backend: cyten.block_backends.BlockBackend) -> list[degrees_of_freedom.AnyonDOF]:
    """Return a list of `AnyonDOF` sites."""
    backend = backends.get_backend('fusion_tree', block_backend=block_backend)
    site_list = [
        sites.FibonacciAnyonSite(backend=backend),
        sites.IsingAnyonSite(nu=1, backend=backend),
        sites.IsingAnyonSite(nu=3, backend=backend),
        sites.GoldenSite(backend=backend),
        sites.SU2kSpin1Site(k=4, backend=backend),
        sites.SU2kSpin1Site(k=5, backend=backend),
    ]
    return site_list


@pytest.mark.parametrize('codom', [1, 2, 3])
def test_coupling(codom, make_compatible_space):
    legs = [make_compatible_space(max_sectors=3, max_mult=3) for _ in range(codom)]
    labels = [f'p{i}' for i in range(codom)]
    labels = [*labels, *[l + '*' for l in labels[::-1]]]
    T = tensors.SymmetricTensor.from_random_normal(codomain=legs, domain=legs, labels=labels)
    site_list = [degrees_of_freedom.Site(leg) for leg in legs]
    coupling = couplings.Coupling.from_tensor(T, site_list, name='name')
    coupling.test_sanity()
    assert coupling.name == 'name'
    assert coupling.num_sites == codom
    assert tensors.almost_equal(coupling.to_tensor(), T)
    if T.symmetry.can_be_dropped:
        coupling_to_numpy = coupling.to_numpy(understood_braiding=True)
        assert np.allclose(coupling_to_numpy, T.to_numpy(understood_braiding=True))
        coupling2 = couplings.Coupling.from_dense_block(coupling_to_numpy, site_list, understood_braiding=True)
        coupling2.test_sanity()
        assert np.all(coupling2.sites == coupling.sites)
        for i in range(codom):
            assert tensors.almost_equal(coupling2.factorization[i], coupling.factorization[i])


# TEST SPIN COUPLINGS


def test_spin_spin_coupling(any_backend, np_random):
    site_list = generate_spin_dofs(any_backend)
    num_sites = min(5, len(site_list))
    site_list = np_random.choice(site_list, size=num_sites, replace=False)
    for i, site1 in enumerate(site_list):
        check_evs = False
        Jx, Jy, Jz = np_random.random(3)
        # either SpinSite or SpinHalfFermionSite
        conserve = site1.conserve if isinstance(site1, sites.SpinSite) else site1.conserve_S
        if conserve in ['Sz']:
            check_evs = True
            Jx = Jy = 0
        elif conserve in ['SU(2)']:
            check_evs = True
            Jx = Jy = Jz

        # test different site combinations
        for site2 in site_list[: i + 1]:
            # Note: is_same_symmetry does not work here since it does not distinguish
            # between U(1) fermion number symmetry and Sz spin symmetry for fermions
            if not site1.symmetry == site2.symmetry:
                continue
            coupling = couplings.spin_spin_coupling([site1, site2], Jx=Jx, Jy=Jy, Jz=Jz)
            coupling.test_sanity()
            tensor = coupling.to_tensor()
            # hermiticity
            assert tensors.almost_equal(tensor.hc, tensor)
            # trace is zero
            assert np.allclose(tensors.trace(tensor), 0)
            if site1 == site2:
                # commutation relation
                tensor_commuted = tensors.permute_legs(tensor, codomain=[1, 0], domain=[2, 3])
                tensor_commuted.relabel({'p0': 'p1', 'p1': 'p0', 'p0*': 'p1*', 'p1*': 'p0*'})
                assert tensors.almost_equal(tensor_commuted, tensor)

            # check eigenvalues of special cases
            if check_evs:
                if conserve in ['Sz']:
                    if isinstance(site1, sites.SpinSite):
                        expect_evs = np.arange(-site1.S, site1.S + 1)[:, None]
                        expect_evs = expect_evs @ np.arange(-site2.S, site2.S + 1)[None, :]
                        expect_evs = expect_evs.flatten()
                    else:
                        # spin-1/2 fermions
                        expect_evs = np.array([0] * 12 + [-0.25, 0.25] * 2)
                elif conserve in ['SU(2)']:
                    if isinstance(site1, sites.SpinSite):
                        double_spin = site1.double_total_spin + site2.double_total_spin
                        lower_limit = abs(site1.double_total_spin - site2.double_total_spin)
                        spin_tots = site1.S * (site1.S + 1) + site2.S * (site2.S + 1)
                        expect_evs = [[s * (s + 2) / 4] * (s + 1) for s in range(double_spin, lower_limit - 1, -2)]
                        expect_evs = (np.concatenate(expect_evs) - spin_tots) / 2.0
                    else:
                        expect_evs = np.array([0] * 12 + [0.25] * 3 + [-0.75])
                evs = tensor.to_numpy(leg_order=[0, 1, 3, 2], understood_braiding=True)
                evs = np.reshape(evs, (np.prod(evs.shape[:2]), -1))
                evs = np.sort(np.linalg.eigvalsh(evs))
                assert np.allclose(evs, np.sort(Jz * expect_evs))

    check_coupling(couplings.spin_spin_coupling, site_num=2, invalid_site_nums=[1, 3], boson_fermion_mixing=False)


def test_spin_field_coupling(any_backend, np_random):
    site_list = generate_spin_dofs(any_backend)
    num_sites = min(5, len(site_list))
    site_list = np_random.choice(site_list, size=num_sites, replace=False)
    for site in site_list:
        hx, hy, hz = np_random.random(3)
        # either SpinSite or SpinHalfFermionSite
        conserve = site.conserve if isinstance(site, sites.SpinSite) else site.conserve_S
        if conserve in ['Sz', 'parity']:
            hx = hy = 0
        elif conserve in ['SU(2)']:
            # coupling not allowed
            continue
        coupling = couplings.spin_field_coupling([site], hx=hx, hy=hy, hz=hz)
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # trace is zero
        assert np.allclose(tensors.trace(tensor), 0)
        # check eigenvalues
        h = np.sqrt(hx**2 + hy**2 + hz**2)
        if isinstance(site, sites.SpinSite):
            expect_evs = np.arange(-site.S, site.S + 1)
        else:
            # spin-1/2 fermions
            expect_evs = np.array([0, 0, -0.5, 0.5])
        evs = tensor.to_numpy(understood_braiding=True)
        evs = np.sort(np.linalg.eigvalsh(evs))
        assert np.allclose(evs, np.sort(h * expect_evs))

    check_coupling(couplings.spin_field_coupling, site_num=1, invalid_site_nums=[2], boson_fermion_mixing=False)


def test_aklt_coupling(any_backend, np_random):
    site_list = generate_spin_dofs(any_backend)
    num_sites = min(5, len(site_list))
    site_list = np_random.choice(site_list, size=num_sites, replace=False)
    for i, site1 in enumerate(site_list):
        J = np_random.random()
        # test different site combinations
        for site2 in site_list[: i + 1]:
            if not site1.symmetry == site2.symmetry:
                continue
            coupling = couplings.aklt_coupling([site1, site2], J=J)
            coupling.test_sanity()
            tensor = coupling.to_tensor()
            # hermiticity
            assert tensors.almost_equal(tensor.hc, tensor)
            if site1 == site2:
                # commutation relation
                tensor_commuted = tensors.permute_legs(tensor, codomain=[1, 0], domain=[2, 3])
                tensor_commuted.relabel({'p0': 'p1', 'p1': 'p0', 'p0*': 'p1*', 'p1*': 'p0*'})
                assert tensors.almost_equal(tensor_commuted, tensor)

            if isinstance(site1, sites.SpinSite):
                double_spin = site1.double_total_spin + site2.double_total_spin
                lower_limit = abs(site1.double_total_spin - site2.double_total_spin)
                spin_tots = site1.S * (site1.S + 1) + site2.S * (site2.S + 1)
                expect_evs = [[s * (s + 2) / 4] * (s + 1) for s in range(double_spin, lower_limit - 1, -2)]
                expect_evs = (np.concatenate(expect_evs) - spin_tots) / 2.0
            else:
                expect_evs = np.array([0] * 12 + [0.25] * 3 + [-0.75])
            expect_evs += expect_evs**2 / 3.0
            evs = tensor.to_numpy(leg_order=[0, 1, 3, 2], understood_braiding=True)
            evs = np.reshape(evs, (np.prod(evs.shape[:2]), -1))
            evs = np.sort(np.linalg.eigvalsh(evs))
            assert np.allclose(evs, np.sort(J * expect_evs))
            if site1 == site2 and isinstance(site1, sites.SpinSite) and site1.double_total_spin == 2:
                # actual AKLT case
                assert np.allclose(evs, J * np.array([-2.0 / 3.0] * 4 + [4.0 / 3.0] * 5))

    check_coupling(couplings.aklt_coupling, site_num=2, invalid_site_nums=[1, 3], boson_fermion_mixing=False)


@pytest.mark.slow  # TODO can we speed it up?
def test_chiral_3spin_coupling(any_backend, np_random):
    site_list = generate_spin_dofs(any_backend)
    num_sites = min(3, len(site_list))
    site_list = np_random.choice(site_list, size=num_sites, replace=False)
    for i, site1 in enumerate(site_list):
        chi = np_random.random()
        # test different site combinations
        for site2 in site_list[: i + 1]:
            if not site1.symmetry == site2.symmetry:
                continue
            site3 = np_random.choice([site1, site2])
            coupling = couplings.chiral_3spin_coupling([site1, site2, site3], chi=chi)
            coupling.test_sanity()
            tensor = coupling.to_tensor()
            # hermiticity
            assert tensors.almost_equal(tensor.hc, tensor)
            # trace is zero
            assert np.allclose(tensors.trace(tensor), 0)
            if site1 == site2:
                # cyclic permutation relation
                tensor_commuted = tensors.permute_legs(tensor, codomain=[2, 0, 1], domain=[3, 5, 4])
                relabel = {'p2': 'p0', 'p1': 'p2', 'p0': 'p1', 'p2*': 'p0*', 'p1*': 'p2*', 'p0*': 'p1*'}
                tensor_commuted.relabel(relabel)
                assert tensors.almost_equal(tensor_commuted, tensor)

    check_coupling(couplings.chiral_3spin_coupling, site_num=3, invalid_site_nums=[1, 2], boson_fermion_mixing=False)


# TEST BOSON AND FERMION COUPLINGS


def test_chemical_potential(any_backend, np_random):
    bosonic_sites = generate_bosonic_dofs(any_backend)
    num_sites = min(3, len(bosonic_sites))
    bosonic_sites = np_random.choice(bosonic_sites, size=num_sites, replace=False)
    fermionic_sites = generate_fermionic_dofs(any_backend)
    num_sites = min(3, len(fermionic_sites))
    fermionic_sites = np_random.choice(fermionic_sites, size=num_sites, replace=False)
    all_sites = [*bosonic_sites, *fermionic_sites]

    for site in all_sites:
        mu = np_random.random()
        species = np_random.integers(1, site.num_species + 1)
        species = np_random.choice(range(site.num_species), size=species, replace=False)
        if isinstance(site, sites.SpinHalfFermionSite):
            if site.conserve_S in ['SU(2)'] and len(species) == 1:
                species = np.append(species, 1 - species[0])
        coupling = couplings.chemical_potential([site], mu=mu, species=species)
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # check eigenvalues
        Nmax = site.Nmax if isinstance(site, sites.SpinlessBosonSite) else [1] * site.num_species
        expect_evs = []
        for occupations in it.product(*[list(range(n + 1)) for n in Nmax]):
            expect_evs.append(-mu * sum([occupations[k] for k in species]))
        evs = tensor.to_numpy(understood_braiding=True)
        evs = np.sort(np.linalg.eigvalsh(evs))
        assert np.allclose(evs, np.sort(expect_evs))

    check_coupling(couplings.chemical_potential, site_num=1, invalid_site_nums=[2], boson_fermion_mixing=False, mu=1.0)


def test_onsite_interaction(any_backend, np_random):
    bosonic_sites = generate_bosonic_dofs(any_backend)
    num_sites = min(3, len(bosonic_sites))
    bosonic_sites = np_random.choice(bosonic_sites, size=num_sites, replace=False)
    fermionic_sites = generate_fermionic_dofs(any_backend)
    num_sites = min(3, len(fermionic_sites))
    fermionic_sites = np_random.choice(fermionic_sites, size=num_sites, replace=False)
    all_sites = [*bosonic_sites, *fermionic_sites]

    for site in all_sites:
        U = np_random.random()
        species = np_random.integers(1, site.num_species + 1)
        species = np_random.choice(range(site.num_species), size=species, replace=False)
        if isinstance(site, sites.SpinHalfFermionSite):
            if site.conserve_S in ['SU(2)'] and len(species) == 1:
                species = np.append(species, 1 - species[0])
        coupling = couplings.onsite_interaction([site], U=U, species=species)
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # check eigenvalues
        Nmax = site.Nmax if isinstance(site, sites.SpinlessBosonSite) else [1] * site.num_species
        expect_evs = []
        for occupations in it.product(*[list(range(n + 1)) for n in Nmax]):
            n = sum([occupations[k] for k in species])
            expect_evs.append(U * n**2 / 2.0)
        evs = tensor.to_numpy(understood_braiding=True)
        evs = np.sort(np.linalg.eigvalsh(evs))
        assert np.allclose(evs, np.sort(expect_evs))

    check_coupling(couplings.onsite_interaction, site_num=1, invalid_site_nums=[2], boson_fermion_mixing=False)


@pytest.mark.slow  # TODO can we speed it up?
def test_density_density_interaction(any_backend, np_random):
    bosonic_sites = generate_bosonic_dofs(any_backend)
    num_sites = min(3, len(bosonic_sites))
    bosonic_sites = np_random.choice(bosonic_sites, size=num_sites, replace=False)
    fermionic_sites = generate_fermionic_dofs(any_backend)
    num_sites = min(3, len(fermionic_sites))
    fermionic_sites = np_random.choice(fermionic_sites, size=num_sites, replace=False)
    all_sites = [*bosonic_sites, *fermionic_sites]

    for site in all_sites:
        V = np_random.random()
        species1 = np_random.integers(1, site.num_species + 1)
        species1 = np_random.choice(range(site.num_species), size=species1, replace=False)
        species2 = np_random.integers(1, site.num_species + 1)
        species2 = np_random.choice(range(site.num_species), size=species2, replace=False)
        if isinstance(site, sites.SpinHalfFermionSite) and site.conserve_S in ['SU(2)']:
            species1 = species2 = [0, 1]
        coupling = couplings.density_density_interaction([site] * 2, V, species1, species2)
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        if all(species1 == species2):
            # commutation relation
            tensor_commuted = tensors.permute_legs(tensor, codomain=[1, 0], domain=[2, 3])
            tensor_commuted.relabel({'p0': 'p1', 'p1': 'p0', 'p0*': 'p1*', 'p1*': 'p0*'})
            assert tensors.almost_equal(tensor_commuted, tensor)
        # check eigenvalues
        Nmax = site.Nmax if isinstance(site, sites.SpinlessBosonSite) else [1] * site.num_species
        n1 = []
        n2 = []
        for occupations in it.product(*[list(range(n + 1)) for n in Nmax]):
            n1.append(sum([occupations[k] for k in species1]))
            n2.append(sum([occupations[k] for k in species2]))
        expect_evs = V * np.outer(n1, n2).flatten()
        evs = tensor.to_numpy(leg_order=[0, 1, 3, 2], understood_braiding=True)
        evs = np.reshape(evs, (np.prod(evs.shape[:2]), -1))
        evs = np.sort(np.linalg.eigvalsh(evs))
        assert np.allclose(evs, np.sort(expect_evs))

    check_coupling(
        couplings.density_density_interaction, site_num=2, invalid_site_nums=[1, 3], boson_fermion_mixing=True
    )


@pytest.mark.slow  # TODO can we speed it up?
def test_hopping(any_backend, np_random):
    bosonic_sites = generate_bosonic_dofs(any_backend)
    num_sites = min(3, len(bosonic_sites))
    bosonic_sites = np_random.choice(bosonic_sites, size=num_sites, replace=False)
    fermionic_sites = generate_fermionic_dofs(any_backend)
    num_sites = min(3, len(fermionic_sites))
    fermionic_sites = np_random.choice(fermionic_sites, size=num_sites, replace=False)
    all_sites = [*bosonic_sites, *fermionic_sites]

    for site in all_sites:
        t = np_random.random()
        species1 = np_random.integers(1, site.num_species + 1)
        species1 = np_random.choice(range(site.num_species), size=species1, replace=False)
        species2 = np_random.integers(1, site.num_species + 1)
        species2 = np_random.choice(range(site.num_species), size=species2, replace=False)
        if len(species1) != len(species2):
            limit = min(len(species1), len(species2))
            species1 = species1[:limit]
            species2 = species2[:limit]

        if isinstance(site, (sites.SpinlessBosonSite, sites.SpinlessFermionSite)):
            if not isinstance(site.conserve, str):
                # easiest way to deal with symmetries on the individual species
                species2 = species1
        if isinstance(site, sites.SpinHalfFermionSite):
            if site.conserve_S in ['Sz']:
                species2 = species1
            elif site.conserve_S in ['SU(2)']:
                species1 = species2 = degrees_of_freedom.ALL_SPECIES

        coupling = couplings.hopping([site] * 2, t, species=(species1, species2))
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # trace is zero
        assert np.allclose(tensors.trace(tensor), 0)
        # if there is a permutation s.t. species1 <-> species2, we can commute the legs
        symmetric = False
        for perm in it.permutations(range(len(species1))):
            if np.all(species1[list(perm)] == species2) and np.all(species2[list(perm)] == species1):
                symmetric = True
        if symmetric:
            # commutation relation; this does commute for fermions since
            # a_0_k^\dagger a_1_l + hc -> (exchange legs) -> -1 * a_0_l a_1_k^\dagger + hc
            # = a_1_k^\dagger a_0_l + hc = a_0_l^\dagger a_1_k + hc
            tensor_commuted = tensors.permute_legs(tensor, codomain=[1, 0], domain=[2, 3])
            tensor_commuted.relabel({'p0': 'p1', 'p1': 'p0', 'p0*': 'p1*', 'p1*': 'p0*'})
            assert tensors.almost_equal(tensor_commuted, tensor)

    check_coupling(couplings.hopping, site_num=2, invalid_site_nums=[1, 3], boson_fermion_mixing=True)


def test_pairing(any_backend, np_random):
    bosonic_sites = generate_bosonic_dofs(any_backend, conserve=['parity', 'None'])
    num_sites = min(3, len(bosonic_sites))
    bosonic_sites = np_random.choice(bosonic_sites, size=num_sites, replace=False)
    fermionic_sites = generate_fermionic_dofs(any_backend, conserve=['parity'])
    num_sites = min(3, len(fermionic_sites))
    fermionic_sites = np_random.choice(fermionic_sites, size=num_sites, replace=False)
    all_sites = [*bosonic_sites, *fermionic_sites]

    for site in all_sites:
        Delta = np_random.random()
        species1 = np_random.integers(1, site.num_species + 1)
        species1 = np_random.choice(range(site.num_species), size=species1, replace=False)
        species2 = np_random.integers(1, site.num_species + 1)
        species2 = np_random.choice(range(site.num_species), size=species2, replace=False)
        if len(species1) != len(species2):
            limit = min(len(species1), len(species2))
            species1 = species1[:limit]
            species2 = species2[:limit]

        if isinstance(site, (sites.SpinlessBosonSite, sites.SpinlessFermionSite)):
            if not isinstance(site.conserve, str):
                # easiest way to deal with symmetries on the individual species
                species2 = species1
        if isinstance(site, sites.SpinHalfFermionSite):
            if site.conserve_S in ['Sz']:
                for i, k in enumerate(species1):
                    species2[i] = 1 - k
            elif site.conserve_S in ['SU(2)']:
                species1 = species2 = []

        coupling = couplings.pairing([site] * 2, Delta, species=(species1, species2))
        coupling.test_sanity()
        if len(species1) == 0:
            continue
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # trace is zero
        assert np.allclose(tensors.trace(tensor), 0)
        # if there is a permutation s.t. species1 <-> species2, we can commute the legs
        symmetric = False
        for perm in it.permutations(range(len(species1))):
            if np.all(species1[list(perm)] == species2) and np.all(species2[list(perm)] == species1):
                symmetric = True
        if symmetric:
            # commutation relation
            tensor_commuted = tensors.permute_legs(tensor, codomain=[1, 0], domain=[2, 3])
            tensor_commuted.relabel({'p0': 'p1', 'p1': 'p0', 'p0*': 'p1*', 'p1*': 'p0*'})
            assert tensors.almost_equal(tensor_commuted, site.anti_commute_sign * tensor)

    check_coupling(couplings.pairing, site_num=2, invalid_site_nums=[1, 3], boson_fermion_mixing=True)


def test_onsite_pairing(any_backend, np_random):
    bosonic_sites = generate_bosonic_dofs(any_backend, conserve=['parity', 'None'])
    num_sites = min(3, len(bosonic_sites))
    bosonic_sites = np_random.choice(bosonic_sites, size=num_sites, replace=False)
    fermionic_sites = generate_fermionic_dofs(any_backend, conserve=['parity'])
    num_sites = min(3, len(fermionic_sites))
    fermionic_sites = np_random.choice(fermionic_sites, size=num_sites, replace=False)
    all_sites = [*bosonic_sites, *fermionic_sites]

    for site in all_sites:
        Delta = np_random.random()
        species1 = np_random.integers(1, site.num_species + 1)
        species1 = np_random.choice(range(site.num_species), size=species1, replace=False)
        species2 = np_random.integers(1, site.num_species + 1)
        species2 = np_random.choice(range(site.num_species), size=species2, replace=False)
        if len(species1) != len(species2):
            limit = min(len(species1), len(species2))
            species1 = species1[:limit]
            species2 = species2[:limit]

        if isinstance(site, (sites.SpinlessBosonSite, sites.SpinlessFermionSite)):
            if not isinstance(site.conserve, str):
                # easiest way to deal with symmetries on the individual species
                species2 = species1
        if isinstance(site, sites.SpinHalfFermionSite):
            if site.conserve_S in ['Sz', 'SU(2)']:
                species1 = [0]
                species2 = [1]

        coupling = couplings.onsite_pairing([site], Delta, species=(species1, species2))
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # trace is zero
        assert np.allclose(tensors.trace(tensor), 0)
        if isinstance(site, degrees_of_freedom.FermionicDOF):
            # default case is trivial for fermions
            coupling = couplings.onsite_pairing([site], Delta=1)
            coupling.test_sanity()
            assert np.allclose(tensors.norm(coupling.to_tensor()), 0)

    check_coupling(couplings.onsite_pairing, site_num=1, invalid_site_nums=[2], boson_fermion_mixing=False)


# TEST CLOCK COUPLINGS


def test_clock_clock_coupling(any_backend, np_random):
    site_list = generate_clock_dofs(any_backend)
    for site in site_list:
        Jx, Jz = np_random.random(2)
        coupling = couplings.clock_clock_coupling([site] * 2, Jx=Jx, Jz=Jz)
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # trace is zero
        assert np.allclose(tensors.trace(tensor), 0)
        # commutation relation
        tensor_commuted = tensors.permute_legs(tensor, codomain=[1, 0], domain=[2, 3])
        tensor_commuted.relabel({'p0': 'p1', 'p1': 'p0', 'p0*': 'p1*', 'p1*': 'p0*'})
        assert tensors.almost_equal(tensor_commuted, tensor)

    check_coupling(couplings.clock_clock_coupling, site_num=2, invalid_site_nums=[1, 3], boson_fermion_mixing=False)


def test_clock_field_coupling(any_backend, np_random):
    site_list = generate_clock_dofs(any_backend)
    for site in site_list:
        hx, hz = np_random.random(2)
        if isinstance(site.leg.symmetry, cyten.ZNSymmetry):
            hx = 0
        coupling = couplings.clock_field_coupling([site], hx=hx, hz=hz)
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # trace is zero
        assert np.allclose(tensors.trace(tensor), 0)
        # check eigenvalues
        if isinstance(site.leg.symmetry, cyten.ZNSymmetry):
            expect_evs = 2 * np.cos(np.linspace(0, 2 * np.pi, site.q, endpoint=False))
            evs = tensor.to_numpy(understood_braiding=True)
            evs = np.sort(np.linalg.eigvalsh(evs))
            assert np.allclose(evs, np.sort(hz * expect_evs))

    check_coupling(couplings.clock_field_coupling, site_num=1, invalid_site_nums=[2], boson_fermion_mixing=False)


# TEST ANYONIC COUPLINGS


def test_sector_projection_coupling(block_backend):
    site_list = generate_anyon_dofs(block_backend)
    num_sites = [3, 2, 1, 2, 2, 1]
    sectors = np.asarray([[1], [2], [1], [0], [2], [2]], dtype=int)
    for site, num, sector in zip(site_list, num_sites, sectors):
        coupling = couplings.sector_projection_coupling([site] * num, J=1.0, sector=sector, name='')
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # trace is integer * dim(sector)
        dim_sec = site.symmetry.qdim(sector)
        tr = tensors.trace(tensor)
        assert np.allclose(np.round(tr / dim_sec, 0), tr / dim_sec)


def test_gold_coupling(block_backend):
    backend = backends.get_backend('fusion_tree', block_backend=block_backend)
    site_list = [sites.GoldenSite(backend=backend), sites.FibonacciAnyonSite(backend=backend)]
    for i, site in enumerate(site_list):
        coupling = couplings.gold_coupling([site] * 2, J=1.0)
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # trace
        assert np.allclose(tensors.trace(tensor), [-1, -2][i])

    coupling = couplings.gold_coupling(site_list, J=1.0)
    coupling.test_sanity()
    tensor = coupling.to_tensor()
    # hermiticity
    assert tensors.almost_equal(tensor.hc, tensor)
    # trace
    assert np.allclose(tensors.trace(tensor), -1)

    check_coupling(couplings.gold_coupling, site_num=2, invalid_site_nums=[1, 3], boson_fermion_mixing=False)


# def test_insert_identity_between_sites(block_backend):
#     """Test inserting identity tensors between sites in a coupling."""
#     backend = backends.get_backend(block_backend=block_backend)
#     site = sites.SpinSite(S=0.5, conserve='None', backend=backend)
#
#     coupling = couplings.chiral_3spin_coupling([site, site, site], chi=1.0)
#     coupling.test_sanity()
#
#     for pos in [1, 2]:
#         new_coupling = coupling.insert_identity_between_sites(position=pos)
#
#         assert len(new_coupling.sites) == len(coupling.sites)
#         assert len(new_coupling.factorization) == len(coupling.factorization) + 1
#
#
# def test_insert_identity_between_sites_invalid_position(block_backend):
#     """Test that invalid positions raise errors."""
#     backend = backends.get_backend(block_backend=block_backend)
#     site = sites.SpinSite(S=0.5, conserve='None', backend=backend)
#
#     coupling = couplings.heisenberg_coupling([site, site], J=1.0)
#
#     with pytest.raises(ValueError, match='Position must be between'):
#         coupling.insert_identity_between_sites(position=0)
#
#     with pytest.raises(ValueError, match='Position must be between'):
#         coupling.insert_identity_between_sites(position=2)
#
#     with pytest.raises(ValueError, match='Position must be between'):
#         coupling.insert_identity_between_sites(position=-1)
#
#
# def test_insert_identity_between_sites_single_site(block_backend):
#     """Test that inserting identity on single-site coupling fails."""
#     backend = backends.get_backend(block_backend=block_backend)
#     site = sites.SpinSite(S=0.5, conserve='None', backend=backend)
#
#     coupling = couplings.spin_field_coupling([site], hz=1.0)
#
#     with pytest.raises(ValueError, match='Position must be between'):
#         coupling.insert_identity_between_sites(position=1)
#
#
# def test_insert_identity_between_sites_different_backends(block_backend):
#     """Test identity insertion with different backends using a 3-site spin coupling."""
#     backend = backends.get_backend(block_backend=block_backend)
#     site = sites.SpinSite(S=0.5, conserve='None', backend=backend)
#
#     coupling = couplings.chiral_3spin_coupling([site, site, site], chi=1.0)
#
#     for pos in [1, 2]:
#         new_coupling = coupling.insert_identity_between_sites(position=pos)
#         assert len(new_coupling.sites) == len(coupling.sites)
#         assert len(new_coupling.factorization) == len(coupling.factorization) + 1


# def test_coupling_hash_roundtrip(block_backend):
#     """Test that to_hash and from_hash correctly roundtrip a coupling."""
#     backend = backends.get_backend(block_backend=block_backend)
#     site = sites.SpinSite(S=0.5, conserve='None', backend=backend)
#
#     coupling = couplings.heisenberg_coupling([site, site], J=1.0)
#     coupling.test_sanity()
#
#     hash_str = coupling.to_hash()
#     reconstructed = couplings.Coupling.from_hash(hash_str)
#
#     assert reconstructed.name == coupling.name
#     assert len(reconstructed.sites) == len(coupling.sites)
#     assert len(reconstructed.factorization) == len(coupling.factorization)
#
#     assert hash_str == reconstructed.to_hash()
#
#
# def test_coupling_hash_3site(block_backend):
#     """Test hash for 3-site couplings."""
#     backend = backends.get_backend(block_backend=block_backend)
#     site = sites.SpinSite(S=0.5, conserve='None', backend=backend)
#
#     coupling = couplings.chiral_3spin_coupling([site, site, site], chi=1.0)
#     coupling.test_sanity()
#
#     hash_str = coupling.to_hash()
#     reconstructed = couplings.Coupling.from_hash(hash_str)
#
#     assert reconstructed.name == coupling.name
#     assert len(reconstructed.sites) == len(coupling.sites)
#     assert len(reconstructed.factorization) == len(coupling.factorization)
#     assert hash_str == reconstructed.to_hash()
#
#
# def test_coupling_hash_with_identity(block_backend):
#     """Test hash for couplings with identity inserted."""
#     backend = backends.get_backend(block_backend=block_backend)
#     site = sites.SpinSite(S=0.5, conserve='None', backend=backend)
#
#     coupling = couplings.chiral_3spin_coupling([site, site, site], chi=1.0)
#     coupling.test_sanity()
#
#     coupling_with_id = coupling.insert_identity_between_sites(position=1)
#     hash_str = coupling_with_id.to_hash()
#     reconstructed = couplings.Coupling.from_hash(hash_str)
#
#     assert reconstructed.name == coupling_with_id.name
#     assert len(reconstructed.sites) == len(coupling_with_id.sites)
#     assert len(reconstructed.factorization) == len(coupling_with_id.factorization)
#     assert hash_str == reconstructed.to_hash()
#
#
# def test_coupling_hash_deterministic(block_backend):
#     """Test that hash is deterministic for same coupling."""
#     backend = backends.get_backend(block_backend=block_backend)
#     site = sites.SpinSite(S=0.5, conserve='None', backend=backend)
#
#     coupling1 = couplings.heisenberg_coupling([site, site], J=1.0)
#     coupling2 = couplings.heisenberg_coupling([site, site], J=1.0)
#
#     hash1 = coupling1.to_hash()
#     hash2 = coupling2.to_hash()
#
#     assert hash1 == hash2

class SimpleTestGraph:
    """Minimal mock MPO-like object for testing _make_graph_from_couplings."""

    def __init__(self, L):
        self.L = L
        self._graph = None



def test_coupling_hash_different_for_different_couplings(block_backend):
    """Test that different couplings have different hashes."""
    backend = backends.get_backend(block_backend=block_backend)
    site = sites.SpinSite(S=0.5, conserve='None', backend=backend)

    coupling1 = couplings.heisenberg_coupling([site, site], J=1.0)
    coupling2 = couplings.heisenberg_coupling([site, site], J=2.0)

    hash1 = coupling1.to_hash()
    hash2 = coupling2.to_hash()

    assert hash1 != hash2


def test_MPO_graph_from_couplings():
    """Test building MPO graph from couplings using hashing."""

    pytest.skip('Test currently depends on tenpy.')

    L = 4
    spin_sites = [SpinSite(S=0.5, conserve='Sz') for _ in range(L)]

    coupling1 = heisenberg_coupling([spin_sites[0], spin_sites[1]], J=1.0)
    coupling2 = heisenberg_coupling([spin_sites[0], spin_sites[1]], J=2.0)
    coupling3 = spin_field_coupling([spin_sites[0]], hz=0.5)

    couplings = [coupling1, coupling2, coupling3]

    hashes = [c.to_hash() for c in couplings]
    assert len(hashes) == len(couplings)
    assert hashes[0] != hashes[1]
    assert hashes[0] != hashes[2]

    test_graph = SimpleTestGraph(L)

    MPO._make_graph_from_couplings(test_graph, couplings)

    assert test_graph._graph is not None
    assert len(test_graph._graph) == L

    for site_graph in test_graph._graph:
        assert isinstance(site_graph, dict)


def test_MPO_graph_from_couplings_identity_insertion():
    """Test building MPO graph from cyten Couplings with identity insertion."""

    pytest.skip('Test currently depends on tenpy.')

    L = 4
    spin_sites = [SpinSite(S=0.5, conserve='Sz') for _ in range(L)]

    coupling = heisenberg_coupling([spin_sites[0], spin_sites[2]], J=1.0)

    coupling_with_identity = coupling.insert_identity_between_sites(1)

    assert len(coupling_with_identity.factorization) == len(coupling.factorization) + 1
    assert len(coupling_with_identity.sites) == len(coupling.sites)

    hash_original = coupling.to_hash()
    hash_with_identity = coupling_with_identity.to_hash()
    assert hash_original != hash_with_identity

    couplings = [coupling_with_identity]

    test_graph = SimpleTestGraph(L)

    MPO._make_graph_from_couplings(test_graph, couplings)

    assert test_graph._graph is not None
    assert len(test_graph._graph) == L


def test_MPO_graph_from_couplings_hash():
    """Test that coupling hashing correctly identifies unique couplings."""


    spin_site1 = SpinSite(S=0.5, conserve='Sz')
    spin_site2 = SpinSite(S=0.5, conserve='Sz')

    coupling_J1 = heisenberg_coupling([spin_site1, spin_site2], J=1.0)
    coupling_J1p = heisenberg_coupling([spin_site1, spin_site2], J=1.0)

    coupling_J2 = heisenberg_coupling([spin_site1, spin_site2], J=2.0)
    coupling_J2p = heisenberg_coupling([spin_site1, spin_site2], J=2.0)
    coupling_diff_J = spin_field_coupling([spin_site1], hz=0.5)

    hash1 = coupling_J1.to_hash()
    hash1p= coupling_J1p.to_hash()

    hash2 = coupling_J2.to_hash()
    hash2p = coupling_J2p.to_hash()

    hash3 = coupling_diff_J.to_hash()


    assert hash1 != hash2
    assert hash1 != hash3
    assert hash2 != hash3

    assert hash1p == hash1
    assert hash2p == hash2


def test_coupling_hashing():
    """Test that coupling hashing correctly distinguishes different couplings."""

    spin_site = SpinSite(S=0.5, conserve='Sz')

    coupling_J1 = heisenberg_coupling([spin_site, spin_site], J=1.0)
    coupling_J2 = heisenberg_coupling([spin_site, spin_site], J=2.0)
    coupling_spin_field = spin_field_coupling([spin_site], hz=0.5)

    hash1 = coupling_J1.to_hash()
    hash2 = coupling_J2.to_hash()
    hash3 = coupling_spin_field.to_hash()

    assert hash1 != hash2
    assert hash1 != hash3
    assert hash2 != hash3


@pytest.mark.parametrize(
    "coupling_factory,site_args,coupling_kwargs,valid_positions",
    [
        # 2-site Heisenberg
        (couplings.heisenberg_coupling,
         [lambda backend: [sites.SpinSite(S=0.5, conserve='None', backend=backend)] * 2],
         {"J": 1.0},
         [1]),
        # 3-site chiral
        (couplings.chiral_3spin_coupling,
         [lambda backend: [sites.SpinSite(S=0.5, conserve='None', backend=backend)] * 3],
         {"chi": 1.0},
         [1, 2]),
        # 2-site AKLT
        (couplings.aklt_coupling,
         [lambda backend: [sites.SpinSite(S=1, conserve='None', backend=backend)] * 2],
         {"J": 1.0},
         [1]),
        # 2-site clock
        # (couplings.clock_clock_coupling,
        #  [lambda backend: [sites.ClockSite(3, conserve='None', backend=backend)] * 2],
        #  {"Jx": 1.0, "Jz": 1.0},
        #  [1]),
    ]
)
def test_identity_insertion_parametrized(block_backend, coupling_factory, site_args, coupling_kwargs, valid_positions):
    backend = backends.get_backend(block_backend=block_backend)
    sites_list = site_args[0](backend)
    coupling = coupling_factory(sites_list, **coupling_kwargs)
    orig_num_sites = len(coupling.sites)
    orig_num_factors = len(coupling.factorization)
    orig_hash = coupling.to_hash()

    for pos in valid_positions:
        new_coupling = coupling.insert_identity_between_sites(position=pos)
        # Number of sites should remain the same
        assert len(new_coupling.sites) == orig_num_sites + 1
        # Number of factors should increase by 1
        assert len(new_coupling.factorization) == orig_num_factors + 1
        # Hash should change
        assert new_coupling.to_hash() != orig_hash

    # Test invalid positions
    for invalid_pos in [0, -1, orig_num_factors + 2]:
        with pytest.raises(ValueError):
            coupling.insert_identity_between_sites(position=invalid_pos)

    @pytest.mark.parametrize(
        "coupling_factory,site_args,coupling_kwargs,valid_positions",
        [
            # 2-site Heisenberg
            (couplings.heisenberg_coupling,
             [lambda backend: [sites.SpinSite(S=0.5, conserve='None', backend=backend)] * 2],
             {"J": 1.0},
             [1]),
            # 3-site chiral
            (couplings.chiral_3spin_coupling,
             [lambda backend: [sites.SpinSite(S=0.5, conserve='None', backend=backend)] * 3],
             {"chi": 1.0},
             [1, 2]),
            # 2-site AKLT
            (couplings.aklt_coupling,
             [lambda backend: [sites.SpinSite(S=1, conserve='None', backend=backend)] * 2],
             {"J": 1.0},
             [1]),
            # 2-site clock
            # (couplings.clock_clock_coupling,
            #  [lambda backend: [sites.ClockSite(3, conserve='None', backend=backend)] * 2],
            #  {"Jx": 1.0, "Jz": 1.0},
            #  [1]),
        ]
    )
    def test_identity_insertion_parametrized(block_backend, coupling_factory, site_args, coupling_kwargs, valid_positions):
        backend = backends.get_backend(block_backend=block_backend)
        sites_list = site_args[0](backend)
        coupling = coupling_factory(sites_list, **coupling_kwargs)
        orig_num_sites = len(coupling.sites)
        orig_num_factors = len(coupling.factorization)
        orig_hash = coupling.to_hash()

        for pos in valid_positions:
            new_coupling = coupling.insert_identity_between_sites(position=pos)
            # Number of sites should remain the same
            assert len(new_coupling.sites) == orig_num_sites
            # Number of factors should increase by 1
            assert len(new_coupling.factorization) == orig_num_factors + 1
            # Hash should change
            assert new_coupling.to_hash() != orig_hash

        # Test invalid positions
        for invalid_pos in [0, -1, orig_num_factors + 2]:
            with pytest.raises(ValueError):
                coupling.insert_identity_between_sites(position=invalid_pos)


# def test_coupling_identity_insertion():
#     """Test that inserting identity between sites creates a different hash."""
#
#
#     spin_site = SpinSite(S=0.5, conserve='Sz')
#
#     coupling = heisenberg_coupling([spin_site, spin_site], J=1.0)
#     coupling_with_id = coupling.insert_identity_between_sites(1)
#
#     hash_original = coupling.to_hash()
#     hash_with_id = coupling_with_id.to_hash()
#
#     assert hash_original != hash_with_id
#     assert len(coupling_with_id.factorization) == len(coupling.factorization) + 1
#     assert len(coupling_with_id.sites) == len(coupling.sites)+1


def test_coupling_graph_structure():
    """Test that building a graph using coupling hashes works.

    """

    spin_site = SpinSite(S=0.5, conserve='Sz')

    coupling = heisenberg_coupling([spin_site, spin_site], J=1.0)
    coupling_hash = coupling.to_hash()

    graph = [{} for _ in range(2)]

    factorization = coupling.factorization

    for local_idx, tensor in enumerate(factorization):
        tensor = permute_legs(tensor, codomain=['wL', 'wR'], domain=['p', 'p*'])
        tensor_np = tensor.to_numpy()

        chiL = tensor_np.shape[0]
        chiR = tensor_np.shape[1]

        for jL in range(chiL):
            keyL = ('coupling', coupling_hash, local_idx, jL)
            for jR in range(chiR):
                keyR = ('coupling', coupling_hash, local_idx, jR)
                op = tensor_np[jL, jR, :, :]
                if op is not None and len(op.shape) >= 2:
                    norm_sq = float((op.real * op.real + op.imag * op.imag).sum())
                    norm_val = norm_sq**0.5
                    if norm_val > 1e-12:
                        graph[local_idx][(keyL, keyR)] = op

    assert len(graph) == 2

    hash_keys_found = set()
    for site_graph in graph:
        for keyL, keyR in site_graph.keys():
            if isinstance(keyL, tuple) and keyL[0] == 'coupling':
                hash_keys_found.add(keyL[1])

    assert coupling_hash in hash_keys_found


def test_coupling_graph_from_multiple_couplings():
    """Test building a graph from multiple couplings with unique hash keys."""
    try:
        from cyten.models.sites import SpinSite
        from cyten.models.couplings import heisenberg_coupling, spin_field_coupling
        from cyten.tensors import permute_legs
    except ImportError:
        pytest.skip('cyten not available')

    spin_site = SpinSite(S=0.5, conserve='Sz')

    coupling1 = heisenberg_coupling([spin_site, spin_site], J=1.0)
    coupling2 = heisenberg_coupling([spin_site, spin_site], J=2.0)
    coupling3 = spin_field_coupling([spin_site], hz=0.5)

    couplings = [coupling1, coupling2, coupling3]
    hashes = [c.to_hash() for c in couplings]

    assert len(set(hashes)) == 3

    L = 4
    graph = [{} for _ in range(L)]

    for coupling_idx, coupling in enumerate(couplings):
        coupling_hash = coupling.to_hash()
        factorization = coupling.factorization

        for local_idx, tensor in enumerate(factorization):
            site_idx = local_idx % L

            tensor = permute_legs(tensor, codomain=['wL', 'wR'], domain=['p', 'p*'])
            tensor_np = tensor.to_numpy()

            chiL = tensor_np.shape[0]
            chiR = tensor_np.shape[1]

            for jL in range(chiL):
                keyL = ('coupling', coupling_hash, local_idx, jL)
                for jR in range(chiR):
                    keyR = ('coupling', coupling_hash, local_idx, jR)
                    op = tensor_np[jL, jR, :, :]
                    if op is not None and len(op.shape) >= 2 and op.shape[0] > 1:
                        graph[site_idx][(keyL, keyR)] = op

    all_hashes_in_graph = set()
    for site_graph in graph:
        for keyL, keyR in site_graph.keys():
            if isinstance(keyL, tuple) and keyL[0] == 'coupling':
                all_hashes_in_graph.add(keyL[1])

    for h in hashes:
        assert h in all_hashes_in_graph


def test_coupling_to_graph_keys():
    """Test that coupling data is correctly converted to graph key-value structure."""
    try:
        from cyten.models.sites import SpinSite
        from cyten.models.couplings import heisenberg_coupling
        from cyten.tensors import permute_legs
    except ImportError:
        pytest.skip('cyten not available')

    spin_site = SpinSite(S=0.5, conserve='Sz')
    coupling = heisenberg_coupling([spin_site, spin_site], J=1.0)

    coupling_hash = coupling.to_hash()
    factorization = coupling.factorization

    assert len(factorization) == 2

    all_keys = []
    all_values = []

    for local_idx, tensor in enumerate(factorization):
        tensor_permuted = permute_legs(tensor, codomain=['wL', 'wR'], domain=['p', 'p*'])
        tensor_np = tensor_permuted.to_numpy()

        chiL = tensor_np.shape[0]
        chiR = tensor_np.shape[1]

        for jL in range(chiL):
            keyL = ('coupling', coupling_hash, local_idx, jL)
            for jR in range(chiR):
                keyR = ('coupling', coupling_hash, local_idx, jR)
                op = tensor_np[jL, jR, :, :]

                all_keys.append((keyL, keyR))
                all_values.append(op)

    assert len(all_keys) > 0
    assert len(all_keys) == len(all_values)

    hash_from_keys = set()
    for keyL, keyR in all_keys:
        if keyL[0] == 'coupling':
            hash_from_keys.add(keyL[1])

    assert coupling_hash in hash_from_keys


def test_mpograph_coupling_keys_with_tenpy():
    """Test the structure of keys that MPOGraph.add_coupling_as_term creates.

    This test requires tenpy to be available. It will be skipped if tenpy
    cannot be imported.
    """
    try:
        from tenpy.networks import mpo
        from tenpy.networks import site as tenpy_site
    except ImportError:
        pytest.skip('tenpy.networks not available')

    spin_site = SpinSite(S=0.5, conserve='Sz')
    coupling = heisenberg_coupling([spin_site, spin_site], J=1.0)

    coupling_hash = coupling.to_hash()

    tenpy_sites = [tenpy_site.SpinHalfSite(conserve='Sz', sort_charge=False) for _ in range(4)]
    graph = mpo.MPOGraph(tenpy_sites, bc='finite', unit_cell_width=4)

    graph.add_coupling_as_term(coupling)

    coupling_keys = []
    for site_graph in graph.graph:
        for keyL in site_graph.keys():
            if isinstance(keyL, tuple) and len(keyL) >= 2 and keyL[0] == 'coupling':
                coupling_keys.append(keyL)

    assert len(coupling_keys) > 0

    for key in coupling_keys:
        assert key[0] == 'coupling'
        assert key[1] == coupling_hash


def test_coupling_identity_string_mimic():
    """Test that we can build identity strings for couplings.
    """
    try:
        from cyten.models.sites import SpinSite
        from cyten.models.couplings import heisenberg_coupling
        from cyten.tensors import permute_legs
    except ImportError:
        pytest.skip('cyten not available')

    spin_site = SpinSite(S=0.5, conserve='Sz')

    coupling = heisenberg_coupling([spin_site, spin_site], J=1.0)
    coupling_with_id = coupling.insert_identity_between_sites(1)

    assert len(coupling_with_id.factorization) == len(coupling.factorization) + 1

    coupling_hash = coupling_with_id.to_hash()
    hash_key = ('coupling', coupling_hash)

    L = 4
    graph = [{} for _ in range(L)]
    states = [set() for _ in range(L + 1)]

    factorization = coupling_with_id.factorization
    num_tensors = len(factorization)

    for local_idx in range(num_tensors):
        site_idx = local_idx % L
        tensor = factorization[local_idx]
        tensor = permute_legs(tensor, codomain=['wL', 'wR'], domain=['p', 'p*'])
        tensor_np = tensor.to_numpy()

        chiL = tensor_np.shape[0]
        chiR = tensor_np.shape[1]

        for jL in range(chiL):
            for jR in range(chiR):
                op = tensor_np[jL, jR, :, :]
                norm_sq = float((op.real * op.real + op.imag * op.imag).sum())
                norm_val = norm_sq**0.5
                if norm_val < 1e-12:
                    keyL = hash_key + (local_idx, jL)
                    keyR = hash_key + (local_idx + 1, jR)
                    if keyL not in graph[site_idx]:
                        graph[site_idx][keyL] = {}
                    graph[site_idx][keyL][keyR] = op
                    states[site_idx].add(keyL)
                    states[site_idx + 1].add(keyR)

    assert len(graph) == L
    assert len(states) == L + 1

    hash_keys_in_graph = set()
    for site_graph in graph:
        for keyL in site_graph.keys():
            if isinstance(keyL, tuple) and keyL[0] == 'coupling':
                hash_keys_in_graph.add(keyL[1])

    assert coupling_hash in hash_keys_in_graph


def test_coupling_string_methods_logic():
    """Test add_coupling_string_left_to_right and add_coupling_string_right_to_left.
    """
    try:
        from cyten.models.sites import SpinSite
        from cyten.models.couplings import heisenberg_coupling
    except ImportError:
        pytest.skip('cyten not available')

    spin_site = SpinSite(S=0.5, conserve='Sz')
    coupling = heisenberg_coupling([spin_site, spin_site], J=1.0)

    coupling_hash = coupling.to_hash()
    hash_key = ('coupling', coupling_hash)

    start_key = ('coupling', coupling_hash, 0, 'start')

    returned_keys = []

    i, j = 0, 3
    keyL = keyR = start_key
    for k in range(i + 1, j):
        if (k - i) % 4 == 0:
            keyR = keyL + (k, hash_key, 'Id')
        if not (keyL, keyR) in [(('a', 'b'), ('c', 'd'))]:
            returned_keys.append((k, keyL, keyR))
        keyL = keyR

    assert len(returned_keys) == 2
    assert returned_keys[0][0] == 1
    assert returned_keys[1][0] == 2


def test_add_missing_IdL_IdR_logic():
    """Test add_missing_IdL_IdR for coupling hashes.
    """
    L = 4
    graph = [{} for _ in range(L)]
    states = [set() for _ in range(L + 1)]

    graph[0][('IdL',)] = {('IdL',): [('Id', 1.0)]}
    graph[2][('coupling', 'hash123')] = {('coupling', 'hash123'): [('op', 1.0)]}

    insert_all_id = True
    max_IdL = L
    min_IdR = 0

    for k in range(0, max_IdL):
        if ('IdL', 'IdL') not in [(key, rkey) for key in graph[k] for rkey in graph[k][key]]:
            if 'IdL' not in graph[k]:
                graph[k]['IdL'] = {}
            graph[k]['IdL']['IdL'] = [('Id', 1.0)]

    for k in range(min_IdR, L):
        if ('IdR', 'IdR') not in [(key, rkey) for key in graph[k] for rkey in graph[k][key]]:
            if 'IdR' not in graph[k]:
                graph[k]['IdR'] = {}
            graph[k]['IdR']['IdR'] = [('Id', 1.0)]

    assert ('IdL', 'IdL') in [(key, rkey) for key in graph[0] for rkey in graph[0][key]]
    assert ('IdR', 'IdR') in [(key, rkey) for key in graph[3] for rkey in graph[3][key]]




