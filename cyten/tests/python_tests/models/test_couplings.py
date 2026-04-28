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
