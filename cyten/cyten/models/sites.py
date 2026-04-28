"""Defines classes that describe the sites of a lattice."""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

from collections.abc import Sequence
from itertools import product as itproduct
from typing import Literal

import numpy as np

from ..backends import TensorBackend
from ..symmetries import (
    ElementarySpace,
    FermionParity,
    FibonacciAnyonCategory,
    IsingAnyonCategory,
    NoSymmetry,
    ProductSymmetry,
    SU2_kAnyonCategory,
    SU2Symmetry,
    Symmetry,
    U1Symmetry,
    ZNSymmetry,
)
from .degrees_of_freedom import AnyonDOF, BosonicDOF, ClockDOF, FermionicDOF, SpinDOF


class SpinSite(SpinDOF):
    """Class for sites that have a single spin degree of freedom.

    TODO find a good format to doc the onsite operators that exist in a site

    Attributes
    ----------
    S : float
        The total spin.
    double_total_spin : int
        Twice the :attr:`S`. We store this in addition because it is an integer.
    conserve : Literal['SU(2)', 'Sz', 'parity', 'None']
        The symmetry to be conserved. We can conserve::

            - SU(2), the full spin rotation symmetry.
            - Sz (= U(1) symmetry), with sector labels corresponding to ``2 * Sz``.
            - Sz parity (= Z_2 symmetry), with sector labels corresponding to ``(Sz + S_tot) % 2``.
            - nothing.

        Conserves nothing by default.

    """

    def __init__(
        self,
        S: float = 0.5,
        conserve: Literal['SU(2)', 'Sz', 'parity', 'None'] = None,
        backend: TensorBackend = None,
        default_device: str = None,
    ):
        self.S = S = float(S)
        two_S = int(round(2 * S, 0))
        self.double_total_spin = two_S
        dim = two_S + 1
        if two_S < 0:
            raise ValueError('Negative spin.')
        if not np.allclose(two_S / 2, S):
            raise ValueError('total_spin must be half integer: 0, 1/2, 1, 3/2, ...')

        # build spin vector
        Sz = np.diag(-S + np.arange(dim))
        Sp = np.zeros((dim, dim))
        for n in range(dim - 1):
            # Sp |m> = sqrt( S(S+1) - m(m+1) ) |m+1>
            m = n - S
            Sp[n + 1, n] = np.sqrt(S * (S + 1) - m * (m + 1))
        spin_vector = self._spin_vector_from_Sp(Sz=Sz, Sp=Sp)

        sym = SpinDOF.conservation_law_to_symmetry(conserve)
        # build leg
        if isinstance(sym, SU2Symmetry):
            leg = ElementarySpace.from_defining_sectors(sym, [[two_S]])
        elif isinstance(sym, U1Symmetry):
            leg = ElementarySpace.from_basis(sym, np.arange(-two_S, two_S + 2, 2)[:, None])
        elif isinstance(sym, ZNSymmetry):
            leg = ElementarySpace.from_basis(sym, np.arange(dim)[:, None] % 2)
        elif isinstance(sym, NoSymmetry):
            leg = ElementarySpace.from_trivial_sector(dim=dim, symmetry=sym)
        else:
            raise ValueError(f'`conserve` invalid for `SpinSite`: {conserve}')
        self.conserve = conserve

        state_labels = {str(n - S): n for n in range(dim)}
        state_labels['down'] = 0
        state_labels['up'] = dim - 1

        SpinDOF.__init__(
            self,
            leg=leg,
            spin_vector=spin_vector,
            state_labels=state_labels,
            backend=backend,
            default_device=default_device,
        )

        if not isinstance(sym, SU2Symmetry):
            self.add_onsite_operator('Sz', spin_vector[:, :, 2], is_diagonal=True)
            if two_S == 1:
                self.add_onsite_operator('Sigmaz', 2.0 * spin_vector[:, :, 2], is_diagonal=True)
        if isinstance(sym, NoSymmetry):
            self.add_onsite_operator('Sx', spin_vector[:, :, 0])
            self.add_onsite_operator('Sy', spin_vector[:, :, 1])
            self.add_onsite_operator('Sp', spin_vector[:, :, 0] + 1.0j * spin_vector[:, :, 1])
            self.add_onsite_operator('Sm', spin_vector[:, :, 0] - 1.0j * spin_vector[:, :, 1])
            if two_S == 1:
                self.add_onsite_operator('Sigmax', 2.0 * spin_vector[:, :, 0])
                self.add_onsite_operator('Sigmay', 2.0 * spin_vector[:, :, 1])

    def test_sanity(self):
        """Perform sanity checks."""
        super().test_sanity()
        S_sq = np.tensordot(self.spin_vector, self.spin_vector, ([-1, 1], [-1, 0]))
        eigenvalue = self.double_total_spin * (self.double_total_spin + 2) / 4
        assert np.allclose(S_sq, eigenvalue * np.eye(self.double_total_spin + 1))

    def __repr__(self):
        return f'SpinSite(S={self.S}, conserve={self.conserve})'


class SpinlessBosonSite(BosonicDOF):
    """Site for (possibly multiple) spinless bosons.

    TODO describe onsite operators

    Parameters
    ----------
    Nmax : int | Sequence[int]
        The maximum occupation of each of the boson species. An `int` corresponds to a single boson
        species. Otherwise, the number of boson species corresponds to `len(Nmax)`.
    conserve : Literal['N', 'parity', 'None'] | Sequence[Literal['N', 'parity', 'None']]
        The symmetry to be conserved. We can conserve::

            - total particle number sum_k N_k (``conserve == 'N'``).
            - individual particle numbers N_k (``conserve[i] == 'N'``).
            - total parity (sum_i N_k) % 2 (``conserve == 'parity'``).
            - individual parities N_k % 2 (``conserve[i] == 'parity'``).
            - nothing (``conserve == 'None'`` or ``conserve[i] == 'None'``).

        A `Literal` corresponds to symmetries involving all boson species, such as the total
        particle number (``conserve == 'N'``) or the total parity (``conserve == 'parity'``).
        For a sequence, the entry ``conserve[i]`` corresponds to the symmetry of boson species `k`,
        such that, e.g., ``conserve[k] == 'N'`` signifies that its particle number is conserved.

        Conserves nothing by default.
    filling : float | None
        Average total filling (that is, filling of all species together). Used to define the
        on-site operators ``dN`` and ``dNdN`` if ``filling is not None``.

    Attributes
    ----------
    conserve : Literal['N', 'parity', 'None'] | list[Literal['N', 'parity', 'None']]
        The conserved symmetry, see above.
    filling : float | None
        Average total filling.
    num_species, Nmax, creators, annihilators
        see :class:`BosonicDOF`

    """

    def __init__(
        self,
        Nmax: int | list[int] | np.ndarray,
        conserve: Literal['N', 'parity', 'None'] | Sequence[Literal['N', 'parity', 'None']] = None,
        filling: float | None = None,
        backend: TensorBackend = None,
        default_device: str = None,
    ):
        Nmax = np.atleast_1d(np.asarray(Nmax, dtype=int))
        # need to manually throw an error for non-integers in Nmax
        assert np.allclose(Nmax, np.asarray(Nmax)), f'Invalid `Nmax`: {Nmax}'
        num_species = len(Nmax)
        if not isinstance(conserve, str) and conserve is not None:
            # strings are sequences, so test for strings
            msg = f'Invalid number of entries in `conserve`: {len(conserve)} != {num_species}'
            assert len(conserve) == num_species, msg
        self.filling = filling

        # states for each species
        states = [list(range(n + 1)) for n in Nmax]
        dims = np.ones_like(Nmax) + Nmax
        total_dim = np.prod(dims, dtype=int)

        sym = BosonicDOF.conservation_law_to_symmetry(conserve)
        if isinstance(sym, ProductSymmetry):
            assert len(conserve) == len(sym.factors)  # TODO delete
            no_sym_idcs = []
            parity_sym_idcs = []
            for i, sym_factor_i in enumerate(sym.factors):
                if isinstance(sym_factor_i, NoSymmetry):
                    no_sym_idcs.append(i)
                elif isinstance(sym_factor_i, ZNSymmetry):
                    parity_sym_idcs.append(i)
                elif isinstance(sym_factor_i, U1Symmetry):
                    pass
                else:
                    msg = f'Entry in `conserve` invalid for `SpinlessBosonSite`: {conserve[i]}'
                    raise ValueError(msg)
            sectors = []
            for occupations in itproduct(*states):
                sector = np.asarray(occupations, dtype=int)
                sector[no_sym_idcs] = 0
                sector[parity_sym_idcs] = np.mod(sector[parity_sym_idcs], 2)
                sectors.append(sector)
            leg = ElementarySpace.from_basis(sym, np.asarray(sectors, dtype=int))
        else:
            if isinstance(sym, (U1Symmetry, ZNSymmetry)):
                # for U(1) and Z_2, iterate over all states in the correct order to
                # get the correct basis_perm in ElementarySpace.from_basis
                sectors = []
                for occupations in itproduct(*states):
                    sectors.append(np.sum(occupations))
                sectors = np.asarray(sectors, dtype=int)[:, None]
                if isinstance(sym, ZNSymmetry):
                    sectors = np.mod(sectors, 2)
                leg = ElementarySpace.from_basis(sym, sectors)
            elif isinstance(sym, NoSymmetry):
                leg = ElementarySpace.from_trivial_sector(dim=total_dim, symmetry=sym)
            else:
                raise ValueError(f'`conserve` invalid for `SpinlessBosonSite`: {conserve}')
        self.conserve = conserve

        # state labels have the form '(n0, n1, ...)' with n0, n1, ... corresponding to the
        # occupations for the species. For a single species, this is changed to 'n0', i.e.,
        # the brackets and comma from the tuple are omitted.
        state_labels = {}
        dim_prod = np.asarray([np.prod(dims[i + 1 :]) for i in range(num_species)], dtype=int)
        for occupations in itproduct(*states):
            label = str(occupations)
            if num_species == 1:
                label = label[1:-2]
            state_labels[label] = np.sum(np.asarray(occupations, dtype=int) * dim_prod)
        # vacuum == no bosons
        state_labels['vac'] = 0

        creators, annihilators = BosonicDOF._creation_annihilation_ops_from_Nmax(Nmax=Nmax)

        BosonicDOF.__init__(
            self,
            leg=leg,
            creators=creators,
            annihilators=annihilators,
            state_labels=state_labels,
            onsite_operators=None,
            backend=backend,
            default_device=default_device,
        )
        self.add_individual_occupation_ops()
        self.add_total_occupation_ops()
        # construct operators relative to filling
        ops = {}
        if filling is not None:
            dN_diag = np.diag(self.n_tot) - filling * np.ones(total_dim)
            dN = np.diag(dN_diag)
            dNdN = np.diag(dN_diag**2)
            ops['dN'] = dN
            ops['dNdN'] = dNdN
        for name, op in ops.items():
            self.add_onsite_operator(name, op, is_diagonal=True)

    def __repr__(self):
        return f'SpinlessBosonSite(Nmax={self.Nmax}, conserve={self.conserve}, filling={self.filling})'


class SpinlessFermionSite(FermionicDOF):
    """Site for (possibly multiple) spinless fermions.

    TODO describe onsite operators

    .. todo ::
        For now, assume that the symmetry needs to capture the fermionic statistics.
        Do not think about JW strings yet...
        That is also the reason why NoSymmetry is not an option here

    Parameters
    ----------
    num_species : int
        Number of fermion species.
    conserve : Literal['N', 'parity'] | Sequence[Literal['N', 'parity', 'None']]
        The symmetry to be conserved. We can conserve::

            - total fermion number sum_i N_k (``conserve == 'N'``).
            - individual fermion numbers N_k (``conserve[i] == 'N'``).
            - total fermion parity (sum_i N_k) % 2 (``conserve == 'parity'``).
            - individual fermion parities N_k % 2 (``conserve[i] == 'parity'``).
            - nothing for an individual fermion (``conserve[i] == 'None'``); .

        A `Literal` corresponds to symmetries involving all fermion species, such as the total
        fermion number (``conserve == 'N'``) or the total fermion parity
        (``conserve == 'parity'``). For a sequence, the entry ``conserve[k]`` corresponds to the
        symmetry of fermion species `k`, such that, e.g., ``conserve[k] == 'N'`` signifies that
        its fermion number is conserved.

        Note that the total fermion parity is always conserved. It is thus always part of the
        symmetry. Hence, ``conserve == 'None'`` is not a valid value. On the other hand,
        ``conserve = ['None']`` is interpreted as valid and the resulting symmetry conserves the
        fermionic parity.

        Conserves total fermion parity by default.
    filling : float | None
        Average total filling (that is, filling of all species together). Used to define the
        on-site operators ``dN`` and ``dNdN`` if ``filling is not None``.

    Attributes
    ----------
    num_species : int
        Number of fermion species.
    conserve : Literal['N', 'parity'] | list[Literal['N', 'parity', 'None']]
        The conserved symmetry, see above.
    filling : float, optional
        Average total filling.
    creators, annihilators
        see :class:`FermionicDOF`

    """

    def __init__(
        self,
        num_species: int,
        conserve: Literal['N', 'parity'] | Sequence[Literal['N', 'parity', 'None']] = 'parity',
        filling: float | None = None,
        backend: TensorBackend = None,
        default_device: str = None,
    ):
        assert isinstance(num_species, int)
        assert num_species > 0, 'Must have at least a single fermion species'
        if not isinstance(conserve, str):
            msg = f'Invalid number of entries in `conserve`: {len(conserve)} != {num_species}'
            assert len(conserve) == num_species, msg
        self.filling = filling

        sym = FermionicDOF.conservation_law_to_symmetry(conserve)
        if isinstance(sym, FermionParity):
            sectors = []
            for occupations in itproduct([0, 1], repeat=num_species):
                sectors.append(np.sum(occupations) % 2)
            leg = ElementarySpace.from_basis(sym, np.asarray(sectors, dtype=int)[:, None])
        elif not isinstance(conserve, str):
            assert len(conserve) + 1 == len(sym.factors)
            no_sym_idcs = []
            parity_sym_idcs = []
            # no need to iterate over the final fermion parity
            for i, sym_factor_i in enumerate(sym.factors[:-1]):
                if isinstance(sym_factor_i, NoSymmetry):
                    no_sym_idcs.append(i)
                elif isinstance(sym_factor_i, ZNSymmetry):
                    parity_sym_idcs.append(i)
                elif isinstance(sym_factor_i, U1Symmetry):
                    pass
                else:
                    msg = f'Entry in `conserve` invalid for `SpinlessFermionSite`: {conserve[i]}'
                    raise ValueError(msg)
            sectors = []
            for occupations in itproduct([0, 1], repeat=num_species):
                sector = np.asarray(occupations, dtype=int)
                sector = np.append(sector, np.sum(sector) % 2)
                sector[no_sym_idcs] = 0
                sectors.append(sector)
            leg = ElementarySpace.from_basis(sym, np.asarray(sectors, dtype=int))
        elif isinstance(sym.factors[0], U1Symmetry):
            # remaining case: conserve total particle number
            assert len(sym.factors) == 2
            sectors = []
            for occupations in itproduct([0, 1], repeat=num_species):
                fermion_number = np.sum(occupations)
                sectors.append([fermion_number, fermion_number % 2])
            leg = ElementarySpace.from_basis(sym, np.asarray(sectors, dtype=int))
        else:
            raise ValueError(f'`conserve` invalid for `SpinlessFermionSite`: {conserve}')
        self.conserve = conserve

        # state labels have the form '(n0, n1, ...)' with n0, n1, ... corresponding to the
        # occupations for the species. For a single species, this is changed to 'n0', i.e.,
        # the brackets and comma from the tuple are omitted.
        state_labels = {}
        for occupations in itproduct([0, 1], repeat=num_species):
            label = str(occupations)
            if num_species == 1:
                label = label[1:-2]
            state_labels[label] = int(''.join(str(n_i) for n_i in occupations), 2)
        # vacuum == no fermions
        state_labels['vac'] = 0

        creators, annihilators = FermionicDOF._creation_annihilation_ops(num_species=num_species)

        FermionicDOF.__init__(
            self,
            leg=leg,
            creators=creators,
            annihilators=annihilators,
            state_labels=state_labels,
            onsite_operators=None,
            backend=backend,
            default_device=default_device,
        )
        self.add_individual_occupation_ops()
        self.add_total_occupation_ops()

        # construct operators relative to filling
        ops = {}
        if filling is not None:
            dN_diag = np.diag(self.n_tot) - filling * np.ones(2**num_species)
            dN = np.diag(dN_diag)
            dNdN = np.diag(dN_diag**2)
            ops['dN'] = dN
            ops['dNdN'] = dNdN
        for name, op in ops.items():
            self.add_onsite_operator(name, op, is_diagonal=True, understood_braiding=True)

    def __repr__(self):
        return f'SpinlessFermionSite(num_species={self.num_species}, conserve={self.conserve}, filling={self.filling})'


class SpinHalfFermionSite(SpinDOF, FermionicDOF):
    """Site for spin-1/2 fermions.

    TODO describe onsite operators

    Parameters
    ----------
    conserve_N : Literal['N', 'parity']
        The fermion symmetry to be conserved. We can conserve::

            - total fermion number N_up + N_down (``conserve == 'N'``).
            - total fermion parity (N_up + N_down) % 2 (``conserve == 'parity'``).

        Note that the total fermion parity is always conserved and is thus always part of the
        total symmetry. Hence, ``conserve == 'None'`` is not a valid choice.
        Conserves total fermion parity by default.
    conserve_S : Literal['SU(2)', 'Sz', 'parity', 'None']
        The spin symmetry to be conserved. We can conserve::

            - SU(2), the full spin rotation symmetry.
            - Sz (= U(1) symmetry), with sector labels corresponding to ``2 * Sz``.
            - Sz parity (= Z_2 symmetry), with sector labels corresponding to ``(Sz + S_tot) % 2``.
            - nothing.

        Conserves nothing by default.
    filling : float | None
        Average total filling (that is, filling of spin up and spin down fermions together). Used
        to define the on-site operators ``dN`` and ``dNdN`` if ``filling is not None``.

    Attributes
    ----------
    conserve_N : Literal['N', 'parity']
        The conserved symmetry, see above.
    conserve_S : Literal['SU(2)', 'Sz', 'parity', 'None']
        The conserved spin symmetry, see above.
    filling : float, optional
        Average total filling.
    creators, annihilators
        see :class:`FermionicDOF`
    spin_vector
        see :class:`SpinDOF`

    """

    def __init__(
        self,
        conserve_N: Literal['N', 'parity'] = 'parity',
        conserve_S: Literal['SU(2)', 'Sz', 'parity', 'None'] = None,
        filling: float | None = None,
        backend: TensorBackend = None,
        default_device: str = None,
    ):
        assert isinstance(conserve_N, str), f'Invalid `conserve_N`: {conserve_N}'
        self.filling = filling

        sym_N = FermionicDOF.conservation_law_to_symmetry(conserve_N)
        if not isinstance(conserve_N, str):
            sym_N.factors[0].descriptive_name = sym_N.factors[0].descriptive_name.replace('species0', 'spin_up')
            sym_N.factors[1].descriptive_name = sym_N.factors[1].descriptive_name.replace('species1', 'spin_down')
            sym_N.descriptive_name = sym_N.descriptive_name.replace('species0', 'spin_up')
            sym_N.descriptive_name = sym_N.descriptive_name.replace('species1', 'spin_down')

        # construct sectors (including spin as U(1)) as: [spin, fermion U(1), fermion parity]
        if isinstance(sym_N, FermionParity):
            sectors = np.asarray([[0, 0], [-1, 1], [1, 1], [0, 0]], dtype=int)
        elif isinstance(sym_N.factors[0], U1Symmetry):
            sectors = np.asarray([[0, 0, 0], [-1, 1, 1], [1, 1, 1], [0, 2, 0]], dtype=int)
        else:
            raise ValueError(f'`conserve_N` invalid for `SpinHalfFermionSite`: {conserve_N}')

        sym_S = SpinDOF.conservation_law_to_symmetry(conserve_S)
        if isinstance(sym_S, U1Symmetry):
            pass  # sectors already correct
        elif isinstance(sym_S, ZNSymmetry):
            sectors[:, 0] = np.mod(sectors[:, 0], 2)
        elif isinstance(sym_S, SU2Symmetry):
            sectors[1, 0] = 1
        elif isinstance(sym_S, NoSymmetry):
            sectors = sectors[:, 1:]
        else:
            raise ValueError(f'`conserve_S` invalid for `SpinHalfFermionSite`: {conserve_S}')

        if isinstance(sym_S, NoSymmetry):
            sym = sym_N
        else:
            sym = [sym_S, *sym_N.factors] if isinstance(sym_N, ProductSymmetry) else [sym_S, sym_N]
            sym = ProductSymmetry(sym)
        leg = ElementarySpace.from_basis(sym, sectors)
        self.conserve_N = conserve_N
        self.conserve_S = conserve_S

        # build spin vector and creation / annihilation ops
        Sz = np.diag([0, -0.5, 0.5, 0])
        Sp = np.zeros((4, 4))
        Sp[2, 1] = 1
        spin_vector = self._spin_vector_from_Sp(Sz=Sz, Sp=Sp)
        creators, annihilators = FermionicDOF._creation_annihilation_ops(num_species=2)

        state_labels = {
            '(0, 0)': 0,
            '(0, 1)': 1,
            '(1, 0)': 2,
            '(1, 1)': 3,
            'empty': 0,
            'vac': 0,
            'down': 1,
            'up': 2,
            'full': 3,
        }

        super().__init__(
            leg=leg,
            spin_vector=spin_vector,
            creators=creators,
            annihilators=annihilators,
            state_labels=state_labels,
            onsite_operators=None,
            backend=backend,
            default_device=default_device,
            species_names=['up', 'down'],
        )

        if not isinstance(sym_S, SU2Symmetry):
            self.add_individual_occupation_ops()
            self.onsite_operators.update({'Nup': self.onsite_operators.pop('N0')})
            self.onsite_operators.update({'Ndown': self.onsite_operators.pop('N1')})
        self.add_total_occupation_ops()

        # spin operators
        ops = {}
        if not isinstance(sym_S, SU2Symmetry):
            ops['Sz'] = spin_vector[:, :, 2]
            ops['Sigmaz'] = 2.0 * spin_vector[:, :, 2]
        if isinstance(sym_S, NoSymmetry):
            self.add_onsite_operator('Sx', spin_vector[:, :, 0], understood_braiding=True)
            self.add_onsite_operator('Sy', spin_vector[:, :, 1], understood_braiding=True)
            self.add_onsite_operator('Sp', spin_vector[:, :, 0] + 1.0j * spin_vector[:, :, 1], understood_braiding=True)
            self.add_onsite_operator('Sm', spin_vector[:, :, 0] - 1.0j * spin_vector[:, :, 1], understood_braiding=True)
            self.add_onsite_operator('Sigmax', 2.0 * spin_vector[:, :, 0], understood_braiding=True)
            self.add_onsite_operator('Sigmay', 2.0 * spin_vector[:, :, 1], understood_braiding=True)

        # construct operators relative to filling
        if filling is not None:
            dN_diag = np.diag(self.n_tot) - filling * np.ones(4)
            dN = np.diag(dN_diag)
            dNdN = np.diag(dN_diag**2)
            ops['dN'] = dN
            ops['dNdN'] = dNdN
        for name, op in ops.items():
            self.add_onsite_operator(name, op, is_diagonal=True, understood_braiding=True)

    def __repr__(self):
        return (
            f'SpinHalfFermionSite(conserve_N={self.conserve_N}, conserve_S={self.conserve_S}, filling={self.filling})'
        )


class ClockSite(ClockDOF):
    """Class for sites that have a single quantum clock degree of freedom.

    TODO describe onsite operators

    Parameters
    ----------
    q : int
        Number of states per site.
    conserve : Literal['Z_N', 'None']
        The symmetry to be conserved. We can conserve::

            - Z_N symmetry.
            - nothing.

    Attributes
    ----------
    conserve : Literal['Z_N', 'None']
        The conserved symmetry, see above.
    q, clock_operators
        see :class:`ClockDOF`

    """

    def __init__(
        self, q: int, conserve: Literal['Z_N', 'None'] = None, backend: TensorBackend = None, default_device: str = None
    ):
        assert isinstance(q, int)

        # build clock operators
        X = np.eye(q, k=1) + np.eye(q, k=1 - q)
        Z = np.diag(np.exp(2.0j * np.pi * np.arange(q, dtype=np.complex128) / q))
        clock_operators = np.stack([X, Z], axis=2)

        # build leg
        if conserve in ['Z_N', 'ZN', 'Z_q', 'Zq']:
            sym = ZNSymmetry(q, 'q')
            leg = ElementarySpace.from_basis(sym, np.arange(q)[:, None])
        elif conserve in ['None', 'none', None]:
            sym = NoSymmetry()
            leg = ElementarySpace.from_trivial_sector(dim=q, symmetry=sym)
        else:
            raise ValueError(f'Invalid `conserve`: {conserve}')
        self.conserve = conserve

        state_labels = {str(n): n for n in range(q)}
        state_labels['up'] = 0
        if q % 2 == 0:
            state_labels['down'] = q // 2

        ClockDOF.__init__(
            self,
            leg=leg,
            q=q,
            clock_operators=clock_operators,
            state_labels=state_labels,
            backend=backend,
            default_device=default_device,
        )

        Xhc = np.conj(clock_operators[:, :, 0].T)
        if isinstance(sym, NoSymmetry):
            self.add_onsite_operator('X', X)
            self.add_onsite_operator('Xhc', Xhc)
            self.add_onsite_operator('Xphc', X + Xhc)

    def __repr__(self):
        return f'ClockSite(q={self.q}, conserve={self.conserve})'


class AnyonSite(AnyonDOF):
    """Class for anyon models where the local Hilbert space contains all sectors once.

    Parameters
    ----------
    symmetry : Symmetry
        The symmetry describing the anyons.
    sector_names : sequence of str or None
        The sector names that appear in the onsite projection operators. The `i`th operator is
        called `f'P_{sector_names[i]}'` and projects onto the `i`th sector in
        `leg.sector_decomposition`. For `None` entries (default), no projection operators are
        constructed.

    """

    def __init__(
        self,
        symmetry: Symmetry,
        sector_names: Sequence[str | None] = None,
        backend: TensorBackend = None,
        default_device: str = None,
    ):
        leg = ElementarySpace.from_defining_sectors(symmetry, symmetry.all_sectors())
        AnyonDOF.__init__(self, leg=leg, sector_names=sector_names, backend=backend, default_device=default_device)

    def __repr__(self):
        return f'AnyonSite(symmetry={self.symmetry}, sector_names={self.sector_names})'


class FibonacciAnyonSite(AnyonSite):
    """Class for sites containing the trivial and the Fibonacci / tau sectors.

    Projectors onto the onsite vacuum and tau sectors are automatically constructed
    and are named `'P_vac'` and `'P_tau'`, respectively.

    Parameters
    ----------
    handedness: Literal['left', 'right']
        The handedness of the anyons.

    """

    def __init__(
        self, handedness: Literal['left', 'right'] = 'left', backend: TensorBackend = None, default_device: str = None
    ):
        sym = FibonacciAnyonCategory(handedness=handedness)
        AnyonSite.__init__(self, sym, sector_names=['vac', 'tau'], backend=backend, default_device=default_device)

    def __repr__(self):
        return f'FibonacciAnyonSite(handedness={self.symmetry.handedness})'


class IsingAnyonSite(AnyonSite):
    """Class for sites containing the trivial, the Ising / sigma, and the fermion / psi sectors.

    Projectors onto the onsite vacuum, sigma and psi sectors are automatically constructed and are
    named `'P_vac'`, `'P_sigma'`, and `'P_psi'`, respectively.

    Parameters
    ----------
    `nu`: odd int
        Specifies the Ising anyons as different `nu` correspond to different topological twists.

    """

    def __init__(self, nu: int = 1, backend: TensorBackend = None, default_device: str = None):
        sym = IsingAnyonCategory(nu=nu)
        AnyonSite.__init__(
            self, sym, sector_names=['vac', 'sigma', 'psi'], backend=backend, default_device=default_device
        )

    def __repr__(self):
        return f'IsingAnyonSite(nu={self.symmetry.nu})'


class GoldenSite(AnyonDOF):
    """Class for Fibonacci anyon models where the local Hilbert space only contains the tau sector.

    Parameters
    ----------
    handedness: Literal['left', 'right']
        The handedness of the anyons.

    """

    def __init__(
        self, handedness: Literal['left', 'right'] = 'left', backend: TensorBackend = None, default_device: str = None
    ):
        sym = FibonacciAnyonCategory(handedness=handedness)
        leg = ElementarySpace.from_defining_sectors(sym, [sym.tau])
        AnyonDOF.__init__(self, leg=leg, backend=backend, default_device=default_device)

    def __repr__(self):
        return f'GoldenSite(handedness={self.symmetry.handedness})'


class SU2kSpin1Site(AnyonDOF):
    """Class for SU(2)_k anyon models where the local Hilbert space only contains the spin-1 sector.

    Parameters
    ----------
    k : int
        Level of the SU(2)_k anyon model / symmetry.
    handedness: Literal['left', 'right']
        The handedness of the anyons.

    """

    def __init__(
        self,
        k: int,
        handedness: Literal['left', 'right'] = 'left',
        backend: TensorBackend = None,
        default_device: str = None,
    ):
        assert k >= 2
        sym = SU2_kAnyonCategory(k, handedness=handedness)
        leg = ElementarySpace.from_defining_sectors(sym, [sym.spin_one])
        AnyonDOF.__init__(self, leg=leg, backend=backend, default_device=default_device)

    def __repr__(self):
        return f'SU2kSpin1Site(k={self.symmetry.k}, handedness={self.symmetry.handedness})'
