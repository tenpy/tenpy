# Copyright (C) TeNPy Developers, Apache license
import itertools
import numpy as np
import scipy as sp
import pytest
from tenpy.models.molecular import MolecularModel
from tenpy.tools.params import Config
from test_model import check_general_model


class MolecularModelTest(MolecularModel):
    r"""Spin-1/2 fermion molecular Hamiltonian (with simplified term loops)."""

    def init_terms(self, params: Config) -> None:
        """Initialize terms."""
        params.touch("one_body_tensor")  # suppress unused key warning
        two_body_tensor = params.get(
            "two_body_tensor",
            np.zeros((self.norb, self.norb, self.norb, self.norb)),
            expect_type="array",
        )
        constant = params.get("constant", 0, expect_type="real")

        # constant
        for p in range(self.norb):
            self.add_onsite(constant / self.norb, p, "Id")

        # one-body terms
        for p, q in itertools.product(range(self.norb), repeat=2):
            self._add_one_body(self.one_body_tensor[p, q], p, q)

        # two-body terms
        for p, q, r, s in itertools.product(range(self.norb), repeat=4):
            self._add_two_body(0.5 * two_body_tensor[p, q, r, s], p, q, r, s)

    def _add_one_body(self, coeff: complex, i: int, j: int) -> None:
        dx0 = np.zeros(2)
        if i == j:
            self.add_onsite(coeff, i, "Ntot")
        else:
            self.add_coupling(coeff, i, "Cdu", j, "Cu", dx0)
            self.add_coupling(coeff, i, "Cdd", j, "Cd", dx0)

    def _add_two_body(self, coeff: complex, i: int, j: int, k: int, ell: int) -> None:
        dx0 = np.zeros(2)
        if i == j == k == ell:
            self.add_onsite(2 * coeff, i, "Nu Nd")
        else:
            self.add_multi_coupling(
                coeff,
                [("Cdu", dx0, i), ("Cdu", dx0, k), ("Cu", dx0, ell), ("Cu", dx0, j)],
            )
            self.add_multi_coupling(
                coeff,
                [("Cdu", dx0, i), ("Cdd", dx0, k), ("Cd", dx0, ell), ("Cu", dx0, j)],
            )
            self.add_multi_coupling(
                coeff,
                [("Cdd", dx0, i), ("Cdu", dx0, k), ("Cu", dx0, ell), ("Cd", dx0, j)],
            )
            self.add_multi_coupling(
                coeff,
                [("Cdd", dx0, i), ("Cdd", dx0, k), ("Cd", dx0, ell), ("Cd", dx0, j)],
            )


@pytest.mark.parametrize(
    "cons_N, cons_Sz, norb",
    [
        ('N', 'Sz', 3),
        ('N', 'parity', 3),
        ('N', None, 3),
        ('N', 'Sz', 4),
        ('N', 'parity', 4),
        ('N', None, 4),
        ('parity', 'Sz', 3),
        ('parity', 'parity', 3),
        ('parity', None, 3),
        ('parity', 'Sz', 4),
        ('parity', 'parity', 4),
        ('parity', None, 4),
        (None, 'Sz', 3),
        (None, 'parity', 3),
        (None, None, 3),
        (None, 'Sz', 4),
        (None, 'parity', 4),
        (None, None, 4),
    ],
)
def test_MolecularModel(cons_N, cons_Sz, norb):
    rng = np.random.default_rng()

    # generate a random one-body tensor
    mat = rng.standard_normal((norb, norb)).astype(complex, copy=False)
    mat += 1j * rng.standard_normal((norb, norb)).astype(complex, copy=False)
    one_body_tensor = mat + mat.T.conj()

    # generate a random two-body tensor
    rank = norb * (norb + 1) // 2
    cholesky_vecs = rng.standard_normal((rank, norb, norb)).astype(complex, copy=False)
    cholesky_vecs += cholesky_vecs.transpose((0, 2, 1))
    two_body_tensor = np.einsum("ipr,iqs->prqs", cholesky_vecs, cholesky_vecs)
    orbital_rotation = sp.stats.unitary_group.rvs(norb)
    two_body_tensor = np.einsum(
        "abcd,aA,bB,cC,dD->ABCD",
        two_body_tensor,
        orbital_rotation,
        orbital_rotation.conj(),
        orbital_rotation,
        orbital_rotation.conj(),
        optimize=True,
    )

    # generate a random constant
    constant = rng.standard_normal()

    # check the model sanity
    model_params = {
        'cons_N': cons_N,
        'cons_Sz': cons_Sz,
        'one_body_tensor': one_body_tensor,
        'two_body_tensor': two_body_tensor,
        'constant': constant
    }
    check_general_model(MolecularModel, model_params)

    # check the model terms
    mpo_model = MolecularModel(model_params)
    mpo_model_test = MolecularModelTest(model_params)
    assert mpo_model.H_MPO.is_equal(mpo_model_test.H_MPO)
