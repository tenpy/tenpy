# Copyright (C) TeNPy Developers, Apache license
import numpy as np
import scipy as sp
import pytest
from tenpy.models.molecular import MolecularModel
from test_model import check_general_model


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

    check_general_model(
        MolecularModel,
        {
            'cons_N': cons_N,
            'cons_Sz': cons_Sz,
            'norb': norb,
            'one_body_tensor': one_body_tensor,
            'two_body_tensor': two_body_tensor,
            'constant': constant
        },
    )
