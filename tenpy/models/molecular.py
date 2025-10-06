"""Molecular models."""
# Copyright (C) TeNPy Developers, Apache license

import itertools
import numpy as np

from .model import CouplingMPOModel
from .lattice import Lattice
from ..networks.site import SpinHalfFermionSite
from ..tools.params import Config

__all__ = ['MolecularModel']


class MolecularModel(CouplingMPOModel):
    r"""Spin-1/2 fermion molecular Hamiltonian.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\sigma, ij} h_{ij} c^{\dagger}_{\sigma, i} c_{\sigma, j}
            + \frac{1}{2} \sum_{\sigma\tau, ijkl} h_{ijkl} c^{\dagger}_{\sigma, i}
            c^{\dagger}_{\tau, k} c_{\tau, l} c_{\sigma, j}
            + \mathrm{constant}

    Here :math:`h_{ij}` is called the one-body tensor and :math:`h_{ijkl}` is called the
    two-body tensor. All parameters are collected in a single dictionary `model_params`,
    which is turned into a :class:`~tenpy.tools.params.Config` object.

    .. note::
        Since molecules do not have a lattice structure, the molecular orbitals are
        mapped to sites in a unit cell of a :class:`~tenpy.models.lattice.Lattice` of
        unit length.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`MolecularModel` below.

    Options
    -------
    .. cfg:config :: MolecularModel
        :include: CouplingMPOModel

        cons_N : {'N' | 'parity' | None}
            Whether particle number is conserved,
            see :class:`~tenpy.networks.site.SpinHalfFermionSite` for details.
        cons_Sz : {'Sz' | 'parity' | None}
            Whether spin is conserved,
            see :class:`~tenpy.networks.site.SpinHalfFermionSite` for details.
        one_body_tensor : array
            One-body tensor with shape `(norb, norb)`.
        two_body_tensor : array
            Two-body tensor with shape `(norb, norb, norb, norb)`.
        constant : float
            Constant.
    """

    def __init__(self, params: dict) -> None:
        if "one_body_tensor" in params and isinstance(params["one_body_tensor"], np.ndarray):
            self.one_body_tensor = params["one_body_tensor"]
        else:
            raise ValueError("required parameter one_body_tensor is undefined or not an array")
        self.norb = self.one_body_tensor.shape[0]
        CouplingMPOModel.__init__(self, params)

    def init_sites(self, params: Config) -> SpinHalfFermionSite:
        """Initialize sites."""
        cons_N = params.get("cons_N", "N")
        cons_Sz = params.get("cons_Sz", "Sz")
        return SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)

    def init_lattice(self, params: Config) -> Lattice:
        """Initialize lattice."""
        site = self.init_sites(params)
        basis = np.array(([self.norb, 0], [0, 1]))
        pos = np.array([[i, 0] for i in range(self.norb)])
        lat = Lattice(
            [1, 1],
            [site] * self.norb,
            basis=basis,
            positions=pos,
        )
        return lat

    def init_terms(self, params: Config) -> None:
        """Initialize terms."""
        params.touch("one_body_tensor")  # suppress unused key warning
        two_body_tensor = params.get(
            "two_body_tensor",
            np.zeros((self.norb, self.norb, self.norb, self.norb)),
            expect_type="array",
        )
        constant = params.get("constant", 0, expect_type="real")

        for p in range(self.norb):
            # one-body tensor
            h1 = self.one_body_tensor[p, p]
            self.add_onsite(h1, p, "Ntot")
            # two-body tensor
            h2 = two_body_tensor[p, p, p, p]
            self.add_onsite(h2, p, "Ntot")
            self.add_onsite(-0.5 * h2, p, "Nu Nu")
            self.add_onsite(-0.5 * h2, p, "Cdu Cd Cdd Cu")
            self.add_onsite(-0.5 * h2, p, "Cdd Cu Cdu Cd")
            self.add_onsite(-0.5 * h2, p, "Nd Nd")
            # constant
            self.add_onsite(constant / self.norb, p, "Id")

        for p, q in itertools.combinations(range(self.norb), 2):
            # one-body tensor
            h1 = self.one_body_tensor[p, q]
            self._add_one_body(h1, p, q, flag_hc=True)
            # two-body tensor
            indices = [(p, p, q, q), (p, q, p, q), (p, q, q, p)]
            for i, j, k, ell in indices:
                h2 = two_body_tensor[i, j, k, ell]
                self._add_two_body(0.5 * h2, i, j, k, ell, flag_hc=True)

        for p, s in itertools.combinations_with_replacement(range(self.norb), 2):
            for q, r in itertools.combinations_with_replacement(range(self.norb), 2):
                values, counts = np.unique([p, q, r, s], return_counts=True)
                if not (len(values) in [1, 2] and len(set(counts)) == 1):
                    # two-body tensor
                    indices = [(p, q, r, s)]
                    if p != s:
                        indices.append((s, q, r, p))  # swap p and s
                    if q != r:
                        indices.append((p, r, q, s))  # swap q and r
                    for idx, (i, j, k, ell) in enumerate(indices):
                        # reverse p, q, r, s by adding hermitian conjugate
                        flag_hc = True if not idx and i != ell and j != k else False
                        h2 = two_body_tensor[i, j, k, ell]
                        self._add_two_body(0.5 * h2, i, j, k, ell, flag_hc=flag_hc)

    def _add_one_body(self, coeff: complex, i: int, j: int, flag_hc: bool = False) -> None:
        dx0 = np.zeros(2)
        self.add_coupling(coeff, i, "Cdu", j, "Cu", dx0, plus_hc=flag_hc)
        self.add_coupling(coeff, i, "Cdd", j, "Cd", dx0, plus_hc=flag_hc)

    def _add_two_body(self,
                      coeff: complex,
                      i: int,
                      j: int,
                      k: int,
                      ell: int,
                      flag_hc: bool = False) -> None:
        dx0 = np.zeros(2)
        self.add_multi_coupling(
            coeff,
            [("Cdu", dx0, i), ("Cdu", dx0, k), ("Cu", dx0, ell), ("Cu", dx0, j)],
            plus_hc=flag_hc,
        )
        self.add_multi_coupling(
            coeff,
            [("Cdu", dx0, i), ("Cdd", dx0, k), ("Cd", dx0, ell), ("Cu", dx0, j)],
            plus_hc=flag_hc,
        )
        self.add_multi_coupling(
            coeff,
            [("Cdd", dx0, i), ("Cdu", dx0, k), ("Cu", dx0, ell), ("Cd", dx0, j)],
            plus_hc=flag_hc,
        )
        self.add_multi_coupling(
            coeff,
            [("Cdd", dx0, i), ("Cdd", dx0, k), ("Cd", dx0, ell), ("Cd", dx0, j)],
            plus_hc=flag_hc,
        )
