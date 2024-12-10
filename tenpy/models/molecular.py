"""Molecular models."""
# Copyright (C) TeNPy Developers, Apache license

import itertools
import numpy as np

from .model import CouplingMPOModel
from .lattice import Lattice
from ..networks.site import SpinHalfFermionSite

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
        norb : int
            Number of molecular orbitals.
        one_body_tensor : array
            One-body tensor with shape `(norb, norb)`.
        two_body_tensor : array
            Two-body tensor with shape `(norb, norb, norb, norb)`.
        constant : float
            Constant.
    """

    def init_sites(self, params):
        cons_N = params.get("cons_N", "N")
        cons_Sz = params.get("cons_Sz", "Sz")
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site

    def init_lattice(self, params):
        L = params.get("L", 1)
        norb = params.get("norb", 4)
        site = self.init_sites(params)
        basis = np.array(([norb, 0.0], [0, 1]))
        pos = np.array([[i, 0] for i in range(norb)])
        lat = Lattice(
            [L, 1],
            [site] * norb,
            order="default",
            bc="open",
            bc_MPS="finite",
            basis=basis,
            positions=pos,
        )
        return lat

    def init_terms(self, params):
        dx0 = np.array([0, 0])
        norb = params.get("norb", 4)
        one_body_tensor = params.get("one_body_tensor", np.zeros((norb, norb)))
        two_body_tensor = params.get("two_body_tensor", np.zeros((norb, norb, norb, norb)))
        constant = params.get("constant", 0)

        for p, q in itertools.product(range(norb), repeat=2):
            h1 = one_body_tensor[q, p]
            if p == q:
                self.add_onsite(h1, p, "Nu")
                self.add_onsite(h1, p, "Nd")
                self.add_onsite(constant / norb, p, "Id")
            else:
                self.add_coupling(h1, p, "Cdu", q, "Cu", dx0)
                self.add_coupling(h1, p, "Cdd", q, "Cd", dx0)

            for r, s in itertools.product(range(norb), repeat=2):
                h2 = two_body_tensor[q, p, s, r]
                if p == q == r == s:
                    self.add_onsite(0.5 * h2, p, "Nu")
                    self.add_onsite(-0.5 * h2, p, "Nu Nu")
                    self.add_onsite(0.5 * h2, p, "Nu")
                    self.add_onsite(-0.5 * h2, p, "Cdu Cd Cdd Cu")
                    self.add_onsite(0.5 * h2, p, "Nd")
                    self.add_onsite(-0.5 * h2, p, "Cdd Cu Cdu Cd")
                    self.add_onsite(0.5 * h2, p, "Nd")
                    self.add_onsite(-0.5 * h2, p, "Nd Nd")
                else:
                    self.add_multi_coupling(
                        0.5 * h2,
                        [
                            ("Cdu", dx0, p),
                            ("Cdu", dx0, r),
                            ("Cu", dx0, s),
                            ("Cu", dx0, q),
                        ],
                    )
                    self.add_multi_coupling(
                        0.5 * h2,
                        [
                            ("Cdu", dx0, p),
                            ("Cdd", dx0, r),
                            ("Cd", dx0, s),
                            ("Cu", dx0, q),
                        ],
                    )
                    self.add_multi_coupling(
                        0.5 * h2,
                        [
                            ("Cdd", dx0, p),
                            ("Cdu", dx0, r),
                            ("Cu", dx0, s),
                            ("Cd", dx0, q),
                        ],
                    )
                    self.add_multi_coupling(
                        0.5 * h2,
                        [
                            ("Cdd", dx0, p),
                            ("Cdd", dx0, r),
                            ("Cd", dx0, s),
                            ("Cd", dx0, q),
                        ],
                    )
