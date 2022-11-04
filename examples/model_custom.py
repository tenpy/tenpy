"""This file ``/examples/model_custom.py`` illustrates customization code for simulations.

Put this file somewhere where it can be importet by python, i.e. either in the working directory
or somewhere in the ``PYTHON_PATH``. Then you can use it with the parameters file
``/examples/model_custom.yml`` from the terminal like this::

    tenpy-run -i model_custom simulation_custom.yml
"""

import tenpy
from tenpy.networks.site import SpinSite
from tenpy.networks.mps import TransferMatrix
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.models.lattice import Chain
from tenpy.linalg import np_conserved as npc
import numpy as np


class AnisotropicSpin1Chain(CouplingMPOModel, NearestNeighborModel):
    r"""An example for a custom model, implementing the Hamiltonian of :arxiv:`1204.0704`.

    .. math ::
        H = J \sum_i \vec{S}_i \cdot \vec{S}_{i+1} + B \sum_i S^x_i + D \sum_i (S^z_i)^2
    """
    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params):
        B = model_params.get('B', 0.)
        conserve = model_params.get('conserve', 'best')
        if conserve == 'best':
            conserve = 'Sz' if not model_params.any_nonzero(['B']) else None
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        sort_charge = model_params.get('sort_charge', True)
        return SpinSite(S=1., conserve=None, sort_charge=sort_charge)

    def init_terms(self, model_params):
        J = model_params.get('J', 1.)
        B = model_params.get('B', 0.)
        D = model_params.get('D', 0.)

        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(J / 2., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(J, u1, 'Sz', u2, 'Sz', dx)

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(B, u, 'Sx')
            self.add_onsite(D, u, 'Sz Sz')


def m_pollmann_turner_inversion(results, psi, model, simulation, tol=0.01):
    """Measurement function for equation 15 of :arxiv:`1204.0704`.

    See :func:`~tenpy.simulations.measurement.measurement_index` for the call structure.
    """
    psi2 = psi.copy()
    psi2.spatial_inversion()
    mixed_TM = TransferMatrix(psi, psi2, transpose=False, charge_sector=0, form='B')
    evals, evecs = mixed_TM.eigenvectors(which='LM', num_ev=1)
    results['pollmann_turner_inversion_eta'] = eta = evals[0]
    U_I = evecs[0].split_legs().transpose().conj()
    U_I *= np.sqrt(U_I.shape[0])  # previously normalized to npc.norm(U_I) = 1, need unitary
    results['pollmann_turner_inversion_U_I'] = U_I
    if abs(eta) < 1. - tol:
        O_I = 0.
    else:
        O_I = npc.inner(U_I, U_I.conj(), axes=[[0, 1], [1, 0]]) / U_I.shape[0]
    results['pollmann_turner_inversion'] = O_I
