"""A collection of tests to check the automatic inclusion of Hermitian 
conjugates."""
# Copyright 2018-2020 TeNPy Developers, GNU GPLv3

import itertools as it
import tenpy.linalg.np_conserved as npc
from tenpy.models.tf_ising import TFIChain
from tenpy.models.spins import SpinChain
from tenpy.algorithms import dmrg
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.networks import mps
import pytest
import numpy as np
from scipy import integrate
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.tools.params import get_parameter
from tenpy.networks.site import SpinHalfSite

__all__ = ['TFIModel', 'TFIChain']


class TFIModel_with_hcs(CouplingMPOModel):
    r"""Transverse field Ising model on a general lattice. The model is
    redefined here to be able to set the h.c. flags at will.

    The Hamiltonian reads:

    .. math ::
        H = - \sum_{\langle i,j\rangle, i < j} \mathtt{J} \sigma^x_i \sigma^x_{j}
            - \sum_{i} \mathtt{g} \sigma^z_i

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs, each pair appearing
    exactly once.
    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    Parameters
    ----------
    conserve : None | 'parity'
        What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
    J, g : float | array
        Couplings as defined for the Hamiltonian above.
    lattice : str | :class:`~tenpy.models.lattice.Lattice`
        Instance of a lattice class for the underlaying geometry.
        Alternatively a string being the name of one of the Lattices defined in
        :mod:`~tenpy.models.lattice`, e.g. ``"Chain", "Square", "HoneyComb", ...``.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions along the x-direction.
        For 'infinite' boundary conditions, repeat the unit cell in x-direction.
        Coupling boundary conditions in x-direction are chosen accordingly.
        Only used if `lattice` is a string.
    order : string
        Ordering of the sites in the MPS, e.g. 'default', 'snake';
        see :meth:`~tenpy.models.lattice.Lattice.ordering`.
        Only used if `lattice` is a string.
    L : int
        Lenght of the lattice.
        Only used if `lattice` is the name of a 1D Lattice.
    Lx, Ly : int
        Length of the lattice in x- and y-direction.
        Only used if `lattice` is the name of a 2D Lattice.
    bc_y : 'ladder' | 'cylinder'
        Boundary conditions in y-direction.
        Only used if `lattice` is the name of a 2D Lattice.
    """
    def __init__(self, model_params):
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        conserve = get_parameter(model_params, 'conserve', 'parity', self.name)
        assert conserve != 'Sz'
        if conserve == 'best':
            conserve = 'parity'
            if self.verbose >= 1.:
                print(self.name + ": set conserve to", conserve)
        site = SpinHalfSite(conserve=conserve)
        return site

    def init_terms(self, model_params):
        J = get_parameter(model_params, 'J', 1., self.name, True)
        g = get_parameter(model_params, 'g', 1., self.name, True)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g, u, 'Sigmaz')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'Sigmax', u2, 'Sigmax', dx, add_hc=True)


def e0_tranverse_ising(g=0.5):
    """Exact groundstate energy of transverse field Ising.

    H = - J sigma_z sigma_z + g sigma_x
    Can be obtained by mapping to free fermions.
    """
    return integrate.quad(_f_tfi, 0, np.pi, args=(g, ))[0]


def _f_tfi(k, g):
    return -2 * np.sqrt(1 + g**2 - 2 * g * np.cos(k)) / np.pi / 2.


params = [
    ('finite', True, True, 1),  # 1-site DMRG with mixer=False is expected to fail.
    ('finite', True, True, 2),
    ('finite', True, False, 2),
    ('finite', False, True, 1),
    ('finite', False, True, 2),
    ('finite', False, False, 2),
    ('infinite', True, True, 1),
    ('infinite', True, True, 2),
    ('infinite', True, False, 2),
    ('infinite', False, True, 1),
    ('infinite', False, True, 2),
    ('infinite', False, False, 2)
]


@pytest.mark.parametrize("bc_MPS, combine, mixer, n", params)
@pytest.mark.slow
def test_dmrg_add_hcs(bc_MPS, combine, mixer, n, g=1.2):
    L = 2 if bc_MPS == 'infinite' else 8
    model_params = dict(L=L, J=1., g=g, bc_MPS=bc_MPS, conserve=None, verbose=0,
                        add_hcs_to_MPO=True)
    M = TFIModel_with_hcs(model_params)
    state = [0] * L  # Ferromagnetic Ising
    psi = mps.MPS.from_product_state(M.lat.mps_sites(), state, bc=bc_MPS)
    dmrg_pars = {
        'verbose': 5,
        'combine': combine,
        'mixer': mixer,
        'chi_list': {
            0: 10,
            5: 30
        },
        'max_E_err': 1.e-12,
        'max_S_err': 1.e-8,
        'N_sweeps_check': 4,
        'mixer_params': {
            'disable_after': 15,
            'amplitude': 1.e-5
        },
        'trunc_params': {
            'svd_min': 1.e-10,
        },
        'max_N_for_ED': 20,  # small enough that we test both diag_method=lanczos and ED_block!
        'max_sweeps': 40,
        'active_sites': n,
    }
    if not mixer:
        del dmrg_pars['mixer_params']  # avoid warning of unused parameter
    if bc_MPS == 'infinite':
        # if mixer is not None:
        #     dmrg_pars['mixer_params']['amplitude'] = 1.e-12  # don't actually contribute...
        dmrg_pars['start_env'] = 1
    res = dmrg.run(psi, M, dmrg_pars)
    if bc_MPS == 'finite':
        ED = ExactDiag(M)
        ED.build_full_H_from_mpo()
        ED.full_diagonalization()
        E_ED, psi_ED = ED.groundstate()
        ov = npc.inner(psi_ED, ED.mps_to_full(psi), 'range', do_conj=True)
        print("E_DMRG={Edmrg:.14f} vs E_exact={Eex:.14f}".format(Edmrg=res['E'], Eex=E_ED))
        print("compare with ED: overlap = ", abs(ov)**2)
        assert abs(abs(ov) - 1.) < 1.e-10  # unique groundstate: finite size gap!
    else:
        # compare exact solution for transverse field Ising model
        Edmrg = res['E']
        Eexact = e0_tranverse_ising(g)
        print("E_DMRG={Edmrg:.12f} vs E_exact={Eex:.12f}".format(Edmrg=Edmrg, Eex=Eexact))
        print("relative energy error: {err:.2e}".format(err=abs((Edmrg - Eexact) / Eexact)))
        print("norm err:", psi.norm_test())
        # Edmrg2 = np.mean(psi.expectation_value(M.H_bond))
        # Edmrg3 = M.H_MPO.expectation_value(psi)
        assert abs((Edmrg - Eexact) / Eexact) < 1.e-10
        # assert abs((Edmrg - Edmrg2) / Edmrg2) < max(1.e-10, np.max(psi.norm_test()))
        # assert abs((Edmrg - Edmrg3) / Edmrg3) < max(1.e-10, np.max(psi.norm_test()))