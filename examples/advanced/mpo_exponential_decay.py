"""Demonstration of the mpo.MPO.from_grids method.

We construct a MPO model for a spin 1/2 Heisenberg chain with an infinite number of
2-sites interactions, with strength that decays exponentially with the distance between the sites.

Because of the infinite number of couplings it is not possible to construct the MPO from a
coupling model. However the tensors that form the MPO have a surprisingly simple
form (see the grid below).

We run the iDMRG algorithm to find the ground state and energy density of the
system in the thermodynamic limit.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import tenpy.linalg.np_conserved as npc

from tenpy.networks.mpo import MPO
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy.models.model import MPOModel
from tenpy.models.lattice import Chain
from tenpy.algorithms import dmrg
from tenpy.tools.params import asConfig


class ExponentiallyDecayingHeisenberg(MPOModel):
    r"""Spin-1/2 Heisenberg Chain with exponentially decaying interactions.

    The Hamiltonian reads:

    .. math ::

        H = \sum_i \sum_{j>i} \exp(-\frac{|j-i-1|}{\mathtt{xi}}) (
                  \mathtt{Jxx}/2 (S^{+}_i S^{-}_j + S^{-}_i S^{+}_j)
                + \mathtt{Jz} S^z_i S^z_j                        ) \\
            - \sum_i \mathtt{hz} S^z_i

    All parameters are collected in a single dictionary `model_params`.

    Parameters
    ----------
    L : int
        Length of the chain.
    Jxx, Jz, hz, xi: float
        Coupling parameters as defined for the Hamiltonian above.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions.
    conserve : 'Sz' | 'parity' | None
        What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
    """
    def __init__(self, model_params):
        # model parameters
        model_params = asConfig(model_params, "ExponentiallyDecayingHeisenberg")
        L = model_params.get('L', 2)
        xi = model_params.get('xi', 0.5)
        Jxx = model_params.get('Jxx', 1.)
        Jz = model_params.get('Jz', 1.5)
        hz = model_params.get('hz', 0.)
        conserve = model_params.get('conserve', 'Sz')
        if xi == 0.:
            g = 0.
        elif xi == np.inf:
            g = 1.
        else:
            g = np.exp(-1 / (xi))

        # Define the sites and the lattice, which in this case is a simple uniform chain
        # of spin 1/2 sites
        site = SpinHalfSite(conserve=conserve)
        lat = Chain(L, site, bc_MPS='infinite', bc='periodic')

        # The operators that appear in the Hamiltonian. Standard spin operators are
        # already defined for the spin 1/2 site, but it is also possible to add new
        # operators using the add_op method
        Sz, Sp, Sm, Id = site.Sz, site.Sp, site.Sm, site.Id

        # yapf:disable
        # The grid (list of lists) that defines the MPO. It is possible to define the
        # operators in the grid in the following ways:
        # 1) NPC arrays, defined above:
        grid = [[Id,   Sp,   Sm,   Sz,   -hz*Sz    ],
                [None, g*Id, None, None, 0.5*Jxx*Sm],
                [None, None, g*Id, None, 0.5*Jxx*Sp],
                [None, None, None, g*Id, Jz*Sz     ],
                [None, None, None, None, Id        ]]
        # 2) In the form [("OpName", strength)], where "OpName" is the name of the
        # operator (e.g. "Sm" for Sm) and "strength" is a number that multiplies it.
        grid = [[[("Id", 1)], [("Sp",1)], [("Sm",1)], [("Sz",1)], [("Sz", -hz)]    ],
                [None       , [("Id",g)], None      , None      , [("Sm", 0.5*Jxx)]],
                [None       , None      , [("Id",g)], None      , [("Sp", 0.5*Jxx)]],
                [None       , None      , None      , [("Id",g)], [("Sz",Jz)]      ],
                [None       , None      , None      , None      , [("Id",1)]       ]]
        # 3) It is also possible to write a single "OpName", equivalent to
        # [("OpName", 1)].
        grid = [["Id"       , "Sp"      , "Sm"      , "Sz"      , [("Sz", -hz)]    ],
                [None       , [("Id",g)], None      , None      , [("Sm", 0.5*Jxx)]],
                [None       , None      , [("Id",g)], None      , [("Sp", 0.5*Jxx)]],
                [None       , None      , None      , [("Id",g)], [("Sz",Jz)]      ],
                [None       , None      , None      , None      , "Id"             ]]
        # yapf:enable
        grids = [grid] * L

        # Generate the MPO from the grid. Note that it is not necessary to specify
        # the physical legs and their charges, since the from_grids method can extract
        # this information from the position of the operators inside the grid.
        H = MPO.from_grids(lat.mps_sites(), grids, bc='infinite', IdL=0, IdR=-1)
        MPOModel.__init__(self, lat, H)


def example_run_dmrg():
    """Use iDMRG to extract information about the ground state of the system."""
    model_params = dict(L=2, Jxx=1, Jz=1.5, xi=0.8, verbose=1)
    model = ExponentiallyDecayingHeisenberg(model_params)
    psi = MPS.from_product_state(model.lat.mps_sites(), ["up", "down"], bc='infinite')
    dmrg_params = {
        'mixer': True,
        'chi_list': {
            0: 100
        },
        'trunc_params': {
            'svd_min': 1.e-10
        },
        'verbose': 1
    }
    results = dmrg.run(psi, model, dmrg_params)
    print("Energy per site: ", results['E'])
    print("<Sz>: ", psi.expectation_value('Sz'))


if __name__ == "__main__":
    example_run_dmrg()
