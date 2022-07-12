"""Definition of a model: the XXZ chain."""

from tenpy.networks.site import SpinSite
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel


class XXZChain(CouplingModel, NearestNeighborModel, MPOModel):
    def __init__(self, L=2, S=0.5, J=1., Delta=1., hz=0.):
        spin = SpinSite(S=S, conserve="Sz")
        # the lattice defines the geometry
        lattice = Chain(L, spin, bc="open", bc_MPS="finite")
        CouplingModel.__init__(self, lattice)
        # add terms of the Hamiltonian
        self.add_coupling(J * 0.5, 0, "Sp", 0, "Sm", 1)  # Sp_i Sm_{i+1}
        self.add_coupling(J * 0.5, 0, "Sp", 0, "Sm", -1)  # Sp_i Sm_{i-1}
        self.add_coupling(J * Delta, 0, "Sz", 0, "Sz", 1)
        # (for site dependent prefactors, the strength can be an array)
        self.add_onsite(-hz, 0, "Sz")

        # finish initialization
        # generate MPO for DMRG
        MPOModel.__init__(self, lat, self.calc_H_MPO())
        # generate H_bond for TEBD
        NearestNeighborModel.__init__(self, lat, self.calc_H_bond())
