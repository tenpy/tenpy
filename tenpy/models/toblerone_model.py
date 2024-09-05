"""Model of spin chains with three coupled chains on antiferromagnetically coupled sites on a triangle

.. todo ::
    test and validate code
"""
# Copyright 2018-2023 TeNPy Developers, GNU GPLv3

from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import NLegLadder
from tenpy.networks.site import SpinHalfSite
import numpy as np

__all__ = ['TobleroneModel']

class TobleroneModel(CouplingMPOModel):
    r"""Spin-half particles in a Toblerone (layered 3-site triangular lattice).
    
    The Hamiltonian is the Ising model:
        
    .. math ::
        H = - \sum_{\langle i,j \rangle} J_{i,j} S_i^z S_j^z
            - h \sum_j \S_j^z
            
          = - \sum_x ( J1 S_(x,0)^z S_(x,1)^z + J1 S_(x,1)^z S_(x,2)^z + J2 S_(x,2)^z S_(x,0)^z )
            - hx \sum_x (S_(x,0)^x + S_(x,2)^x)
            - hxt \sum_(x) S_(x,1)^x
            - \sum_x ( Jp (S_(x,0)^z S_(x+1,0)^z + S_(x,2)^z S_(x+1,2)^z ) +  Jpt S_(x,1)^z S_(x+1,1)^z)
            
    where the prior term is the nearest neighbour interaction and the latter is
    the on-site interaction with an external magnetic field.
    
    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`TobleroneModel` below.
        
    Options
    -------
    .. cfg.config :: TobleroneModel
        :include: CouplingMPOModel
        
        J1, J2, Jp, Jpt : float
            Strength of the interactions between spins.
        hx : float
            Strength of the externally applied magnetic field for ladder spins (z component).
        hxt : float
            Strength of the externally applied magnetic field for top spins (z component).
        S : array
            The spin matrices, e.g. S^z = [[0,1],[1,0]].
        L : int
            Length of the ladder (lattice).
    """
    defaults = {
        "bc_MPS": "infinite",
    }
    
    def init_terms(self, model_params):
        #Interaction strengths.
        J1 = model_params.get('J1', 1.) #"." forces float instead of int.
        J2 = model_params.get('J2', 2.)
        Jp = model_params.get('Jp', 1.)
        Jpt = model_params.get('Jpt', 0.)
        hx = model_params.get('hx', 1.)
        hxt = model_params.get('hxt', 1.)
        
        #External magnetic field interaction.

        self.add_onsite(hx, 0, 'Sx')
        self.add_onsite(hxt, 1, 'Sx')
        self.add_onsite(hx, 2, 'Sx')
            
        #Coupling between lattice sites in the same unit cell.
        dx = 0 ##sort array
        self.add_coupling(J1, 0, 'Sz', 1, 'Sz', dx)
        self.add_coupling(J1, 1, 'Sz', 2, 'Sz', dx)
        self.add_coupling(J2, 0, 'Sz', 2, 'Sz', dx)
        
        #Coupling between lattice sites in adjacent unit cells.
        dx = 1
        
        self.add_coupling(Jp, 0, 'Sz', 0, 'Sz', dx)
        self.add_coupling(Jpt, 1, 'Sz', 1, 'Sz', dx)
        self.add_coupling(Jp, 2, 'Sz', 2, 'Sz', dx)
        
    def init_sites(self, model_params):
        spin = SpinHalfSite(conserve=None) #Defines the type of site (spin-half).
        return spin
    
    def init_lattice(self, model_params):
        bc_MPS = model_params.get('bc_MPS', 'finite')
        sites = self.init_sites(model_params)
        L = model_params.get('L', 1) #Length of the ladder
        lat = NLegLadder(L, 3, sites) #Define order and bc_MPS?:
        #lat = NLegLadder(L, 3, sites, order, bcMPS)
        return lat
