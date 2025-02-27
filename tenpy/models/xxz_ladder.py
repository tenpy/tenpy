"""Model of spin chains with three coupled chains on antiferromagnetically coupled sites on a triangle

.. todo ::
    test and validate code
"""
# Copyright 2018-2023 TeNPy Developers, GNU GPLv3

from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Ladder
from tenpy.networks.site import SpinHalfSite
import numpy as np

__all__ = ['XXZLadderModel']

class XXZLadderModel(CouplingMPOModel):
    r"""Spin-half particles in on a ladder with XXZ Hamiltonian along each of the coupled chains, and XXZ or Heisenberg coupling along each rung.
    
    The Hamiltonian is as follows:
        
    .. math ::
        H =  J \sum_{i,\alpha=\{A,B\}} \frac{1}{2}(S_{\alpha,i}^+ S_{\alpha,i+1}^- + S_{\alpha,i}^- S_{\alpha,i+1}^+ \Delta S_{\alpha,i}^z S_{\alpha,i+1}^z
            
            + Jp \sum_i \frac{1}{2}(S_{A,i}^+ S_{B,i+1}^- + S_{A,i}^- S_{\B,i+1}^+ \Delta_p S_{A,i}^z S_{B,i+1}^z
            + \sum_i \sum_{\gamma=1}^3 h_\gamma (S_{A,i}^\gamma + S_{B,i}^\gamma)
                
    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`XXZLadderModel` below.
        
    Options
    -------
    .. cfg.config :: XXZLadderModel
        :include: CouplingMPOModel
        
        J : float
            Strength of coupling along chains
        Delta : float
            Coupling anisotropy along each of the chains (Delta=1 yields the Heisenberg model along chains)
        Jp : float
            Strength of Heisenberg coupling between chains
        Deltap : float
            Coupling anisotropy along rungs of the latter (Deltap=1, yields Heisenberg coupling)
        hx, hy, hz: float
            Magnetic field components along each axis
        L : int
            Length of the ladder (lattice).
        
    """
    defaults = {
        "bc_MPS": "infinite",
    }
    
    def init_terms(self, model_params):
        #Interaction strengths.
        J = model_params.get('J', 1., float) #"." forces float instead of int.
        Jp = model_params.get('Jp', 0.5, float)
        Delta = model_params.get('Delta', 1.0, float)
        Deltap = model_params.get('Deltap', 1.0, float)
        hx = model_params.get('hx', 0.0, float)
        hy = model_params.get('hy', 0.0, float)
        hz = model_params.get('hz', 0.0, float)
        
        # Onsite terms: External magnetic field interaction.
        for alpha in range(2):
            self.add_onsite(hx, alpha, 'Sx')
            self.add_onsite(hy, alpha, 'Sy')
            self.add_onsite(hz, alpha, 'Sz')
            
        # XXZ - coupling along rungs
        dx = 0
        self.add_coupling(Jp, 0, 'Sp', 1, 'Sm', dx)
        self.add_coupling(Jp, 0, 'Sm', 1, 'Sp', dx)
        self.add_coupling(Deltap*Jp, 0, 'Sz', 1, 'Sz', dx)
        
        # XXZ - coupling between sites in adjacent unit cells along each chain.
        dx = 1
        for alpha in range(2):
            self.add_coupling(J, alpha, 'Sp', alpha, 'Sm', dx)
            self.add_coupling(J, alpha, 'Sm', alpha, 'Sp', dx)
            self.add_coupling(Delta*J, alpha, 'Sz', alpha, 'Sz', dx)
        
    def init_sites(self, model_params):
        spin = SpinHalfSite(conserve=None) #Defines the type of site (spin-half).
        return spin
    
    def init_lattice(self, model_params):
        bc_MPS = model_params.get('bc_MPS', 'finite')
        sites = self.init_sites(model_params)
        L = model_params.get('L', 1) #Length of the ladder
        lat = Ladder(L, sites) #Define order and bc_MPS?:
        return lat
