#!/usr/bin/python2
import numpy as np
import copy
import pickle
import tenpy.linalg.np_conserved as npc
import tenpy.models.spins
import tenpy.networks.mps as mps
import tenpy.networks.site as site
from tenpy.algorithms import tdvp
from tenpy.algorithms import tebd
import sys
import tenpy.networks.mpo
import tenpy.models.model as model
import tenpy.models.lattice  
from tenpy.networks.mps import MPS
import matplotlib.pyplot as plt
def random_prod_state(L,a_model):

    product_state=[]
    #the numpy mps used to compare
    psi_compare=[]
    sz= 2.*np.random.randint(0,2,size=L)-1.0
    for i in range(L):
        psi_compare.append(np.zeros((2,1,1)))
        if sz[i]>0:
            product_state += ["up"]
        else:
            product_state += ["down"]
    psi = MPS.from_product_state(a_model.lat.mps_sites(), product_state, bc=a_model.lat.bc_MPS,form='B')
    return psi
def run_out_of_equilibrium():
    L=10
    J=1
    chi=20
    delta_t=0.01
    chinfo = npc.ChargeInfo([])  # the second argument is just a descriptive name
    parameters= {
        'L':L,
        'S':0.5,
        'conserve':'Sz',
        'Jz':1.0,
        'Jy':1.0,
        'Jx':1.0,
        'hx':0.0,
        'hy':0.0,
        'hz':0.0,
        'muJ':0.0,
        'bc_MPS':'finite',
    }

    heisenberg=tenpy.models.spins.SpinChain(parameters)
    H_MPO=heisenberg.H_MPO
    np.random.seed(0)
    psi=random_prod_state(heisenberg.lat.N_sites,heisenberg)
    N_steps=1 

    tdvp_params = {
        'start_time': 0,
        'dt':delta_t,
    }
    trunc_params= {
        'chi_max': 50,
        'svd_min': 1.e-10,
        'trunc_cut':None 
    }
    tdvp_engine=tdvp.Engine(psi=psi,model=heisenberg,TDVP_params=tdvp_params,trunc_params=trunc_params)
    times=[]
    S_mid=[]
    for i in range(30):
        tdvp_engine.run_two_sites(N_steps=1)
        times.append(tdvp_engine.get_evolved_time())
        s_values=psi.get_SR(int(L/2))
        S_mid.append(np.sum(-2*s_values**2*np.log(s_values)))
    for i in range(30):
        tdvp_engine.run_one_site(N_steps=1) 
        times.append(tdvp_engine.get_evolved_time())
        s_values=psi.get_SR(int(L/2))
        S_mid.append(np.sum(-2*s_values**2*np.log(s_values)))
    plt.figure()
    plt.plot(times,S_mid)
    plt.xlabel('t')
    plt.ylabel('S')
    plt.axvline(x=0.3,color='red')
    plt.text(0.0,0.0000015,"One site update")
    plt.text(0.31,0.0000015,'Two sites update')
    plt.show()
if __name__ == "__main__":
    #This demonstrates that the two sites update allow the bond dimension, and thus the entanglement, to grow.
    #However this is not true for the one site update
    run_out_of_equilibrium()

                
        


