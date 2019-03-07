#!/usr/bin/python2
import numpy as np
import tenpy.linalg.np_conserved as npc
import tenpy.models.spins
import tenpy.networks.mps as mps
import tenpy.networks.site as site
from tenpy.algorithms import tdvp
from tenpy.networks.mps import MPS

def random_prod_state(L,a_model):
    product_state=[]
    #the numpy mps used to compare
    sz= 2.*np.random.randint(0,2,size=L)-1.0
    for i in range(L):
        if sz[i]>0:
            product_state += ["up"]
        else:
            product_state += ["down"]
    print(product_state)
    psi = MPS.from_product_state(a_model.lat.mps_sites(), product_state, bc=a_model.lat.bc_MPS,form='B')
    return psi

def run_out_of_equilibrium():
    L=10
    chi=20
    delta_t=0.01
    model_params = {
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

    heisenberg=tenpy.models.spins.SpinChain(model_params)
    np.random.seed(0)  # TODO why? This seed gives a state with just 2 down spins, and almost now entanglement. Artificial!!!
    psi=random_prod_state(heisenberg.lat.N_sites,heisenberg)

    tdvp_params = {
        'start_time': 0,
        'dt':delta_t,
        'trunc_params': {
            'chi_max': 50,
            'svd_min': 1.e-10,
            'trunc_cut':None
        }
    }
    tdvp_engine=tdvp.Engine(psi=psi,model=heisenberg,TDVP_params=tdvp_params)
    times=[]
    S_mid=[]
    for i in range(30):
        tdvp_engine.run_two_sites(N_steps=1)
        times.append(tdvp_engine.evolved_time)
        S_mid.append(psi.entanglement_entropy(bonds=[L//2])[0])
    for i in range(30):
        tdvp_engine.run_one_site(N_steps=1)
        times.append(tdvp_engine.evolved_time)
        S_mid.append(psi.entanglement_entropy(bonds=[L//2])[0])
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(times,S_mid)
    plt.xlabel('t')
    plt.ylabel('S')
    plt.axvline(x=0.3,color='red')
    plt.text(0.0,0.0000015,"Two sites update")
    plt.text(0.31,0.0000015,"One site update")
    plt.show()

if __name__ == "__main__":
    #This demonstrates that the two sites update allow the bond dimension, and thus the entanglement, to grow.
    #However this is not true for the one site update
    run_out_of_equilibrium()
