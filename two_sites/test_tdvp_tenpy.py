#!/usr/bin/python2
import numpy as np
import misc
import copy
#import h5py
import pickle
import tenpy.linalg.np_conserved as npc
import tenpy.models.spins
import tenpy.networks.mps as mps
import tenpy.networks.site as site
import tdvp as tdvp
import tenpy.algorithms 
from tenpy.algorithms import tebd
import sys
import tdvp_fast
import tenpy.networks.mpo
import tenpy.models.model as model
import tenpy.models.lattice  
import tebd as tebd_frank
from tenpy.networks.mps import MPS
from tenpy.tools.misc import inverse_permutation
def overlap(mps1, mps2):
    """ Calculate overlap <self|mps2>. Performs conjugation of self! """
    X=np.ones((1,1))
    L=len(mps1)
    for i in range(0, L):
        print(X.shape,np.conj(mps1[i]).shape,'axes=[[0],[1]]')
        tmp = np.tensordot(X, np.conj(mps1[i]), axes=[[0],[1]])
        X = np.tensordot(tmp, mps2[i], axes=[[0,1],[1,0]])
    overlap=X.reshape(())
    return overlap



def random_product_state(L,chi):
    d=2
    sz= 2.*np.random.randint(0,2,size=L)-1.0
    mps=[]
    for i in range(L):
        D1 = np.min([d**np.min([i,L-i]),chi])
        D2 = np.min([d**np.min([i+1,L-i-1]),chi])

        mps.append(np.zeros((2,D1,D2)))
        if sz[i]>0:
            mps[-1][0,0,0]=1.
        else:
            mps[-1][1,0,0]=1.
    return mps,sz

def fixed_product_state(L,chi,seed):
    np.random.seed(seed)
    mps,sz=random_product_state(L,chi)
    return mps,sz
#########################################################
#fixed_product_state with charge conservation


def random_product_state_charge_shalf2(L,chi):
    d=2
    np.random.seed(1)
    sz= 2.*np.random.randint(0,2,size=L)-1.0
    print("sz")
    print(sz)
    return_mps=[]
    chinfo = npc.ChargeInfo([])  # the second argument is just a descriptive name
    p_leg = npc.LegCharge.from_trivial(2)  # charges for up, down
    v_left_old=npc.LegCharge.from_qflat(chinfo,[[]])#arbitrary, set to zero
    if sz[0]>0:
        v_right_old=npc.LegCharge.from_qflat(chinfo,[[]])
        B=npc.zeros([v_left_old,v_right_old.conj(),p_leg])
        B[0,0,1]=1.0 #up
        print(0,"up")
    else:
        v_right_old=npc.LegCharge.from_qflat(chinfo,[[]])
        B=npc.zeros([v_left_old,v_right_old.conj(),p_leg])
        B[0,0,0]=1.0 #down
        print(0,"down")
    B.iset_leg_labels(['vL', 'vR', 'p'])  # virtual left/right, physical
    return_mps.append(B)
    for i in range(1,L):
        if sz[i]>0:
            v_left_new=v_right_old
            new_charge=0
            v_right_new=npc.LegCharge.from_qflat(chinfo,[[]])
            B=npc.zeros([v_left_new,v_right_new.conj(),p_leg])
            B[0,0,1]=1.0 #up
            print(i,"up")
        else:
            v_left_new=v_right_old
            new_charge=0
            v_right_new=npc.LegCharge.from_qflat(chinfo,[[]])
            B=npc.zeros([v_left_new,v_right_new.conj(),p_leg])
            B[0,0,0]=1.0 #down
            print(i,"down")
        B.iset_leg_labels(['vL', 'vR', 'p'])  # virtual left/right, physical
        return_mps.append(B)
        #prepare the next iteration
        v_left_old=v_left_new
        v_right_old=v_right_new
    #define singular values
    legs=B.get_leg_labels()
    Ss = [np.ones(1)]*L
    Sx =  0.5*np.array([[0,1],[1,0]])
    Sz =  0.5*np.array([[1,0],[0,-1]]) 
    one_site=site.Site(p_leg, ['up', 'down'],Sz=Sz)
    lattice=[]
    for i_site in range(0,L):
        lattice.append(one_site)

    full_mps=mps.MPS(lattice,return_mps,Ss,'finite','B')


    return full_mps,sz

def mpo_charge(lattice,hx,hz):
    chinfo = npc.ChargeInfo([1], ['2*Sz'])  # the second argument is just a descriptive name
    p_leg = npc.LegCharge.from_qflat(chinfo, [[1], [-1]])  # charges for up, down
    Sx = 0.5*npc.Array.from_ndarray( [[0., 1.], [1., 0.]],[p_leg, p_leg.conj()])
    Sz = 0.5*npc.Array.from_ndarray([[1., 0.], [0., -1.]],[p_leg, p_leg.conj()])
    Id = npc.eye_like(Sz)  # identity
    for op in [Sz,Sx, Id]:
        op.iset_leg_labels(['p', 'p*'])  # physical in, physical out
    
    mpo_leg = npc.LegCharge.from_qflat(chinfo, [[0], [1], [-1], [0]])
    W_grid=[[Id,None,None,None],
       [Sz,None,None,None],
       [Sx,None,None,None],
       [hz*Sz + hx*Sx,Jz*Sz,Id]]
    W = npc.grid_outer(W_grid, [mpo_leg, mpo_leg.conj()])
    W.iset_leg_labels(['wL', 'wR', 'p', 'p*'])  # wL/wR = virtual left/right of the MPO
    Ws = [W] * L
    H_MPO=tenpy.networks.mpo.MPO(lattice,Ws)
    
    return H_MPO


def convertMps(mps):
    wftensor=mps[0]

    for i in range(1,len(mps)):

        wftensor = np.tensordot(wftensor, mps[i], axes=[[-1],[1]])
    return wftensor.reshape(mps[0].shape[0]**len(mps))

def exact_heisenberg(L):
    d=2
    sx_list=[]
    sy_list=[]
    sz_list = []

    Id=np.eye(2,2)
    sx=np.array([[0.,1.],[1.,0.]])
    sy=np.array([[0.,-1j],[1j,0.]])
    sz=np.array([[1.,0.],[0.,-1.]])

    def OpAverage(B,s,Op,i):
        L=len(B)-1
        lambdaSquare=np.dot(np.conj(np.diag(s[i-1])),np.diag(s[i-1]))
        Bdag=np.conj(np.transpose(B[i],(0,2,1)))
        C=np.tensordot(lambdaSquare,B[i],(1,1))
        C=np.transpose(C,(1,0,2))
        C=np.tensordot(Bdag,C,(2,1))
        C=np.tensordot(Op,C,([0,1],[0,2]))
        return np.trace(C)


    def Entropy(s):
        x=s[s>10**-20]**2
        return -np.inner(np.log(x),x)

    for i_site in range(L):
        if i_site==0:
            X=sx
            Y=sy
            Z=sz
        else:
            X= np.eye(d)
            Y= np.eye(d)
            Z= np.eye(d)

        for j_site in range(1,L):
            if j_site==i_site:
                X=np.kron(X,sx)
                Y=np.kron(Y,sy)
                Z=np.kron(Z,sz)
            else:
                X=np.kron(X,np.eye(d))
                Y=np.kron(Y,np.eye(d))
                Z=np.kron(Z,np.eye(d))

        sx_list.append(X)
        sy_list.append(Y)
        sz_list.append(Z)

    H=np.zeros((2**L,2**L))
    for i in range(L-1):
        H=H+np.dot(sz_list[i],sz_list[i+1])+np.dot(sx_list[i],sx_list[i+1])+np.dot(sy_list[i],sy_list[i+1])
    return H


if __name__ == "__main__":
    L=50
    J=1
    chi=20
    delta_t=0.01
    chinfo = npc.ChargeInfo([])  # the second argument is just a descriptive name
    # create LegCharges on physical leg and even/odd bonds
    p_leg = npc.LegCharge.from_trivial(2)  # charges for up, down
    v_leg_even = npc.LegCharge.from_qflat(chinfo, [[]])
    v_leg_odd = npc.LegCharge.from_qflat(chinfo, [[]])
    #create site and list of sites
    a_site=site.Site(p_leg, ['up', 'down'])
    lattice=[]
    for i_site in range(0,L):
        lattice.append(a_site)
    lat=tenpy.models.lattice.Lattice(Ls=[L], unit_cell=[a_site], order='default', bc_MPS='finite')   
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
        # 'lattice':lat
    }
    heisenberg=tenpy.models.spins.SpinChain(parameters)
    H_MPO=heisenberg.H_MPO
    h_test=[]
    for i_sites in range(H_MPO.L):
        h_test.append(H_MPO.get_W(i_sites).transpose(['wL','wR','p*','p']).to_ndarray())


    def random_prod_state_tenpy(L,a_model):
        product_state=[]
        #the numpy mps used to compare
        psi_compare=[]
        sz= 2.*np.random.randint(0,2,size=L)-1.0
        for i in range(L):
            psi_compare.append(np.zeros((2,1,1)))
            if sz[i]>0:
                product_state += ["up"]
                psi_compare[-1][0,0,0]=1
            else:
                product_state += ["down"]
                psi_compare[-1][1,0,0]=1

        psi = MPS.from_product_state(a_model.lat.mps_sites(), product_state, bc=a_model.lat.bc_MPS,form='B')
        psi_converted=[]
        for i in range(L):
            site=psi.sites[i]
            perm=site.perm
            B_tmp=psi.get_B(i).transpose(['p','vL','vR']).to_ndarray()
            B=B_tmp[inverse_permutation(perm),:,:]
            B=B[::-1,:,:]
            psi_converted.append(B)

        return psi


    np.random.seed(0)
    psi=random_prod_state_tenpy(heisenberg.lat.N_sites,heisenberg)
    N_steps=10 
    tebd_params = {
          'order': 2,
          'dt': delta_t,
          'N_steps': N_steps,
          'trunc_params': {
              'chi_max': 50,
              'svd_min': 1.e-10,
              'trunc_cut':None 
          }
      }
    
    tdvp_params = {
        'start_time': 0,
        'dt':1j*delta_t,
        'N_steps':N_steps
    }
    trunc_params= {
        'chi_max': 50,
        'svd_min': 1.e-10,
        'trunc_cut':None 
    }
    
    print("psi before TEBD")
    print(psi.chi)
    psi_tdvp2=copy.deepcopy(psi)
    engine=tebd.Engine(psi=psi,model=heisenberg,TEBD_params=tebd_params)
    tdvp_engine=tdvp.Engine(psi=psi_tdvp2,model=heisenberg,TDVP_params=tdvp_params,check_error=True,trunc_params=trunc_params)
    engine.run()
    #tdvp_engine.run_two()
    ov=psi.overlap(psi_tdvp2)
    psi=engine.psi
    print("overlap")
    print(ov)
    print("psi after TEBD")
    print(psi.chi)
   
    # test that the initial conditions are the same
     
    tdvp_engine=tdvp.Engine(psi=psi,model=heisenberg,TDVP_params=tdvp_params,check_error=True,trunc_params=trunc_params)
    psit_compare=[]
    for i in range(L):
        B_tmp=psi.get_B(i).transpose(['p','vL','vR']).to_ndarray()
        B=B_tmp[::-1,:,:]
        psit_compare.append(B)
    print(overlap(psit_compare,psit_compare),"overlapp init")
#**********************************************************************************************************
#Initialize TDVP
    tdvp_params = {
        'start_time': 0,
        'dt':1j*delta_t,
        'N_steps':1
    }
    trunc_params= {
        'chi_max': 50,
        'svd_min': 1.e-10,
        'trunc_cut':None 
    }
    tdvp_engine=tdvp.Engine(psi=psi,model=heisenberg,TDVP_params=tdvp_params,check_error=True,trunc_params=trunc_params)
    for t in range(10):
        tdvp_engine.run() 
        psit_compare,Rp_list,spectrum=tdvp_fast.tdvp(psit_compare,h_test,0.5*1j*delta_t, Rp_list=None)
        psit_=[]
    for i in range(L):
        B=psi.get_B(i).transpose(['p','vL','vR']).to_ndarray()
        B=B[::-1,:,:]
        psit_.append(B)
    if np.abs(np.abs(overlap(psit_,psit_compare))-1.0)<1e-13:
        print("test passed")
    else:
        print("Bug TDVP")
        print("overlap=",overlap(psit_,psit_compare))

                
        


