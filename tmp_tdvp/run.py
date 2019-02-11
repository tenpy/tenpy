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
import tdvp_charge as tdvp
import tenpy.algorithms 
from tenpy.algorithms import tebd
import sys
import tdvp_fast
import tenpy.networks.mpo
import tenpy.models.model as model
import tenpy.models.lattice  
sys.path.append('../../tdvp/code/tdvp_otoc/')
# for testing
import tebd as tebd_frank
def overlap(mps1, mps2):
    """ Calculate overlap <self|mps2>. Performs conjugation of self! """
    X=np.ones((1,1))
    L=len(mps1)
    for i in range(0, L):
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
    sz= 2.*np.random.randint(0,2,size=L)-1.0
    print("sz")
    print(sz)
    return_mps=[]
    chinfo = npc.ChargeInfo([1], ['2*Sz'])  # the second argument is just a descriptive name
    p_leg = npc.LegCharge.from_qflat(chinfo, [[-1], [1]])  # charges for up, down
    v_left_old=npc.LegCharge.from_qflat(chinfo,[[0]])#arbitrary, set to zero
    if sz[0]>0:
        v_right_old=npc.LegCharge.from_qflat(chinfo,[[1]])
        B=npc.zeros([v_left_old,v_right_old.conj(),p_leg])
        B[0,0,1]=1.0 #up
        print(0,"up")
    else:
        v_right_old=npc.LegCharge.from_qflat(chinfo,[[-1]])
        B=npc.zeros([v_left_old,v_right_old.conj(),p_leg])
        B[0,0,0]=1.0 #down
        print(0,"down")
    B.iset_leg_labels(['vL', 'vR', 'p'])  # virtual left/right, physical
    return_mps.append(B)
    for i in range(1,L):
        if sz[i]>0:
            v_left_new=v_right_old
            new_charge=1+v_right_old.to_qflat()[0,0]
            v_right_new=npc.LegCharge.from_qflat(chinfo,[[new_charge]]+[[0]]*(chi-1))
            B=npc.zeros([v_left_new,v_right_new.conj(),p_leg])
            B[0,0,1]=1.0 #up
            print(i,"up")
        else:
            v_left_new=v_right_old
            new_charge=-1+v_right_old.to_qflat()[0,0]
            v_right_new=npc.LegCharge.from_qflat(chinfo,[[new_charge]])
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
    one_site=site.Site(p_leg, ['up', 'down'],Smixed=(Sz+Sx)*(Sz+Sx))
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
    #W[0,0] += Id    
    #W[0,1] += Sz
    #W[0,2] += Sx 
    #W[0,3] += hz*Sz + hx*Sx
            
    #W[1,3] += Jz*Sz 
    #W[2,3] += Jx*Sx 
    #W[3,3] += Id



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


def initiate_TEBD(heisenberg):

    #create a nearest neighbour model in order to use TEBD
    #First get the H_bond
    hbond=heisenberg.calc_H_bond()
    #initialize the lattice
    lat=tenpy.models.lattice.Lattice(Ls=[L], unit_cell=[a_site], order='default', bc_MPS='finite', basis=None, positions=None)
    #initialize the model
    nnm=model.NearestNeighborModel(lat=lat,H_bond=hbond)

    delta_t=times[1]-times[0]
    tebd_params = {
          'order': 2,
          'dt': delta_t,
          'N_steps': 5000,
          'trunc_params': {
              'chi_max': 50,
              'svd_min': 1.e-10,
              'trunc_cut': None
          }
      }
    engine=tebd.Engine(psi=psi0,model=nnm,TEBD_params=tebd_params)
    
if __name__ == "__main__":
    L=3
    J=1
    chi=2
    chinfo = npc.ChargeInfo([1], ['2*Sz'])  # the second argument is just a descriptive name
    # create LegCharges on physical leg and even/odd bonds
    p_leg = npc.LegCharge.from_qflat(chinfo, [[1], [-1]])  # charges for up, down
    v_leg_even = npc.LegCharge.from_qflat(chinfo, [[0]])
    v_leg_odd = npc.LegCharge.from_qflat(chinfo, [[1]])
    #create site and list of sites
    a_site=site.Site(p_leg, ['up', 'down'])
    lattice=[]
    for i_site in range(0,L):
        lattice.append(a_site)
    parameters= {
        'L':L,
        'S':0.5,
        'conserve':'Sz',
        'Jz':1.0,
        'Jy':0.0,
        'Jx':0.0,
        'hx':0.0,
        'hy':0.0,
        'hz':0.0,
        'muJ':0.0,
        'bc_MPS':'finite',
    }
    heisenberg=tenpy.models.spins.SpinChain(parameters)

    H_MPO=heisenberg.calc_H_MPO()
   
    times0 = np.linspace(0.0,30,301)
    times=times0[1:len(times0)]
    times2=times0[2:len(times0)]
    
    #get a random product state
    psi0,sz0=random_product_state_charge_shalf2(L,chi)
    #convert it to numpy
    psi0_test=[]
    for i in range(L):
        psi0_tmp=psi0.get_B(i)
        psi0_tmp=psi0_tmp.transpose(['p','vL','vR'])
        psi0_test.append(psi0_tmp.to_ndarray())
    
    tdvp.h_eff(H_MPO,psi0)
    #*************************************************************************************
    #Definition of everything needed for test
    
    model_par = {
            'L': L,
            'S': 0.5,
            'Jz': J,
            'Jx': 0.,
            'Jy': 0.,
            'hz': 0.,
            'hx': 0.,
            'fracture_mpo': False,
            'verbose': 0,
            'dtype': float,
            'conserve_Sz':False,
            'bc' : 'finite'
    }

    Sx =  0.5*np.array([[0,1],[1,0]])
    Sz =  0.5*np.array([[1,0],[0,-1]]) 

    h = tdvp_fast.MPO_TFI(model_par['Jx'],model_par['Jz'],model_par['hx'],model_par['hz'])
    HMPO= [h]*L


    # test that the initial conditions are the same
    
    psit_compare=[]
    for i in range(L):
        psit_compare.append(psi0.get_B(i).transpose(['p','vL','vR']).to_ndarray())
    print(overlap(psit_compare,psi0_test),"overlapp init")
    told=0.0
    for t in times2:
        delta_t = t-told
        told=t
        hmpo_list=[]
        psit,environment,theta_test=tdvp.tdvp(psi0,H_MPO,-0.5*1j*delta_t,chinfo)
        psi0_test,plop,theta_true=tdvp_fast.tdvp(psi0_test,HMPO,-0.5*1j*delta_t, Rp_list=None)
        psit_compare=[]
        for i in range(L):
            psit_compare.append(psi0.get_B(i).transpose(['p','vL','vR']).to_ndarray())
        print("overlap")
        print(overlap(psit_compare,psi0_test))
    print("end")
            
    


