#!/usr/bin/python2
import tdvp_fast as tdvp
import numpy as np
import misc
import copy
#import h5py
import pickle

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


def convertMps(mps):
    wftensor=mps[0]

    for i in range(1,len(mps)):

        wftensor = np.tensordot(wftensor, mps[i], axes=[[-1],[1]])
    return wftensor.reshape(mps[0].shape[0]**len(mps))


if __name__ == "__main__":
    

    L = 6
    chi = 10

    times0 = np.linspace(0.0,10,61)
    times=times0[1:len(times0)]
    print(times)
    print('times',len(times))

    hx =0.0 
    hz =1.0
    J = 1.0

    model_par = {
            'L': L,
            'S': 0.5,
            'Jz': J,
            'Jx': 0.,
            'Jy': 0.,
            'hz': hz,
            'hx': hx,
            'fracture_mpo': False,
            'verbose': 0,
            'dtype': float,
            'conserve_Sz':False,
            'bc' : 'finite'
    }

    Sx =  np.array([[0,1],[1,0]])
    Sz =  np.array([[1,0],[0,-1]])

    O1= Sz
    O2= Sz # WARNING:  Must be Sz (initial state must be eigenstate)
    SITE0 = int(L/2)
    num_random_states=1



    h = tdvp.MPO_TFI(model_par['Jx'],model_par['Jz'],model_par['hx'],model_par['hz'])
    HMPO= [h]*L


    OTOCs=[]
    #psi0,sz0=random_product_state(L,chi)
    psi0,sz0=fixed_product_state(L,chi,0)
    told=0
    for t in times:
        print(t)
        delta_t = t-told
        told=t
        # TODO Here, we should only evolve by delta_t. Does not work right now for some reason
        #psit= copy.deepcopy(psi0)
        psi0,Rpp,spect=tdvp.tdvp(psi0,HMPO,-0.5*1j*delta_t, Rp_list=None)
        #psit= tdvp_evolve(psit,HMPO,t,delta_t)
    


