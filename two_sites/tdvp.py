"""
Time Dependant Variational Principle (TDVP) with MPS (finite version only).
The TDVP MPS algorithm was first proposed by Jutho Haegeman (https://arxiv.org/abs/1103.0936). However
the stability of the algorithm was later improved in another article (https://arxiv.org/abs/1408.5056), that we are following in this implementation. 
The general idea of the algorithm is to project the quantum time evolution in the manyfold of MPS with a given bond dimension.
The algorithm has several advantage (among others: conervation of the unitarity of the time evolution and conservation of energy), and is suitable for time evolution
of Hamiltonian with arbitrary long range.
We have implemented the on site formulation which DOES NOT allow for growth of the bond dimension and the two site algorithm
which allow the bond dimension (but requires truncation as in the TEBD case).
"""

import sys
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
import numpy as np
import pylab as pl
from scipy.sparse.linalg import onenormest
import tenpy.networks.mpo as mpo        
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.np_conserved import svd
from tenpy.tools.params import get_parameter
import scipy.linalg
import warnings
import tenpy.linalg.lanczos
from tenpy.algorithms.truncation import svd_theta


value_test=True
number_test=0

class Engine(object):
    """
    Time dependant variational principle 'engine'
        
    Parameters
    ----------
    psi : :class:`~tenpy.networs.mps.MPS`
        Initial state to be time evolved. Modified in place.
    model : :class:`~tenpy.models.model.`
        The model representing the Hamiltonian for which we want to find the ground state.
    TEBD_params : dict
        Further optional parameters as described in the following table.
        Use ``verbose>0`` to print the used parameters during runtime.
     
        ============== =================================     ===============================================================
        key            type                                  description
        ==============  ================================     ===============================================================
        psi            :class:`~tenpy.networks.mps.MPS`                                   wave function to evolve 
        -------------- ---------------------------------     ---------------------------------------------------------------
        model          :class:`~tenpy.models.MPOModel`
        -------------- ---------------------------------     ---------------------------------------------------------------
        TDVP_params    dict of parameters                    verbose, start_time,dt (time step of the Trotter error)
        -------------- ---------------------------------     ---------------------------------------------------------------
        trunc_params   dict                                  Truncation parameters as described in
                                                             :func:`~tenpy.algorithms.truncation.truncate`.
        -------------- ---------------------------------     --------------------------------------------------------------
        environment    :class:                               initial environment. None by default, will be calculated at the
                       '~tenpy.networks.mpo.MPOEnvironment'  beginning of the algorithm if not provided.
        ============== =================================     ===============================================================
      
    """

    def __init__(self, psi, model, TDVP_params,trunc_params=None,environment=None):
        self.verbose = get_parameter(TDVP_params, 'verbose', 2, 'TDVP')
        self.TDVP_params = TDVP_params
        self.model = model
        self.evolved_time = get_parameter(TDVP_params, 'start_time', 0., 'TDVP')
        self.W =model.H_MPO
        self._update_index = None
        self.environment=None
        self.spectrum=None
        self.psi=psi
        self.L=self.psi.L
        self.dt=get_parameter(TDVP_params, 'dt', 2, 'TDVP')
        self.trunc_params=trunc_params
        #Since we have a second order TDVP, we evolve from 0.5dt during left_right sweep and of 0.5dt again
        #during the right_left sweep

    def del_correct(self,i):
        #LP
        if i+1<self.psi.L:
            self.environment.del_LP(i+1)


        #Rp
        if i-1>-1:
            self.environment.del_RP(i-1)


    def sweep_left_right(self):
        """Performs the sweep left->right of the second order TDVP scheme. Evolve from 0.5*dt"""
        spectrum = []        
        spectrum.append(np.array([1]))
        for j in range(self.L):
            B=self.psi.get_B(j)
            # Get theta
            if j==0:
                theta=B
            else:
                theta=npc.tensordot(s,B,axes=('vR','vL'))   #theta[vL,p,vR]=s[vL,vR]*self.psi[p,vL,vR]
            theta=theta.transpose(['p','vL','vR'])
            #d,chiA,chiB = theta.to_ndarray().shape
            Lp=self.environment.get_LP(j)
            Rp=self.environment.get_RP(j)
            theta=self.update_theta_h1(Lp, Rp, theta, self.W.get_W(j), -1j*0.5*self.dt)
            # SVD and update environment
            U,s,V=self.theta_svd_left_right(theta)
            spectrum.append(s/np.linalg.norm(s.to_ndarray()))
            self.psi.set_B(j,U,form='A')
            self.del_correct(j)
            if j < self.L-1 :
                # Apply expm (-dt H) for 0-site
                
                B=self.psi.get_B(j+1)
                B_jp1=npc.tensordot(V, B, axes=['vR', 'vL'])
                self.psi.set_B(j+1, B_jp1,form='B')
                Lpp=self.environment.get_LP(j+1)
                Rp=npc.tensordot(Rp,V,axes=['vL','vR'])
                Rp=npc.tensordot(Rp,V.conj(),axes=['vL*','vR*'])
                H = H0_mixed(Lpp,Rp)
                
                s=self.update_s_h0(s,H,1j*0.5*self.dt)
                s= s/np.linalg.norm(s.to_ndarray())
                
        #return self.psi, self.environment, spectrum

    def sweep_left_right_two(self):
        """Performs the sweep left->right of the second order TDVP scheme. Evolve from 0.5*dt"""
        spectrum = []        
        spectrum.append(np.array([1]))
        theta_old=self.psi.get_theta(0,1)
        for j in range(self.L-1):
               
            #theta=self.psi.get_theta(j,2)
            theta=npc.tensordot(theta_old,self.psi.get_B(j+1),('vR','vL'))
            theta.ireplace_label('p','p1')
            Lp=self.environment.get_LP(j)
            Rp=self.environment.get_RP(j+1)
            theta=self.update_theta_h2(Lp, Rp, theta, self.W.get_W(j),self.W.get_W(j+1), -0.5*1j*self.dt)
            theta=theta.combine_legs(['vR','p1'])
            theta=theta.combine_legs(['vL','p0'])
            theta.itranspose(['(vL.p0)','(vR.p1)'])
            # SVD and update environment
            U,s,V,err,renorm=svd_theta(theta,self.trunc_params)
            s=s/npc.norm(s)
            U=U.split_legs('(vL.p0)')
            U.ireplace_label('p0','p')
            V=V.split_legs('(vR.p1)')
            V.ireplace_label('p1','p')
            self.psi.set_B(j,U,form='A')
            self.del_correct(j)
            self.psi.set_SR(j,s)
            self.psi.set_B(j+1,V,form='B')
            self.del_correct(j+1)
            if j<self.L-2: 
                # Apply expm (-dt H) for 1-site
                theta=self.psi.get_theta(j+1,1)
                theta.ireplace_label('p0','p')
                Lp=self.environment.get_LP(j+1)
                Rp=self.environment.get_RP(j+1)
                theta=self.update_theta_h1(Lp, Rp, theta, self.W.get_W(j+1), 1j*0.5*self.dt)
                theta_old=theta
                theta_old.ireplace_label('p','p0')



    def sweep_right_left(self):
        """Performs the sweep right->left of the second order TDVP scheme. Evolve from 0.5*dt"""
        spectrum = []        
        spectrum.append(np.array([1]))
        expectation_O = []
        for j in range(self.L-1,-1,-1):
            B=self.psi.get_B(j,form='A')
            # Get theta
            if j==self.L-1:
                theta=B
            else:
                theta=npc.tensordot(s,B,axes=('vL','vR'))   #theta[vL,p,vR]=s[vL,vR]*self.psi[p,vL,vR]
            theta=theta.transpose(['p','vL','vR',])
            
            # Apply expm (-dt H) for 1-site
            chiB,chiA,d = theta.to_ndarray().shape
            Lp=self.environment.get_LP(j)
            Rp=self.environment.get_RP(j)
            theta=self.update_theta_h1(Lp, Rp, theta, self.W.get_W(j), -1j*0.5*self.dt)
            # SVD and update environment
            U,s,V=self.theta_svd_right_left(theta)
            spectrum.append(s/np.linalg.norm(s.to_ndarray()))
            self.psi.set_B(j,U,form='B')
            self.del_correct(j)
            if j > 0:
                # Apply expm (-dt H) for 0-site
                
                B=self.psi.get_B(j-1,form='A')
                B_jm1=npc.tensordot(V, B, axes=['vL', 'vR'])
                self.psi.set_B(j-1, B_jm1,form='A')
                Lp=npc.tensordot(Lp,V,axes=['vR','vL'])
                Lp=npc.tensordot(Lp,V.conj(),axes=['vR*','vL*'])
                H = H0_mixed(Lp,self.environment.get_RP(j-1))
                

                s=self.update_s_h0(s,H,1j*0.5*self.dt)
                s= s/np.linalg.norm(s.to_ndarray())
    
        #return self.psi, self.environment, spectrum

    def sweep_right_left_two(self):
        """Performs the sweep left->right of the second order TDVP scheme. Evolve from 0.5*dt"""
        spectrum = []        
        spectrum.append(np.array([1]))
        theta_old=self.psi.get_theta(self.L-1,1)
        for j in range(self.L-2,-1,-1):
            theta=npc.tensordot(theta_old,self.psi.get_B(j,form='A'),('vL','vR'))
            theta.ireplace_label('p0','p1')
            theta.ireplace_label('p','p0')
            #theta=self.psi.get_theta(j,2)
            Lp=self.environment.get_LP(j)
            Rp=self.environment.get_RP(j+1)
            theta=self.update_theta_h2(Lp, Rp, theta, self.W.get_W(j),self.W.get_W(j+1), -1j*0.5*self.dt)
            theta=theta.combine_legs(['vR','p1'])
            theta=theta.combine_legs(['vL','p0'])
            theta.itranspose(['(vL.p0)','(vR.p1)'])
            # SVD and update environment
            U,s,V,err,renorm=svd_theta(theta,self.trunc_params)
            s=s/npc.norm(s)
            U=U.split_legs('(vL.p0)')
            U.ireplace_label('p0','p')
            V=V.split_legs('(vR.p1)')
            V.ireplace_label('p1','p')
            self.psi.set_B(j,U,form='A')
            self.del_correct(j)
            self.psi.set_SR(j,s)
            self.psi.set_B(j+1,V,form='B')
            self.del_correct(j+1)
            if j>0: 
                # Apply expm (-dt H) for 1-site
                theta=self.psi.get_theta(j,1)
                theta.ireplace_label('p0','p')
                Lp=self.environment.get_LP(j)
                Rp=self.environment.get_RP(j)
                theta=self.update_theta_h1(Lp, Rp, theta, self.W.get_W(j), 1j*0.5*self.dt)
                theta_old=theta
                theta.ireplace_label('p','p0')

    def update_theta_h1(self,Lp, Rp, theta, W, dt):
        """Update with the one site Hamiltonian """
        H = H1_mixed(Lp,Rp,W)
        theta=theta.transpose(['p','vR','vL'])
        theta=theta.combine_legs(['p','vR','vL'])
        #Initialize Lanczos
        parameters_lanczos_h1= {
            'delta':dt
        }
        lanczos_h1=tenpy.linalg.lanczos.LanczosEvolution(H=H, psi0=theta, params=parameters_lanczos_h1)
        theta,N_h1=lanczos_h1.run(dt)
        theta=theta.split_legs(['(p.vR.vL)'])
        return theta
    
    def update_theta_h2(self,Lp, Rp, theta, W0,W1, dt):
        """Update with the one site Hamiltonian with the two site implementation """
        H = H2_mixed(Lp,Rp,W0,W1)
        theta=theta.combine_legs(['p0','p1','vR','vL'])
        #Initialize Lanczos
        parameters_lanczos_h1= {
            'delta':dt
        }
        lanczos_h1=tenpy.linalg.lanczos.LanczosEvolution(H=H, psi0=theta, params=parameters_lanczos_h1)
        theta,N_h1=lanczos_h1.run(dt)
        theta=theta.split_legs(['(p0.p1.vR.vL)'])
        return theta
    
    def theta_svd_left_right(self,theta):
        """Performs the SVD from left to right"""
        theta=theta.transpose(['p','vL','vR'])
        theta=theta.combine_legs(['p','vL'])
        theta=theta.transpose(['(p.vL)','vR'])
        U,s,V = svd(theta,full_matrices=0)
        U=U.split_legs(['(p.vL)'])
        U=self.set_anonymous_svd(U,'vR')
        V=self.set_anonymous_svd(V,'vL')
        s_ndarray = np.diag(s)
        vR_U=U.get_leg('vR')
        vL_V=V.get_leg('vL')
        s=npc.Array.from_ndarray(s_ndarray,[vR_U.conj(),vL_V.conj()], dtype=None, qtotal=None, cutoff=None)
        s.iset_leg_labels(['vL','vR'])
        return U,s,V        

    def set_anonymous_svd(self,U,new_label):
        """Relabel the svd"""
        list_labels=list(U.get_leg_labels())
        for i in range(len(list_labels)):
            if list_labels[i]==None:
                list_labels[i]='None'
        U=U.iset_leg_labels(list_labels)
        U=U.replace_label('None',new_label)
        return U

    def theta_svd_right_left(self,theta):
        """Performs the SVD from right to left"""
        theta=theta.transpose(['vL','p','vR'])
        theta=theta.combine_legs(['p','vR'])
        theta=theta.transpose(['vL','(p.vR)'])
        V,s,U = svd(theta,full_matrices=0)
        U=U.split_legs(['(p.vR)'])
        U=self.set_anonymous_svd(U,'vL')
        V=self.set_anonymous_svd(V,'vR')
        s_ndarray = np.diag(s)
        vL_U=U.get_leg('vL')
        vR_V=V.get_leg('vR')
        s=npc.Array.from_ndarray(s_ndarray,[vR_V.conj(),vL_U.conj()], dtype=None, qtotal=None, cutoff=None)
        s.iset_leg_labels(['vL','vR'])
        return U,s,V 
    
    def update_s_h0(self,s,H,dt):
        """Update with the zero site Hamiltonian (update of the singular value)"""
        #Initialize Lanczos
        parameters_lanczos_h1= {
            'delta':dt
        }
        lanczos_h0=tenpy.linalg.lanczos.LanczosEvolution(H=H, psi0=s.combine_legs(['vL','vR']), params=parameters_lanczos_h1)
        s_new,N_h0=lanczos_h0.run(dt)
        ##s_new=lanczos(H,s.combine_legs(['vL','vR']),dt,{'N_max':np.min([chiA*chiB,6])})
        s_new=s_new.split_legs(['(vL.vR)'])
        return s_new
    
    def calculate_error_left_right(self,i):
        Lp=self.environment.get_LP(i)
        Rp=self.environment.get_RP(i)
        W=self.W.get_W(i)
        B=self.psi.get_B(i)
        p_h_phi=npc.tensordot(Lp,W,axes=['wR','wL'])
        p_h_phi=npc.tensordot(p_h_phi,Rp,axes=['wR','wL'])
        p_h_phi=npc.tensordot(p_h_phi,B,axes=[('p*','vR','vL'),('p','vL','vR')])
        npc.norm(p_h_phi)

                

        # Actual calculation
    def run_one_site(self,N_steps):
        """
        Run the TDVP algorithm with the one site algorithm. Be aware that the bond dimension will not increase
        Parameters
        ----------
        N_steps : integer. Number of steps
        """
        N_steps = get_parameter(self.TDVP_params, 'N_steps', 10, 'TDVP')
        D = self.W._W[0].shape[0]
        if self.environment==None:
            self.environment=mpo.MPOEnvironment(self.psi,self.W,self.psi)
        for i in range(N_steps):
            self.sweep_left_right()
            self.sweep_right_left()
    
    def run_two_sites(self,N_steps):
        """ 
        Run the TDVP algorithm with two sites update. The bond dimension will increase. Truncation happens at every step of the
        sweep, according to the parameters set in trunc_params.
        Parameters
        ----------
        N_steps : integer. Number of steps
        """
        N_steps = get_parameter(self.TDVP_params, 'N_steps', 10, 'TDVP')
        D = self.W._W[0].shape[0]
        #if self.environment==None:
        self.environment=mpo.MPOEnvironment(self.psi,self.W,self.psi)
        for i in range(N_steps):
            self.sweep_left_right_two()
            self.sweep_right_left_two()
            self.evolved_time=self.evolved_time+self.dt

class H0_mixed(object):
    """Class defining the zero site Hamiltonian for Lanczos"""
    def __init__(self,Lp,Rp,dtype=float):
        self.Lp = Lp 
        self.Rp = Rp 
        self.chi1 = Lp.to_ndarray().shape[0]        
        self.chi2 = Rp.to_ndarray().shape[0]
        self.shape = np.array([self.chi1*self.chi2,self.chi1*self.chi2])
        self.dtype = dtype
                
    def matvec(self,x):
        x=x.split_legs(['(vL.vR)'])
        x=npc.tensordot(self.Lp,x,axes=('vR','vL'))
        x=npc.tensordot(x,self.Rp,axes=(['vR','wR'],['vL','wL']) )
        x=x.transpose(['vR*','vL*'])
        x=x.iset_leg_labels(['vL','vR'])
        x=x.combine_legs(['vL','vR'])
        return(x)


class H1_mixed(object):
    """Class defining the one site Hamiltonian for Lanczos"""

    def __init__(self,Lp,Rp,M,dtype=float):
        self.Lp = Lp # a,ap,m
        self.Rp = Rp # b,bp,n
        self.M = M # m,n,i,ip
        self.d = M.shape[3]
        self.chi1 = Lp.shape[0]        
        self.chi2 = Rp.shape[0]
        self.shape = np.array([self.d*self.chi1*self.chi2,self.d*self.chi1*self.chi2])
        self.dtype = dtype
                
    def matvec(self,theta):
        global value_test
        theta=theta.split_legs(['(p.vR.vL)'])
        Lp=self.Lp 
        Rp=self.Rp
        x=npc.tensordot(Lp,theta,axes=('vR','vL'))
        x=npc.tensordot(x,self.M,axes=(['p','wR'],['p*','wL']))
        x=npc.tensordot(x,Rp,axes=(['vR','wR'],['vL','wL']))
        x=x.transpose(['p','vR*','vL*'])
        x=x.iset_leg_labels(['p','vL','vR'])
        h=x.combine_legs(['p','vR','vL'])
        return h

class H2_mixed(object):
    """Class defining the one site Hamiltonian for Lanczos for the two sites algorithm"""

    def __init__(self,Lp,Rp,W0,W1,dtype=float):
        self.Lp = Lp # a,ap,m
        self.Rp = Rp # b,bp,n
        self.W0 = W0 # m,n,i,ip
        self.W1 = W1 
    def matvec(self,theta):
        theta=theta.split_legs(['(p0.p1.vR.vL)'])
        Lp=self.Lp 
        Rp=self.Rp
        x=npc.tensordot(Lp,theta,axes=('vR','vL'))
        x=npc.tensordot(x,self.W0,axes=(['p0','wR'],['p*','wL']))
        x.ireplace_label('p','p0')
        x=npc.tensordot(x,self.W1,axes=(['p1','wR'],['p*','wL']))
        x.ireplace_label('p','p1')
        x=npc.tensordot(x,Rp,axes=(['vR','wR'],['vL','wL']))
        x.ireplace_label('vL*','vR')
        x.ireplace_label('vR*','vL')
        h=x.combine_legs(['p0','p1','vR','vL'])
        return h
