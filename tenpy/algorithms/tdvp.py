"""Time Dependant Variational Principle (TDVP) with MPS (finite version only).

The TDVP MPS algorithm was first proposed by [Haegeman2011]_. However the stability of the
algorithm was later improved in [Haegeman2016]_, that we are following in this implementation.
The general idea of the algorithm is to project the quantum time evolution in the manyfold of MPS
with a given bond dimension. Compared to e.g. TEBD, the algorithm has several advantages:
e.g. it conserves the unitarity of the time evolution and the energy (for the single-site version),
and it is suitable for time evolution of Hamiltonian with arbitrary long range in the form of MPOs.
We have implemented the one-site formulation which **does not** allow for growth of the bond dimension,
and the two-site algorithm which does allow the bond dimension to grow - but requires truncation as in the TEBD case.

.. todo ::
    This is still a beta version, use with care.
    The interface might still change.

.. todo ::
    long-term: Much of the code is similar as in DMRG. To avoid too much duplicated code,
    we should have a general way to sweep through an MPS and updated one or two sites, used in both
    cases.
"""

import numpy as np
from tenpy.networks.mpo import MPOEnvironment
import tenpy.linalg.np_conserved as npc
from tenpy.tools.params import get_parameter
from tenpy.linalg.lanczos import LanczosEvolution
from tenpy.algorithms.truncation import svd_theta


class Engine:
    """Time dependant variational principle 'Engine'

    You can call :meth:`run_one_site` for single-site TDVP, or
    :meth:`run_two_sites` for two-site TDVP.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        Initial state to be time evolved. Modified in place.
    model : :class:`~tenpy.models.model.MPOModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    TDVP_params : dict
        Further optional parameters as described in the following table.
        Use ``verbose>0`` to print the used parameters during runtime.

        ============== ========= ===============================================================
        key            type      description
        ============== ========= ===============================================================
        start_time     float     Initial value for :attr:`evolved_time`
        -------------- --------- ---------------------------------------------------------------
        dt             float     Time step of the Trotter error
        -------------- --------- ---------------------------------------------------------------
        trunc_params   dict      Truncation parameters as described in
                                 :func:`~tenpy.algorithms.truncation.truncate`
        ============== ========= ===============================================================
    environment :  :class:'~tenpy.networks.mpo.MPOEnvironment` | None
        Initial environment. If ``None`` (default), it will be calculated at the beginning.

    Attributes
    ----------
    verbose : int
        Level of verbosity (i.e. how much status information to print); higher=more output.
    evolved_time : float | complex
        Indicating how long `psi` has been evolved, ``psi = exp(-i * evolved_time * H) psi(t=0)``.
    psi : :class:`~tenpy.networks.mps.MPS`
        The MPS, time evolved in-place.
    TDVP_params: dict
        Optional parameters, see :func:`run` and :func:`run_GS` for more details.
    environment : :class:`~tenpy.networks.mpo.MPOEnvironment`
        The environment, storing the `LP` and `RP` to avoid recalculations.
    """

    def __init__(self, psi, model, TDVP_params, environment=None):
        self.verbose = get_parameter(TDVP_params, 'verbose', 1, 'TDVP')
        self.TDVP_params = TDVP_params
        if environment is None:
            environment = MPOEnvironment(psi, model.H_MPO, psi)
        self.evolved_time = get_parameter(TDVP_params, 'start_time', 0., 'TDVP')
        self.H_MPO = model.H_MPO
        self.environment = environment
        if not psi.finite:
            raise ValueError("TDVP is only implemented for finite boundary conditions")
        self.psi = psi
        self.L = self.psi.L
        self.dt = get_parameter(TDVP_params, 'dt', 2, 'TDVP')
        self.trunc_params = get_parameter(TDVP_params, 'trunc_params', {}, 'TDVP')
        self.N_steps = get_parameter(TDVP_params, 'N_steps', 10, 'TDVP')

    # Actual calculation
    def run_one_site(self, N_steps=None):
        """Run the TDVP algorithm with the one site algorithm.

        .. warning ::
            Be aware that the bond dimension will not increase!

        Parameters
        ----------
        N_steps : integer. Number of steps
        """
        if N_steps != None:
            self.N_steps = N_steps
        D = self.H_MPO._W[0].shape[0]
        #Initialize in the correct order
        for i in range(self.L):
            self.psi.get_B(i).itranspose(('vL', 'p', 'vR'))
        for i in range(self.N_steps):
            self.sweep_left_right()
            self.sweep_right_left()
            self.evolved_time = self.evolved_time + self.dt

    def run_two_sites(self, N_steps=None):
        """Run the TDVP algorithm with two sites update.

        The bond dimension will increase. Truncation happens at every step of the
        sweep, according to the parameters set in trunc_params.

        Parameters
        ----------
        N_steps : integer. Number of steps
        """
        if N_steps != None:
            self.N_steps = N_steps
        D = self.H_MPO._W[0].shape[0]
        #Initialize in the correct order
        for i in range(self.L):
            self.psi.get_B(i).itranspose(('vL', 'p', 'vR'))
        for i in range(self.N_steps):
            self.sweep_left_right_two()
            self.sweep_right_left_two()
            self.evolved_time = self.evolved_time + self.dt

    def _del_correct(self, i):
        """Delete correctly the environment once the tensor at site i is updated.

        Parameters
        ----------
        i : int
            Site at which the tensor has been updated
        """

        if i + 1 < self.L:
            self.environment.del_LP(i + 1)
        if i - 1 > -1:
            self.environment.del_RP(i - 1)

    def sweep_left_right(self):
        """Performs the sweep left->right of the second order TDVP scheme with one site update.

        Evolve from 0.5*dt.
        """
        for j in range(self.L):
            B = self.psi.get_B(j)
            # Get theta
            if j == 0:
                theta = B
            else:
                theta = npc.tensordot(B, s,
                                      axes=('vL',
                                            'vR'))  #theta[vL,p,vR]=s[vL,vR]*self.psi[p,vL,vR]
            Lp = self.environment.get_LP(j)
            Rp = self.environment.get_RP(j)
            W1 = self.environment.H.get_W(j)
            theta = self.update_theta_h1(Lp, Rp, theta, W1, -1j * 0.5 * self.dt)
            # SVD and update environment
            U, s, V = self.theta_svd_left_right(theta)
            self.psi.set_B(j, U, form='A')
            self._del_correct(j)
            if j < self.L - 1:
                # Apply expm (-dt H) for 0-site

                B = self.psi.get_B(j + 1)
                B_jp1 = npc.tensordot(V, B, axes=['vR', 'vL'])
                self.psi.set_B(j + 1, B_jp1, form='B')
                Lpp = self.environment.get_LP(j + 1)
                Rp = npc.tensordot(Rp, V, axes=['vL', 'vR'])
                Rp = npc.tensordot(Rp, V.conj(), axes=['vL*', 'vR*'])
                H = H0_mixed(Lpp, Rp)

                s = self.update_s_h0(s, H, 1j * 0.5 * self.dt)
                s = s / np.linalg.norm(s.to_ndarray())

    def sweep_left_right_two(self):
        """Performs the sweep left->right of the second order TDVP scheme with two sites update.

        Evolve from 0.5*dt"""
        theta_old = self.psi.get_theta(0, 1)
        for j in range(self.L - 1):

            theta = npc.tensordot(theta_old, self.psi.get_B(j + 1), ('vR', 'vL'))
            theta.ireplace_label('p', 'p1')
            Lp = self.environment.get_LP(j)
            Rp = self.environment.get_RP(j + 1)
            W1 = self.environment.H.get_W(j)
            W2 = self.environment.H.get_W(j + 1)
            theta = self.update_theta_h2(Lp, Rp, theta, W1, W2, -0.5 * 1j * self.dt)
            theta = theta.combine_legs([['vL', 'p0'], ['vR', 'p1']], qconj=[+1, -1])
            # SVD and update environment
            U, s, V, err, renorm = svd_theta(theta, self.trunc_params)
            s = s / npc.norm(s)
            U = U.split_legs('(vL.p0)')
            U.ireplace_label('p0', 'p')
            V = V.split_legs('(vR.p1)')
            V.ireplace_label('p1', 'p')
            self.psi.set_B(j, U, form='A')
            self._del_correct(j)
            self.psi.set_SR(j, s)
            self.psi.set_B(j + 1, V, form='B')
            self._del_correct(j + 1)
            if j < self.L - 2:
                # Apply expm (-dt H) for 1-site
                theta = self.psi.get_theta(j + 1, 1)
                theta.ireplace_label('p0', 'p')
                Lp = self.environment.get_LP(j + 1)
                Rp = self.environment.get_RP(j + 1)
                theta = self.update_theta_h1(Lp, Rp, theta, W2, 1j * 0.5 * self.dt)
                theta_old = theta
                theta_old.ireplace_label('p', 'p0')

    def sweep_right_left(self):
        """Performs the sweep right->left of the second order TDVP scheme with one site update.

        Evolve from 0.5*dt"""
        expectation_O = []
        for j in range(self.L - 1, -1, -1):
            B = self.psi.get_B(j, form='A')
            # Get theta
            if j == self.L - 1:
                theta = B
            else:
                theta = npc.tensordot(B, s,
                                      axes=('vR',
                                            'vL'))  #theta[vL,p,vR]=s[vL,vR]*self.psi[p,vL,vR]

            # Apply expm (-dt H) for 1-site
            chiB, chiA, d = theta.to_ndarray().shape
            Lp = self.environment.get_LP(j)
            Rp = self.environment.get_RP(j)
            W1 = self.environment.H.get_W(j)
            theta = self.update_theta_h1(Lp, Rp, theta, W1, -1j * 0.5 * self.dt)
            # SVD and update environment
            U, s, V = self.theta_svd_right_left(theta)
            self.psi.set_B(j, U, form='B')
            self._del_correct(j)
            if j > 0:
                # Apply expm (-dt H) for 0-site

                B = self.psi.get_B(j - 1, form='A')
                B_jm1 = npc.tensordot(V, B, axes=['vL', 'vR'])
                self.psi.set_B(j - 1, B_jm1, form='A')
                Lp = npc.tensordot(Lp, V, axes=['vR', 'vL'])
                Lp = npc.tensordot(Lp, V.conj(), axes=['vR*', 'vL*'])
                H = H0_mixed(Lp, self.environment.get_RP(j - 1))

                s = self.update_s_h0(s, H, 1j * 0.5 * self.dt)
                s = s / np.linalg.norm(s.to_ndarray())

    def sweep_right_left_two(self):
        """Performs the sweep left->right of the second order TDVP scheme with two sites update.

        Evolve from 0.5*dt"""
        theta_old = self.psi.get_theta(self.L - 1, 1)
        for j in range(self.L - 2, -1, -1):
            theta = npc.tensordot(theta_old, self.psi.get_B(j, form='A'), ('vL', 'vR'))
            theta.ireplace_label('p0', 'p1')
            theta.ireplace_label('p', 'p0')
            #theta=self.psi.get_theta(j,2)
            Lp = self.environment.get_LP(j)
            Rp = self.environment.get_RP(j + 1)
            W1 = self.environment.H.get_W(j)
            W2 = self.environment.H.get_W(j + 1)
            theta = self.update_theta_h2(Lp, Rp, theta, W1, W2, -1j * 0.5 * self.dt)
            theta = theta.combine_legs([['vL', 'p0'], ['vR', 'p1']], qconj=[+1, -1])
            # SVD and update environment
            U, s, V, err, renorm = svd_theta(theta, self.trunc_params)
            s = s / npc.norm(s)
            U = U.split_legs('(vL.p0)')
            U.ireplace_label('p0', 'p')
            V = V.split_legs('(vR.p1)')
            V.ireplace_label('p1', 'p')
            self.psi.set_B(j, U, form='A')
            self._del_correct(j)
            self.psi.set_SR(j, s)
            self.psi.set_B(j + 1, V, form='B')
            self._del_correct(j + 1)
            if j > 0:
                # Apply expm (-dt H) for 1-site
                theta = self.psi.get_theta(j, 1)
                theta.ireplace_label('p0', 'p')
                Lp = self.environment.get_LP(j)
                Rp = self.environment.get_RP(j)
                theta = self.update_theta_h1(Lp, Rp, theta, W1, 1j * 0.5 * self.dt)
                theta_old = theta
                theta.ireplace_label('p', 'p0')

    def update_theta_h1(self, Lp, Rp, theta, W, dt):
        """Update with the one site Hamiltonian.

        Parameters
        ----------
        Lp : :class:`~tenpy.linalg.np_conserved.Array`
            tensor representing the left environment
        Rp :  :class:`~tenpy.linalg.np_conserved.Array`
            tensor representing the right environment
        theta :  :class:`~tenpy.linalg.np_conserved.Array`
            the theta tensor which needs to be updated
        W : :class:`~tenpy.linalg.np_conserved.Array`
            MPO which is applied to the 'p' leg of theta
        """
        H = H1_mixed(Lp, Rp, W)
        theta = theta.combine_legs(['vL', 'p', 'vR'])
        #Initialize Lanczos
        parameters_lanczos_h1 = {'delta': dt}
        lanczos_h1 = LanczosEvolution(H=H, psi0=theta, params=parameters_lanczos_h1)
        theta, N_h1 = lanczos_h1.run(dt)
        theta = theta.split_legs(['(vL.p.vR)'])
        return theta

    def update_theta_h2(self, Lp, Rp, theta, W0, W1, dt):
        """Update with the two sites Hamiltonian

        Parameters
        ----------
        Lp : :class:`tenpy.linalg.np_conserved.Array`
            tensor representing the left environment
        Rp : :class:`tenpy.linalg.np_conserved.Array`
            tensor representing the right environment
        theta : :class:`tenpy.linalg.np_conserved.Array`
            the theta tensor which needs to be updated
        W : :class:`tenpy.linalg.np_conserved.Array`
            MPO which is applied to the 'p0' leg of theta
        W1 : :class:`tenpy.linalg.np_conserved.Array`
            MPO which is applied to the 'p1' leg of theta
        """
        H = H2_mixed(Lp, Rp, W0, W1)
        theta = theta.combine_legs(['vL', 'p0', 'p1', 'vR'])
        #Initialize Lanczos
        parameters_lanczos_h1 = {'delta': dt}
        lanczos_h1 = LanczosEvolution(H=H, psi0=theta, params=parameters_lanczos_h1)
        theta, N_h1 = lanczos_h1.run(dt)
        theta = theta.split_legs(['(vL.p0.p1.vR)'])
        return theta

    def theta_svd_left_right(self, theta):
        """Performs the SVD from left to right

        Parameters
        ----------
        theta: :class:`tenpy.linalg.np_conserved.Array`
            the theta tensor on which the SVD is applied
        """
        theta = theta.combine_legs(['vL', 'p'])
        U, s, V = npc.svd(theta, full_matrices=0)
        U = U.split_legs(['(vL.p)'])
        U = self.set_anonymous_svd(U, 'vR')
        V = self.set_anonymous_svd(V, 'vL')
        s_ndarray = np.diag(s)
        vR_U = U.get_leg('vR')
        vL_V = V.get_leg('vL')
        s = npc.Array.from_ndarray(s_ndarray, [vR_U.conj(), vL_V.conj()],
                                   dtype=None,
                                   qtotal=None,
                                   cutoff=None)
        s.iset_leg_labels(['vL', 'vR'])
        return U, s, V

    def set_anonymous_svd(self, U, new_label):
        """Relabel the svd

        Parameters
        ----------
        U : :class:`tenpy.linalg.np_conserved.Array`
            the tensor which lacks a leg_label
        """
        list_labels = list(U.get_leg_labels())
        for i in range(len(list_labels)):
            if list_labels[i] == None:
                list_labels[i] = 'None'
        U = U.iset_leg_labels(list_labels)
        U = U.replace_label('None', new_label)
        return U

    def theta_svd_right_left(self, theta):
        """Performs the SVD from right to left

        Parameters
        ----------
        theta : :class:`tenpy.linalg.np_conserved.Array`,
            The theta tensor on which the SVD is applied
        """
        theta = theta.combine_legs(['p', 'vR'])
        V, s, U = npc.svd(theta, full_matrices=0)
        U = U.split_legs(['(p.vR)'])
        U = self.set_anonymous_svd(U, 'vL')
        V = self.set_anonymous_svd(V, 'vR')
        s_ndarray = np.diag(s)
        vL_U = U.get_leg('vL')
        vR_V = V.get_leg('vR')
        s = npc.Array.from_ndarray(s_ndarray, [vR_V.conj(), vL_U.conj()],
                                   dtype=None,
                                   qtotal=None,
                                   cutoff=None)
        s.iset_leg_labels(['vL', 'vR'])
        return U, s, V

    def update_s_h0(self, s, H, dt):
        """Update with the zero site Hamiltonian (update of the singular value)

        Parameters
        ----------
        s : :class:`tenpy.linalg.np_conserved.Array`
            representing the singular value matrix which is updated
        H : H0_mixed
            zero site Hamiltonian that we need to apply on the singular value matrix
        dt : complex number
            time step of the evolution
        """
        #Initialize Lanczos
        parameters_lanczos_h1 = {'delta': dt}
        lanczos_h0 = LanczosEvolution(H=H,
                                      psi0=s.combine_legs(['vL', 'vR']),
                                      params=parameters_lanczos_h1)
        s_new, N_h0 = lanczos_h0.run(dt)
        s_new = s_new.split_legs(['(vL.vR)'])
        return s_new


class H0_mixed:
    """Class defining the zero site Hamiltonian for Lanczos

    Parameters
    ----------
    Lp : :class:`tenpy.linalg.np_conserved.Array`
        left part of the environment
    Rp : :class:`tenpy.linalg.np_conserved.Array`
        right part of the environment

    Attributes
    ----------
    Lp : :class:`tenpy.linalg.np_conserved.Array`
        left part of the environment
    Rp : :class:`tenpy.linalg.np_conserved.Array`
        right part of the environment
    """

    def __init__(self, Lp, Rp):
        self.Lp = Lp
        self.Rp = Rp

    def matvec(self, x):
        x = x.split_legs(['(vL.vR)'])
        x = npc.tensordot(self.Lp, x, axes=('vR', 'vL'))
        x = npc.tensordot(x, self.Rp, axes=(['vR', 'wR'], ['vL', 'wL']))
        #TODO:next line not needed. Since the transpose does not do anything, should not cost anything. Keep for safety ?
        x = x.transpose(['vR*', 'vL*'])
        x = x.iset_leg_labels(['vL', 'vR'])
        x = x.combine_legs(['vL', 'vR'])
        return (x)


class H1_mixed:
    """Class defining the one site Hamiltonian for Lanczos

    Parameters
    ----------
    Lp : :class:`tenpy.linalg.np_conserved.Array`
        left part of the environment
    Rp : :class:`tenpy.linalg.np_conserved.Array`
        right part of the environment
    M : :class:`tenpy.linalg.np_conserved.Array`
        MPO which is applied to the 'p' leg of theta

    Attributes
    ----------
    Lp : :class:`tenpy.linalg.np_conserved.Array`
        left part of the environment
    Rp : :class:`tenpy.linalg.np_conserved.Array`
        right part of the environment
    W : :class:`tenpy.linalg.np_conserved.Array`
        MPO which is applied to the 'p0' leg of theta
    """

    def __init__(self, Lp, Rp, W):
        self.Lp = Lp  # a,ap,m
        self.Rp = Rp  # b,bp,n
        self.W = W  # m,n,i,ip

    def matvec(self, theta):
        theta = theta.split_legs(['(vL.p.vR)'])
        Lp = self.Lp
        Rp = self.Rp
        x = npc.tensordot(Lp, theta, axes=('vR', 'vL'))
        x = npc.tensordot(x, self.W, axes=(['p', 'wR'], ['p*', 'wL']))
        x = npc.tensordot(x, Rp, axes=(['vR', 'wR'], ['vL', 'wL']))
        #TODO:next line not needed. Since the transpose does not do anything, should not cost anything. Keep for safety ?
        x = x.transpose(['vR*', 'p', 'vL*'])
        x = x.iset_leg_labels(['vL', 'p', 'vR'])
        h = x.combine_legs(['vL', 'p', 'vR'])
        return h


class H2_mixed:
    """Class defining the two sites Hamiltonian for Lanczos

    Parameters
    ----------
    Lp : :class:`tenpy.linalg.np_conserved.Array`
        left part of the environment
    Rp : :class:`tenpy.linalg.np_conserved.Array`
        right part of the environment
    W : :class:`tenpy.linalg.np_conserved.Array`
        MPO which is applied to the 'p0' leg of theta

    Attributes
    ----------
    Lp : :class:`tenpy.linalg.np_conserved.Array`
        left part of the environment
    Rp : :class:`tenpy.linalg.np_conserved.Array`
        right part of the environment
    W0 : :class:`tenpy.linalg.np_conserved.Array`
        MPO which is applied to the 'p0' leg of theta
    W1 : :class:`tenpy.linalg.np_conserved.Array`
        MPO which is applied to the 'p1' leg of theta
    """

    def __init__(self, Lp, Rp, W0, W1):
        self.Lp = Lp  # a,ap,m
        self.Rp = Rp  # b,bp,n
        self.H_MPO0 = W0  # m,n,i,ip
        self.H_MPO1 = W1

    def matvec(self, theta):
        theta = theta.split_legs(['(vL.p0.p1.vR)'])
        Lp = self.Lp
        Rp = self.Rp
        x = npc.tensordot(Lp, theta, axes=('vR', 'vL'))
        x = npc.tensordot(x, self.H_MPO0, axes=(['p0', 'wR'], ['p*', 'wL']))
        x.ireplace_label('p', 'p0')
        x = npc.tensordot(x, self.H_MPO1, axes=(['p1', 'wR'], ['p*', 'wL']))
        x.ireplace_label('p', 'p1')
        x = npc.tensordot(x, Rp, axes=(['vR', 'wR'], ['vL', 'wL']))
        x.ireplace_label('vL*', 'vR')
        x.ireplace_label('vR*', 'vL')
        h = x.combine_legs(['vL', 'p0', 'p1', 'vR'])
        return h
