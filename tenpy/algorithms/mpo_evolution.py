r"""Time evolution using the WI or WII approximation of the time
evolution operator"""

import numpy as np
import time
from scipy.linalg import expm

from ..linalg import np_conserved as npc
from .truncation import svd_theta, TruncationError
from ..tools.params import asConfig
from ..networks import mps, mpo
from .mps_compress import apply_mpo

__all__ = ['Engine']

class Engine:
    """Time evolution of an MPS using the WI or WII method

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        Initial state to be time evolved. Modified in place.
    model : :class:`~tenpy.models.model.MPOModel`
        The model representing the Hamiltonian which we want to 
        time evolve psi with.
    options : dict
        Further optional parameters are described in the tables in
        :func: `run`.
        Use ``verbose=1`` to print the used parameters during runtime.

    Options
    -------
    .. cfg:config :: MPO_Evo

        trunc_params : dict
            Truncation parameters as described in :cfg:config:`truncate`.
        start_time : float
            Initial value for :attr:`evolved_time`.
        start_trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            Initial truncation error for :attr:`trunc_err`

    Attributes
    ----------
    verbose : int
    options : :class:`~tenpy.tools.params.Config`
        Optional parameters, see :meth:`run` for more details
    trunc_params : truncation parameters (cf. Options)
    compression : ``'SVD' | 'VAR'``
        Specifies by which method the MPS is compressed after each time 
        evolution step.
        For details see :class:`mps_compress`
    evolved_time : float
        Indicating how long `psi` has been evolved, ``psi = exp(-i * evolved_time * H) psi(t=0)``.
    trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
        The error of the represented state which is introduced due to the truncation during
        the sequence of update steps
    psi : :class:`~tenpy.networks.mps.MPS`
        The MPS, time evolved in-place.
    model : :class:`~tenpy.models.model.MPOModel`
        The model defining the Hamiltonian.
    _U : list of :class:`~tenpy.networks.mps.MPO`
        Exponentiated `H_MPO`;
    _U_param : dict
        A dictionary containing the information of the latest created `_U`.
        We won't recalculate `_U` if those parameters didn't change.
    """
    def __init__(self, psi, model, options):
        self.options = options = asConfig(options, "MPO_Evo")
        self.verbose = options.verbose
        self.trunc_params = options.subconfig('trunc_params')
        self.compression = options.get('compression', 'SVD')
        self.psi = psi
        self.model = model
        self.evolved_time = options.get('start_time', 0.)
        self.trunc_err = options.get('start_trunc_err', TruncationError())
        self._U = None
        self._U_param = {}

    def run(self):
        """Time evolution with the WI/WII method.
        
        .. cfg:configoptions :: MPO_Evo
            
            dt : float
                Time step.
            N_steps : int
                Number of time steps `dt` to evolve
            which : 'I' or 'II'
                Specifies which method is applied. Generally, 'II' is
                preferable.
            order : int
                Order of the algorithm. The total error scales as ``O(t*dt^order)``.
                Implemented are order = 1 and order = 2.
            E_offset : float
                Energy offset subtracted from the Hamiltonian.
        """
        dt = self.options.get('dt', 0.01)
        N_steps = self.options.get('N_steps', 1)
        which = self.options.get('which', 'II')
        order = self.options.get('order', 2)
        E_offset = self.options.get('E_offset', None)

        self.calc_U(dt, order, which, E_offset = E_offset)

        self.update(N_steps)

        return self.psi


    def calc_U(self, dt, order = 2, which = 'II', E_offset = None):
        """Calculate ``self._U``
        
        This function calculates the approximation 
        :math: `_U ~= exp(-i dt_ (H - E_offset))` with

        * dt_ = `dt` for ``order = 1``
        * dt_ = (1 - 1j)/2 `dt`  and dt_ = (1 + 1j)/2 `dt` for ``order = 2``

        Parameters
        ----------
        dt : float
            Size of the time-step used in calculating `_U`
        order : int
            1 or 2
        which : 'I' or 'II'
            Type of approximation for the time evolution operator
        E_offset : None | float
            Possible offset subtracted from H for real-time evolution.
            TODO: Implement this.
        """
        U_param = dict(dt=dt, order=order, which=which, E_offset=E_offset)
        if self._U_param == U_param:
            return # nothing to do: _U is cached
        self._U_param = U_param
        if self.verbose >= 1:
            print("Calculate U for ", U_param)

        H_MPO = self.model.H_MPO

        if order == 1:
            U = H_MPO.make_U(dt * -1j, which=which)
            self._U = [U]
        
        elif order == 2:
            U1 = H_MPO.make_U(- (1. + 1j)/2. * dt * 1j, which=which)
            U2 = H_MPO.make_U(- (1. - 1j)/2. * dt * 1j, which=which)
            self._U = [U1, U2]

    def update(self, N_steps):
        """Time evolve by N_steps steps

        Parameters
        ----------
        N_steps: int
            The number of time steps psi is evolved by.

        Returns
        -------
        trunc_err: :class:`~tenpy.algorithms.truncation.TruncationError'
        """
        trunc_err = TruncationError()
        
        for _ in np.arange(N_steps):
            for U_mpo in self._U:
                trunc_err += self.apply_mpo(U_mpo)
                # self.psi = apply_mpo(U_mpo, self.psi, self.trunc_params)
        self.evolved_time = self.evolved_time + N_steps * self._U_param['dt']
        self.trunc_err = self.trunc_err + trunc_err # not += : make a copy!
        # (this is done to avoid problems of users storing self.trunc_err after each `update`)
        return trunc_err

    def apply_mpo(self, U_mpo):
        """Apply time evolution operator(s) to self.psi using
        the compression method `self.compression`.

        The truncation is performed in accordance with `self.trunc_params`.

        Parameters
        ----------
        U_mpo : :class:`~tepy.networks.mpo.MPO`
            Representing the time evolution operator to be applied to `self.psi`.

        Returns
        -------
        trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The error of the represented state which is introduced due to the truncation
            during this update step.
        """
        if self.compression == 'SVD':
            return self.apply_mpo_svd(U_mpo)

        else:
            raise ValueError("Compression method not implemented.")

    def mps_compress(self):
        r"""Takes `self.psi` and compresses it; in place.
        The truncation is performed according to 
        `self.trunc_params`.
        """
        bc = self.psi.bc
        L = self.psi.L
        if bc == 'finite':
            # TODO: could we simply replace this with MPS.canonical_form_finite()?
            # Do QR starting from the left
            B = self.psi.get_B(0, form='Th')
            for i in range(self.psi.L - 1):
                B = B.combine_legs(['vL', 'p'])
                q, r = npc.qr(B, inner_labels=['vR', 'vL'])
                B = q.split_legs()
                self.psi.set_B(i, B, form=None)
                B = self.psi.get_B((i + 1) % L, form='B')
                B = npc.tensordot(r, B, axes=('vR', 'vL'))
            # Do SVD from right to left, truncate the singular values according to trunc_params
            for i in range(self.psi.L - 1, 0, -1):
                B = B.combine_legs(['p', 'vR'])
                u, s, vh, err, norm_new = svd_theta(B, self.trunc_params)
                self.psi.norm *= norm_new
                vh = vh.split_legs()
                self.psi.set_B(i % L, vh, form='B')
                B = self.psi.get_B(i - 1, form=None)
                B = npc.tensordot(B, u, axes=('vR', 'vL'))
                B.iscale_axis(s, 'vR')
                self.psi.set_SL(i % L, s)
            self.psi.set_B(0, B, form='Th')
        if bc == 'infinite':
            for i in range(self.psi.L):
                self.svd_two_site(i, self.psi)
            for i in range(self.psi.L - 1, -1, -1):
                self.svd_two_site(i, self.psi, self.trunc_params)

    @staticmethod
    def svd_two_site(i, mps, trunc_params=None):
        r"""Builds a theta and splits it using svd for an MPS.

        Parameters
        ----------
        i : int
            First site.
        mps : :class:`tenpy.networks.mps.MPS`
            MPS to use on.
        trunc_params : None | dict
        If None no truncation is done. Else dict as in :func:`~tenpy.algorithms.truncation.truncate`.
        """
        # TODO: this is already implemented somewhere else....
        theta = mps.get_theta(i, n=2)
        theta = theta.combine_legs([['vL', 'p0'], ['p1', 'vR']], qconj=[+1, -1])
        if trunc_params is None:
            trunc_params = {'chi_max': 100000, 'svd_min': 1.e-15, 'trunc_cut': 1.e-15}
        u, s, vh, err, renorm = svd_theta(theta, trunc_params)
        mps.norm *= renorm
        u = u.split_legs()
        vh = vh.split_legs()
        u.ireplace_label('p0', 'p')
        vh.ireplace_label('p1', 'p')
        mps.set_B(i, u, form='A')
        mps.set_B((i + 1) % mps.L, vh, form='B')
        mps.set_SR(i, s)

    
    def apply_mpo_svd(self, U_mpo):
        """Applies an MPO and truncates the resulting MPS using SVD in-place.

        Parameters
        ----------
        U_mpo : :class:`~tenpy.networks.mpo.MPO`
            MPO to apply. Usually one of :func:`make_U_I` or :func:`make_U_II`.
            The approximation being made are uncontrolled for other MPOs and infinite bc.

        Returns
        -------
        trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The error of the represented state which is introduced due to the truncation
            during this update step.
        """
        trunc_err = TruncationError()    # TODO: Implement trunc_err
        bc = self.psi.bc
        if bc != U_mpo.bc:
            raise ValueError("Boundary conditions of MPS and MPO are not the same")
        if self.psi.L != U_mpo.L:
            raise ValueError("Length of MPS and MPO not the same")
        Bs = [
            npc.tensordot(self.psi.get_B(i, form='B'), U_mpo.get_W(i), axes=('p', 'p*'))
            for i in range(self.psi.L)
        ]
        if bc == 'finite':
            Bs[0] = npc.tensordot(self.psi.get_theta(0, 1), U_mpo.get_W(0), axes=('p0', 'p*'))
        for i in range(self.psi.L):
            if i == 0 and bc == 'finite':
                Bs[i] = Bs[i].take_slice(U_mpo.get_IdL(i), 'wL')
                Bs[i] = Bs[i].combine_legs(['wR', 'vR'], qconj=[-1])
                Bs[i].ireplace_labels(['(wR.vR)'], ['vR'])
                Bs[i].legs[Bs[i].get_leg_index('vR')] = Bs[i].get_leg('vR').to_LegCharge()
            elif i == self.psi.L - 1 and bc == 'finite':
                Bs[i] = Bs[i].take_slice(U_mpo.get_IdR(i), 'wR')
                Bs[i] = Bs[i].combine_legs(['wL', 'vL'], qconj=[1])
                Bs[i].ireplace_labels(['(wL.vL)'], ['vL'])
                Bs[i].legs[Bs[i].get_leg_index('vL')] = Bs[i].get_leg('vL').to_LegCharge()
            else:
                Bs[i] = Bs[i].combine_legs([['wL', 'vL'], ['wR', 'vR']], qconj=[+1, -1])
                Bs[i].ireplace_labels(['(wL.vL)', '(wR.vR)'], ['vL', 'vR'])
                Bs[i].legs[Bs[i].get_leg_index('vL')] = Bs[i].get_leg('vL').to_LegCharge()
                Bs[i].legs[Bs[i].get_leg_index('vR')] = Bs[i].get_leg('vR').to_LegCharge()

        if bc == 'infinite':
            #calculate good (rather arbitrary) guess for S[0] (no we don't like it either)
            weight = np.ones(U_mpo.get_W(0).shape[U_mpo.get_W(0).get_leg_index('wL')]) * 0.05
            weight[U_mpo.get_IdL(0)] = 1
            weight = weight / np.linalg.norm(weight)
            S = [np.kron(weight, self.psi.get_SL(0))]  # order dictated by '(wL,vL)'
        else:
            S = [np.ones(Bs[0].get_leg('vL').ind_len)]
        #Wrong S values but will be calculated in mps_compress
        for i in range(self.psi.L):
            S.append(np.ones(Bs[i].get_leg('vR').ind_len))

        forms = ['B' for i in range(self.psi.L)]
        if bc == 'finite':
            forms[0] = 'Th'
        self.psi = mps.MPS(self.psi.sites, Bs, S, form=forms, bc=self.psi.bc)
        self.mps_compress()
        return trunc_err


    
