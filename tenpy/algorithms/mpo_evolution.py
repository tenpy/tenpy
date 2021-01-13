"""Time evolution using the WI or WII approximation of the time evolution operator."""

# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import time
from scipy.linalg import expm

from .algorithm import Algorithm
from ..linalg import np_conserved as npc
from .truncation import TruncationError
from ..tools.params import asConfig

__all__ = ['ExpMPOEvolution']


class ExpMPOEvolution(Algorithm):
    """Time evolution of an MPS using the W_I or W_II approximation for ``exp(H dt)``.

    :cite:`zaletel2015` described a method to obtain MPO approximations :math:`W_I` and
    :math:`W_{II}` for the exponential ``U = exp(i H dt)`` of an MPO `H`, implemented in
    :meth:`~tenpy.networks.mpo.MPO.make_U_I` and :meth:`~tenpy.networks.mpo.MPO.make_U_II`.
    This class uses it for real-time evolution.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        Initial state to be time evolved. Modified in place.
    model : :class:`~tenpy.models.model.MPOModel`
        The model representing the Hamiltonian which we want to
        time evolve psi with.
    options : dict
        Further optional parameters are described in :cfg:config:`ExpMPOEvolution`.

    Options
    -------
    .. cfg:config :: ExpMPOEvolution
        :include: ApplyMPO

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
        super().__init__(psi, model, options)
        options = self.options
        self.verbose = options.verbose
        self.evolved_time = options.get('start_time', 0.)
        self.trunc_err = options.get('start_trunc_err', TruncationError())
        self._U_MPO = None
        self._U_param = {}
        options.setdefault('start_env_sites', model.H_MPO.max_range)

    def run(self):
        """Run the real-time evolution with the WI/WII approximation.

        Options
        -------
        .. cfg:configoptions :: ExpMPOEvolution

            dt : float
                Time step.
            N_steps : int
                Number of time steps `dt` to evolve
            approximation : 'I' or 'II'
                Specifies which approximation is applied. The default 'II' is more precise.
                See :cite:`zaletel2015` and :meth:`~tenpy.networks.mps.MPO.make_U` for more details.
            order : int
                Order of the algorithm. The total error scales as ``O(t*dt^order)``.
                Implemented are order = 1 and order = 2.
        """
        dt = self.options.get('dt', 0.01)
        N_steps = self.options.get('N_steps', 1)
        approximation = self.options.get('approximation', 'II')
        order = self.options.get('order', 2)

        self.calc_U(dt, order, approximation)

        self.update(N_steps)

        return self.psi

    def calc_U(self, dt, order=2, approximation='II'):
        """Calculate ``self._U_MPO``

        This function calculates the approximation ``U ~= exp(-i dt_ H)`` with
        ``dt_ = dt` for ``order=1``, or
        ``dt_ = (1 - 1j)/2 dt`` and ``dt_ = (1 + 1j)/2 dt`` for ``order=2``.

        Parameters
        ----------
        dt : float
            Size of the time-step used in calculating `_U`
        order : int
            1 or 2
        approximation : 'I' or 'II'
            Type of approximation for the time evolution operator.
        """
        U_param = dict(dt=dt, order=order, approximation=approximation)
        if self._U_param == U_param:
            return  # nothing to do: _U is cached
        self._U_param = U_param
        if self.verbose >= 1:
            print("Calculate U for ", U_param)

        H_MPO = self.model.H_MPO
        if order == 1:
            U_MPO = H_MPO.make_U(dt * -1j, approximation=approximation)
            self._U_MPO = [U_MPO]
        elif order == 2:
            U1 = H_MPO.make_U(-(1. + 1j) / 2. * dt * 1j, approximation=approximation)
            U2 = H_MPO.make_U(-(1. - 1j) / 2. * dt * 1j, approximation=approximation)
            self._U_MPO = [U1, U2]
        else:
            raise ValueError("order {0:d} not implemented".format(order=order))

    def update(self, N_steps):
        """Time evolve by `N_steps` steps.

        Parameters
        ----------
        N_steps: int
            The number of time steps psi is evolved by.

        Returns
        -------
        trunc_err: :class:`~tenpy.algorithms.truncation.TruncationError`
            Truncation error induced during the update.
        """
        trunc_err = TruncationError()

        for _ in np.arange(N_steps):
            for U_MPO in self._U_MPO:
                trunc_err += U_MPO.apply(self.psi, self.options)
        self.evolved_time = self.evolved_time + N_steps * self._U_param['dt']
        self.trunc_err = self.trunc_err + trunc_err  # not += : make a copy!
        # (this is done to avoid problems of users storing self.trunc_err after each `update`)
        return trunc_err
