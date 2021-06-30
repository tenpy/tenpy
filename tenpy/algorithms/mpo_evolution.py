"""Time evolution using the WI or WII approximation of the time evolution operator."""

# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import time
from scipy.linalg import expm
import logging
logger = logging.getLogger(__name__)

from .algorithm import TimeEvolutionAlgorithm
from ..linalg import np_conserved as npc
from .truncation import TruncationError
from ..tools.params import asConfig

__all__ = ['ExpMPOEvolution', 'TimeDependentExpMPOEvolution']


class ExpMPOEvolution(TimeEvolutionAlgorithm):
    """Time evolution of an MPS using the W_I or W_II approximation for ``exp(H dt)``.

    :cite:`zaletel2015` described a method to obtain MPO approximations :math:`W_I` and
    :math:`W_{II}` for the exponential ``U = exp(i H dt)`` of an MPO `H`, implemented in
    :meth:`~tenpy.networks.mpo.MPO.make_U_I` and :meth:`~tenpy.networks.mpo.MPO.make_U_II`.
    This class uses it for real-time evolution.

    Parameters are the same as for :class:`~tenpy.algorithms.algorithm.Algorithm`.

    Options
    -------
    .. cfg:config :: ExpMPOEvolution
        :include: ApplyMPO, TimeEvolutionAlgorithm

        start_trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            Initial truncation error for :attr:`trunc_err`
        approximation : 'I' | 'II'
            Specifies which approximation is applied. The default 'II' is more precise.
            See :cite:`zaletel2015` and :meth:`~tenpy.networks.mpo.MPO.make_U`
            for more details.
        order : int
            Order of the algorithm. The total error up to time `t` scales as ``O(t*dt^order)``.
            Implemented are order = 1 and order = 2.

    Attributes
    ----------
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
    def __init__(self, psi, model, options, **kwargs):
        super().__init__(psi, model, options, **kwargs)
        options = self.options
        self.evolved_time = options.get('start_time', 0.)
        self.trunc_err = options.get('start_trunc_err', TruncationError())
        self._U_MPO = None
        self._U_param = {}

    def run(self):
        """Run the real-time evolution with the W_I/W_II approximation.  """
        dt = self.options.get('dt', 0.01)
        N_steps = self.options.get('N_steps', 1)
        approximation = self.options.get('approximation', 'II')
        order = self.options.get('order', 2)

        self.calc_U(dt, order, approximation)

        self.update(N_steps)

        return self.psi

    def calc_U(self, dt, order=2, approximation='II'):
        """Calculate ``self._U_MPO``.

        This function calculates the approximation ``U ~= exp(-i dt_ H)`` with
        ``dt_ = dt` for ``order=1``, or
        ``dt_ = (1 - 1j)/2 dt`` and ``dt_ = (1 + 1j)/2 dt`` for ``order=2``.

        Parameters
        ----------
        dt : float
            Size of the time-step used in calculating `_U`
        order : int
            The order of the algorithm. Only 1 and 2 are allowed.
        approximation : 'I' or 'II'
            Type of approximation for the time evolution operator.
        """
        U_param = dict(dt=dt, order=order, approximation=approximation)
        if self._U_param == U_param:
            return  # nothing to do: _U is cached
        self._U_param = U_param
        logger.info("Calculate U for %s", U_param)

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

        for _ in range(N_steps):
            for U_MPO in self._U_MPO:
                trunc_err += U_MPO.apply(self.psi, self.options)
        self.evolved_time = self.evolved_time + N_steps * self._U_param['dt']
        self.trunc_err = self.trunc_err + trunc_err  # not += : make a copy!
        # (this is done to avoid problems of users storing self.trunc_err after each `update`)
        return trunc_err


class TimeDependentExpMPOEvolution(ExpMPOEvolution):
    """Variant of :class:`ExpMPOEvolution` that can handle time-dependent hamiltonians.

    As of now, it only supports first :cfg:option:`ExpMPOEvolution.order` with a very basic
    implementation, that just reinitializes the model after each time evolution steps with an
    updated model parameter `time` set to :attr:`evolved_time`.
    The model class should read that parameter.

    .. todo ::
        This is still under development and lacks rigorous tests.
    """
    time_dependent_H = True

    def run(self):
        N_steps = self.options.get('N_steps', 1)
        self.update(N_steps)
        return self.psi

    def update(self, N_steps):
        dt = self.options.get('dt', 0.01)
        approximation = self.options.get('approximation', 'II')
        order = self.options.get('order', 1)

        trunc_err = TruncationError()
        for _ in range(N_steps):
            self.calc_U(dt, order, approximation)
            for U_MPO in self._U_MPO:
                trunc_err += U_MPO.apply(self.psi, self.options)
            self.evolved_time = self.evolved_time + dt
            self.model = self.reinit_model()  # use the updated model for the next measurement!
        self.trunc_err = self.trunc_err + trunc_err  # not += : make a copy!
        # (this is done to avoid problems of users storing self.trunc_err after each `update`)
        return trunc_err

    def calc_U(self, dt, order, approximation):
        U_param = dict(dt=dt, order=order, approximation=approximation, time=self.evolved_time)
        if self._U_param == U_param:
            return  # nothing to do: _U is cached
        self._U_param = U_param
        logger.info("Calculate U for %s", U_param)

        if order != 1:
            raise NotImplementedError("order > 1 with time-dependent H requires re-derivation")
        U_MPO = self.model.H_MPO.make_U(dt * -1j, approximation=approximation)
        self._U_MPO = [U_MPO]

    def reinit_model(self):
        """Re-initialize a new model at current time.

        Returns
        -------
        model :
            New instance of the model initialized at ``model_params['time'] = self.evolved_time``.
        """
        cls = self.model.__class__
        model_params = self.model.options  # if you get an error, set this in your custom model
        model_params['time'] = self.evolved_time
        return cls(model_params)
