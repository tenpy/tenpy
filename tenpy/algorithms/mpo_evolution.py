"""Time evolution using the WI or WII approximation of the time evolution operator."""

# Copyright (C) TeNPy Developers, Apache license

import logging

logger = logging.getLogger(__name__)

from .algorithm import TimeEvolutionAlgorithm, TimeDependentHAlgorithm
from ..linalg.truncation import TruncationError
from ..tools.misc import consistency_check

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

        approximation : 'I' | 'II'
            Specifies which approximation is applied. The default 'II' is more precise.
            See :cite:`zaletel2015` and :meth:`~tenpy.networks.mpo.MPO.make_U`
            for more details.
        order : int
            Order of the algorithm. The total error up to time `t` scales as ``O(t*dt^order)``.
            Implemented are order = 1 and order = 2.
        max_dt : float | None
            Threshold for raising errors on too large time steps. Default ``1.0``.
            See :meth:`~tenpy.tools.misc.consistency_check`.
            The trotterization in the time evolution operator assumes that the time step is small.
            We raise an error if it is not.
            Can be downgraded to a warning by setting this option to ``None``.

    Attributes
    ----------
    _U : list of :class:`~tenpy.networks.mps.MPO`
        Exponentiated `H_MPO`;
    _U_param : dict
        A dictionary containing the information of the latest created `_U`.
        We won't recalculate `_U` if those parameters didn't change.
    """
    def __init__(self, psi, model, options, **kwargs):
        super().__init__(psi, model, options, **kwargs)
        options = self.options
        self._U_MPO = None
        self._U_param = {}

    # run from TimeEvolutionAlgorithm

    def prepare_evolve(self, dt):
        order = self.options.get('order', 2, int)
        approximation = self.options.get('approximation', 'II', str)

        self.calc_U(dt, order, approximation)

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
        if self._U_param == U_param and not self.force_prepare_evolve:
            return  # nothing to do: _U is cached
        self._U_param = U_param
        logger.info("Calculate U for %s", U_param)
        consistency_check(dt, self.options, 'max_dt', 1.,
                          'delta_t > ``max_delta_t`` is unreasonably large for trotterization.')
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
        self.force_prepare_evolve = False

    def evolve_step(self, dt):
        trunc_err = TruncationError()
        for U_MPO in self._U_MPO:
            trunc_err += U_MPO.apply(self.psi, self.options)
        return trunc_err


class TimeDependentExpMPOEvolution(TimeDependentHAlgorithm,ExpMPOEvolution):
    """Variant of :class:`ExpMPOEvolution` that can handle time-dependent hamiltonians.

    See details in :class:`~tenpy.algorithms.algorithm.TimeDependentHAlgorithm` as well.
    """
    # uses run from TimeDependentHAlgorithm
    # so nothing to redefine here
