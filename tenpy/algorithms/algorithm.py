"""This module contains some base classes for algorithms."""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

import warnings
import time
import numpy as np
import logging
logger = logging.getLogger(__name__)

from .truncation import TruncationError
from ..tools.events import EventHandler
from ..tools.params import asConfig
from ..tools.cache import DictCache

__all__ = ['Algorithm', 'TimeEvolutionAlgorithm', 'TimeDependentHAlgorithm']


class Algorithm:
    """Base class and common interface for a tensor-network based algorithm in TeNPy.

    Parameters
    ----------
    psi :
        Tensor network to be updated by the algorithm.
    model : :class:`~tenpy.models.model.Model` | None
        Model with the representation of the hamiltonian suitable for the algorithm.
        None for algorithms which don't require a model.
    options : dict-like
        Optional parameters for the algorithm.
        In the online documentation, you can find the correct set of options in the
        :ref:`cfg-config-index`.
    resume_data : None | dict
        Can only be passed as keyword argument.
        By default (``None``) ignored. If a `dict`, it should contain the data returned by
        :meth:`get_resume_data` when intending to continue/resume an interrupted run.
        If it contains `psi`, this takes precedence over the argument `psi`.
    cache : None | :class:`DictCache`
        The cache to be used to reduce memory usage.
        None defaults to a new, trivial :class:`DictCache` which keeps everything in RAM.

    Options
    -------
    .. cfg:config :: Algorithm

        trunc_params : dict
            Truncation parameters as described in :cfg:config:`truncation`.

    Attributes
    ----------
    psi :
        Tensor network to be updated by the algorithm.
    model : :class:`~tenpy.models.model.Model`
        Model with the representation of the hamiltonian suitable for the algorithm.
    options : :class:`~tenpy.tools.params.Config`
        Optional parameters.
    checkpoint : :class:`~tenpy.tools.events.EventHandler`
        An event that the algorithm emits at regular intervalls when it is in a
        "well defined" step, where an intermediate status report, measurements and/or
        interrupting and saving to disk for later resume make sense.
    cache : :class:`DictCache` or subclass
        The cache to be used.
    resume_data : dict
        Data given as parameter `resume_data` and/or to be returned by :meth:`get_resume_data`.
    _resume_psi :
        Possibly a copy of `psi` to be used for :meth:`get_resume_data`.
    """
    def __init__(self, psi, model, options, *, resume_data=None, cache=None):
        self.options = asConfig(options, self.__class__.__name__)
        self.trunc_params = self.options.subconfig('trunc_params')
        self.psi = psi
        self.model = model
        if resume_data is None:
            resume_data = {}
        elif 'psi' in resume_data:
            self.psi = resume_data['psi']
        self.resume_data = resume_data
        if cache is None:
            cache = DictCache.trivial()
        self.cache = cache
        self.checkpoint = EventHandler("algorithm")
        self._resume_psi = None

    @classmethod
    def switch_engine(cls, other_engine, *, options=None, **kwargs):
        """Initialize algorithm from another algorithm instance of a different class.

        You can initialize one engine from another, not too different subclasses.
        Internally, this function calls :meth:`get_resume_data` to extract data from the
        `other_engine` and then initializes the new class.

        Note that it transfers the data **without** making copies in most case; even the options!
        Thus, when you call `run()` on one of the two algorithm instances, it will modify the
        state, environment, etc. in the other.
        We recommend to make the switch as ``engine = OtherSubClass.switch_engine(engine)``
        directly replacing the reference.

        Parameters
        ----------
        cls : class
            Subclass of :class:`Algorithm` to be initialized.
        other_engine : :class:`Algorithm`
            The engine from which data should be transfered. Another, but not too different
            algorithm subclass-class; e.g. you can switch from the
            :class:`~tenpy.algorithms.dmrg.TwoSiteDMRGEngine` to the
            :class:`~tenpy.algorithms.dmrg.OneSiteDMRGEngine`.
        options : None | dict-like
            If not None, these options are used for the new initialization.
            If None, take the options from the `other_engine`.
        **kwargs :
            Further keyword arguments for class initialization.
            If not defined, `resume_data` is collected with :meth:`get_resume_data`.
        """
        # If `resume_data` is defined in the kwargs, use that.
        # This allows subclasses to overwritinstead of calling :meth:`get_resume_data`.
        if 'resume_data' not in kwargs:
            kwargs['resume_data'] = other_engine.get_resume_data()
        if options is None:
            options = other_engine.options
        kwargs.setdefault('cache', other_engine.cache)
        obj = cls(other_engine.psi, other_engine.model, options, **kwargs)
        obj.checkpoint = other_engine.checkpoint  # TODO: do this?
        return obj

    @property
    def verbose(self):
        warnings.warn(
            "verbose is deprecated, we're using logging now! \n"
            "See https://tenpy.readthedocs.io/en/latest/intro/logging.html", FutureWarning, 2)
        return self.options.get('verbose', 1.)

    def run(self):
        """Actually run the algorithm.

        Needs to be implemented in subclasses.
        """
        raise NotImplementedError("Sublcasses should implement this.")

    def resume_run(self):
        """Resume a run that was interrupted.

        In case we saved an intermediate result at a :class:`checkpoint`, this function
        allows to resume the :meth:`run` of the algorithm (after re-initialization with the
        `resume_data`).
        Since most algorithms just have a while loop with break conditions,
        the default behaviour implemented here is to just call :meth:`run`.
        """
        if self._resume_psi is not None:
            self.psi = self._resume_psi
            self._resume_psi = None
        return self.run()

    def get_resume_data(self, sequential_simulations=False):
        """Return necessary data to resume a :meth:`run` interrupted at a checkpoint.

        At a :attr:`checkpoint`, you can save :attr:`psi`, :attr:`model` and :attr:`options`
        along with the data returned by this function.
        When the simulation aborts, you can resume it using this saved data with::

            eng = AlgorithmClass(psi, model, options, resume_data=resume_data)
            eng.resume_run()

        An algorithm which doesn't support this should override `resume_run` to raise an Error.

        Parameters
        ----------
        sequential_simulations : bool
            If True, return only the data for re-initializing a sequential simulation run,
            where we "adiabatically" follow the evolution of a ground state (for variational
            algorithms), or do series of quenches (for time evolution algorithms);
            see :func:`~tenpy.simulations.simulation.run_seq_simulations`.

        Returns
        -------
        resume_data : dict
            Dictionary with necessary data (apart from copies of `psi`, `model`, `options`)
            that allows to continue the simulation from where we are now.
            It might contain an explicit copy of `psi`.
        """
        psi = self._resume_psi
        if psi is not None:
            return {'psi': psi}
        else:
            return {'psi': self.psi}


class TimeEvolutionAlgorithm(Algorithm):
    """Common interface for (real) time evolution algorithms.

    Parameters are the same as for :class:`Algorithm`.

    Options
    -------
    .. cfg:config :: TimeEvolutionAlgorithm
        :include: Algorithm

        start_time : float
            Initial value for :attr:`evolved_time`.
        dt : float
            Minimal time step by which to evolve.
        N_steps : int
            Number of time steps `dt` to evolve by in :meth:`run`.
            Adjusting `dt` and `N_steps` at the same time allows to keep the evolution time
            done in :meth:`run` fixed.
            Further, e.g., the Trotter decompositions of order > 1 are slightly more efficient
            if more than one step is performed at once.
        preserve_norm : bool
            Whether the state will be normalized to its initial norm after each time step.

    Attributes
    ----------
    evolved_time : float | complex
        Indicating how long `psi` has been evolved, ``psi = exp(-i * evolved_time * H) psi(t=0)``.
        Not that the real-part of `t` is increasing for a real-time evolution,
        while the imaginary-part of `t` is *decreasing* for a imaginary time evolution.
    """
    time_dependent_H = False  #: whether the algorithm supports time-dependent H

    def __init__(self, psi, model, options, **kwargs):
        super().__init__(psi, model, options, **kwargs)
        self.evolved_time = self.options.get('start_time', 0.)
        self.trunc_err = self.options.get('start_trunc_err', TruncationError())
        self.force_prepare_evolve = False
        if self.resume_data:
            self.evolved_time = self.resume_data['evolved_time']

    def get_resume_data(self, sequential_simulations=False):
        data = super().get_resume_data(sequential_simulations)
        data['evolved_time'] = self.evolved_time
        return data

    def run(self):
        """Perform a (real-)time evolution of :attr:`psi` by `N_steps` * `dt`.

        You probably want to call this in a loop along with measurements.
        The recommended way to do this is via the
        :class:`~tenpy.simulations.time_evolution.RealTimeEvolution`.
        """
        dt = self.options.get('dt', 0.1)
        N_steps = self.options.get('N_steps', 1)

        start_time = time.time()
        Sold = np.mean(self.psi.entanglement_entropy())

        self.run_evolution(N_steps, dt)

        S = self.psi.entanglement_entropy()
        logger.info(
            "--> time=%(t)3.3f, max(chi)=%(chi)d, max(S)=%(S).5f, "
            "avg DeltaS=%(dS).4e, since last update: %(wall_time).1fs", {
                't': self.evolved_time.real,
                'chi': max(self.psi.chi),
                'S': max(S),
                'dS': np.mean(S) - Sold,
                'wall_time': time.time() - start_time,
            })
        return self.psi

    def run_evolution(self, N_steps, dt):
        """Perform a (real-)time evolution of :attr:`psi` by `N_steps` * `dt`.

        This is the inner part of :meth:`run` without the logging.
        For parameters see :cfg:config:`TimeEvolutionAlgorithm`.
        """
        preserve_norm = self.options.get('preserve_norm', None)
        if preserve_norm is None:  # default: preserve norm for real time evolution
            preserve_norm = not np.iscomplex(dt)
        if preserve_norm:
            old_norm = self.psi.norm

        self.prepare_evolve(dt)
        trunc_err = self.evolve(N_steps, dt)

        self.trunc_err = self.trunc_err + trunc_err  # not += : make a copy!
        if preserve_norm:
            self.psi.norm = old_norm

    def prepare_evolve(self, dt):
        """Prepare an evolution step.

        This method is used to prepare repeated calls of :meth:`evolve` given the :attr:`model`.
        For example, it may generate approximations of ``U=exp(-i H dt)``.
        To avoid overhead, it may cache the result depending on parameters/options;
        but it should always regenerate it if :attr:`force_prepare_evolve` is set.

        Parameters
        ----------
        dt : float
            The time step to be used.
        """
        # this function can e.g. calculate an approximation
        raise NotImplementedError("Sublcasses should implement this.")

    def evolve(self, N_steps, dt):
        """Evolve by N_steps*dt.

        Subclasses may override this with a more efficient way of do `N_steps` `update_step`.

        Parameters
        ----------
        N_steps : int
            The number of time steps by `dt` to take at once.
        dt : float
            Small time step. Might be ignored if already used in :meth:`prepare_update`.

        Returns
        -------
        trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            Sum of truncation errors introduced during evolution.
        """
        trunc_err = TruncationError()

        for _ in range(N_steps):
            trunc_err += self.evolve_step(dt)

        self.evolved_time = self.evolved_time + N_steps * dt
        # (this is done to avoid problems of users storing self.trunc_err after each `update`)
        return trunc_err

    def evolve_step(self, dt):
        raise NotImplementedError("Subclasses should implement this.")


class TimeDependentHAlgorithm(TimeEvolutionAlgorithm):
    r"""Time evolution under a time dependent Hamiltonian.

    TimeEvolutionAlgorithm subclasses approximate the evolution by many small time steps of `dt`.
    If we have a time-dependent Hamiltonian ``H(t)``, we can to **first order** in dt approximate
    the evolution by just updating ``H(t)`` after each time step, keeping it constant during the
    update step, i.e., we approximate as follows:

    .. math ::
        U(t_0, t) = T{exp(-i int_{t_0}^t ds H(s))}
             \approx prod_{i=0}^{N-1} exp(-i \Delta t H(t_0 + i*\Delta t))
             \textrm{ where } \Delta t = (t-t_0) / N

    .. note ::
        Even if the algorithm approximation for :math:`exp(-i \Delta t H(t_0 + i*\Delta t))`
        might be precise to higher order in dt, the approximation of the time dependence
        (and hence the overall scaling of the error with `dt`) is only correct to first order!
        Yet, if the time dependence of H is weak, it might still be better to use order > 1.

    .. todo ::
        This is still under development and lacks rigorous tests.
    """
    time_dependent_H = True

    def __init__(self, psi, model, options, **kwargs):
        super().__init__(psi, model, options, **kwargs)
        self.reinit_model()  # can have non-trivial initial `evolved_time`
        # e.g when starting from checkpoint or for sequential runs

    def run_evolution(self, N_steps, dt):
        """Run the time evolution for N_steps * dt.

        Updates the model after each time step `dt` to account for changing H(t).
        For parameters see :cfg:config:`TimeEvolutionAlgorithm`.
        """
        preserve_norm = self.options.get('preserve_norm', None)
        if preserve_norm is None:  # default: preserve norm for real time evolution
            preserve_norm = not np.iscomplex(dt)
        if preserve_norm:
            old_norm = self.psi.norm

        trunc_err = TruncationError()
        # explicit loop over N_steps updating H after each step!
        for _ in range(N_steps):
            self.prepare_evolve(dt)
            trunc_err += self.evolve(1, dt)  # update changes self.evolved_time

            self.reinit_model()


        if preserve_norm:
            self.psi.norm = old_norm

    def reinit_model(self):
        """Re-initialize a new :attr:`model` at current :attr:`evolved_time`.

        Skips re-initialization if the ``model.options['time']`` is the same as `evolved_time`.
        The model should read out the option ``'time'`` and initialize the corresponding ``H(t)``.
        """
        model_time = self.model.options.get('time', None)
        if model_time is not None and model_time == self.evolved_time:
            return  # already had that time defined during model init, so no need to update
        self.model = self.model.update_time_parameter(self.evolved_time)
        self.force_prepare_evolve = True
