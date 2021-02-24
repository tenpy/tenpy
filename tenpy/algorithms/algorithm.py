"""This module contains some base classes for algorithms."""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

import warnings

from ..tools.events import EventHandler
from ..tools.params import asConfig

__all__ = ['Algorithm', 'TimeEvolutionAlgorithm']


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
    """
    def __init__(self, psi, model, options, *, resume_data=None):
        self.options = asConfig(options, self.__class__.__name__)
        self.trunc_params = self.options.subconfig('trunc_params')
        self.psi = psi
        self.model = model
        self.resume_data = resume_data
        self.checkpoint = EventHandler("algorithm")

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
        self.run()

    def get_resume_data(self):
        """Return necessary data to resume a :meth:`run` interrupted at a checkpoint.

        At a :attr:`checkpoint`, you can save :attr:`psi`, :attr:`model` and :attr:`options`
        along with the data returned by this function.
        When the simulation aborts, you can resume it using this saved data with::

            eng = AlgorithmClass(psi, model, options, resume_data=resume_data)
            eng.resume_run(resume_data)

        An algorithm which doesn't support this should override `resume_run` to raise an Error.

        Returns
        -------
        resume_data : dict
            Dictionary with necessary data (apart from copies of `psi`, `model`, `options`)
            that allows to continue the simulation from where we are now.
        """
        return {}


class TimeEvolutionAlgorithm(Algorithm):
    """Common interface for (real) time evolution algorithms.

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
        By default (``None``) ignored. If a `dict`, it should contain the data returned by
        :meth:`get_resume_data` when intending to continue/resume an interrupted run.

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

    Attributes
    ----------
    evolved_time : float | complex
        Indicating how long `psi` has been evolved, ``psi = exp(-i * evolved_time * H) psi(t=0)``.
        Not that the real-part of `t` is increasing for a real-time evolution,
        while the imaginary-part of `t` is *decreasing* for a imaginary time evolution.
    """
    def __init__(self, psi, model, options, *, resume_data=None):
        super().__init__(psi, model, options, resume_data=resume_data)
        self.evolved_time = self.options.get('start_time', 0.)
        if resume_data is not None:
            self.evolved_time = resume_data['evolved_time']

    def get_resume_data(self):
        data = super().get_resume_data()
        data['evolved_time'] = self.evolved_time
        return data

    def run(self):
        """Perform a real-time evolution of :attr:`psi` by `N_steps`*`dt`.

        You probably want to call this in a loop.
        """
        raise NotImplementedError("Subclasses should override this")
