"""This module contains some base classes for algorithms."""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

from ..tools.events import EventHandler
from ..tools.params import asConfig

__all__ = ['Algorithm']


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
    verbose : float
        Level of verboseness, higher=more output.
    """
    def __init__(self, psi, model, options):
        self.options = asConfig(options, self.__class__.__name__)
        self.trunc_params = self.options.subconfig('trunc_params')
        self.psi = psi
        self.model = model
        self.checkpoint = EventHandler("algorithm")
        self.verbose = self.options.verbose

    def run(self):
        """Actually run the algorithm.

        Needs to be implemented in subclasses.
        """
        raise NotImplementedError("Sublcasses should implement this.")

    def resume_run(self):
        """Resume a run that was interrupted.

        In case we saved an intermediate result at a :class:`checkpoint`, this function
        allows to resume the :meth:`run` of the algorithm.
        Since most algorithms just have a while loop with break conditions,
        the default behaviour implemented here is to just call :meth:`run`.
        """
        self.run()
