"""This module contains base classes for simulations.

The :class:`Simulation` class tries to put everything needed for a simulation in a structured form
and collects task like initializing the tensor network state, model and algorithm classes,
running the actual algorithm, possibly performing measurements and saving the results.

See :doc:`/intro/simulations` for an overview and
:doc:`/examples` for a list of example parameter yaml files.
"""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

import os
import sys
from pathlib import Path
import time
import importlib
import warnings
import functools
import numpy as np
import logging
import copy

from ..models.model import Model
from ..algorithms.algorithm import Algorithm
from ..networks.mps import InitialStateBuilder
from ..models.model import NearestNeighborModel
from ..tools import hdf5_io
from ..tools.cache import CacheFile
from ..tools.params import asConfig
from ..tools.events import EventHandler
from ..tools.misc import find_subclass, update_recursive, get_recursive, set_recursive
from ..tools.misc import setup_logging as setup_logging_
from .. import version
from .measurement import (measurement_wrapper, _m_psi_method, _m_psi_method_wrapped,
                          _m_model_method, _m_model_method_wrapped)

__all__ = [
    'Simulation',
    'Skip',
    'init_simulation',
    'run_simulation',
    'init_simulation_from_checkpoint',
    'resume_from_checkpoint',
    'run_seq_simulations',
    'output_filename_from_dict',
]


class Simulation:
    """Base class for simulations.

    The prefered way to run simulations is in a `with` statement, which allows us to redirect
    error messages to the log files, timely warn about unused parameters and to properly close any
    open files. In other words, use the simulation class like this::

        with Simulation(options, ...) as sim:
            results = sim.run()

    The wrappers :func:`run_simulation` and :func:`run_seq_simulations` do that.

    Parameters
    ----------
    options : dict-like
        The simulation parameters as outlined below.
        Ideally, these options should be enough to fully specify all parameters of a simulation
        to ensure reproducibility.
    setup_logging : bool
        Whether to call :meth:`setup_logging` at the beginning of initialization.
    resume_data : None | dict
        Ignored if None. If a dictionary, it should contain the data for resuming the simulation,
        ``results['resume_data']`` (see :attr:`results`).
        Note that the dict is cleared after readout to allow freeing memory.

    Options
    -------
    .. cfg:config :: Simulation

        directory : str
            If not None (default), switch to that directory at the beginning of the simulation.
        log_params : dict
            Log parameters; see :cfg:config:`log`.
        overwrite_output : bool
            Whether an exisiting file may be overwritten.
            Otherwise, if the file already exists we try to replace
            ``filename.ext`` with ``filename_01.ext`` (and further increasing numbers).
        random_seed : int | None
            If not ``None``, initialize the (legacy) numpy random generator with the given seed.
            **Note** that models have their own :attr:`~tenpy.models.model.Model.rng` with
            a separate (default) :cfg:option:`CouplingMPOModel.random_seed` in the `model_params`.
            If this `random_seed` is set, we call
            ``model_params('random_seed', random_seed + 123456)``
        sequential : dict
            Parameters for running simulations sequentially, see :cfg:config:`sequential`.
            Ignored by the simulation itself, but used by :func:`run_seq_simulations` and
            :func:`resume_from_checkpoint` to run a whole sequence of simulations passing on the
            state (and possible more).

    Attributes
    ----------
    options : :class:`~tenpy.tools.params.Config`
        Simulation parameters.
    model : :class:`~tenpy.models.model.Model`
        The model to be simulated.
    psi :
        The tensor network state updated by the algorithm.
    engine :
        The engine of the algorithm.
    results : dict
        Collection of all the results to be saved in the end.
        In a standard simulation, it will have the following entries.

        simulation_parameters: nested dict
            The simulation parameters passed as `options`.
        version_info : dict
            Information of the used library/code versions and simulation class.
            See :meth:`get_version_info`.
        finished_run : bool
            Usefull to check whether the output file finished or was generated at a checkpoint.
            This flag is set to `True` only right at the end of :meth:`run`
            (or :meth:`resume_run`) before saving.
        measurements : dict
            Data of all the performed measurements.
        psi :
            The final tensor network state.
            Only included if :cfg:option:`Simulation.save_psi` is True (default).
        resume_data : dict
            Additional data for resuming the algorithm run.
            Not part of `self.results`, but only added in :meth:`prepare_results_for_save` with
            the most up-to-date `resume_data` from
            :meth:`~tenpy.algorithms.algorithm.Algorithm.get_resume_data`.
            Only included if :cfg:option:`Simultion.save_resume_data` is True.
            Note that this contains anoter (reference or even copy of) `psi`.

    cache : :class:`~tenpy.tools.cache.DictCache`
        Cache that can be used by algorithms.
    measurement_event : :class:`~tenpy.tools.events.EventHandler`
        An event that gets emitted each time when measurements should be performed.
        The callback functions should take :attr:`psi`, the simulation class itself,
        and a dictionary `results` as arguments.
        They should directly write the results into that dictionary.
    output_filename : str
        Filename for output.
    _backup_filename : str
        When writing a file a second time, instead of simply overwriting it, move it to there.
        In that way, we still have a non-corrupt version if something fails during saving.
    _init_walltime : float
        Walltime at initialization of the simulation class.
        Used as reference point in :meth:`walltime`.
    _last_save : float
        Time of the last call to :meth:`save_results`, initialized to :attr:`_init_walltime`.
    loaded_from_checkpoint : bool
        True when the simulation is loaded with :meth:`from_saved_checkpoint`.
    grouped : int
        By how many sites we grouped in :meth:`group_sites_for_algorithm`.
    model_ungrouped :
        Only set if `grouped` > 1. In that case, :attr:`model` is the modified/grouped model,
        and `model_ungrouped` is the original ungrouped model.
    """
    #: name of the default algorithm `engine` class
    default_algorithm = 'TwoSiteDMRGEngine'

    #: tuples as for :cfg:option:`Simulation.connect_measurements` that get added if
    #: the :cfg:option:`Simulation.use_default_measurements` is True.
    default_measurements = [
        ('tenpy.simulations.measurement', 'm_measurement_index', {}, 1),
        ('tenpy.simulations.measurement', 'm_bond_dimension'),
        ('tenpy.simulations.measurement', 'm_energy_MPO'),
        ('tenpy.simulations.measurement', 'm_entropy'),
    ]

    #: logger : An instance of a logger; see :doc:`/intro/logging`. NB: class attribute.
    logger = logging.getLogger(__name__ + ".Simulation")

    def __init__(self, options, *, setup_logging=True, resume_data=None):
        self._init_walltime = time.time()
        if not hasattr(self, 'loaded_from_checkpoint'):
            self.loaded_from_checkpoint = False
        self.options = options  # delay conversion to Config: avoid logging before setup_logging
        cwd = self.options.setdefault("directory", None)
        if cwd is not None:
            if not os.path.exists(cwd):
                os.mkdir(cwd)
            os.chdir(cwd)
        self.fix_output_filenames()
        if setup_logging:
            log_params = self.options.setdefault('log_params', {})
            if 'logging_params' in self.options:
                # when you remove this if clause, also clean up the 'logging_params' from the
                # self.options.touch(..., 'logging_params') below
                warnings.warn("Renamed `logging_params` to `log_params` for simulation.",
                              FutureWarning, 2)
                log_params = self.options['logging_params']
            setup_logging_(**log_params, output_filename=self.output_filename)
        # now that we have logging running, catch up with log messages
        self.logger.info("new simulation\n%s\n%s\n%s", "=" * 80, self.__class__.__name__, "=" * 80)
        self.options = asConfig(self.options, self.__class__.__name__)
        self.options.touch('directory', 'output_filename', 'output_filename_params',
                           'overwrite_output', 'skip_if_output_exists', 'safe_write', 'log_params',
                           'logging_params')
        if cwd is not None:
            self.logger.info("change directory to %s", cwd)  # os.chdir(cwd) above
        self.logger.info("output filename: %s", self.output_filename)

        random_seed = self.options.get('random_seed', None)
        if random_seed is not None:
            if self.loaded_from_checkpoint:
                warnings.warn("resetting `random_seed` for a simulation loaded from checkpoint."
                              "Depending on where you use random numbers, "
                              "this might or might not be what you want!")
            np.random.seed(random_seed)
            self.options.subconfig('model_params').setdefault('random_seed', random_seed + 123456)
        self.results = {
            'simulation_parameters': self.options,
            'version_info': self.get_version_info(),
            'finished_run': False,
        }
        self._last_save = time.time()
        self.measurement_event = EventHandler("psi, simulation, model, results")
        if resume_data is not None:
            if 'psi' in resume_data:
                self.psi = resume_data['psi']
            if 'model' in resume_data:  # usually not: we can cheaply regenerate a model
                self.model = resume_data['model']
            self.results['resume_data'] = resume_data
        self.options.touch('sequential')  # added by :func:`run_seq_simulations` for completeness
        self.cache = CacheFile.open()
        self.grouped = 1

    def __enter__(self):
        self.init_cache()
        self.cache = self.cache.__enter__()  # start cache context
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cache.__exit__(exc_type, exc_value, traceback)  # exit cache context
        if exc_type is not None:
            self.logger.exception("simulation abort with the following exception",
                                  exc_info=(exc_type, exc_value, traceback))
        self.options.warn_unused(True)

    @property
    def verbose(self):
        warnings.warn(
            "verbose is deprecated, we're using logging now! \n"
            "See https://tenpy.readthedocs.io/en/latest/intro/logging.html", FutureWarning, 2)
        return self.options.get('verbose', 1.)

    def run(self):
        """Run the whole simulation.

        Returns
        -------
        results : dict
            The :attr:`results` as returned by :meth:`prepare_results_for_save`.
        """
        if self.loaded_from_checkpoint:
            warnings.warn("called `run()` on a simulation loaded from checkpoint. "
                          "You should probably call `resume_run()` instead!")
        self.init_model()
        self.init_state()
        self.group_sites_for_algorithm()
        self.init_algorithm()
        self.init_measurements()

        self.run_algorithm()

        self.group_split()
        self.final_measurements()
        self.results['finished_run'] = True
        results = self.save_results()
        self.logger.info('finished simulation run\n' + "=" * 80)
        self.options.warn_unused(True)
        return results

    @classmethod
    def from_saved_checkpoint(cls, filename=None, checkpoint_results=None, **kwargs):
        """Re-initialize a given simulation class from checkpoint results.

        You should probably call :meth:`resume_run` after sucessfull initialization.

        Instead of calling this directly, consider using :func:`resume_from_checkpoint`.

        Parameters
        ----------
        filename : None | str
            The filename of the checkpoint to be loaded.
            You can either specify the `filename` or the `checkpoint_results`.
        checkpoint_results : None | dict
            Alternatively to `filename` the results of the simulation so far, i.e. directly the
            data dicitonary saved at a simulation checkpoint.
        **kwargs :
            Further keyword arguments given to the `Simulation.__init__`.
        """
        if filename is not None:
            if checkpoint_results is not None:
                raise ValueError("pass either filename or checkpoint_results")
            checkpoint_results = hdf5_io.load(filename)
        if checkpoint_results is None:
            raise ValueError("you need to pass `filename` or `checkpoint_results`")
        options = checkpoint_results['simulation_parameters']
        # usually, we would say `sim = cls(options)`.
        # the following 3 lines provide an additional hook setting :attr:`loaded_from_checkpoint`
        # before calling the `__init__()`, such that other methods can be customized to this case.
        sim = cls.__new__(cls)
        sim.loaded_from_checkpoint = True  # hook to disable parts of the __init__()
        if 'resume_data' in checkpoint_results:
            kwargs.setdefault('resume_data', checkpoint_results['resume_data'])
        sim.__init__(options, **kwargs)
        sim.results = checkpoint_results
        if 'measurements' in checkpoint_results:
            sim.results['measurements'] = {k: list(v)
                                           for k, v in sim.results['measurements'].items()}
        return sim

    def resume_run(self):
        """Resume a simulation that was initialized from a checkpoint.

        Returns
        -------
        results : dict
            The :attr:`results` as returned by :meth:`prepare_results_for_save`.
        """
        if not self.loaded_from_checkpoint:
            warnings.warn("called `resume_run()` on a simulation *not* loaded from checkpoint. "
                          "You probably want `run()` instead!")
        self.init_model()

        if not hasattr(self, 'psi'):
            # didn't get psi in resume_data, but might still have it in the results
            if 'psi' not in self.results:
                raise ValueError("psi not saved in the results: can't resume!")
            self.psi = self.results['psi']
        self.init_state()  # does (almost) nothing if self.psi is already initialized
        self.group_sites_for_algorithm()
        self.init_algorithm()  # automatically reads out and del's ``self.results['resume_data']``

        # the relevant part from init_measurements(), but don't make a measurement
        self._connect_measurements()
        self.options.touch('measure_initial')

        self.resume_run_algorithm()  # continue with the actual algorithm
        self.group_split()
        self.final_measurements()
        self.results['finished_run'] = True
        results = self.save_results()
        self.logger.info('finished simulation (resume_)run\n' + "=" * 80)
        self.options.warn_unused(True)
        return results

    def init_cache(self):
        """Initialize the :attr:`cache` from the options.

        This method is only called automatically when the simulation is used in a
        ``with ...`` statement.
        This is the case if you use :func:`run_simulation`, etc.

        Options
        -------
        .. cfg:configoptions :: Simulation

            cache_threshold_chi : int
                If the `algorithm_params.trunc_params.chi_max` in :attr:`options` is smaller than
                this threshold, do not initialize a (non-trivial) cache.
            cache_params : dict
                Dictionary with parameters for the cache, see
                :meth:`~tenpy.tools.cache.CacheFile.open`.
        """
        cache_threshold_chi = self.options.get("cache_threshold_chi", 2000)
        cache_params = self.options.get("cache_params", {})
        chi = get_recursive(self.options, "algorithm_params.trunc_params.chi_max", default=None)
        if chi is not None and chi < cache_threshold_chi:
            self.cache = CacheFile.open()  # default = keep in RAM.
            return
        self.cache.close()
        self.logger.info("initialize new cache")
        self.cache = CacheFile.open(**cache_params)
        # note: can't use a `with self.cache` statement, but emulate it:
        # self.__enter__() calls this function followed by
        # self.cache = self.cache.__enter__()

    def init_model(self):
        """Initialize a :attr:`model` from the model parameters.

        Skips initialization if :attr:`model` is already set.

        Options
        -------
        .. cfg:configoptions :: Simulation

            model_class : str | class
                Mandatory. Class or name of a subclass of :class:`~tenpy.models.model.Model`.
            model_params : dict
                Dictionary with parameters for the model; see the documentation of the
                corresponding `model_class`.
        """
        model_class_name = self.options["model_class"]  # no default value!
        if hasattr(self, 'model'):
            self.options.touch('model_params')
            return  # skip actually regenerating the model
        ModelClass = find_subclass(Model, model_class_name)
        params = self.options.subconfig('model_params')
        self.model = ModelClass(params)

    def init_state(self):
        """Initialize a tensor network :attr:`psi`.

        Skips initialization if :attr:`psi` is already set.

        Options
        -------
        .. cfg:configoptions :: Simulation

            initial_state_builder_class : str | class
                Class or name of a subclass of :class:`~tenpy.networks.mps.InitialStateBuilder`.
                Used to initialize `psi` according to the `initial_state_params`.
            initial_state_params : dict
                Dictionary with parameters for building `psi`; see the decoumentation of the
                `initial_state_builder_class`, e.g. :cfg:config:`InitialStateBuilder`.
            save_psi : bool
                Whether the final :attr:`psi` should be included into the output :attr:`results`.
        """
        if not hasattr(self, 'psi'):
            builder_class = self.options.get('initial_state_builder_class', 'InitialStateBuilder')
            Builder = find_subclass(InitialStateBuilder, builder_class)
            params = self.options.subconfig('initial_state_params')
            initial_state_builder = Builder(self.model.lat, params, self.model.dtype)
            self.psi = initial_state_builder.run()
        else:
            self.logger.info("initial state as given")  # nothing to do
            # but avoid warnings about unused parameters
            self.options.touch('initial_state_builder_class', 'initial_state_params')
        if self.options.get('save_psi', True):
            self.results['psi'] = self.psi

    def group_sites_for_algorithm(self):
        """Coarse-grain the model and state for the algorithm.

        Options
        -------
        .. cfg:configoptions :: Simulation

            group_sites : int
                How many sites to group. 1 means no grouping.
            group_to_NearestNeighborModel : bool
                If True, convert the grouped model to a
                :class:`~tenpy.models.model.NearestNeighborModel`.
                Use this if you want to run TEBD with a model that was originally next-nearest
                neighbor.
        """
        group_sites = self.grouped = self.options.get("group_sites", 1)
        to_NN = self.options.get("group_to_NearestNeighborModel", False)
        if group_sites < 1:
            raise ValueError("invalid `group_sites` = " + str(group_sites))
        if group_sites > 1:
            if not self.loaded_from_checkpoint or self.psi.grouped < group_sites:
                self.psi.group_sites(group_sites)
            self.model_ungrouped = self.model.copy()
            self.model.group_sites(group_sites)
            if to_NN:
                self.model = NearestNeighborModel.from_MPOModel(self.model)

    def group_split(self):
        """Split sites of psi that were grouped in  :meth:`group_sites_for_algorithm`."""
        if self.grouped > 1:
            self.psi.group_split(self.options['algorithm_params']['trunc_params'])
            self.model = self.model_ungrouped
            del self.model_ungrouped
            self.grouped = 1

    def init_algorithm(self, **kwargs):
        """Initialize the algorithm.

        If :attr:`results` has `'resume_data'`, it is read out, used for initialization
        and removed from the results.

        Parameters
        ----------
        **kwargs :
            Extra keyword arguments passed on to the Algorithm.__init__(),
            for example the `resume_data` when calling `resume_run`.

        Options
        -------
        .. cfg:configoptions :: Simulation

            algorithm_class : str | class
                Class or name of a subclass of :class:`~tenpy.algorithms.algorithm.Algorithm`.
                The engine of the algorithm to be run.
            algorithm_params : dict
                Dictionary with parameters for the algortihm; see the decoumentation of the
                `algorithm_class`.
            connect_algorithm_checkpoint : list of tuple
                Functions to connect to the :attr:`~tenpy.algorithms.Algorith.checkpoint` event
                of the algorithm.
                Each tuple can be of length 2 to 4, with entries
                ``(module, function, kwargs, priority)``, the last two optionally.
                The mandatory `module` and `function` specify a callback measurement function.
                `kwargs` can specify extra keyword-arguments for the function,
                `priority` allows to tune the order in which the measurement functions get called.
                See :meth:`~tenpy.tools.events.EventHandler.connect_by_name` for more details.
        """
        alg_class_name = self.options.get("algorithm_class", self.default_algorithm)
        AlgorithmClass = find_subclass(Algorithm, alg_class_name)
        if 'resume_data' in self.results:
            self.logger.info("use `resume_data` for initializing the algorithm engine")
            kwargs.setdefault('resume_data', self.results['resume_data'].copy())
            # clean up: they are no longer up to date after algorithm initialization!
            # up to date resume_data is added in :meth:`prepare_results_for_save`
            self.results['resume_data'].clear()
            del self.results['resume_data']
        kwargs.setdefault('cache', self.cache)
        params = self.options.subconfig('algorithm_params')
        self.engine = AlgorithmClass(self.psi, self.model, params, **kwargs)
        self.engine.checkpoint.connect(self.save_at_checkpoint)
        con_checkpoint = list(self.options.get('connect_algorithm_checkpoint', []))
        for entry in con_checkpoint:
            self.engine.checkpoint.connect_by_name(*entry)

    def init_measurements(self):
        """Initialize and prepare measurements.

        Options
        -------
        .. cfg:configoptions :: Simulation

            connect_measurements : list of tuple
                Functions to connect to the :attr:`measurement_event`.
                Each tuple can be of length 2 to 4, with entries
                ``(module, function, kwargs, priority)``, the last two optionally.
                The mandatory `module` and `function` specify a callback measurement function.
                `kwargs` can specify extra keyword-arguments for the function,
                `priority` allows to tune the order in which the measurement functions get called.
                See :meth:`~tenpy.tools.events.EventHandler.connect_by_name` for more details.
            use_default_measurements : bool
                Each Simulation class defines a list of :attr:`default_measurements` in the same
                format as :cfg:option:`Simulation.connect_measurements`.
                This flag allows to explicitly disable them.
            measure_initial: bool
                Whether to perform a measurement on the initial state, i.e., before starting the
                algorithm run.
        """
        self._connect_measurements()
        if self.options.get('measure_initial', True):
            self.make_measurements()  # sets up self.results['measurements'] if necesssary

    def _connect_measurements(self):
        if self.options.get('use_default_measurements', True):
            def_meas = self.default_measurements
        else:
            def_meas = []
        con_meas = list(self.options.get('connect_measurements', []))
        for entry in def_meas + con_meas:
            # (module_name, func_name, kwargs=None, priority=0) = entry
            self._connect_measurements_fct(*entry)

    def _connect_measurements_fct(self, module_name, func_name, extra_kwargs=None, priority=0):
        if extra_kwargs is None:
            extra_kwargs = {}
        wrap = False
        if func_name.startswith('wrap'):
            wrap = True
            func_name = func_name.split()[1]

        # find measurement function
        if module_name == 'psi_method':
            # psi might change/only be created at beginning of measurement
            # so the function needs to be extracted dynamically during measurement
            # this is done in `tenpy.simulations.measurement._m_psi_method{_wrapped}()`
            extra_kwargs['func_name'] = func_name
            func = _m_psi_method_wrapped if wrap else _m_psi_method
            wrap = False
        elif module_name == 'model_method':
            # analogous to psi_method
            extra_kwargs['func_name'] = func_name
            func = _m_model_method_wrapped if wrap else _m_model_method
            wrap = False
        elif module_name == 'simulation_method':
            # the simulation class already exists, so we can directly get the corresponding method
            func = getattr(self, func_name)
        else:
            # global functions should also exist already, so we can directly get them
            func = hdf5_io.find_global(module_name, func_name)

        if wrap:
            if 'results_key' in extra_kwargs:
                results_key = extra_kwargs['results_key']
                del extra_kwargs['results_key']
            else:
                results_key = func_name
            func = measurement_wrapper(func, results_key=results_key)

        self.measurement_event.connect(func, priority, extra_kwargs)

    def run_algorithm(self):
        """Run the algorithm.

        Calls ``self.engine.run()``.
        """
        self.engine.run()

    def resume_run_algorithm(self):
        """Resume running the algorithm.

        Calls ``self.engine.resume_run()``.
        """
        # usual algorithms have a loop with break conditions, which we can just resume
        self.engine.resume_run()

    def make_measurements(self):
        """Perform measurements and merge the results into ``self.results['measurements']``."""
        self.logger.info("make measurements")
        results = self.perform_measurements()
        self._merge_measurement_results(results)

    def _merge_measurement_results(self, results):
        """Merge dictionary `results` from measurements into ``self.results['measurement']``."""
        # merge the results into self.results['measurements']
        previous_results = self.results.get('measurements', None)
        if previous_results is None:
            self.results['measurements'] = {k: [v] for k, v in results.items()}
            return

        previous_keys = set(previous_results.keys())
        new_keys = set(results.keys())
        new_keys_not_previous = new_keys - previous_keys
        if new_keys_not_previous:
            warnings.warn(f"measurement gave new keys {new_keys_not_previous!r} "
                            "fill up with `None` for previous measurements.")
            some_previous_measurement = next(iter(previous_results.values()))
            measurement_len = len(some_previous_measurement)
            for key in new_keys_not_previous:
                previous_results[key] = [None] * measurement_len

        # actual merge
        for k, v in results.items():   # only new keys
            previous_results[k].append(v)

        previous_keys_not_new = previous_keys - new_keys
        if previous_keys_not_new:
            warnings.warn(f"measurement didn't give keys {previous_keys_not_new!r} "
                          "we have from previous measurements, fill up with `None`")
            for key in previous_keys_not_new:
                previous_results[key].append(None)
        # done

    def perform_measurements(self):
        """Emits the :attr:`measurement_event` to call measurement functions and collect results.

        Returns
        -------
        results : dict
            The results from calling the measurement functions.
        """
        # TODO: safe-guard measurements with try-except?
        # in case of a failed measurement, we should raise the exception at the end of the
        # simulation?
        results = {}
        psi, model = self.get_measurement_psi_model(self.psi, self.model)

        returned = self.measurement_event.emit(results=results,
                                               psi=psi,
                                               model=model,
                                               simulation=self)
        # check for returned values, although there shouldn't be any
        returned = [entry for entry in returned if entry is not None]
        if len(returned) > 0:
            msg = ("Some measurement function returned a value instead of writing to `results`.\n"
                   "Add it to measurement results as 'UNKNOWN'.")
            warnings.warn(msg)
            results['UNKNOWN'] = returned
        return results

    def get_measurement_psi_model(self, psi, model):
        """Get psi for measurements.

        Sometimes, the `psi` we want to use for measurements is different from the one the
        algorithm actually acts on.
        Here, we split sites, if they were grouped in :meth:`group_sites_for_algorithm`.

        Parameters
        ----------
        psi :
            Tensor network; initially just ``self.psi``.
            The method should make a copy before modification.
        model :
            Model matching `psi` (in terms of indexing, MPS order, grouped sites, ...)
            Initially just ``self.model``.

        Returns
        -------
        psi :
            The psi suitable as argument for generic measurement functions.
        model :
            Model matching `psi` (in terms of indexing, MPS order, grouped sites, ...)
        """
        if self.grouped > 1:
            if psi is self.psi:
                psi = psi.copy()  # make copy before
            psi.group_split(self.options['algorithm_params']['trunc_params'])
            model = self.model_ungrouped
        return psi, model

    def final_measurements(self):
        """Perform a last set of measurements."""
        self.make_measurements()

    def get_version_info(self):
        """Try to save version info which is necessary to allow reproducability."""
        sim_module = self.__class__.__module__
        # also try to extract git revision of the simulation class
        if sim_module.startswith('tenpy') or sim_module == "__main__":
            cwd = os.getcwd()
        else:
            # use the cwd of the file where the simulation class is defined
            module = importlib.import_module(sim_module)  # get module object
            cwd = os.path.dirname(os.path.abspath(module.__file__)),
        git_rev = version._get_git_revision(cwd)

        version_info = {
            'tenpy': version.version_summary,
            'simulation_class': self.__class__.__qualname__,
            'simulation_module': sim_module,
            'simulation_git_HEAD': git_rev,
        }
        return version_info

    def get_output_filename(self):
        """Read out the `output_filename` from the options.

        You can easily overwrite this method in subclasses to customize the outputfilename
        depending on the options passed to the simulations.

        Options
        -------
        .. cfg:configoptions :: Simulation

            output_filename : path_like | None
                If ``None`` (default), no output is written to files.
                If a string, this filename is used for output (up to modifications by
                :meth:`fix_output_filenames` to avoid overwriting previous results).
            output_filename_params : dict
                Instead of specifying the `output_filename` directly, this dictionary describes
                the parameters that should be included into it.
                Entries of the dictionary are keyword arguments to
                :func:`output_filename_from_dict` with the simulation parameters
                (:cfg:option:`Simulation`, or equivalently :attr:`options`) as `options`.

        Returns
        -------
        output_filename : str | None
            Filename for output; None disables any writing to files.
            Relative to :cfg:option:`Simulation.directory`, if specified.
            The file ending determines the output format.
        """
        # note: this function shouldn't use logging: it's called before setup_logging()
        output_filename_params = self.options.setdefault('output_filename_params', None)
        if output_filename_params is not None:
            default = output_filename_from_dict(self.options, **output_filename_params)
        else:
            default = None
        output_filename = self.options.setdefault('output_filename', default)
        return output_filename

    def fix_output_filenames(self):
        """Determine the output filenames.

        This function determines the :attr:`output_filename` and writes a one-line text into
        that file to indicate that we're running a simulation generating it.
        Further, :attr:`_backup_filename` is determined.

        Options
        -------
        .. cfg:configoptions :: Simulation

            skip_if_output_exists : bool
                If True, raise :class:`Skip` if the output file already exists.
            overwrite_output : bool
                Only makes a difference if `skip_if_output_exists` is False and the file exists.
                In that case, with `overwrite_output`, just save everything under that name again,
                or with `overwrite_output`=False, replace
                ``filename.ext`` with ``filename_01.ext`` (and further increasing numbers)
                until we get a filename that doesn't exist yet.
            safe_write : bool
                If True (default), perform a "safe" overwrite of `output_filename` as described
                in :meth:`save_results`.
        """
        # note: this function shouldn't use logging: it's called before setup_logging()
        # hence, assume that `options` is still a pure dict, not a tenpy.tools.misc.Config
        output_filename = self.get_output_filename()
        overwrite_output = self.options.setdefault("overwrite_output", False)
        skip_if_exists = self.options.setdefault("skip_if_output_exists", False)
        if output_filename is None:
            self.output_filename = None
            self._backup_filename = None
            return
        out_fn = Path(output_filename)  # convert to Path
        self.output_filename = out_fn
        self._backup_filename = self.get_backup_filename(out_fn)

        if out_fn.exists():
            if skip_if_exists:
                # no need to touch options: not yet converted to config
                raise Skip("simulation output filename already exists", out_fn)
            if not overwrite_output and not self.loaded_from_checkpoint:
                # adjust output filename to avoid overwriting stuff
                root, ext = os.path.splitext(out_fn)
                for i in range(1, 100):
                    new_out_fn = Path(root + '_' + str(i) + ext)
                    if not new_out_fn.exists():
                        break
                else:
                    raise ValueError("Refuse to make another copy. CLEAN UP!")
                warnings.warn(f"changed output filename to {new_out_fn!s}")
                self.output_filename = out_fn = new_out_fn
                self._backup_filename = self.get_backup_filename(out_fn)
            # else: overwrite stuff in `save_results`
            if overwrite_output and not self.loaded_from_checkpoint:
                # move logfile to *.backup.log
                log_fn = out_fn.with_suffix('.log')
                backup_log_fn = self.get_backup_filename(log_fn)
                if log_fn.exists() and backup_log_fn is not None:
                    if backup_log_fn.exists():
                        backup_log_fn.unlink()
                    log_fn.rename(backup_log_fn)
        if self._backup_filename is not None and not self._backup_filename.exists():
            import socket
            text = "simulation initialized on {host!r} at {time!s}\n"
            text = text.format(host=socket.gethostname(), time=time.asctime())
            with self._backup_filename.open('w') as f:
                f.write(text)

    def get_backup_filename(self, output_filename):
        """Extract the name used for backups of `output_filename`.

        Parameters
        ----------
        output_filename : pathlib.Path
            The filename where data is saved.

        Returns
        -------
        backup_filename : pathlib.Path
            The filename where to keep a backup while writing files to avoid.
        """
        # note: this function shouldn't use logging
        if self.options.setdefault("safe_write", True):
            return output_filename.with_suffix('.backup' + output_filename.suffix)
        else:
            return None

    def save_results(self, results=None):
        """Save the :attr:`results` to an output file.

        Performs a "safe" overwrite of :attr:`output_filename` by first moving the old file
        to :attr:`_backup_filename`, then writing the new file, and finally removing the backup.

        Parameters
        ----------
        results : dict | None
            The results to be safed. If not specified, call :meth:`prepare_results_for_save`
            to allow last-minute adjustments to the saved :attr:`results`.
        """
        if results is None:
            results = self.prepare_results_for_save()

        output_filename = self.output_filename
        backup_filename = self._backup_filename
        if output_filename is None:
            return results  # don't save to disk
        start_time = time.time()

        if output_filename.exists():
            # keep a single backup, previous backups are overwritten.
            if backup_filename is not None:
                if backup_filename.exists():
                    backup_filename.unlink()  # remove if exists
                output_filename.rename(backup_filename)
            else:
                output_filename.unlink()  # remove

        # actually save the results to disk
        self._save_to_file(results, output_filename)

        if backup_filename is not None and backup_filename.exists():
            # successfully saved, so we can safely remove the old backup
            backup_filename.unlink()

        self._last_save = time.time()
        self.logger.info("saving results to disk; took %.1fs", self._last_save - start_time)
        return results

    def _save_to_file(self, results, output_filename):
        hdf5_io.save(results, output_filename)

    def prepare_results_for_save(self):
        """Bring the `results` into a state suitable for saving.

        For example, this can be used to convert lists to numpy arrays, to add more meta-data,
        or to clean up unnecessarily large entries.

        Options
        -------
        :cfg:configoptions :: Simulation

            save_resume_data : bool
                If True, include data from :meth:`~tenpy.algorithms.Algorithm.get_resume_data`
                into the output as `resume_data`.

        Returns
        -------
        results : dict
            A copy of :attr:`results` containing everything to be saved.
            Measurement results are converted into a numpy array (if possible).
        """
        results = self.results.copy()
        results['simulation_parameters'] = self.options.as_dict()
        if 'measurements' in results:
            # try to convert measurements into numpy arrays to store more compactly
            results['measurements'] = measurements = results['measurements'].copy()
            for k, v in measurements.items():
                try:
                    v = np.array(v)
                except:
                    continue
                if v.dtype != np.dtype(object):
                    measurements[k] = v
        if self.options.get('save_resume_data', self.options['save_psi']):
            results['resume_data'] = self.engine.get_resume_data()
        return results

    def save_at_checkpoint(self, alg_engine):
        """Save the intermediate results at the checkpoint of an algorithm.

        Parameters
        ----------
        alg_engine : :class:`~tenpy.algorithms.Algorithm`
            The engine of the algorithm. Not used in this function, mostly there for compatibility
            with the :attr:`tenpy.algorithms.Algorithm.checkpoint` event.

        Options
        -------
        .. cfg:configoptions :: Simulation

            save_every_x_seconds : float | None
                By default (``None``), this feature is disabled.
                If given, save the :attr:`results` obtained so far at each
                :attr:`tenpy.algorithm.Algorithm.checkpoint` when at least `save_every_x_seconds`
                seconds evolved since the last save (or since starting the algorithm).
                To avoid unnecessary, slow disk input/output, the value will be increased if
                saving takes longer than 10% of `save_every_x_seconds`.
                Use ``0.`` to force saving at each checkpoint.
        """
        save_every = self.options.get('save_every_x_seconds', None)
        now = time.time()
        if save_every is not None and now - self._last_save > save_every:
            self.save_results()
            time_to_save = time.time() - now
            if time_to_save > 0.1 * save_every > 0.:
                save_every = 20 * time_to_save
                self.logger.warning(
                    "Saving took longer than 10%% of `save_every_x_seconds`. "
                    "Increase the latter to %.1f", save_every)
                self.options['save_every_x_seconds'] = save_every
        # done

    def walltime(self):
        """Wall time evolved since initialization of the simulation class.

        Utility measurement method. To measure it, add the following entry to the
        :cfg:option:`Simulation.connect_measurements` option::

            - - simulation_method
              - wrap walltime

        Returns
        -------
        seconds : float
            Elapsed (wall clock) time in seconds since the initialization of the simulation.
        """
        return time.time() - self._init_walltime


class Skip(ValueError):
    """Error raised if simulation output already exists.

    Parameters
    ----------
    msg : str
        Error message.
    filename : str
        Filename of the existing output file due to which the simulation is skipped.
    """
    def __init__(self, msg, filename):
        filename = str(filename)
        super().__init__(msg + '\n' + filename)
        self.filename = filename


_deprecated_not_set = object()


def init_simulation(simulation_class='GroundStateSearch',
                    simulation_class_kwargs=None,
                    **simulation_params):
    """Run the simulation with a simulation class.

    If you need to run the simulation, you can use a `with` statement for proper context
    management::

        with sim:
            results = sim.run()

    Parameters
    ----------
    simulation_class : str
        The name of a (sub)class of :class:`~tenpy.simulations.simulations.Simulation`
        to be used for running the simulation.
    simulation_class_kwargs : dict | None
        A dictionary of keyword-arguments to be used for the initializing the simulation.
    **simulation_params :
        Further keyword arguments as documented in the corresponding simulation class,
        see :cfg:config:`Simulation`.

    Returns
    -------
    results : dict
        The results of the Simulation, i.e., what
        :meth:`tenpy.simulations.simulation.Simulation.run()` returned.
    """
    SimClass = find_subclass(Simulation, simulation_class)
    if simulation_class_kwargs is None:
        simulation_class_kwargs = {}
    sim = SimClass(simulation_params, **simulation_class_kwargs)
    return sim


def run_simulation(simulation_class='GroundStateSearch',
                   simulation_class_kwargs=None,
                   *,
                   simulation_class_name=_deprecated_not_set,
                   **simulation_params):
    """Run the simulation with a simulation class.

    .. deprecated :: 0.9.0
        The `simulation_class_name` argument has been renamed to just `simulation_class`.

    Parameters
    ----------
    simulation_class : str
        The name of a (sub)class of :class:`~tenpy.simulations.simulations.Simulation`
        to be used for running the simulation.
    simulation_class_kwargs : dict | None
        A dictionary of keyword-arguments to be used for the initializing the simulation.
    **simulation_params :
        Further keyword arguments as documented in the corresponding simulation class,
        see :cfg:config:`Simulation`.

    Returns
    -------
    results : dict
        The results of the Simulation, i.e., what
        :meth:`tenpy.simulations.simulation.Simulation.run()` returned.
    """
    if simulation_class_name is not _deprecated_not_set:
        assert simulation_class == 'GroundStateSearch'
        warnings.warn(
            "The `simulation_class_name` argument has been renamed to `simulation_class`"
            " for more consistency with remaining parameters.", FutureWarning)
        simulation_class = simulation_class_name
    sim = init_simulation(simulation_class, simulation_class_kwargs, **simulation_params)
    with sim:
        results = sim.run()
    return results


def init_simulation_from_checkpoint(*,
                                    filename=None,
                                    checkpoint_results=None,
                                    update_sim_params=None,
                                    simulation_class_kwargs=None):
    """Re-initialize a simulation from a given checkpoint without running it.

    (All parameters have to be given as keyword arguments.)

    If you need to run the simulation, you can use a `with` statement for proper context
    management::

        with sim:
            results = sim.run()

    Parameters
    ----------
    filename : None | str
        The filename of the checkpoint to be loaded.
        You can either specify the `filename` or the `checkpoint_results`.
    checkpoint_results : None | dict
        Alternatively to `filename` the results of the simulation so far, i.e. directly the data
        dicitonary saved at a simulation checkpoint.
    update_sim_params : None | dict
        Allows to update specific :cfg:config:`Simulation` parameters; ignored if `None`.
        Uses :func:`~tenpy.tools.misc.update_recursive` to update values, such that the keys of
        `update_sim_params` can be recursive, e.g. `algorithm_params/max_sweeps`.
    simlation_class_kwargs : None | dict
        Further keyword arguemnts given to the simulation class, ignored if `None`.

    Returns
    -------
    results :
        The results from running the simulation, i.e.,
        what :meth:`tenpy.simulations.Simulation.resume_run()` returned.

    Notes
    -----
    The `checkpoint_filename` should be relative to the current working directory. If you use the
    :cfg:option:`Simulation.directory`, the simulation class will attempt to change to that
    directory during initialization. Hence, either resume the simulation from the same directory
    where you originally started, or update the :cfg:option:`Simulation.directory`
    (and :cfg:option`Simulation.output_filename`) parameter with `update_sim_params`.
    """
    if filename is not None:
        if checkpoint_results is not None:
            raise ValueError("pass either filename or checkpoint_results")
        checkpoint_results = hdf5_io.load(filename)
    if checkpoint_results is None:
        raise ValueError("you need to pass `filename` or `checkpoint_results`")
    if checkpoint_results['finished_run']:
        raise Skip("Simulation already finished", filename)
    sim_class_mod = checkpoint_results['version_info']['simulation_module']
    sim_class_name = checkpoint_results['version_info']['simulation_class']
    SimClass = hdf5_io.find_global(sim_class_mod, sim_class_name)
    if simulation_class_kwargs is None:
        simulation_class_kwargs = {}

    options = checkpoint_results['simulation_parameters']
    if update_sim_params is not None:
        update_recursive(options, update_sim_params)

    sim = SimClass.from_saved_checkpoint(checkpoint_results=checkpoint_results,
                                        **simulation_class_kwargs)
    return sim


def resume_from_checkpoint(*,
                           filename=None,
                           checkpoint_results=None,
                           update_sim_params=None,
                           simulation_class_kwargs=None):
    """Resume a simulation run from a given checkpoint.

    (All parameters have to be given as keyword arguments.)

    Parameters
    ----------
    filename : None | str
        The filename of the checkpoint to be loaded.
        You can either specify the `filename` or the `checkpoint_results`.
    checkpoint_results : None | dict
        Alternatively to `filename` the results of the simulation so far, i.e. directly the data
        dicitonary saved at a simulation checkpoint.
    update_sim_params : None | dict
        Allows to update specific :cfg:config:`Simulation` parameters; ignored if `None`.
        Uses :func:`~tenpy.tools.misc.update_recursive` to update values, such that the keys of
        `update_sim_params` can be recursive, e.g. `algorithm_params/max_sweeps`.
    simlation_class_kwargs : None | dict
        Further keyword arguemnts given to the simulation class, ignored if `None`.

    Returns
    -------
    results :
        The results from running the simulation, i.e.,
        what :meth:`tenpy.simulations.Simulation.resume_run()` returned.

    Notes
    -----
    The `checkpoint_filename` should be relative to the current working directory. If you use the
    :cfg:option:`Simulation.directory`, the simulation class will attempt to change to that
    directory during initialization. Hence, either resume the simulation from the same directory
    where you originally started, or update the :cfg:option:`Simulation.directory`
    (and :cfg:option`Simulation.output_filename`) parameter with `update_sim_params`.
    """
    sim = init_simulation_from_checkpoint(filename=filename,
                                          checkpoint_results=checkpoint_results,
                                          update_sim_params=update_sim_params,
                                          simulation_class_kwargs=simulation_class_kwargs)
    del checkpoint_results  # possibly free memory
    options = sim.options
    with sim:
        results = sim.resume_run()
        if 'sequential' in options:
            sequential = options['sequential']
            sequential['index'] += 1
            resume_data = sim.engine.get_resume_data(sequential_simulations=True)
    if 'sequential' in options:
        # note: it is important to exit the with ... as sim`` statement before continuing
        # to free memory and cache
        SimClass = sim.__class__
        if simulation_class_kwargs is None:
            simulation_class_kwargs = {}
        del sim  # free memory
        return run_seq_simulations(sequential,
                                   SimClass,
                                   simulation_class_kwargs,
                                   resume_data=resume_data,
                                   **options)
    return results


def run_seq_simulations(sequential,
                        simulation_class='GroundStateSearch',
                        simulation_class_kwargs=None,
                        *,
                        simulation_class_name=_deprecated_not_set,
                        resume_data=None,
                        collect_results_in_memory=False,
                        **simulation_params):
    """Sequentially run (variational) simulations.

    Uses the results (in particular the state) from one simulation to intialize another one.
    This allows to "adiabatically" or "smoothly" follow the evolution of the ground state as
    certain model (or algorithm) parameters change.

    Options
    -------
    .. cfg:config :: sequential

        recursive_keys : list of str
            Mandatory.
            The list of recursive keys for the `simulation_params` to be changed.
            for example an entry ``'model_params.Jz'`` indicates that
            ``simulation_params['model_params']['Jz']`` should be changed,
            see :func:`~tenpy.tools.misc.get_recursive`.
        value_lists : list of list
            For each entry of `recursive_keys` the list of values that this parameter should take.
            If `value_lists` is not given at the beginning of this function, it is read
            out from the `simulation_params`, i.e. you can alternatively directly change the
            values in you simulation_params options to be lists.
            We iterate through all values with ``zip(*values)``.
        format_strs : list of str
            For each of the `recursive_keys` a formatting string `format_str` to be formatted with
            ``format_str.format(value)``. If non-zero, the `format_strs` are used for `parts` in
            :func:`output_filename_from_dict` to find a unique `output_filename` .
            For example, for ``'model_params.Jz'`` a good choice would be ``'Jz_{0:.3f}'``.
            If `format_strs` is not given at all, it defaults to
            ``[rkey.split(separator)[-1] + '_{0!s}' for rkey in recursive_keys]``.
            Exception: if `output_filename` or `directory` is part of the `recursive_keys`,
            the whole list of `format_strs` is ignored and the `output_filename` is not updated.
        separator : str
            Separator to split recursive keys in :func:`~tenpy.tools.misc.get_recursive` etc.
            Defaults to ``'.'``.
        index : int
            The first index for each of the `value_lists` to run things with.
        base_directory : pathlike
            Working directory relative to which the :cfg:option:`Simulation.directory` of the
            individual simulations is specified.
            Defaults to the current working directory at the beginning of this function.

    Parameters
    ----------
    sequential : dict
        Paramters specifying the sequential simulation, see :cfg:config:`sequential` above.
    resume_data : None | dict
        Usually None if you didn't already run a simulation that you want to continue.
        Otherwise the `resume_data` as given to the Simulation class.
    collect_results_in_memory : bool
        If False (default), just save the results to the corresponding output files.
        If True, collect the results by keeping *copies* of psi and all simulation results
        *in memory*. (This can kill your available RAM quickly!)
    simulation_class : str
    simulation_class_kwargs : dict | None
    **simulation_params :
        Further arguments as in :func:`run_simulation`.

    Returns
    -------
    results: list | dict
        If `collect_results_in_memory`, a list of dictionaries with the results for each
        simulation. Otherwise just the results of the last simulation run.
    """
    sequential = asConfig(sequential, 'sequential')
    separator = sequential.get('separator', '.')
    recursive_keys = sequential['recursive_keys']
    N_keys = len(recursive_keys)
    format_strs = [rkey.split(separator)[-1] + '_{0!s}' for rkey in recursive_keys]
    format_strs = sequential.get('format_strs', format_strs)
    value_lists = [get_recursive(simulation_params, r_key) for r_key in recursive_keys]
    value_lists = sequential.get('value_lists', value_lists)
    index = sequential.get('index', 0)
    base_directory = sequential.get('base_directory', os.getcwd())

    if N_keys > 0:
        N_sims = len(value_lists[0])
        for vl in value_lists[1:]:
            if len(vl) != N_sims:
                raise ValueError("Different lengths for the ``sequential['value_lists']``")
        for k in recursive_keys:
            # goal of sequential simulation: keep the initial state from previous simulation!
            for check in ['initial_state', 'output_filename_params']:
                assert not k.startswith(check), "really?!?"
    else:
        N_sims = 1

    if simulation_class_name is not _deprecated_not_set:
        assert simulation_class == 'GroundStateSearch'
        warnings.warn(
            "The `simulation_class_name` argument has been renamed to `simulation_class`"
            " for more consistency with remaining parameters.", FutureWarning)
        simulation_class = simulation_class_name

    SimClass = find_subclass(Simulation, simulation_class)
    if simulation_class_kwargs is None:
        simulation_class_kwargs = {}

    # try to create varying output filenames
    # do we save to file at all?
    if simulation_params.get('output_filename', None) is not None or \
            simulation_params.get('output_filename_params', None) is not None:
        if 'output_filename' not in recursive_keys and 'directory' not in recursive_keys:
            # need to update the output_filename for each simulation
            output_filename_params = simulation_params.get('output_filename_params', {})
            output_filename = simulation_params.get('output_filename', None)
            if output_filename is not None:
                output_filename = os.fspath(output_filename)
                prefix, suffix = os.path.splitext(output_filename)
                output_filename_params.update({'prefix': prefix, 'suffix': suffix})
                # rather regenerate in Simulation.get_output_filenames
                del simulation_params['output_filename']
            parts = output_filename_params.setdefault('parts', {})
            for k, v in zip(recursive_keys, format_strs):
                if k not in parts and v:
                    parts[k] = v
            simulation_params['output_filename_params'] = output_filename_params
    else:  # we don't save results to files
        if not collect_results_in_memory:
            raise ValueError("Refuse to run without producing output")
    if collect_results_in_memory:
        all_results = []

    simulation_params['sequential'] = sequential

    for index in range(index, N_sims):
        os.chdir(base_directory)
        # update simulation parameters
        sequential['index'] = index
        sim_params = copy.deepcopy(simulation_params)
        for rec_key, values in zip(recursive_keys, value_lists):
            val = values[index]
            set_recursive(sim_params, rec_key, val, separator, insert_dicts=True)

        if resume_data is not None:
            simulation_class_kwargs['resume_data'] = resume_data

        with SimClass(sim_params, **simulation_class_kwargs) as sim:
            results = sim.run()
            if collect_results_in_memory:
                all_results.append(results)
            # save results for the next simulation
            resume_data = sim.engine.get_resume_data(sequential_simulations=True)
        del sim  # but free memory to avoid too many copies (e.g. the whole environment)
        if index + 1 < N_sims:
            del results
    # all simulations are done!
    if collect_results_in_memory:
        return all_results
    else:
        return results


def output_filename_from_dict(options,
                              parts={},
                              prefix='result',
                              suffix='.h5',
                              joint='_',
                              parts_order=None,
                              separator='.'):
    """Format a `output_filename` from parts with values from nested `options`.

    The results of a simulation are ideally fixed by the simulation class and the `options`.
    Unique filenames could be obtained by including *all* options into the filename, but this
    would be a huge overkill: it suffices if we include the options that we actually change.
    This function helps to keep the length of the output filename at a sane level
    while ensuring (hopefully) sufficient uniqueness.

    Parameters
    ----------
    options : (nested) dict
        Typically the simulation parameters, i.e., options passed to :class:`Simulation`.
    parts :: dict
        Entries map a `recursive_key` for `options` to a `format_str` used
        to format the value, i.e. we extend the filename with
        ``format_str.format(get_recursive(options, recursive_key, separator))``.
        If `format_str` is empty, no part is added to the filename.
    prefix, suffix : str
        First and last part of the filename.
    joint : str
        Individual filename parts (except the suffix) are joined by this string.
    parts_order : None | list of keys
        Optionally, an explicit order for the keys of `parts`.
        By default (None), just the keys of `parts`, i.e. the order in which they appear in the
        dictionary; before python 3.7 (where the order is not defined) alphabetically sorted.
    separator : str
        Separator for :func:`~tenpy.tools.misc.get_recursive`.

    Returns
    -------
    output_filename : str
        (Hopefully) sufficiently unique filename.

    Examples
    --------
    >>> from tenpy.simulations.simulation import output_filename_from_dict
    >>> options = {  # some simulation parameters
    ...    'algorithm_params': {
    ...         'dt': 0.01,  # ...
    ...    },
    ...    'model_params':  {
    ...         'Lx': 3,
    ...         'Ly': 4, # ...
    ...    }, # ... and many more options ...
    ... }
    >>> output_filename_from_dict(options)
    'result.h5'
    >>> output_filename_from_dict(options, suffix='.pkl')
    'result.pkl'
    >>> output_filename_from_dict(options, parts={'model_params.Ly': 'Ly_{0:d}'}, prefix='check')
    'check_Ly_4.h5'
    >>> output_filename_from_dict(options, parts={
    ...         'algorithm_params.dt': 'dt_{0:.3f}',
    ...         'model_params.Ly': 'Ly_{0:d}'})
    'result_dt_0.010_Ly_4.h5'
    >>> output_filename_from_dict(options, parts={
    ...         'algorithm_params.dt': 'dt_{0:.3f}',
    ...         ('model_params.Lx', 'model_params.Ly'): '{0:d}x{1:d}'})
    'result_dt_0.010_3x4.h5'
    >>> output_filename_from_dict(options, parts={
    ...         'algorithm_params.dt': '_dt_{0:.3f}',
    ...         'model_params.Lx': '_{0:d}',
    ...         'model_params.Ly': 'x{0:d}'}, joint='')
    'result_dt_0.010_3x4.h5'
    """
    formatted_parts = [prefix]
    if parts_order is None:
        if sys.version_info < (3, 7):
            # dictionaries are not ordered -> sort keys alphabetically
            parts_order = sorted(parts.keys(), key=lambda x: x[0] if isinstance(x, tuple) else x)
        else:
            parts_order = parts.keys()  # dictionaries are ordered, so use that order
    else:
        assert set(parts_order) == set(parts.keys())
    for recursive_key in parts_order:
        format_str = parts[recursive_key]
        if not format_str:
            continue
        if not isinstance(recursive_key, tuple):
            recursive_key = (recursive_key, )
        vals = [get_recursive(options, r_key, separator) for r_key in recursive_key]
        part = format_str.format(*vals)
        formatted_parts.append(part)
    return joint.join(formatted_parts) + suffix
