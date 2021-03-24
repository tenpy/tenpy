"""This module contains base classes for simulations.

The :class:`Simulation` class tries to put everything need for a simulation in a structured form
and collects task like initializing the tensor network state, model and algorithm classes,
running the actual algorithm, possibly performing measurements and saving the results.


.. todo ::
    provide examples,
    give user guide
"""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

import os
from pathlib import Path
import time
import importlib
import warnings
import numpy as np
import logging

from ..models.model import Model
from ..algorithms.algorithm import Algorithm
from ..networks.mps import InitialStateBuilder
from ..tools import hdf5_io
from ..tools.params import asConfig
from ..tools.events import EventHandler
from ..tools.misc import find_subclass, update_recursive
from ..tools.misc import setup_logging as setup_logging_
from .. import version

__all__ = ['Simulation', 'Skip', 'run_simulation', 'resume_from_checkpoint']


class Simulation:
    """Base class for simulations.

    Parameters
    ----------
    options : dict-like
        The simulation parameters as outlined below.
        Ideally, these options should be enough to fully specify all parameters of a simulation
        to ensure reproducibility.
    setup_logging : bool
        Whether to call :meth:`setup_logging` at the beginning of initialization.

    Options
    -------
    .. cfg:config :: Simulation

        directory : str
            If not None (default), switch to that directory at the beginning of the simulation.
        output_filename : string | None
            Filename for output. The file ending determines the output format.
            None (default) disables any writing to files.
        logging_params : dict
            Logging parameters; see :cfg:config:`logging`.
        overwrite_output : bool
            Whether an exisiting file may be overwritten.
            Otherwise, if the file already exists we try to replace
            ``filename.ext`` with ``filename_01.ext`` (and further increasing numbers).
        random_seed : int | None
            If not ``None``, initialize the numpy random generator with the given seed.

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
            The data fro resuming the algorithm run.
            Only included if :cfg:option:`Simultion.save_resume_data` is True.

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
    _last_save : float
        Time of the last call to :meth:`save_results`, initialized to startup time.
    loaded_from_checkpoint : bool
        True when the simulation is loaded with :meth:`from_saved_checkpoint`.

    """
    #: name of the default algorithm `engine` class
    default_algorithm = 'TwoSiteDMRGEngine'

    #: tuples as for :cfg:option:`Simulation.connect_measurements` that get added if
    #: the :cfg:option:`Simulation.use_default_measurements` is True.
    default_measurements = [
        ('tenpy.simulations.measurement', 'measurement_index', {}, 1),
        ('tenpy.simulations.measurement', 'bond_dimension'),
        ('tenpy.simulations.measurement', 'energy_MPO'),
        ('tenpy.simulations.measurement', 'entropy'),
    ]

    #: logger : An instance of a logger; see :doc:`/intro/logging`. NB: class attribute.
    logger = logging.getLogger(__name__ + ".Simulation")

    def __init__(self, options, *, setup_logging=True):
        if not hasattr(self, 'loaded_from_checkpoint'):
            self.loaded_from_checkpoint = False
        self.options = asConfig(options, self.__class__.__name__)
        cwd = self.options.get("directory", None)
        if cwd is not None:
            os.chdir(cwd)
        self.fix_output_filenames()
        if setup_logging:
            log_params = self.options.subconfig('logging_params')
            setup_logging_(log_params, self.output_filename)  # needs self.output_filename
        self.logger.info("simulation class %s", self.__class__.__name__)
        # catch up with logging messages
        if cwd is not None:
            self.logger.info("change directory to %r", cwd)
        self.logger.info("output filename: %r", self.output_filename)

        random_seed = self.options.get('random_seed', None)
        if random_seed is not None:
            if self.loaded_from_checkpoint:
                warnings.warn("resetting `random_seed` for a simulation loaded from checkpoint."
                              "Depending on where you use random numbers, "
                              "this might or might not be what you want!")
            np.random.seed(random_seed)
        self.results = {
            'simulation_parameters': self.options,
            'version_info': self.get_version_info(),
            'finished_run': False,
        }
        self._last_save = time.time()
        self.measurement_event = EventHandler("psi, simulation, results")

    @property
    def verbose(self):
        warnings.warn(
            "verbose is deprecated, we're using logging now! \n"
            "See https://tenpy.readthedocs.io/en/latest/intro/logging.html", FutureWarning, 2)
        return self.options.get('verbose', 1.)

    def run(self):
        """Run the whole simulation."""
        if self.loaded_from_checkpoint:
            warnings.warn("called `run()` on a simulation loaded from checkpoint. "
                          "You should probably call `resume_run()` instead!")
        self.init_model()
        self.init_state()
        self.init_algorithm()
        self.init_measurements()
        self.run_algorithm()
        self.final_measurements()
        self.results['finished_run'] = True
        results = self.save_results()
        self.logger.info('finished simulation run')
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
        sim.__init__(options, **kwargs)
        sim.results = checkpoint_results
        sim.results['measurements'] = {k: list(v) for k, v in sim.results['measurements'].items()}
        return sim

    def resume_run(self):
        """Resume a simulation that was initialized from a checkpoint."""
        if not self.loaded_from_checkpoint:
            warnings.warn("called `resume_run()` on a simulation *not* loaded from checkpoint. "
                          "You probably want `run()` instead!")
        self.init_model()
        # init_state() equivalent
        if not hasattr(self, 'psi'):
            if 'psi' not in self.results:
                raise ValueError("psi not saved in the results: can't resume!")
            self.psi = self.results['psi']
        self.options.touch('initial_state_builder_class', 'initial_state_params', 'save_psi')

        kwargs = {}
        if 'resume_data' in self.results:
            kwargs['resume_data'] = self.results['resume_data']
        self.init_algorithm(**kwargs)
        # the relevant part from init_measurements()
        self._connect_measurements()

        self.resume_run_algorithm()  # continue with the actual algorithm
        self.final_measurements()
        self.results['finished_run'] = True
        results = self.save_results()
        self.logger.info('finished simulation run')
        return results

    def init_model(self):
        """Initialize a :attr:`model` from the model parameters.

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
        if isinstance(model_class_name, str):
            ModelClass = find_subclass(Model, model_class_name)
            if ModelClass is None:
                raise ValueError("can't find Model called " + repr(model_class_name))
        else:
            ModelClass = model_class_name
        params = self.options.subconfig('model_params')
        self.model = ModelClass(params)

    def init_state(self):
        """Initialize a tensor network :attr:`psi`.

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
        builder_class = self.options.get('initial_state_builder_class', 'InitialStateBuilder')
        if isinstance(builder_class, str):
            Builder = find_subclass(InitialStateBuilder, builder_class)
            if Builder is None:
                raise ValueError("can't find InitialStateBuilder called " + repr(builder_class))
        else:
            InitStateBuilder = builder_class
        params = self.options.subconfig('initial_state_params')
        initial_state_builder = Builder(self.model.lat, params, self.model.dtype)
        self.psi = initial_state_builder.run()
        if self.options.get('save_psi', True):
            self.results['psi'] = self.psi

    def init_algorithm(self, **kwargs):
        """Initialize the algorithm.

        Parameters
        ----------
        **kwargs :
            Extra keyword arguments passed on to the algorithm __init__(),
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
        if isinstance(alg_class_name, str):
            AlgorithmClass = find_subclass(Algorithm, alg_class_name)
            if AlgorithmClass is None:
                raise ValueError("can't find Algorithm called " + repr(alg_class_name))
        else:
            AlgorithmClass = alg_class_name
        self._init_algorithm(AlgorithmClass, **kwargs)
        self.engine.checkpoint.connect(self.save_at_checkpoint)
        con_checkpoint = list(self.options.get('connect_algorithm_checkpoint', []))
        for entry in con_checkpoint:
            self.engine.checkpoint.connect_by_name(*entry)

    def _init_algorithm(self, AlgorithmClass, **kwargs):
        params = self.options.subconfig('algorithm_params')
        self.engine = AlgorithmClass(self.psi, self.model, params, **kwargs)

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
        """
        self._connect_measurements()
        results = self.perform_measurements()
        results = {k: [v] for k, v in results.items()}
        self.results['measurements'] = results

    def _connect_measurements(self):
        if self.options.get('use_default_measurements', True):
            def_meas = self.default_measurements
        else:
            def_meas = []
        con_meas = list(self.options.get('connect_measurements', []))
        for entry in def_meas + con_meas:
            self.measurement_event.connect_by_name(*entry)

    def run_algorithm(self):
        """Run the algorithm. Calls ``self.engine.run()``."""
        self.engine.run()

    def resume_run_algorithm(self):
        """Resume running the algorithm. Calls ``self.engine.resume_run()``."""
        # usual algorithms have a loop with break conditions, which we can just resume
        self.engine.resume_run()

    def make_measurements(self):
        """Perform measurements and merge the results into ``self.results['measurements']``."""
        results = self.perform_measurements()
        previous_results = self.results['measurements']
        for k, v in results.items():
            previous_results[k].append(v)
        # done

    def perform_measurements(self):
        """Emits the :attr:`measurement_event` to call measurement functions and collect results.

        Returns
        -------
        results : dict
            The results from calling the measurement functions.
        """
        # TODO: save-guard measurements with try-except?
        # in case of a failed measurement, we should raise the exception at the end of the
        # simulation?
        results = {}
        returned = self.measurement_event.emit(results=results, simulation=self, psi=self.psi)
        # still save the values returned
        returned = [entry for entry in returned if entry is not None]
        if len(returned) > 0:
            msg = ("Some measurement function returned a value instead of writing to `results`.\n"
                   "Add it to measurement results as 'UNKNOWN'.")
            warnings.warn(msg)
            results['UNKNOWN'] = returned
        return results

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

    def fix_output_filenames(self):
        """Determine the output filenames.

        This function determines the :attr:`output_filename` and writes a one-line text into
        that file to indicate that we're running a simulation generating it.
        Further, :attr:`_backup_filename` is determined.

        Options
        -------
        .. cfg:configoptions :: Simulation

            output_filename : string | None
                Filename for output. The file ending determines the output format.
                None (default) disables any writing to files.
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
        output_filename = self.options.get("output_filename", None)
        overwrite_output = self.options.get("overwrite_output", False)
        skip_if_exists = self.options.get("skip_if_output_exists", False)
        if output_filename is None:
            self.output_filename = None
            self._backup_filename = None
            return
        out_fn = Path(output_filename)  # convert to Path
        self.output_filename = out_fn
        self._backup_filename = self.get_backup_filename(out_fn)

        if out_fn.exists():
            if skip_if_exists:
                self.options.touch(*self.options.unused)
                raise Skip("simulation output filename already exists: " + str(fn))
            if not overwrite_output and not self.loaded_from_checkpoint:
                # adjust output filename to avoid overwriting stuff
                for i in range(1, 100):
                    new_out_fn = out_fn.with_suffix('_' + str(i) + out_fn.suffix)
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
        if self.options.get("safe_write", True):
            return output_filename.with_suffix('.backup' + output_filename.suffix)
        else:
            return None

    def save_results(self):
        """Save the :attr:`results` to an output file.

        Performs a "safe" overwrite of :attr:`output_filename` by first moving the old file
        to :attr:`_backup_filename`, then writing the new file, and finally removing the backup.

        Calls :meth:`prepare_results_for_save` to allow last-minute adjustments to the saved
        :attr:`results`.
        """
        results = self.prepare_results_for_save()

        output_filename = self.output_filename
        backup_filename = self._backup_filename
        if output_filename is None:
            return results  # don't save to disk

        if output_filename.exists():
            # keep a single backup, previous backups are overwritten.
            if backup_filename is not None:
                if backup_filename.exists():
                    backup_filename.unlink()  # remove if exists
                output_filename.rename(backup_filename)
            else:
                output_filename.unlink()  # remove

        self.logger.info("saving results to disk")  # save results to disk
        hdf5_io.save(results, output_filename)

        if backup_filename is not None and backup_filename.exists():
            # successfully saved, so we can safely remove the old backup
            backup_filename.unlink()

        self._last_save = time.time()
        return results

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
        """
        results = self.results.copy()
        results['simulation_parameters'] = self.options.as_dict()
        # try to convert measurements into sigle arrays
        measurements = results['measurements'].copy()
        results['measurements'] = measurements
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
                warnings.warn("Saving took longer than 10% of `save_every_x_seconds`."
                              "Increase the latter to {0:.1f}".format(save_every))
                self.options['save_every_x_seconds'] = save_every
        # done


class Skip(ValueError):
    """Error raised if simulation output already exists."""
    pass


def run_simulation(simulation_class_name='GroundStateSearch',
                   simulation_class_kwargs=None,
                   **simulation_params):
    """Run the simulation with a simulation class.

    Parameters
    ----------
    simulation_class_name : str
        The name of a (sub)class of :class:`~tenpy.simulations.simulations.Simulation`
        to be used for running the simulaiton.
    simulation_class_kwargs : dict | None
        A dictionary of keyword-arguments to be used for the initializing the simulation.
    **simulation_params :
        Further keyword arguments as documented in the corresponding simulation class,
        see :cfg:config`Simulation`.

    Returns
    -------
    results :
        The results from running the simulation, i.e.,
        what :meth:`tenpy.simulations.Simulation.run()` returned.
    """
    SimClass = find_subclass(Simulation, simulation_class_name)
    if SimClass is None:
        raise ValueError("can't find simulation class called " + repr(simulation_class_name))
    if simulation_class_kwargs is None:
        simulation_class_kwargs = {}
    try:
        sim = SimClass(simulation_params, **simulation_class_kwargs)
        results = sim.run()
    except:
        # include the traceback into the log
        # this might cause a duplicated traceback if logging to std out is on,
        # but that's probably better than having no error messages in the log.
        Simulation.logger.exception("simulation abort with the following exception")
        raise  # raise the same error again
    return results


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
        Allows to update specific :cfg:config:`Simulation` parameters, ignored if `None`.
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
        raise Skip("Simulation already finished")
    sim_class_mod = checkpoint_results['version_info']['simulation_module']
    sim_class_name = checkpoint_results['version_info']['simulation_class']
    SimClass = hdf5_io.find_global(sim_class_mod, sim_class_name)
    if simulation_class_kwargs is None:
        simulation_class_kwargs = {}

    options = checkpoint_results['simulation_parameters']
    if update_sim_params is not None:
        update_recursive(options, update_sim_params)

    try:
        sim = SimClass.from_saved_checkpoint(checkpoint_results=checkpoint_results,
                                             **simulation_class_kwargs)
        results = sim.resume_run()
    except:
        # include the traceback into the log
        # this might cause a duplicated traceback if logging to std out is on,
        # but that's probably better than having no error messages in the log.
        Simulation.logger.exception("simulation abort with the following exception")
        raise  # raise the same error again
    return results
