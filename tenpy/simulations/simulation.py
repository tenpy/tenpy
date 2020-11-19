"""This module contains base classes for simulations.

The "simulation" class tries to put everything need for a simulation in a structured form and
collects task like initializing the tensor network state, model and algorithm classes,
running the actual algorithm, possibly performing measurements and saving the results.
"""
# Copyright 2020 TeNPy Developers, GNU GPLv3

import os
import time
import importlib

from ..tools import hdf5_io
from ..tools.params import asConfig
from ..tools.events import EventHandler
from ..tools.misc import find_subclass
from .. import version
from ..networks.mps import InitialStateBuilder

__all__ = ['Simulation', 'MPSSimulation']


class Simulation:
    """Base class for simulations.

    Parameters
    ----------
    options : dict-like
        The simulation parameters. Ideally, these options should be enough to fully specify all
        parameters of a simulation to ensure reproducibility.
        In addition, you should save the used version of the TeNPy library (and of your
        own model files etc.)
        (In practice, one should not hard-code parameters in the model files and algorithms.)

    Options
    -------
    .. cfg:config :: Simulation

        directory : string
            If not None (default), switch to that directory at the beginning of the simulation.
        output_filename : string | None
            Filename for output. The file ending determines the output format.
            None (defaul) disables any writing to files.
        overwrite_output : bool
            Whether an exisiting file may be overwritten.
            Otherwise, if the file already exists we try to replace
            ``filename.ext`` with ``filename_01.ext`` (and further increasing numbers).

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
    output_filename : str
        Filename for output.
    _backup_filename : str
        When writing a file a second time, instead of simply overwriting it, move it to there.
        In that way, we still have a non-corrupt version if something fails during saving.
    _last_save : float
        Time of the last call to :meth:`save_results`, initialized to startup time.

    """
    def __init__(self, options):
        self.options = options = asConfig(options, 'Simulation')
        cwd = options.get("directory", None)
        if cwd is not None:
            print(f"going into directory {cwd!r}")
            os.chdir(cwd)
        self.results = {
            'simulation_parameters': options.as_dict(),
            'version_info': self.get_version_info(),
        }
        self._last_save = time.time()
        self._fix_output_filenames()

    def run(self):
        """Run the whole simulation."""
        self.init_model()
        self.init_state()
        self.init_algorithm()
        self.init_measurements()
        self.run_algorithm()
        self.final_measurements()
        self.save_results()

    def init_model(self):
        """Initialize a model from the model parameters.

        Options
        -------
        .. cfg:configoptions :: Simulation

            model_class : string | class
                Mandatory. Name or class for the model to be initialized.
            model_params : dict
                Dictionary with parameters for the model; see the documentation of the
                corresponding `model_class`.

        """
        model_class_name = self.options["model_class"]  # no default value!
        if isinstance(model_class_name, str):
            ModelClass = find_subclass(tenpy.models.model.Model, model_class_name)
        else:
            ModelClass = model_class_name
        model_params = self.options.subconfig('model_params')
        self.model = ModelClass(model_params)

    def init_state(self):
        init_state_builder_class = self.options.get('init_state_builder', 'InitialStateBuilder')
        if isinstance(init_state_builder_class, str):
            InitStateBuilder = find_subclass(tenpy.networks.mps.InitialStateBuilder,
                                             init_state_builder_class)
        else:
            InitStateBuilder = init_state_builder_class
        init_state_params = self.options.subconfig('init_state_params', 'InitialStateBuilder')
        builder = InitStateBuilder(self.model.lat, init_state_params, self.model.dtype)
        self.psi = builder.build()

    def init_algorithm(self):
        # TODO need common algorithm base class
        raise NotImplementedError("subclasses should implement this")
        self.engine = AlgorithmClass(psi, model, algorithm_params)

    def run_algorithm(self):
        self.engine.run()

    def init_measurements(self):
        raise NotImplementedError("TODO")

    def final_measurements(self):
        raise NotImplementedError("TODO")

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

    def _fix_output_filenames(self):
        self.output_filename = options.get("output_filename", None)
        overwrite_output = options.get("overwrite_output", False)
        if output_filename is None:
            self.output_filename = None
            self._backup_filename = None
            return
        if os.path.exists(self.output_filename):
            if overwrite_output:
                path, filename = os.path.split(output_filename)
                backup_filename = os.path.join(path, "__old__" + filename)
                os.path.move(output_filename, backup_filename)
            else:
                # adjust output filename to avoid overwriting stuff
                root, ext = os.path.splitext(output_filename)
                for i in range(1, 100):
                    output_filename = '{0}_{1:2d}.{2}'.format(root, i, ext)
                    if not os.path.exists(output_filename):
                        break
                else:
                    raise ValueError("Refuse to make another copy. CLEAN UP!")
                warnings.warn("changed output filename to {0!r}".format(output_filename))
                path, filename = os.path.split(output_filename)
                backup_filename = os.path.join(path, "__old__" + filename)
        # we made sure that `output_filename` doesn't exist yet,
        # so create it as empty file to indicated that we want to save something there.
        open(output_filename, 'w').close()
        self.output_filename = output_filename
        self._backup_filename = backup_filename

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
            return  # do nothing

        if os.path.exists(output_filename):
            # keep a single backup, previous backups are overwritten.
            os.rename(output_filename, self._backup_filename)

        hdf5_io.save(self.results, output_filename)

        if os.path.exists(backup_filename):
            # successfully saved, so we can savely remove the old backup
            os.remove(backup_filename)
        self._last_save = time.time()

    def prepare_results_for_save(self):
        """Bring the `results` into a state suitable for saving.

        For example, this can be used to convert lists to arrays, to add more meta-data,
        or to clean up unnecessaily large entries.

        Returns
        -------
        results : dict
            A copy of :attr:`results` containing everything to be saved.
        """
        self.results['simulation_parameters'] = self.options.as_dict()  # TODO: necessary?
        # TODO: convert lists to arrays

    def save_at_checkpoint(self, alg_engine):
        save_every = self.options.get('save_every_x_seconds', None)
        if save_every is not None and time.time() - self._last_save > save_every:
            time_to_save = time.time()
            self.save_results()
            time_to_save = self._last_save - time_to_save
            assert time_to_save > 0.
            if time_to_save > 0.1 * save_every:
                save_every = 20 * time_to_save
                warnings.warn("Saving took longer than 10% of `save_every_x_seconds`."
                              "Increase the latter to {0:.1f}".format(save_every))
                self.options['save_every_x_seconds'] = save_every
        # done


class MPSSimulation(Simulation):
    InitStateBuilder = InitialStateBuilder

    def init_psi(self):
        sim_args = self.options
        init_state = sim_args['initial_state']
        model = self.model
        if isinstance(init_state, dict) and init_state.get('category', "") == 'from_file':
            changed = dmrg_submit.get_changed_parameters(sim_args, sim_args["change_parameters"])
            init_state['filename'] = init_state['filename'].format(**changed)
        self.psi = prepare_initial_state(init_state, lattice=model.lat, dtype=model.H_MPO.dtype)

        E_initial = self.results['E_initial'] = model.H_MPO.expectation_value(self.psi)
        print(f"E_initial = <psi_init|H|psi_init> = {E_initial:.12f}")

        if self.sim_args.get('environment_from_file', False) and \
                isinstance(init_state, dict) and init_state['category'] == 'from_file':
            filename = init_state['filename']
            print("loading environment from ", repr(filename))
            with h5py.File(filename, 'r') as f:
                env_data = hdf5_io.load_from_hdf5(f, "/env_data")
            # TODO: this renaming shouldn't be necessary; add corresponding DMRG parameter in TeNPy...
            self.dmrg_params['init_env_data'] = env_data

    def run_dmrg(self):
        self.eng = eng = dmrg.TwoSiteDMRGEngine(self.psi, self.model, self.dmrg_params)
        eng.checkpoint.connect(self.save_at_checkpoint)
        E, psi = eng.run()

        self.results['E_dmrg'] = E
        self.results['E_mpo'] = self.model.H_MPO.expectation_value(psi)
        print("E_mpo =", self.results['E_mpo'])
        print("N = ", psi.expectation_value('N'))

        self.results['psi'] = psi
        self.results['env_data'] = eng.env.get_initialization_data()
        self.results['sweep_stats'] = eng.sweep_stats
        self.results['update_stats'] = eng.update_stats

    def prepare_results_for_save(self):
        init_env_data = self.dmrg_params.get('init_env_data', {})
        for k in ['init_LP', 'init_RP']:
            if k in init_env_data:
                if isinstance(init_env_data[k], npc.Array):
                    init_env_data[k] = repr(init_env_data[k])
        super().save_results()
