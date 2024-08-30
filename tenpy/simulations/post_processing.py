"""Simple post-processing class and functions.

This module contains the :class:`DataLoader` class. This rationale behind this class is to make
loading data from a (finished) :class:`~tenpy.simulations.simulation.Simulation` run easier and more efficient.
This includes not loading the full ``.hdf5`` file into memory without having to directly interact with the
:class:`~tenpy.tools.hdf_io.Hdf5Loader`. Furthermore, an instance of the model, the lattice and the Brillouin Zone
of the Simulation can be directly accessed.
Similar to the :mod:`~tenpy.simulations.measurement` the functions provided in this module can be used by
the simulation class in a post-processing step. They follow the syntax
``def pp_function(DL, *, kwarg1, kwarg_2=default_2):``.
"""
# Copyright (C) TeNPy Developers, GNU GPLv3


import os
import warnings
from pathlib import Path
import numpy as np
import logging

from ..tools.spectral_function_tools import spectral_function, plot_correlations_on_lattice
from ..tools import hdf5_io
from ..tools.misc import to_iterable, get_recursive, set_recursive, find_subclass
from ..tools.params import Config
from ..models import Model

try:
    import h5py
    h5py_version = h5py.version.version_tuple
except ImportError:
    h5py_version = (0, 0)

__all__ = [
    'DataLoader', 'DataFiles', 'pp_spectral_function', 'pp_plot_correlations_on_lattice'
]


class DataLoader:
    r"""Post-processing class to handle IO and get Model and MPS from saved simulation data.

    Parameters
    ----------
    filename : str | Path, optional
        Path to a hdf5 file.
    simulation :
        An instance of a :class:`~tenpy.simulations.simulation.Simulation`
    data : dict, optional
        dictionary of simulation results (to be used in e.g. Jupyter Notebooks)

    Attributes
    ----------
    filename : str | Path
        Path to the hdf5 file.
    sim_params : dict
        Simulation parameters loaded from the hdf5 file.
        This includes the model parameters and algorithm parameters

    .. todo ::
        Include an Option for saving data into a ``.hdf5`` file without overwriting any results.
    """
    logger = logging.getLogger(__name__ + ".DataLoader")

    def __init__(self, filename=None, simulation=None, data=None):
        self.logger.info("Initializing\n%s\n%s\n%s", "=" * 80, self.__class__.__name__, "=" * 80)

        self._measurements = None
        self.sim_params = None

        if filename is not None:
            self.filename = Path(filename)
            self.logger.info(f"Loading data from {self.filename!s}")
            if self.filename.suffix == '.h5' or self.filename.suffix == '.hdf5':
                # create a h5group (which is open)
                self.logger.info(
                    f'Open file {self.filename.name}, when no context manager is used, it might be useful to '
                    f'call self.close()')

                h5group = h5py.File(self.filename, 'r')
                self._Hdf5Loader = hdf5_io.Hdf5Loader(h5group)
            else:
                self.logger.info(f"Not using hdf5 data-format.\nLoading data can be slow")
                # all data is loaded as other filenames
                self._all_data = hdf5_io.load(self.filename)

            self.sim_params = self._load('simulation_parameters')

        elif simulation is not None:
            self.sim = simulation
            self.logger.info(f"Initializing from {self.sim.__class__.__name__}")
            self.sim_params = self.sim.options.as_dict()
            self._all_data = self.sim.results

            self._model = self.sim.model
            if hasattr(self.sim, 'psi'):
                self._psi = self.sim.psi

        elif data is not None:
            self.logger.info(f"Initializing data loader from passed results")
            # all data is loaded as other filenames
            self._all_data = data
            self.sim_params = self._load('simulation_parameters')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if hasattr(self, '_Hdf5Loader'):
            self._Hdf5Loader.h5group.close()
            self.logger.info(f"Closed {self.filename}")

    def __repr__(self):
        if self.filename is not None:
            return f"DataLoader(filename={self.filename!r})"
        if hasattr(self, 'sim'):
            return f"DataLoader(simulation={self.sim!r})"
        return "Dataloader(data=...)"

    @property
    def measurements(self):
        if self._measurements is None:
            self._measurements = self._load('measurements', convert_to_numpy=True)
        return self._measurements

    def _load_recursive(self, paths, **kwargs):
        """Load data recursively into a dictionary.

        Parameters
        ----------
        paths : str or list of str
            Path(s) to load from the hdf5 file.
        **kwargs :
            keyword arguments to :meth:`_load`

        Returns
        -------
        dict
            data loaded from paths as dictionary
        """
        paths = to_iterable(paths)
        res = dict()
        for path in paths:
            value = self._load(path, **kwargs)
            set_recursive(res, path, value, separator='/', insert_dicts=True)
        return res

    def _load(self, path, prefix='', convert_to_numpy=False):
        """Load data from either the hdf5 file or from _all_data.

        For hdf5 files, this function enables one to load data from a file, without loading the whole file.
        I.e. only the data written into ``file[path]`` for path in paths is loaded.

        Parameters
        ----------
        path : str
            Path to load from either the hdf5 file or _all_data
        prefix : str, optional
            Prefix for paths.
        convert_to_numpy : bool, optional
            Try to convert loaded data to NumPy arrays.

        Returns
        -------
        res :
            data corresponding to path
        """
        key = prefix + path
        try:
            if hasattr(self, '_Hdf5Loader'):
                value = self._Hdf5Loader.load(key)
            elif hasattr(self, '_all_data'):
                value = get_recursive(self._all_data, key, separator='/')
            else:
                raise ValueError("Can't find any results.")
            if isinstance(value, Config):
                value = value.as_dict()
            if convert_to_numpy:
                value = self.convert_list_to_ndarray(value, key=key)
            return value
        except KeyError:
            warnings.warn(f"{key} does not exist!")

    def get_data_m(self, key, prefix='measurements/', convert_to_numpy=True):
        return self._load(key, prefix=prefix, convert_to_numpy=convert_to_numpy)

    def get_data(self, key, prefix='', convert_to_numpy=False):
        return self._load(key, prefix=prefix, convert_to_numpy=convert_to_numpy)

    def convert_list_to_ndarray(self, value, key):
        if isinstance(value, list):
            converted_value = np.array(value)
            if converted_value.dtype == np.dtype(object):
                self.logger.info("Can't convert %s to numpy array, proceed without conversion",
                                 key)
            else:
                value = converted_value
        return value

    @property
    def model(self):
        if not hasattr(self, '_model'):
            self._model = self._get_model()
        return self._model

    def get_model(self):
        """Deprecated in favor of the simpler property access via :attr:`DataLoader.model`."""
        warnings.warn("Use ``DataLoader.model`` instead of ``DataLoader.get_model()``",
                      FutureWarning, 2)
        return self.model

    def _get_model(self):
        model_class_name = self.sim_params['model_class']
        model_params = self.sim_params['model_params']
        model_class = find_subclass(Model, model_class_name)
        return model_class(model_params)

    @property
    def lat(self):
        return self.model.lat

    @property
    def BZ(self):
        return self.lat.BZ

    @property
    def psi(self):
        if not hasattr(self, '_psi'):
            self._psi = self.get_data('psi')
        return self._psi

    def get_all_keys_as_dict(self):
        if hasattr(self, '_Hdf5Loader'):
            return self._Hdf5Loader.get_all_hdf5_keys()
        elif hasattr(self, '_all_data'):
            return self._all_data
        else:
            raise ValueError("Can't find any results.")


class DataFiles:
    """Hold multiple DataLoader instances open, indexed by the filename.

    Acts like a dictionary mapping filenames to :class:`DataLoader`.
    Item access implicitly opens files that are not yet loaded.

    Parameters
    ----------
    files : list of str
        Filenames of output files to be opened.

    Examples
    --------
    .. doctest ::
        :skipif: True

        >>> data_files = DataFiles(['results/output_1.h5',
        ...                         'results_other/output_3.h5'])
        >>> data_files['results/output_1.h5']
        DataLoader(filename='results/output_1.h5')
        >>> data_files['results/output_2.h5']
        loading results/output_2.h5 ... successful
    """
    def __init__(self, files=None, folder=None):
        self._open_files = {} # filename -> DataLoader
        self._resolve_filenames = {}
        self._keys = []
        if files:
            for file in files:
                _ = self[file]
        if folder:
            self.load_from_folder(folder)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Close all files held open by self."""
        for dl in self._open_files.values():
            dl.close()

    def __getitem__(self, filename):
        normalized = self._normalize_filename(filename)
        data = self._open_files.get(normalized, None)
        if data is None:
            filename = str(filename)
            try:
                data = DataLoader(filename)
            except OSError as e:
                print(f"Erorr: failed to open {filename}")
                raise e from None
            self._keys.append(filename)
            self._open_files[normalized] = data
        return data

    def __setitem__(self, filename, data_loader):
        if filename is None:
            filename = data_loader.filename
        normalized = self._normalize_filename(filename)
        self._open_files[normalized] = data_loader
        return data_loader  # TODO: do we need to return this?

    def _normalize_filename(self, filename):
        filename = str(filename)
        resolved = self._resolve_filenames.get(filename, None)
        if resolved is None:
            resolved = str(Path(filename).resolve())
            self._resolve_filenames[filename] = resolved
        return resolved

    def keys(self):
        """Return paths of the files opened."""
        return self._keys

    def values(self):
        """Return iterator over the :class:`DataLoader` instances."""
        return self._open_files.values()

    def items(self):
        return zip(self._keys, self.values())

    def __delitem__(self, filename):
        normalized = self._normalize_filename(filename)
        self._open_files[normalized].close()
        del self._open_files[normalized]
        for key in self._keys:
            if self._resolve_filenames[key] == normalized:
                self._keys.remove(key)
                break
        for key, val in list(self._resolve_filenames.items()):
            if val == normalized:
                del self._resolve_filenames[key]

    def __repr__(self):
        if self._open_files:
            return "<DataFiles() with files\n    " + '\n    '.join(self.keys()) + ">"

    def load_from_folder(self, folder, glob="*.h5"):
        """Load all data files from a given folder."""
        files = Path(folder).glob(glob)
        for file in files:
            print(f"loading {file!s}", end=' ')
            try:
                _ = self[file]
            except OSError as e:
                print("... FAILED! Ignoring.")
            else:
                print("... successful")
        # done

    # TODO: tests for this

    # TODO: semi-automatically analyze sim_params and find differences?
    # TODO get pandas.DataFrame from changing keys


def pp_spectral_function(DL: DataLoader,
                         *,
                         correlation_key,
                         conjugate_correlation=False,
                         **kwargs):
    r"""Given a time dependent correlation function C(t, r), calculate its Spectral Function.

    After a run of :class:`~tenpy.simulations.time_evolution.TimeDependentCorrelation`, a :class:`DataLoader` instance
    should be passed, from which the underlying lattice and additional parameters (e.g. ``dt``) can be extracted.
    The `correlation_key` must coincide with the key of the time-dep. correlation function in the output of the
    Simulation.

    Parameters
    ----------
    DL : DataLoader
    correlation_key : str
    conjugate_correlation : bool | False
    **kwargs
        keyword arguments to :func:`~tenpy.tools.spectral_function_tools.spectral_function`
    """
    dt: float = DL.sim_params['algorithm_params']['dt']
    N_steps = DL.sim_params['algorithm_params'].get('N_steps', None)
    if N_steps is not None:
        dt *= N_steps

    time_dep_corr = DL.get_data_m(correlation_key)
    # conjugate correlation (i.e. to put r_0 to the right site)
    if conjugate_correlation is True:
        time_dep_corr = np.conjugate(time_dep_corr)

    return spectral_function(time_dep_corr, DL.lat, dt, **kwargs)


def pp_plot_correlations_on_lattice(DL: DataLoader,
                                    *,
                                    data_key,
                                    t_step=0,
                                    keys='nearest_neighbors',
                                    default_dir: str = 'plots',
                                    save_as: str = 'Correlations.pdf',
                                    markers='D',
                                    figsize=(8, 8),
                                    **kwargs):
    """Save a plot during post-processing to plot correlations on a lattice.

    Parameters
    ----------
    DL : :class:`DataLoader`
    data_key: str
        key for correlation function
    t_step: int
        time step to plot correlations on
    keys : str or list
        Valid keys are the ones defined in the corresponding lattice :attr:`~tenpy.models.lattice.Lattice.pairs`,
        e.g. `'nearest_neighbors'`
    markers : str or list
          a str for a single or a list of symbols for different sites within a unit cell given to plot sites
    figsize : tuple
    save_as : str
        string under which to save the plot (and extension)
    default_dir : str
        default (sub-) directory under which to save the plot
    kwargs :
        kwargs to :func:`~tenpy.tools.spectral_function_tools.plot_correlations_on_lattice`
    """
    import matplotlib.pyplot as plt
    if not os.path.exists(default_dir):
        os.mkdir(default_dir)

    keys = to_iterable(keys)
    markers = to_iterable(markers)
    lat = DL.lat
    correlations = DL.get_data_m(data_key)
    # loop over nearest_neighbors, next_nearest_neighbors, etc.
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f'Correlations {data_key}')
    if correlations.ndim == 3:
        correlations = correlations[t_step]
        ax.set_title(f'Correlations {data_key}, timestep {t_step}')
    for key in keys:
        plot_correlations_on_lattice(ax, lat, correlations, pairs=key, **kwargs)
    lat.plot_sites(ax, markers=markers)
    saving_path = os.path.join(default_dir, save_as)
    plt.savefig(saving_path, bbox_inches='tight')
