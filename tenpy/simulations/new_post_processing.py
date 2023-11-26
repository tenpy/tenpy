"""Simple post-processing"""
# Copyright 2020-2023 TeNPy Developers, GNU GPLv3

import os
import numpy as np
import logging

from ..tools import hdf5_io
from ..tools.misc import to_iterable, get_recursive, set_recursive, find_subclass
from ..tools.prediction import linear_prediction
from ..models import Model

try:
    import h5py

    h5py_version = h5py.version.version_tuple
except ImportError:
    h5py_version = (0, 0)

__all__ = ['DataLoader', 'to_lat_geometry', 'to_mps_geometry', 'spectral_function', 'fourier_transform_space',
           'fourier_transform_time', 'apply_gaussian_windowing', 'get_all_hdf5_keys']


class DataLoader:
    r"""PostProcessor class to handle IO and instantiating a model.

    Parameters
    ----------
    filename : str
        Path to a hdf5 file.

    Attributes
    ----------
    filename : str
        Path to the hdf5 file.
    sim_params : dict
        Simulation parameters loaded from the hdf5 file.
        This includes the model parameters and algorithm parameters
    measurement_keys : list or str
        List or str of passed measurements results
    """
    logger = logging.getLogger(__name__ + ".DataLoader")

    # somehow read out all keys of a filename recursively
    # set filename=None, simulation=None, measurement_keys=None in init,
    # then load based on whether ... instead of classmethod from simulation, from_file...
    # provide method to .get_data('key') and .get_simulation('key')

    def __init__(self, filename=None, simulation=None, measurement_keys=None):
        self.logger.info("Initializing\n%s\n%s\n%s", "=" * 80, self.__class__.__name__,
                         "=" * 80)
        if filename is not None:
            self.filename = filename
            self.logger.info(f"Loading data from {self.filename}")
            if self.filename.endswith('.h5') or self.filename.endswith('.hdf5'):
                # create a h5group (which is open)
                self.logger.info(f'Open file {self.filename}, when no context manager is used, it might be useful to '
                                 f'call self.close()')

                h5group = h5py.File(self.filename, 'r')
                self._Hdf5Loader = hdf5_io.Hdf5Loader(h5group)
            else:
                self.logger.info(f"Not using hdf5 data-format.\nLoading data can be slow")
                # all data is loaded as other filenames
                self._all_data = hdf5_io.load(self.filename)

            self.sim_params = self._load_recursive('simulation_parameters')['simulation_parameters']

        elif simulation is not None:
            self.sim = simulation
            self.logger.info(f"Initializing from {self.sim.__class__.__name__}")
            self.sim_params = self.sim.simulation_parameters
            self._model = self.sim.model
            if hasattr(self.sim, 'psi'):
                self._psi = self.sim.psi
            self._measurements = self.sim.results['measurements']

        self.measurement_keys = measurement_keys

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        self.save()

    def close(self):
        if hasattr(self, '_Hdf5Loader'):
            self._Hdf5Loader.h5group.close()
            self.logger.info(f"Closed {self.filename}")

    @property
    def measurements(self):
        if not hasattr(self, '_measurements'):
            if self.measurement_keys is not None:
                self._measurements = self._load_recursive(self.measurement_keys, prefix='measurements/',
                                                          convert_to_numpy=True)
            else:
                self._measurements = self._load('measurements', convert_to_numpy=True)
        return self._measurements

    def _load_recursive(self, paths, **kwargs):
        """Load data recursively into a dictionary

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
            data corresponding to path`
        """
        key = prefix + path
        try:
            if hasattr(self, '_Hdf5Loader'):
                value = self._Hdf5Loader.load(key)
            elif hasattr(self, '_all_data'):
                value = get_recursive(self._all_data, key, separator='/')
            else:
                raise ValueError("Can't find any results.")
            if convert_to_numpy is True:
                value = self.convert_list_to_ndarray(value)
            return value
        except KeyError:
            self.logger.warning(f"{key} does not exist!")

    def get_data(self, key, prefix='measurements/', convert_to_numpy=True):
        return self._load(key, prefix=prefix, convert_to_numpy=convert_to_numpy)

    @staticmethod
    def generate_unique_filename(filename, append_str=''):
        base, extension = os.path.splitext(filename)
        base += append_str
        new_filename = f"{base}{extension}"
        count_append_number = 1
        while os.path.exists(new_filename):
            new_filename = f"{base}_{count_append_number}{extension}"
            count_append_number += 1
        return new_filename

    def save(self):
        filename = self.generate_unique_filename(self.filename, append_str='_processed')
        self.logger.info(f"Saving Results to file: {filename}")
        return hdf5_io.save(self.measurement_keys, filename)

    @staticmethod
    def convert_list_to_ndarray(value):
        try:
            if isinstance(value, list):
                value = np.array(value)
                if value.dtype == np.dtype(object):
                    raise Exception("Can't convert results to numpy array")
        except Exception as e:
            logging.warning(f"{e}, proceeding without converting")
        return value

    @property
    def model(self):
        if not hasattr(self, '_model'):
            self._model = self.get_model()
        return self._model

    def get_model(self):
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
            self._psi = self.get_psi()
        return self._psi

    def get_psi(self):
        raise NotImplementedError()

    def get_all_keys_as_dict(self):
        if hasattr(self, '_Hdf5Loader'):
            h5_group = self._Hdf5Loader.h5group
            return get_all_hdf5_keys(h5_group)
        elif hasattr(self, '_all_data'):
            return self._all_data
        else:
            raise ValueError("Can't find any results.")

    # def apply(self, function: callable, *args, save_as: str = None, **kwargs):
    #     """
    #     Apply a function (which might depend on the model) and update the results dictionary
    #
    #     Parameters
    #     ----------
    #     function : callable
    #         Function to apply.
    #     args : tuple
    #         Positional arguments for the function.
    #     save_as : str, optional
    #         Key to save the result in the results' dictionary.
    #     kwargs : dict
    #         Keyword arguments for the function.
    #
    #     Returns
    #         -------
    #         any
    #             Result of the applied function.
    #         """
    #         result = function(self.model, *args, **kwargs)
    #         if save_as is not None:
    #             self.results[save_as] = result
    #         return result


def spectral_function(DL: DataLoader, time_dep_corr_key, *,
                      gaussian_window: bool = False, sigma: float = 0.4,
                      linear_predict: bool = False, m: int = None, p: int = None,
                      truncation_mode: str = 'renormalize', split: float = 0):
    # get lattice
    lat = DL.lat
    dt = DL.sim_params['algorithm_params']['dt']
    time_dep_corr = DL.get_data(time_dep_corr_key)

    # first we fourier transform in space C(r, t) -> C(k, t)
    time_dep_corr_lat = to_lat_geometry(lat, time_dep_corr, axes=-1)
    ft_space, k = fourier_transform_space(lat, time_dep_corr_lat)
    k_reduced = lat.BZ.reduce_points(k)
    # optional linear prediction
    if linear_predict is True:
        axis = 0  # since we assume that the time-series values are along the first dimension (n_tsteps, n_sites, ...)
        n_tsteps = ft_space.shape[axis]
        # linear prediction parameters
        if m is None:
            m = n_tsteps
        if p is None:
            p = n_tsteps // 3
        ft_space = linear_prediction(ft_space, m, p, axis=axis, truncation_mode=truncation_mode, split=split)
    # optional gaussian windowing
    if gaussian_window is True:
        ft_space = apply_gaussian_windowing(ft_space, sigma, axes=0)
    # fourier transform in time C(k, t) -> C(k, w) = S
    s_k_w, w = fourier_transform_time(ft_space, dt)
    return {'S': s_k_w, 'k': k, 'k_reduced': k_reduced, 'w': w}


def fourier_transform_space(lat, a):
    # make sure a is in lattice form not mps form
    if lat.dim == 1:
        ft_space = np.fft.fftn(a, axes=(1,))
        k = np.fft.fftfreq(ft_space.shape[1])
        # shifting
        ft_space = np.fft.fftshift(ft_space, axes=1)
        k = np.fft.fftshift(k)
        # make sure k is returned in correct basis
        k = (k * lat.reciprocal_basis).flatten()  # model is 1d
    else:
        # only transform over dims (1, 2), since 3 could hold unit cell index
        ft_space = np.fft.fftn(a, axes=(1, 2))
        k_x = np.fft.fftfreq(ft_space.shape[1])
        k_y = np.fft.fftfreq(ft_space.shape[2])
        # shifting
        ft_space = np.fft.fftshift(ft_space, axes=(1, 2))
        k_x = np.fft.fftshift(k_x)
        k_y = np.fft.fftshift(k_y)
        # make sure k is returned in correct basis
        b1, b2 = lat.reciprocal_basis
        k_x = b1 * k_x.reshape(-1, 1)  # multiply k_x by its basis vector (b1)
        k_y = b2 * k_y.reshape(-1, 1)  # multiply k_y by its basis vector (b2)
        # if k is indexed like (kx, ky) a coordinate (2d) is returned.
        k = k_x[:, np.newaxis, :] + k_y[np.newaxis, :, :]
        # e.g., if k_x, k_y hold the following (2d) points, the above is equivalent to
        # k_x = np.array([[1, 2, 3], [1, 1, 1]]).T
        # k_y = np.array([[-2, -2], [1, 2]]).T
        # k = np.zeros((3, 2, 2))
        # for i in range(len(k_y)):
        #     k[:, i, :] = k_x + k_y[i]
        # # or equivalently
        # # for i in range(len(k_x)):
        # #     k[i, :, :] = k_x[i] + k_y
    return ft_space, k


def fourier_transform_time(a, dt, axis=0):
    # fourier transform in time
    # (note that ifft is used, resulting in a minus sign in the exponential)
    ft_time = np.fft.ifft(a, axis=axis) * a.shape[axis]  # renormalize
    w = np.fft.fftfreq(len(ft_time), dt / (2 * np.pi))
    # shifting
    ft_time = np.fft.fftshift(ft_time, axes=axis)
    w = np.fft.fftshift(w)
    return ft_time, w


def apply_gaussian_windowing(a, sigma: float = 0.4, axes=0):
    """Simple gaussian windowing function along an axes.

    Applying a windowing function avoids Gibbs oscillation. tn are time steps 0, 1, ..., N

    Parameters
    ----------
    a : ndarray
        a ndarray where the time series is along axis `axes`
    sigma : float
        sigma used in gaussian window
    axes : int
        axes along which to apply the gaussian window

    Returns
    -------
    np.ndarray
    """
    # extract number of time-steps
    n_tsteps = a.shape[axes]
    tn = np.arange(n_tsteps)
    # create gaussian windowing function with the right shape
    gaussian_window = np.exp(-0.5 * (tn / (n_tsteps * sigma)) ** 2)
    # swap dimension which should be weighted (window applied to)
    # to last dim, so np broadcasting can be used
    swapped_a = np.swapaxes(a, -1, axes)
    # apply window
    weighted_arr = swapped_a * gaussian_window
    # swap back to original position
    return np.swapaxes(weighted_arr, axes, -1)


def to_lat_geometry(lat, a, axes=-1):
    return lat.mps2lat_values(a, axes=axes)


# TODO: make this function available on the lattice level, but allow an axis parameter for that
def to_mps_geometry(lat, a):
    """Bring measurement in lattice geometry to mps geometry.

    This assumes that the array a has shape (..., Lx, Ly, Lu),
    or if Lu = 1, (..., Lx, Ly)
    """
    mps_idx_flattened = np.ravel_multi_index(tuple(lat.order.T), lat.shape)
    dims_until_lat_dims = a.ndim - (lat.dim + 1)  # add unit cell dim
    if lat.Lu == 1:
        dims_until_lat_dims += 1
    a = a.reshape(a.shape[:dims_until_lat_dims] + (-1,))
    a = np.take(a, mps_idx_flattened, axis=-1)
    return a


# TODO: move this to hdf5_io
def get_all_hdf5_keys(h5_group):
    results = dict()
    for key in h5_group.keys():
        if isinstance(h5_group[key], h5py.Group):
            results[key] = get_all_hdf5_keys(h5_group[key])
        else:
            results[key] = h5_group[key]

    # if we are on the lowest recursion level, we only give the keys as sets
    if not any([isinstance(h5_group[key], h5py.Group) for key in h5_group.keys()]):
        results = set(results)
    return results
