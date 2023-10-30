"""This module contains classes for post-processing for tenpy :class:`Simulation`.

The :class:`SimulationPostProcessor` mostly handles initialization and is
meant as a base class for post-processing. The :class:`SpectralPostProcessor`
should be used together with the simulation :class:`SpectralSimulation`.
This module also provides basic function for linear prediction for multidimensional time
series data.
"""
# Copyright 2020-2023 TeNPy Developers, GNU GPLv3

import numpy as np
import scipy
import logging

from . import simulation
from .simulation import *
from ..tools import hdf5_io
from ..tools.misc import update_recursive
from ..tools.params import asConfig

# __all__ = simulation.__all__ + ['SimulationPostProcessor', 'SpectralFunctionProcessor']
__all__ = ['SimulationPostProcessor', 'SpectralFunctionProcessor', 'init_simulation_for_processing',
           'linear_prediction', 'get_lpc', 'alpha_and_c', 'autocorrelation']


class SimulationPostProcessor:
    """Base class for post-processing.

    This class is intended to mostly handle data loading from a result of a :class:`Simulation`.
    This is done by reinitializing parts of the simulation under the attribute :attr:`sim` and reinitializing the model.
    This way, the post-processing steps of a quantity directly have access to the model parameters and
    the simulation parameters. The post-processing results are either saved in the same file, or written
    into a different file.

    Parameters
    ----------
    sim_results : str or None
        Filename of a finished simulation run # TODO: include option to pass as dictionary
    processing_params : dict-like
        For command line use, a .yml file should hold the information.
        These parameters are converted to a (dict-like Config object) :class:`~tenpy.tools.params.Config`.
        This should hold the following information
        processing_params = {'append_results': False,
        'output_filename': 'post_processed_results.h5',
        'processing_step1': {...}, # (of subclass)
        'processing_step2': {...}, # (of subclass)}

    Attributes
    ----------
    sim
        Instance of a :class:`Simulation`
    measurements
        Computed measurements from running the simulation
    """

    logger = logging.getLogger(__name__ + ".PostProcessing")
    inside_simulation = False

    def __init__(self, sim_results, *, processing_params: dict = None):
        # Config can't take None, but empty dict
        processing_params = processing_params if processing_params is not None else dict()
        # convert parameters to Config object
        self.options = asConfig(processing_params, self.__class__.__name__)
        # setup_logging_()
        self.logger.info("Initializing post-processing:\n%s\n%s\n%s", "=" * 80, self.__class__.__name__,
                         "=" * 80)
        # TODO: fix logging of simulation ... why does simulation_class_kwargs disable logging not work (see below)
        if not self.inside_simulation:
            self.sim = init_simulation_for_processing(filename=sim_results,
                                                      simulation_class_kwargs={'setup_logging': False})
            self.sim.init_model()
            # TODO: do we want to initialize the state again (e.g., for measurements) ?
            # self.sim.init_state()
        # shorthand notation
        self.model = self.sim.model
        self.lat = self.sim.model.lat
        self.BZ = self.sim.model.lat.BZ

        self.measurements = self.sim.results['measurements']
        self.sim_params = self.sim.results['simulation_parameters']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.options.warn_unused(True)

    @classmethod
    def from_simulation(cls, sim: Simulation, *, processing_params: dict = None):
        if (not hasattr(sim, 'model')) or (not hasattr(sim, 'results')):
            raise TypeError("Can't perform post-processing on simulation w/o model or results")
        post_processing_sim = cls.__new__(cls)
        post_processing_sim.inside_simulation = True
        post_processing_sim.sim = sim  # TODO: bad idea?
        post_processing_sim.__init__(None, processing_params=processing_params)
        return post_processing_sim

    def run(self):
        results, key = self.run_processing()
        results = self.save_results(results, key)
        self.logger.info('finished post-processing run\n%s', "=" * 80)
        return results

    def run_processing(self):
        raise NotImplementedError("Subclass must define :meth:`run` for post-processing")

    def save_results(self, results, key: str):
        """Saving results from a post-processing step.

        If post-processing was done inside a simulation, append results
        to dictionary of simulation results.

        Parameters
        ----------
        results : dict or ndarray
        key : str
            The name under which the results should be stored.
        """
        if self.options.get('append_results', False) is False:
            results_dict = dict()
            results_dict['simulation_parameters'] = self.sim_params
            results_dict[key] = results
            # TODO: generate output filename automatically and don't overwrite output filename
            output_filename = self.options.get('output_filename', None)
            if output_filename is not None:
                hdf5_io.save(results_dict, output_filename)
            else:
                self.logger.warning("Missing an output filename, results are returned\
                                     but not saved")
            results = results_dict
        else:
            if self.sim.results.get(key, None) is not None:
                self.logger.warning('Overwriting previous post_processing_results')
            # append results to sim.results
            self.sim.results[key] = results
            if self.inside_simulation:
                # saving is handled by Simulation class
                results = None
            else:
                results = self.sim.save_results(results=self.sim.results)
                # TODO: unnecessarily writes all results to disk again ?
                # if we don't pass results, we still need to avoid:
                # if self.options.get('save_resume_data', self.options['save_psi']):
                #     results['resume_data'] = self.engine.get_resume_data()
                # in prepare_results_for_save()
        return results

    @staticmethod
    def convert_to_ndarray(value):
        try:
            if not isinstance(value, np.ndarray):
                value = np.array(value)
                if value.dtype == np.dtype(object):
                    raise Exception("Can't convert results to numpy array")
        except Exception:
            raise Exception("Can't convert results to numpy array")
        return value


class SpectralFunctionProcessor(SimulationPostProcessor):
    """Post-processing class for the :class:`SpectralSimulation`.

    This class helps to calculate spectral functions from the given correlations of
    a run of a :class:`SpectralSimulation`. The options to perform additional post-processing steps,
    namely applying a windowing function and using linear prediction are provided and
    controlled by the processing_params of the class.

    Parameters
    ----------
    sim_results : dict-like
    processing_params : dict-like
        processing_params = {'linear_prediction': {...},
        'windowing_function': {...},
        'append_results': False,
        'output_filename': 'post_processed_results.h5'}

    Attributes
    ----------
    linear_prediction : Config
        holding all parameters for a linear prediction, example:
        linear_prediction = {'m': 50,
        'p': 30,
        'cyclic: False,
        'trunc_mode': 'cutoff',  # or 'renormalize', or 'conjugate'
        'epsilon': 10e-7,
        'mode': 'individual'}  # or 'full'

    windowing_function : Config
        holding all parameters for a windowing function, example:
        windowing_function = {'window': 'gaussian',
        'sigma': 0.4}
    """

    def __init__(self, sim_results, *, processing_params: dict = None):
        super().__init__(sim_results, processing_params=processing_params)
        self.linear_prediction = self.options.get('linear_prediction', default=None)
        self.windowing_function = self.options.get('windowing_function', default=None)
        # convert processing steps to Config, if used.
        if self.linear_prediction is not None:
            self.linear_prediction_params = asConfig(self.linear_prediction, "Linear Prediction setup")
        if self.windowing_function is not None:
            self.windowing_function_params = asConfig(self.windowing_function, "Windowing function setup")

    def run_processing(self):
        results = {'post_process_params': dict()}
        if hasattr(self, "linear_prediction_params"):
            results['post_process_params']['linear_prediction'] = self.linear_prediction_params.as_dict()
        if hasattr(self, "windowing_function_params"):
            results['post_process_params']['windowing'] = self.windowing_function_params.as_dict()

        for key, value in self.measurements.items():
            if "spectral_function_t" in key:
                # convert results to numpy array if possible, should we just copy the result here?
                value = self.convert_to_ndarray(value)
                # compute spectral function
                self.logger.info(f'computing spectral function of {key}')
                S, k, w = self.compute_spectral_function(value)
                k_reduced = self.BZ.reduce_points(k)
                # store results
                results[key] = dict()
                results[key]['spectral_function'] = S
                results[key]['k'] = k
                results[key]['k_reduced'] = k_reduced
                results[key]['w'] = w
                # TODO: storing simulation output again, should only be a hard link for .hdf5
                results[key]['raw_sim_data'] = value
        dict_key = 'spectral_function'
        return results, dict_key

    def compute_spectral_function(self, m_corr):
        m_corr_lat = self._to_lat_geometry(m_corr)
        ft_space, k = self.fourier_transform_space(m_corr_lat)

        if hasattr(self, 'linear_prediction_params'):
            ft_space = self._to_mps_geometry(ft_space)
            ft_space = self.linear_predict(ft_space)
            ft_space = self._to_lat_geometry(ft_space)

        if hasattr(self, 'windowing_function_params'):
            ft_space = self.apply_windowing_function(ft_space)

        S, w = self.fourier_transform_time(ft_space)
        return S, k, w

    def check_lattice_consistency(self, a):
        assert isinstance(a, np.ndarray), "Result should be given back as a numpy array"
        assert a.ndim == 2, "result of Spectral Simulation should be of shape (t_steps, mps_idx)"
        assert a.shape[1] == self.lat.N_sites, "Number of Sites differs from number of MPS tensors"

    def _to_lat_geometry(self, a):
        self.check_lattice_consistency(a)
        return self.lat.mps2lat_values(a, axes=-1)

    def _to_mps_geometry(self, a):
        """This assumes that the array a has shape (..., Lx, Ly, Lu), if Lu = 1, (..., Lx, Ly)"""
        mps_idx_flattened = np.ravel_multi_index(tuple(self.lat.order.T), self.lat.shape)
        dims_until_lat_dims = a.ndim - (self.lat.dim + 1)  # add unit cell dim
        if self.lat.Lu == 1:
            dims_until_lat_dims += 1
        a = a.reshape(a.shape[:dims_until_lat_dims] + (-1,))
        a = np.take(a, mps_idx_flattened, axis=-1)
        return a
        # this is the same as (which linting doesn't accept unfortunately)
        # order = self.lat.order
        # if self.lat.Lu == 1:
        #     order = order[..., :-1]
        # return a[..., *order.T]

    def gaussian(self, n_steps: int):
        """Simple gaussian windowing function. Applying a windowing function
        avoids Gibbs oscillation. tn are time steps 0, 1, ..., N"""
        tn = np.arange(n_steps)
        sigma = self.windowing_function_params.get("sigma", 0.4)
        return np.exp(-0.5 * (tn/(n_steps * sigma)) ** 2)

    def lorentzian(self):
        eta = self.windowing_function_params.get("eta", 0.4)
        raise NotImplementedError("More windowing functions will be implemented in the future")

    def apply_windowing_function(self, a, axis=0):
        # TODO: include other options (Lorentzian, ...)
        window_name = self.windowing_function_params.get('window', 'gaussian')
        if not hasattr(self, window_name):
            self.logger.warning(f"{window_name} not defined, continuing without applying a windowing function.")
            return a
        window_function = getattr(self, window_name)
        window = window_function(a.shape[axis])
        # make window compatible with numpy broadcasting
        window = window.reshape(window.shape + (1,) * (len(a.shape) - 1))
        self.windowing_function_params.warn_unused(True)
        return np.swapaxes(np.swapaxes(a, 0, axis) * window, axis, 0)

    def linear_predict(self, x):
        args = self.readout_linear_pred_params(x)  # m, p, trunc_mode, cyclic, epsilon, mode
        return linear_prediction(x, *args)

    def readout_linear_pred_params(self, data):
        split = self.linear_prediction_params.get('split', 0)
        assert split < 1, "Split must be a float between 0 and 1"
        x_length = int(len(data)*split)
        m = self.linear_prediction_params.get('m', x_length // 2)
        cyclic = self.linear_prediction_params.get('cyclic', False)
        p = self.linear_prediction_params.get('p', (x_length - 1) // 3)
        epsilon = self.linear_prediction_params.get('epsilon', 10e-7)
        trunc_mode = self.linear_prediction_params.get('trunc_mode', 'cutoff')
        mode = self.linear_prediction_params.get('mode', 'individual')
        self.linear_prediction_params.warn_unused(True)
        return m, p, split, trunc_mode, mode, cyclic, epsilon

    def fourier_transform_space(self, a):
        if self.lat.dim == 1:
            ft_space = np.fft.fftn(a, axes=(1,))
            k = np.fft.fftfreq(ft_space.shape[1])
            # shifting
            ft_space = np.fft.fftshift(ft_space, axes=1)
            k = np.fft.fftshift(k)
            # make sure k is returned in correct basis
            k = (k*self.lat.reciprocal_basis).flatten()  # model is 1d
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
            b1, b2 = self.lat.reciprocal_basis
            k_x = b1*k_x.reshape(-1, 1)  # multiply k_x by its basis vector (b1)
            k_y = b2*k_y.reshape(-1, 1)  # multiply k_y by its basis vector (b2)
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

    def fourier_transform_time(self, a):
        # fourier transform in time (note that ifft is used, resulting in a minus sign in the exponential)
        ft_time = np.fft.ifftn(a, axes=(0,)) * a.shape[0]  # renormalize
        dt = self.sim_params['algorithm_params']['dt']
        w = np.fft.fftfreq(len(ft_time), dt/(2*np.pi))
        # shifting
        ft_time = np.fft.fftshift(ft_time, axes=0)
        w = np.fft.fftshift(w)
        return ft_time, w


def init_simulation_for_processing(*,
                                    filename=None,
                                    checkpoint_results=None,
                                    update_sim_params=None,
                                    simulation_class_kwargs=None):
    """Re-initialize a simulation from a given checkpoint without running it.

    This is the same as :func:`init_simulation_from_checkpoint` but still initializes
    the simulation if finished_run is True.

    (All parameters have to be given as keyword arguments.)

    Parameters
    ----------
    filename : None | str
        The filename of the checkpoint to be loaded.
        You can either specify the `filename` or the `checkpoint_results`.
    checkpoint_results : None | dict
        Alternatively to `filename` the results of the simulation so far, i.e. directly the data
        dictionary saved at a simulation checkpoint.
    update_sim_params : None | dict
        Allows to update specific :cfg:config:`Simulation` parameters; ignored if `None`.
        Uses :func:`~tenpy.tools.misc.update_recursive` to update values, such that the keys of
        `update_sim_params` can be recursive, e.g. `algorithm_params/max_sweeps`.
    simulation_class_kwargs : None | dict
        Further keyword arguments given to the simulation class, ignored if `None`.

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
    # TODO: handle logging correctly
    if checkpoint_results['finished_run'] is False:
        logging.warning("Performing post-processing on an unfinished simulation run")

    sim_class_mod = checkpoint_results['version_info']['simulation_module']
    sim_class_name = checkpoint_results['version_info']['simulation_class']
    SimClass = hdf5_io.find_global(sim_class_mod, sim_class_name)

    if simulation_class_kwargs is None:
        simulation_class_kwargs = {}

    # TODO: does it make sense to update parameters for post-processing ? -> probably not
    options = checkpoint_results['simulation_parameters']
    if update_sim_params is not None:
        update_recursive(options, update_sim_params)

    sim = SimClass.from_saved_checkpoint(checkpoint_results=checkpoint_results, **simulation_class_kwargs)
    return sim


# Linear Prediction
def linear_prediction(x: np.ndarray, m: int, p: int, split: float = 0, trunc_mode: str = 'renormalize',
                      two_d_mode: str = 'individual', cyclic: bool = False, epsilon: float = 10e-7) -> np.ndarray:
    """Linear prediction for m time steps, based on the last p data points of the time series x.

    Parameters
    ----------
    x : ndarray
        time series data (n_tsteps, ...)
    m : int
        number of timesteps to predict
    p : int
        number of last points to base linear prediction on
    split : float
        where 0<=split<1; this splits the dataset into a portion that is cut off in order to discard effects
        not relevant for the long term behaviour of the system
    trunc_mode : str
        the truncation mode (default is 'renormalize', meaning their absolute
        value will be set to 1) used for truncating the eigenvalues. Other options are 'cutoff'  and 'conjugate'
    two_d_mode : str
        the mode to use for 2D data (default is 'full', in which case the mse of the entire datapoints
        along the second (non-time) direction is taken). An alternative is 'individual', where the
        mse calculation is only base on the time series data points along each 'row' in the second dimension
    cyclic : bool
        whether to use the cyclic autocorrelation or not (see :meth:`autocorrelation`)
    epsilon : float
        regularization constant, in case matrix can not be inverted

    Returns
    -------
    ndarray
        with shape (n_tsteps + m, ...)
    """
    # data validation
    if not isinstance(m, int) or not isinstance(p, int):
        raise TypeError("Parameters 'm' and 'p' must be integers.")
    if not isinstance(cyclic, bool):
        raise TypeError("Cyclic must be a boolean value")
    if not isinstance(epsilon, float):
        raise TypeError("Parameter 'epsilon' must be a float.")
    if not isinstance(trunc_mode, str) or not isinstance(two_d_mode, str):
        raise TypeError("Parameter 'trunc_mode' and 'two_d_mode' must be a strings.")
    if trunc_mode.casefold() not in ('cutoff', 'renormalize', 'conjugate'):
        raise ValueError("Parameter 'trunc_mode' must be 'renormalize' (default), 'cutoff', or 'conjugate'.")
    if two_d_mode.casefold() not in ('full', 'individual'):
        raise ValueError("Parameter 'two_d_mode' must be 'individual' (default) or 'full'.")

    # get index of where to split dataset
    split_idx = int(len(x)*split)
    if two_d_mode == 'full':
        correlations = list()
        # get the last p correlations
        for j in range(p + 1):
            correlations.append(autocorrelation(x[split_idx:], 0, j, cyclic=cyclic))
        correlations = np.array(correlations)
        lpc = get_lpc(correlations, epsilon=epsilon)
        alpha, c = alpha_and_c(x[split_idx:], lpc, trunc_mode=trunc_mode)
        predictions = [np.tensordot(c, alpha ** m_i, axes=(0, 0)) for m_i in range(1, m + 1)]
        return np.concatenate([x, predictions])
    else:
        predictions = list()
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        for x_indiv_site in x.T:  # transpose array since it is given in (t_steps, site)
            correlations = list()
            # get the last p correlations
            for j in range(p + 1):
                correlations.append(autocorrelation(x_indiv_site[split_idx:], 0, j, cyclic=cyclic))
            correlations = np.array(correlations)
            lpc = get_lpc(correlations, epsilon=epsilon)
            alpha, c = alpha_and_c(x_indiv_site[split_idx:], lpc, trunc_mode=trunc_mode)
            predictions_site = [np.tensordot(c, alpha ** m_i, axes=(0, 0)) for m_i in range(1, m + 1)]
            predictions.append(np.concatenate([x_indiv_site, predictions_site]))
        return np.array(predictions).T  # transpose back


def autocorrelation(x, i, j, cyclic=False, cut_idx=0):  # TODO: introduce cut_pad_zero: bool = False
    """Compute the autocorrelation :math:`R_{XX}(i, j) = E\{x(n-i) \cdot x(n-j)\}`. Note
    that this can be rewritten as :math:`R_{XX}(i - j) = E\{x(n) \cdot x(n-(j-i))\}`

    Parameters
    ----------
    x : ndarray
        time series data, first dimension must be corresponding to the time
    i: delay of i steps of input signal
    j: delay of j steps of copy of input signal
    cyclic : bool
        whether the cyclic autocorrelation is used or not. If set to False (default), the
        data points with indices smaller than 0 are all set to zero. If set to True all indices
        are interpreted mod N (where N is the length of x) -> cyclic.
    cut_idx : int
        index at which data is cut off (e.g. to remove short time effects)
    Returns
    -------
    float
    """
    N = len(x)
    if cyclic is True:  # interprets the indices mod N
        # essentially we compute: np.sum([np.sum(x[n - i] * np.conj(x[n - j])) for n in range(N)])
        x_shift_i = np.roll(x[cut_idx:], -i, 0)
        x_shift_j = np.roll(x[cut_idx:], -j, 0)
        return np.sum(x_shift_i * np.conj(x_shift_j))
    else:
        # for 1D, this is equivalent to: np.correlate(x, x, mode='full')[x.size - 1 + (j - i)]
        # however, we allow a general input
        corrs = []
        for n in range(cut_idx, N):
            if (n - j + i) >= N:
                corrs.append(0)
            elif (n - j + i) < 0:
                corrs.append(0)
            else:
                corrs.append(np.sum(x[n] * np.conj(x[n - j + i])))
        return np.sum(corrs)


def get_lpc(correlations, epsilon=10e-7):
    """Function to obtain the linear prediction coefficients (lpc),
    for given correlations of a time series x.

    Parameters
    ----------
    correlations : ndarray
        containing the last p+1 correlations of the time series
        [E{x(n)*x(n-0)}, E{x(n)*x(n-1)}, ..., E{x(n)*x(n-p)}
    epsilon : float
        regularization constant, in case matrix can not be inverted

    Returns
    -------
    ndarray
        1D array containing the p linear prediction coefficients [a_p, a_{p-1}, ..., a_1] from
        the correlations of the time series x
    """
    r = correlations[1:]
    R = scipy.linalg.toeplitz(correlations[:-1])
    # getting array of coefficients
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        try:
            R_inv = np.linalg.inv(R + np.eye(len(R))*epsilon)
        except Exception:
            raise Exception(f"Matrix could not be inverted, even after applying regularization \
                            with epsilon = {epsilon}, maybe try a higher regularization parameter.")
    return R_inv @ r


def alpha_and_c(x, lpc, trunc_mode='cutoff'):
    """Get the eigenvalues and coefficients for linearly predicting the time series x.
    If necessary, truncate the eigenvalues.

    Parameters
    ----------
    x : ndarray
        time series data (n_tsteps, other)
    lpc : ndarray
        1D array containing the p linear prediction coefficients [a_p, a_{p-1}, ..., a_1] from the correlations of x
    trunc_mode : str
        the truncation mode (default is 'cutoff', which means that those eigenvalues will be cut)
        used for truncating the eigenvalues. Other options are 'renormalize' (meaning their absolute
        value will be set to 1) and 'conjugate'

    Returns
    -------
    evals : ndarray
    c : ndarray
    """
    A = np.diag(np.ones(len(lpc) - 1), -1).astype(lpc.dtype)
    A[0] = lpc

    evals, evects = np.linalg.eig(A)  # Note that A is not symmetric!
    if trunc_mode == 'renormalize':
        evals[np.abs(evals) > 1] = evals[np.abs(evals) > 1] / np.abs(evals[np.abs(evals) > 1])
    elif trunc_mode == 'cutoff':
        evals[np.abs(evals) > 1] = 0
    elif trunc_mode == 'conjugate':
        evals[np.abs(evals) > 1] = 1 / np.conj(evals[np.abs(evals) > 1])

    x_tilde_N = x[-len(lpc):][::-1]
    shape = (-1,) + (x.ndim - 1) * (1,)
    try:
        evects_inv = np.linalg.inv(evects)
    except np.linalg.LinAlgError:
        # Regularization is only done here to avoid an Exception for ill defined correlations (e.g. all zero)
        evects_inv = np.linalg.inv(evects + np.eye(len(evects))*10e-07)
        logging.warning("Matrix of eigenvectors could not be inverted, linear prediction will probably fail...")

    c = np.tensordot(evects_inv, x_tilde_N, axes=(1, 0)) * evects[0, :].reshape(shape)
    return evals, c
