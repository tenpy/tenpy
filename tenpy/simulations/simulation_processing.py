import numpy as np
import logging
from ..tools import hdf5_io
from ..tools.misc import update_recursive
from ..tools.params import asConfig


class SimulationPostProcessor:
    """Base class for post-processing. This class is intended to mostly handle
    data loading from a result of a :class:`Simulation`. This is done by reinitializing parts
    of the simulation under the attribute :attr:`sim` and reinitializing the model.
    This way, the post-processing steps of a quantity directly have access to the model parameters and
    the simulation parameters. The post-processing results are either saved in the same file, or written
    into a different file.

    Parameters
    ----------
    sim_results : str
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
    ...
    """

    logger = logging.getLogger(__name__ + ".PostProcessing")

    def __init__(self, sim_results, *, processing_params: dict = None):
        # convert parameters to Config object
        self.options = asConfig(processing_params, self.__class__.__name__)
        # setup_logging_()
        self.logger.info("Initializing\n%s\n%s\n%s", "=" * 80, self.__class__.__name__, "=" * 80)

        # TODO: fix logging of simulation ... why does simulation_class_kwargs disable logging not work (see below)
        self.sim = init_simulation_for_processing(filename=sim_results, simulation_class_kwargs={'setup_logging': False})
        self.sim.init_model()
        # self.sim.init_state() # TODO: do we want to initialize the state again (e.g., for measurements) ?

        self.model = self.sim.model
        self.lat = self.sim.model.lat
        self.BZ = self.sim.model.lat.BZ

        self.measurements = self.sim.results['measurements']
        self.sim_params = self.sim.results['simulation_parameters']

    @classmethod
    def from_file(cls):
        raise NotImplementedError("Not yet implemented")

    @classmethod
    def from_dict(cls):
        raise NotImplementedError("Not yet implemented")

    def run(self):
        raise NotImplementedError("Subclass must define :meth:`run`for post-processing")

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

    def save_results(self, results, key: str):
        self.sim.results[key] = results
        if self.options.get('append_results', True) is True:
            results = self.sim.save_results(results=self.sim.results)
            # if we don't pass results, we still need to avoid:
            # if self.options.get('save_resume_data', self.options['save_psi']):
            #     results['resume_data'] = self.engine.get_resume_data()
            # in prepare_results_for_save()
        else:
            output_filename = self.options.get('output_filename', None)
            if output_filename is None:
                raise KeyError("Missing an output filename")
            # TODO: generate output filename automatically
            # don't overwrite output filename
            results['simulation_parameters'] = self.sim_params
            hdf5_io.save(results, output_filename)
        return results


class SpectralFunctionProcessor(SimulationPostProcessor):
    """Post-processing class for the :class:`SpectralSimulation`.
    This class helps calculating spectral functions from the given correlations of
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
    windowing_function : Config
        holding all parameters for a windowing function
        windowing_function = {'window': gaussian,
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

    def run(self):
        results = {'post_process_params': dict()}
        if hasattr(self, "linear_pred_params"):
            results['post_process_params']['linear_prediction'] = self.linear_pred_params.as_dict()
        if hasattr(self, "windowing_function_params"):
            results['post_process_params']['windowing'] = self.windowing_function_params.as_dict()

        for key, value in self.measurements.items():
            if "spectral_function_t" in key:
                # convert results to numpy array if possible, should we just copy the result here?
                value = self.convert_to_ndarray(value)
                # compute spectral function
                S, k, w = self.compute_spectral_function(value)
                # store results
                results[key] = dict()
                results[key]['spectral_function'] = S
                results[key]['k'] = k
                results[key]['k_reduced'] = self.BZ.reduce_points(k)
                results[key]['w'] = w
                # TODO: storing simulation output again, should only be a hard link for .hdf5
                results[key]['raw_sim_data'] = value

        if self.sim.results.get('spectral_function') is not None:
            self.logger.warning('Overwriting previous post_processing_results')
        # save results
        self.save_results(results, 'spectral_function')

    def compute_spectral_function(self, m_corr):
        m_corr_lat = self._to_lat_geometry(m_corr)
        ft_space, k = self.fourier_transform_space(m_corr_lat)

        if hasattr(self, 'linear_prediction_params'):
            raise NotImplementedError("Linear prediction is not yet implemented")
            # ft_space = self._to_mps_geometry(ft_space)
            # TODO: provide linear prediction
            # ft_space = self.linear_predict(ft_space)
            # ft_space = self._to_lat_geometry(ft_space)

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
        # if length of the unit cell is 1, order still returns it
        if self.lat.Lu == 1:
            return a[:, *tuple(self.lat.order[..., :-1].T)]
        else:
            return a[:, *tuple(self.lat.order.T)]

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
        dicitonary saved at a simulation checkpoint.
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

    sim = SimClass.from_saved_checkpoint(checkpoint_results=checkpoint_results,
                                        **simulation_class_kwargs)
    return sim
