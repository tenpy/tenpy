"""Functions to perform measurments.

All measurement functions provided in this module support the interface used by the simulation
class, i.e. they take the parameters documented in :func:`measurement_index` and write
the measurement results into the `results` dictionary taken as argument.

As explained in :doc:`/intro/simulations`, you can easily add custom measurement functions.
"""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

from ..networks.mpo import MPOEnvironment
from ..tools.misc import get_recursive
from ..tools import process
import warnings

__all__ = [
    'measurement_index', 'bond_dimension', 'bond_energies', 'simulation_parameter', 'energy_MPO',
    'entropy', 'onsite_expectation_value', 'correlation_length', 'psi_method', 'evolved_time'
]


def measurement_index(results, psi, model, simulation, key='measurement_index'):
    """'Measure' the index of how many mearuements have been performed so far.

    The parameter description below documents the common interface of all measurement
    functions that can be registered to simulations.

    See :doc:`/intro/simulations` for the general setup using measurements.

    .. versionadded:: 0.10.0

        The `model` parameter is new! Any measurement function for simulations now has to accept
        this as keyword argument!

    Parameters
    ----------
    results : dict
        A dictionary with measurement results performed so far.
        Instead of returning the result, the output should be written into this dictionary
        under an appropriate key (or multiple keys, if applicable).
    psi :
    model :
        Tensor network state and matching model (with same sites/indexing) to be measured.
        Usually shorthand for ``simulation.psi`` and ``simulation.model``, respectively,
        but can be different, e.g., when grouping sites.
        See :meth:`~tenpy.simulations.simulation.Simulation.get_measurement_psi_model`.
    simulation : :class:`~tenpy.simulations.simulation.Simulation`
        The simulation class. This gives also access to the `model`, algorithm `engine`, etc.
    key : str
        (Optional.) The key under which to save in `results`.
    **kwargs :
        Other optional keyword arguments for individual measurement functions.
        Those are documented inside each measurement function.
    """
    index = len(simulation.results.get('measurements', {}).get(key, []))
    results[key] = index


def bond_dimension(results, psi, model, simulation, key='bond_dimension'):
    """'Measure' the bond dimension of an MPS.

    Parameters
    ----------
    results, psi, model, simulation, key :
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    """
    results[key] = psi.chi


def bond_energies(results, psi, model, simulation, key='bond_energies'):
    """Measure the energy of an MPS.

    Parameters
    ----------
    results, psi, model, simulation, key :
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    """
    results[key] = simulation.model.bond_energies(psi)


def simulation_parameter(results, psi, model, simulation, recursive_key, key=None, **kwargs):
    """Dummy meausurement of a simulation parameter.

    This can be useful e.g. if you have a time-dependent hamiltonian parameter.

    Parameters
    ----------
    results, psi, simulation, key :
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    recursive_key : str
        Recursive key of the simulation parameter to be read out.
    default, separators :
        Remaining arguments of :func:`~tenpy.tools.misc.get_recursive`.
    """
    if key is None:
        key = recursive_key
    results[key] = get_recursive(simulation.options, recursive_key, **kwargs)


def energy_MPO(results, psi, model, simulation, key='energy_MPO'):
    """Measure the energy of an MPS by evaluating the MPS expectation value.

    Parameters
    ----------
    results, psi, model, simulation, key :
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    """
    psi = simulation.psi  # take original psi, possibly grouped, but compatible with model
    if psi.bc == 'segment':
        init_env_data = simulation.init_env_data
        E = MPOEnvironment(psi, simulation.model.H_MPO, psi, **init_env_data).full_contraction(0)
        results[key] = E - simulation.results['ground_state_energy']
    else:
        results[key] = simulation.model.H_MPO.expectation_value(psi)


def entropy(results, psi, model, simulation, key='entropy'):
    """Measure the entropy at all bonds of an MPS.

    Parameters
    ----------
    results, psi, simulation, key :
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    """
    results['entropy'] = psi.entanglement_entropy()


def onsite_expectation_value(results, psi, model, simulation, opname, key=None, fix_u=None,
                             **kwargs):
    """Measure expectation values of an onsite operator.

    The resulting array of measurements is indexed by *lattice* indices ``(x, y, u)``
    (possibly dropping y and/or u if they are trivial), not by the MPS index.
    Note that this makes the result independent of the way the MPS winds through the lattice.

    The key defaults to ``f"<{opname}>"``.

    Parameters
    ----------
    results, psi, model, simulation, key:
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    opname : str
        The operator to be measured.
        Passed on to :meth:`~tenpy.networks.mps.MPS.expectation_value`.
    fix_u : None | int
        Select a (lattice) unit cell index to restrict measurements to.
    """
    if key is None:
        if not isinstance(opname, str):
            raise ValueError("can't auto-determine key for operator " + repr(opname))
        key = f"<{opname}>"
    if key in results:
        raise ValueError(f"key {key!r} already exists in results")
    lattice = model.lat
    if fix_u is not None:
        kwargs['sites'] = lattice.mps_idx_fix_u(fix_u)
    exp_vals = psi.expectation_value(opname, **kwargs)
    assert exp_vals.ndim == 1  # here exp_vals is given in MPS indices i
    # now reshape/reorder to index (x, y, u)
    if fix_u is None and 'sites' in kwargs:
        exp_vals = lattice.mps2lat_values_masked(exp_vals, mps_inds=kwargs['sites'])
    else:
        exp_vals = lattice.mps2lat_values(exp_vals, u=fix_u)
    results[key] = exp_vals


def correlation_length(results, psi, model, simulation, key='correlation_length', unit=None, **kwargs):
    """Measure the correlaiton of an infinite MPS.

    Parameters
    ----------
    results, psi, model, simulation, key:
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    unit : ``'MPS_sites' | 'MPS_sites_ungrouped' | 'lattice_rings'``
        The unit in which the correlation length is returned, see the warning in
        :meth:`~tenpy.networks.mps.MPS.correlation_length`.

        MPS_sites :
            Units of the current MPS site
        MPS_sites_ungrouped :
            If `psi` is an MPS upon which :meth:`~tenpy.networks.mps.MPS.group_sites` was called,
            this is in units of the ungrouped sites.
        lattice_rings :
            In units of lattice "rings" around the cylinder, for correlations along the
            ``lattice.basis[0]``.
        lattice_spacing :
            In units of lattice spacings (as defined by the lattice basis vectors!)
            for correlations, along the cylinder axis (for periodic boundary conditions along y)
            or along ``lattice.basis[0]`` (for "ladders" with open boundary conditions).

    **kwargs :
        Further keywoard arguments given to :meth:`~tenpy.networks.mps.MPS.correlation_length`.
    """
    corr = psi.correlation_length(**kwargs)
    if unit is None:
        warnings.warn(
            "`unit` for correlation_length not specified."
            "Defaults now to `MPS_sites`, but might change. Specify it explicitly!", FutureWarning)
        unit = 'MPS_sites'
    if unit == 'MPS_sites':
        pass
    elif unit == 'MPS_sites_ungrouped':
        corr = corr * psi.grouped
    elif unit == 'lattice_rings':
        lat = model.lattice
        if lat.N_sites_per_ring is None:
            raise ValueError("lattice doesn't define N_sites_per_ring")
        corr = corr * psi.grouped / lat.N_sites_per_ring
    elif unit == 'lattice_spacing':
        lat = model.lattice
        if lat.N_sites_per_ring is None:
            raise ValueError("lattice doesn't define N_sites_per_ring")
        corr = corr * psi.grouped / lat.N_sites_per_ring / np.inner(lat.basis[0], lat.cylinder_axis)
    else:
        raise ValueError("can't understand unit=" + repr(unit))
    results[key] = corr


def psi_method(results, psi, model, simulation, method, key=None, **kwargs):
    """Generic function to measure arbitrary method of psi with given additional kwargs.

    Instead of using `tenpy.simulations.measurement.psi_method` as a measurement function,
    you can now directly use "psi_method" as `module_name` and replace the `connect_measurements`
    simulation parameter entries as follows::

        An old python entry for connect_measurements
            ['tenpy.simulations.measurement',
             'psi_method',
             'correlation_function',
             {'key': '<Sp_i Sm_j>',
              'ops1': 'Sp',
              'ops2': 'Sm'}]
        can get replaced with new entry:
            ['psi_method',
             'correlation_function',
             {'key': '<Sp_i Sm_j>',
              'ops1': 'Sp',
              'ops2': 'Sm'}]
        Similarly, an old yaml entry for connect_measurements
            - - tenpy.simulations.measurement
              - psi_method
              - method: correlation_function
                key: '<Sp_i Sm_j>'
                ops1: Sp
                ops2: Sm
        can get replaced with new yaml:
            - - psi_method
              - correlation_function
              - key: '<Sp_i Sm_j>'
                ops1: Sp
                ops2: Sm

    The new way is now the preferred way of measuring psi methods, the old way is deprecated.

    Parameters
    ----------
    results, psi, model, simulation, key:
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    method : str
        Name of the method of `psi` to call. `key` defaults to this if not specified.
    **kwargs :
        further keyword arguments given to the method
    """
    # extract the deprecation comment from the doc string
    _psi_method_deprecated_msg = '\n'.join([line[4:] for line in
                                            psi_method.__doc__.splitlines()[2:32]])
    warnings.warn(_psi_method_deprecated_msg, FutureWarning)
    _psi_method_wrapper(results, psi, model, simulation, method, key=key, **kwargs)


def _psi_method_wrapper(results, psi, model, simulation, method, key=None, **kwargs):
    """Wrapper function to allow measuring psi methods.

    This function is used in :meth:`tenpy.simulations.Simulation._connect_measurements_method`
    to handle cases where the `module` of
    :cfg:option:`Simulaiton.connect_measurements` is "psi_method".
    """
    if key is None:
        key = method
    if key in results:
        raise ValueError(f"key {key!r} already exists in results, duplicated measurement!")
    method = getattr(psi, method)
    results[key] = method(**kwargs)


def _simulation_method_wrapper(results, psi, model, simulation, method, key, **kwargs):
    """Wrapper function to allow measuring psi methods.

    This function is used in :meth:`tenpy.simulations.Simulation._connect_measurements_method`
    to handle cases where the `module` of
    :cfg:option:`Simulaiton.connect_measurements` is "simulation_method".
    """
    if key in results:
        raise ValueError(f"key {key!r} already exists in results, duplicated measurement!")
    results[key] = method(psi=psi, **kwargs)


def evolved_time(results, psi, model, simulation, key='evolved_time'):
    """Measure the time evolved by the engine, ``engine.evolved_time``.

    "Measures" :attr:`tenpy.algorithms.algorithm.TimeEvolutionAlgorithm.evolved_time`.

    Parameters
    ----------
    results, psi, model, simulation, key:
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    """
    results[key] = simulation.engine.evolved_time
