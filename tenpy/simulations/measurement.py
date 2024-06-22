"""Functions to perform measurements during simulations.

All measurement functions provided in this module support the interface used by the simulation
class, i.e. they take the parameters ``results, psi, model, simulation`` as keyword arguments,
as documented in :func:`measurement_index` and write
the measurement results into the `results` dictionary taken as argument.
It can take extra keyword arguments that can be specified in

As briefly explained in :doc:`/intro/simulations`, you can easily add custom measurement functions.

Full description and details in :doc:`/intro/measurements`.
"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import warnings
import functools

from ..networks.mpo import MPOEnvironment
from ..tools.misc import get_recursive

__all__ = [
    'measurement_wrapper', 'm_measurement_index', 'm_bond_dimension', 'm_bond_energies',
    'm_simulation_parameter', 'm_energy_MPO', 'm_entropy', 'm_onsite_expectation_value',
    'm_correlation_length', 'm_evolved_time',
]


def measurement_wrapper(function, results_key, **kwargs):
    """Decorator to transform a function into a measurement function.

    As documented in :func:`m_measurement_index`, measurement functions need to take
    positional arguments ``results, psi, model, simulation``.
    This function is a decorator that wraps a given `function` not taking those arguments
    into a measurement function by simply discarding these arguments
    and saving the returned value in the `results`.
    """
    if results_key is None:
        results_key = function.__name__

    @functools.wraps(function)
    def measurement_call(results, psi, model, simulation, **kwargs):
        if results_key in results:
            raise ValueError(f"key {results_key!r} already exists in `results`, "
                             "measurement would overwrite data. "
                             "Probably a measurement function used multiple times!")
        res = function(**kwargs)
        results[results_key] = res

    return measurement_call


def m_measurement_index(results, psi, model, simulation, results_key='measurement_index'):
    """'Measure' the index of how many measurements have been performed so far.

    The parameter description below documents the common interface of all measurement
    functions that can be registered to simulations.

    See :doc:`/intro/measurements` for the general setup using measurements.


    .. versionchanged:: 0.10.0

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
        Note that ``psi`` is compatible with ``model``,
        and ``simulation.psi`` should be compatible with ``simulation.model``,
        but you should not mix them.
    simulation : :class:`~tenpy.simulations.simulation.Simulation`
        The simulation class. This gives also access to the `model`, algorithm `engine`, etc.
    results_key : str
        The key under which to save data in `results`.
        For some measurement functions optional, e.g in this case.
        Note that a single measurement function can add results under multiple keys, if desired.
    **kwargs :
        Other optional keyword arguments for individual measurement functions.
        Those are documented inside each measurement function.
    """
    index = len(simulation.results.get('measurements', {}).get(results_key, []))
    results[results_key] = index


def m_bond_dimension(results, psi, model, simulation, results_key='bond_dimension'):
    """'Measure' the bond dimension of an MPS.

    Parameters
    ----------
    results, psi, model, simulation, results_key :
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    """
    results[results_key] = psi.chi


def m_bond_energies(results, psi, model, simulation, results_key='bond_energies'):
    """Measure the energy of an MPS.

    Parameters
    ----------
    results, psi, model, simulation, results_key :
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    """
    # use simulation.psi and simulation.model since default 'energy' measurement is
    # obtained from model.get_extra_default_measurements() in _connect_measurements() after regrouping
    results[results_key] = simulation.model.bond_energies(simulation.psi)


def m_simulation_parameter(results, psi, model, simulation, recursive_key, results_key=None, **kwargs):
    """Dummy measurement of a simulation parameter.

    This can be useful e.g. if you have a time-dependent hamiltonian parameter.

    Parameters
    ----------
    results, psi, simulation, results_key :
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    recursive_key : str
        Recursive key of the simulation parameter to be read out.
    default, separators :
        Remaining arguments of :func:`~tenpy.tools.misc.get_recursive`.
    """
    if results_key is None:
        results_key = recursive_key
    results[results_key] = get_recursive(simulation.options, recursive_key, **kwargs)


def m_energy_MPO(results, psi, model, simulation, results_key='energy_MPO'):
    """Measure the energy of an MPS by evaluating the MPS expectation value.

    Parameters
    ----------
    results, psi, model, simulation, results_key :
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    """
    psi = simulation.psi  # take original psi, possibly grouped, but compatible with simulation.model
    if psi.bc == 'segment':
        init_env_data = simulation.init_env_data
        E = MPOEnvironment(psi, simulation.model.H_MPO, psi, **init_env_data).full_contraction(0)
        results[results_key] = E - simulation.results['ground_state_energy']
    else:
        results[results_key] = simulation.model.H_MPO.expectation_value(psi)


def m_entropy(results, psi, model, simulation, results_key='entropy'):
    """Measure the entropy at all bonds of an MPS.

    Parameters
    ----------
    results, psi, simulation, results_key :
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    """
    results[results_key] = psi.entanglement_entropy()


def m_onsite_expectation_value(results, psi, model, simulation, opname, results_key=None, fix_u=None,
                             **kwargs):
    """Measure expectation values of an onsite operator.

    The resulting array of measurements is indexed by *lattice* indices ``(x, y, u)``
    (possibly dropping y and/or u if they are trivial), not by the MPS index.
    Note that this makes the result independent of the way the MPS winds through the lattice.

    The results_key defaults to ``f"<{opname}>"``.

    Parameters
    ----------
    results, psi, model, simulation, results_key:
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    opname : str
        The operator to be measured.
        Passed on to :meth:`~tenpy.networks.mps.MPS.expectation_value`.
    fix_u : None | int
        Select a (lattice) unit cell index to restrict measurements to.
    """
    if results_key is None:
        if not isinstance(opname, str):
            raise ValueError("can't auto-determine key for operator " + repr(opname))
        results_key = f"<{opname}>"
    if results_key in results:
        raise ValueError(f"key {results_key!r} already exists in results")
    if fix_u is not None:
        kwargs['sites'] = model.lat.mps_idx_fix_u(fix_u)

    # the actual "measurement"
    exp_vals = psi.expectation_value(opname, **kwargs)

    assert exp_vals.ndim == 1  # here exp_vals is given in MPS indices i
    # now reshape/reorder to index (x, y, u)
    if fix_u is None and 'sites' in kwargs:
        exp_vals = model.lat.mps2lat_values_masked(exp_vals, mps_inds=kwargs['sites'])
    else:
        exp_vals = model.lat.mps2lat_values(exp_vals, u=fix_u)
    results[results_key] = exp_vals


def m_correlation_length(results, psi, model, simulation, results_key='correlation_length', unit=None, **kwargs):
    """Measure the correlation of an infinite MPS.

    Parameters
    ----------
    results, psi, model, simulation, results_key:
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
        Further keyword arguments given to :meth:`~tenpy.networks.mps.MPS.correlation_length`.
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
        lat = model.lat
        if lat.N_sites_per_ring is None:
            raise ValueError("lattice doesn't define N_sites_per_ring")
        corr = corr * psi.grouped / lat.N_sites_per_ring
    elif unit == 'lattice_spacing':
        lat = model.lat
        if lat.N_sites_per_ring is None:
            raise ValueError("lattice doesn't define N_sites_per_ring")
        corr = corr * psi.grouped / lat.N_sites_per_ring / np.inner(lat.basis[0], lat.cylinder_axis)
    else:
        raise ValueError("can't understand unit=" + repr(unit))
    results[results_key] = corr

def m_evolved_time(results, psi, model, simulation, results_key='evolved_time'):
    """Measure the time evolved by the engine, ``engine.evolved_time``.

    "Measures" :attr:`tenpy.algorithms.algorithm.TimeEvolutionAlgorithm.evolved_time`.

    Parameters
    ----------
    results, psi, model, simulation, results_key:
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    """
    results[results_key] = simulation.engine.evolved_time


# the following wrapper functions _m_{psi,model}_method_[wrapped] are used
# in simulation._connect_measurement_fct  for entries with 'psi_method' and 'model_method' in
# :cfg:option:`Simulation.connect_measurement`

def _m_psi_method(results, psi, model, simulation, func_name, **kwargs):
    psi_method = getattr(psi, func_name)
    psi_method(results, model, simulation, **kwargs)


def _m_psi_method_wrapped(results, psi, model, simulation, func_name, results_key, **kwargs):
    if results_key in results:
        raise ValueError(f"key {results_key!r} already exists in `results`, "
                         "measurement would overwrite data. "
                         "Probably a measurement function used multiple times!")
    psi_method = getattr(psi, func_name)
    res = psi_method(**kwargs)
    results[results_key] = res


def _m_model_method(results, psi, model, simulation, func_name, **kwargs):
    model_method = getattr(model, func_name)
    model_method(results, psi, simulation, **kwargs)


def _m_model_method_wrapped(results, psi, model, simulation, func_name, results_key, **kwargs):
    if results_key in results:
        raise ValueError(f"key {results_key!r} already exists in `results`, "
                         "measurement would overwrite data. "
                         "Probably a measurement function used multiple times!")
    model_method = getattr(model, func_name)
    res = model_method(**kwargs)
    results[results_key] = res
