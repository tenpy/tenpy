"""Functions to perform measurments.

All measurement functions provided in this module support the interface used by the simulation
class, i.e. they take the parameters documented in :func:`measurement_index` and write
the measurement results into the `results` dictionary taken as argument.

.. todo ::
    test, provide more.
"""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

from ..networks.mpo import MPOEnvironment
import warnings

__all__ = [
    'measurement_index', 'bond_dimension', 'bond_energies', 'energy_MPO', 'entropy',
    'onsite_expectation_value', 'correlation_length', 'psi_method', 'evolved_time'
]


def measurement_index(results, psi, simulation, key='measurement_index'):
    """'Measure' the index of how many mearuements have been performed so far.

    The parameter description below also documents the common interface of all measurement
    functions, that can be registered to simulations.

    Parameters
    ----------
    results : dict
        A dictionary with measurement results performed so far.
        Instead of returning the result, the output should be written into this dictionary
        under an appropriate key (or multiple keys, if applicable).
    psi :
        Tensor network state to be measured. Shorthand for ``simulation.psi``.
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


def bond_dimension(results, psi, simulation, key='bond_dimension'):
    """'Measure' the bond dimension of an MPS.

    Parameters
    ----------
    results, psi, simulation, key :
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    """
    results[key] = psi.chi


def bond_energies(results, psi, simulation, key='bond_energies'):
    """Measure the energy of an MPS.

    Parameters
    ----------
    results, psi, simulation, key :
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    """
    results[key] = simulation.model.bond_energies(psi)


def energy_MPO(results, psi, simulation, key='energy_MPO'):
    """Measure the energy of an MPS by evaluating the MPS expectation value.

    Parameters
    ----------
    results, psi, simulation, key :
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    """
    if psi.bc == 'segment':
        init_env_data = simulation.init_env_data
        E = MPOEnvironment(psi, simulation.model.H_MPO, psi, **init_env_data).full_contraction(0)
        results[key] = E - simulation.results['ground_state_energy']
    else:
        results[key] = simulation.model.H_MPO.expectation_value(psi)


def entropy(results, psi, simulation, key='entropy'):
    """Measure the entropy at all bonds of an MPS.

    Parameters
    ----------
    results, psi, simulation, key :
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    """
    results['entropy'] = psi.entanglement_entropy()


def onsite_expectation_value(results, psi, simulation, opname, key=None):
    """Measure expectation values of an onsite operator.

    The resulting array of measurements is indexed by *lattice* indices ``(x, y, u)``
    (possibly dropping y and/or u if they are trivial), not by the MPS index.
    Note that this makes the result independent of the way the MPS winds through the lattice.

    The key defaults to ``f"<{opname}>"``.

    Parameters
    ----------
    results, psi, simulation, key:
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    opname : str
        The operator to be measured.
        Passed on to :meth:`~tenpy.networks.mps.MPS.expectation_value`.
    """
    if key is None:
        if not isinstance(opname, str):
            raise ValueError("can't auto-determine key for operator " + repr(opname))
        key = "<{0!s}>".format(opname)
    if key in results:
        raise ValueError(f"key {key!r} already exists in results")
    exp_vals = psi.expectation_value(opname)
    lattice = simulation.model.lat
    exp_vals = lattice.mps2lat_values(exp_vals)
    results[key] = exp_vals


def correlation_length(results, psi, simulation, key='correlation_length', unit=None, **kwargs):
    """Measure the correlaiton of an infinite MPS.

    Parameters
    ----------
    results, psi, simulation, key:
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
            In units of lattice spacings for correlations along the cylinder axis (for periodic
            boundary conditions along y) or along ``lattice.basis[0]`` (for "ladders" with open
            bboundary conditions).

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
        lat = simulation.model.lattice
        if lat.N_sites_per_ring is None:
            raise ValueError("lattice doesn't define N_sites_per_ring")
        corr = corr * psi.grouped / lat.N_sites_per_ring
    elif unit == 'latitce_spacing':
        raise NotImplementedError("TODO")
    else:
        raise ValueError("can't understand unit=" + repr(unit))
    results[key] = corr


def psi_method(results, psi, simulation, method, key=None, **kwargs):
    """General method to measure arbitrary method of psi with given additional kwargs.

    Parameters
    ----------
    results, psi, simulation, key:
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    method : str
        Name of the method of `psi` to call. `key` defaults to this if not specified.
    **kwargs :
        further keyword arguments given to the method
    """
    if key is None:
        key = method
    if key in results:
        raise ValueError(f"key {key!r} already exists in results")
    method = getattr(psi, method)
    results[key] = method(**kwargs)


def evolved_time(results, psi, simulation, key='evolved_time'):
    """Measure the time evolved by the engine, ``engine.evolved_time``.

    "Measures" :attr:`tenpy.algorithms.algorithm.TimeEvolutionAlgorithm.evolved_time`.

    Parameters
    ----------
    results, psi, simulation, key:
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    """
    results[key] = simulation.engine.evolved_time
