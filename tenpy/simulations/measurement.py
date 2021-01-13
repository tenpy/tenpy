"""Functions to perform measurments.

All measurement functions provided in this module support the interface used by the simulation
class, i.e. they take the parameters documented in :func:`measurement_index` and write
the measurement results into the `results` dictionary taken as argument.

.. todo ::
    test, provide more.
"""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

__all__ = [
    'measurement_index', 'bond_dimension', 'bond_energies', 'energy_MPO', 'entropy',
    'onsite_expectation_value', 'correlation_length', 'evolved_time'
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
    exp_vals = psi.expectation_value(opname)
    lattice = simulation.model.lat
    exp_vals = lattice.mps2lat_values(exp_vals)
    results[key] = exp_vals


def correlation_length(results, psi, simulation, key='correlation_length', **kwargs):
    """Measure the correlaiton of an infinite MPS.

    Parameters
    ----------
    results, psi, simulation, key:
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    **kwargs :
        Further keywoard arguments given to :meth:`~tenpy.networks.mps.MPS.correlation_length`.
    """
    corr = psi.correlation_length(**kwargs)
    results[key] = corr


def evolved_time(results, psi, simulation, key='evolved_time'):
    """Measure the time evolved by the engine, ``engine.evolved_time``.

    See e.g. :attr:`tenpy.algorithms.tebd.TEBDEngine.evolved_time`.

    Parameters
    ----------
    results, psi, simulation, key:
        See :func:`~tenpy.simulation.measurement.measurement_index`.
    """
    results[key] = simulation.engine.evolved_time
