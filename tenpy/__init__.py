"""TeNPy - a Python library for Tensor Network Algorithms

TeNPy is a library for algorithms working with tensor networks,
e.g., matrix product states and -operators,
designed to study the physics of strongly correlated quantum systems.
The code is intended to be accessible for newcommers
and yet powerful enough for day-to-day research.
"""
# Copyright 2018-2020 TeNPy Developers, GNU GPLv3
# This file marks this directory as a python package.

# load and provide subpackages on first input
from . import version
from . import tools
from . import linalg
from . import algorithms
from . import networks
from . import models
from . import simulations

#: hard-coded version string
__version__ = version.version

#: full version from git description, and numpy/scipy/python versions
__full_version__ = version.full_version

__all__ = [
    "algorithms", "linalg", "models", "networks", "simulations", "tools", "version", "show_config"
]


def show_config():
    """Print information about the version of tenpy and used libraries.

    The information printed is :attr:`tenpy.version.version_summary`.
    """
    print(version.version_summary)


def run_simulation(simulation_class_name='GroundStateSearch', **simulation_params):
    """Run the simulation with a simulation class.

    Parameters
    ----------
    simulation_class_name : str
        The name of a (sub)class of :class:`~tenpy.simulations.simulations.Simulation`
        to be used for running the simulaiton.
    **simulation_params :
        Further keyword arguments as documented in the corresponding simulation class,
        see :cfg:config`Simulation`.

    Returns
    -------
    results : dict
        The results from running the simulation.
    """
    SimClass = tools.misc.find_subclass(simulations.simulation.Simulation, simulation_class_name)
    sim = SimClass(simulation_params)
    results = sim.run()
    return results
