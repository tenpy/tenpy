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
    "algorithms", "linalg", "models", "networks", "simulations", "tools", "version", "show_config",
    "run_simulation", "console_main"
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
    if SimClass is None:
        raise ValueError("can't find simulation class called " + repr(simulation_class_name))
    sim = SimClass(simulation_params)
    results = sim.run()
    return results


def console_main():
    import argparse
    import textwrap
    import numpy as np
    import scipy
    import sys
    import importlib
    from .tools.params import Config

    desc = "Command line interface to run a TeNPy simulation."
    epilog = textwrap.dedent("""\
        Examples:

        In the simplest case, you just give a single yaml file with all the parameters as argument.

            tenpy-run my_params.yml

        If you implemented a custom simulation class called "MyGreatSimulation" in a
        file 'my_simulations.py', you can use it like this:

            tenpy-run -i my_simulations -c MyGreatSimulation my_params.yml

        Further, you can overwrite one or multiple options of the parameters file like this:

            tenpy-run my_params.yml -o output_filename '"output.h5"' -o model_params/Jz 2.

        Here the quotes around the last argument are required to combine it into a single value.
        Also note that string values for the options require double quotes on the command line.
        """)
    parser = argparse.ArgumentParser(description=desc,
                                     epilog=epilog,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--import-module',
                        '-i',
                        metavar='MODULE',
                        action='append',
                        help="""
                        Import the given python MODULE before setting up the simulation.
                        This is useful if the module contains user-defined subclasses.
                        Use python-style names like `numpy` without the .py ending.""")
    parser.add_argument('--sim-class',
                        '-c',
                        default='GroundStateSearch',
                        help="""
                        selects the Simulation (sub)class, e.g. 'GroundStateSearch' (default)
                        or 'RealTimeEvolution'.""")
    parser.add_argument('parameters_file',
                        nargs='?',
                        help="""
                        A yaml (*.yml) file with the simulation parameters/options.""")
    parser.add_argument('--option',
                        '-o',
                        nargs=2,
                        action='append',
                        metavar=('KEY', 'VALUE'),
                        help=r"""
        Allows overwriting some options from the yaml files.
        KEY can be recursive separated by '/', e.g. 'dmrg_params/trunc_params/chi'.
        VALUE is initialized by python's 'eval(VALUE)' with 'np', 'scipy' and 'tenpy' defined.
        Thus '1.2' and '2.*np.pi*1.j/6' or 'np.linspace(0., 1., 6)' will work, as long as you
        include quotes on the command line to ensure that the VALUE is passed as a single argument.
        """)

    args = parser.parse_args()
    # import extra modules
    context = {'tenpy': globals(), 'np': np, 'scipy': scipy}
    if args.import_module:
        import sys
        sys.path.insert(0, '.')
        for module_name in args.import_module:
            module = importlib.import_module(module_name)
            context[module_name] = module
    # load parameters_file
    options = {}
    if args.parameters_file:
        import yaml
        with open(args.parameters_file, 'r') as stream:
            options = yaml.safe_load(stream)
    # update extra options
    if args.option:
        for key, val_string in args.option:
            val = eval(val_string, context)
            set_recursive(options, key, val, insert_dicts=True)
    run_simulation(args.sim_class, **options)
