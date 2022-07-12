"""TeNPy - a Python library for Tensor Network Algorithms

TeNPy is a library for algorithms working with tensor networks,
e.g., matrix product states and -operators,
designed to study the physics of strongly correlated quantum systems.
The code is intended to be accessible for newcommers
and yet powerful enough for day-to-day research.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3
# This file marks this directory as a python package.

import warnings
import logging
logger = logging.getLogger(__name__)  # main logger for tenpy

# load and provide subpackages on first input
# note that the order matters!
from . import tools
from . import linalg
from . import algorithms
from . import networks
from . import models
from . import simulations
from . import version  # needs to be after linalg!
from .simulations.simulation import run_simulation, resume_from_checkpoint, run_seq_simulations

#: hard-coded version string
__version__ = version.version

#: full version from git description, and numpy/scipy/python versions
__full_version__ = version.full_version

__all__ = [
    "algorithms", "linalg", "models", "networks", "simulations", "tools", "version", "show_config",
    "run_simulation", "resume_from_checkpoint", "run_seq_simulations", "console_main"
]


def show_config():
    """Print information about the version of tenpy and used libraries.

    The information printed is :attr:`tenpy.version.version_summary`.
    """
    print(version.version_summary)


def console_main(*command_line_args):
    """Command line interface.

    For the python interface see :func:`~tenpy.simulations.simulation.run_simulation` and
    :func:`~tenpy.simulations.simulation.run_seq_simulations`.

    When tenpy is installed correctly via pip/conda, a command line script called ``tenpy-run``
    is set up, which calls this function, i.e., you can do the following in the terminal::

        tenpy-run --help

    Equivalently, you can also invoke the tenpy module from your python interpreter like this::

        python -m tenpy --help

    ..
        Sphinx includes the output of ``tenpy-run --help`` here, setup in doc/conf.py.
    """
    import numpy as np
    import scipy
    import sys
    import importlib
    parser = _setup_arg_parser()

    args = parser.parse_args(args=command_line_args if command_line_args else None)
    # import extra modules
    context = {'tenpy': globals(), 'np': np, 'scipy': scipy}
    if args.import_module:
        sys.path.insert(0, '.')
        for module_name in args.import_module:
            module = importlib.import_module(module_name)
            context[module_name] = module
    # load parameters_file
    options = {}
    if args.parameters_file:
        import yaml
        options_files = []
        for fn in args.parameters_file:
            with open(fn, 'r') as stream:
                options = yaml.safe_load(stream)
            options_files.append(options)
        if len(options_files) > 1:
            options = tools.misc.merge_recursive(*options_files, conflict=args.merge)
    # update extra options
    if args.option:
        for key, val_string in args.option:
            val = eval(val_string, context)
            tools.misc.set_recursive(options, key, val, insert_dicts=True)
    if len(options) == 0:
        raise ValueError("No options supplied! Check your command line arguments!")
    if 'output_filename' not in options and 'output_filename_params' not in options:
        raise ValueError("No output filename specified - refuse to run without saving anything!")
    if args.sim_class is not None:  # non-default
        if 'simulation_class_name' in options:
            warnings.warn('command line overrides deprecated `simulation_class_name` parameter',
                          FutureWarning)
            del options['simulation_class_name']
        options['simulation_class'] = args.sim_class
    if 'sequential' not in options:
        run_simulation(**options)
    else:
        run_seq_simulations(**options)


def _setup_arg_parser(width=None):
    import argparse
    import textwrap

    desc = "Command line interface to run a TeNPy simulation."
    epilog = textwrap.dedent("""\
    Examples
    --------

    In the simplest case, you just give a single yaml file with all the parameters as argument:

        tenpy-run my_params.yml

    If you implemented a custom simulation class called ``MyGreatSimulation`` in a
    file ``my_simulations.py``, you can use it like this:

        tenpy-run -i my_simulations -c MyGreatSimulation my_params.yml

    Further, you can overwrite one or multiple options of the parameters file:

        tenpy-run my_params.yml -o output_filename '"rerun_Jz_2.h5"' -o model_params.Jz 2.

    Note that string values for the options require double quotes on the command line.
    """)

    def formatter(prog):
        return argparse.RawDescriptionHelpFormatter(prog,
                                                    indent_increment=4,
                                                    max_help_position=8,
                                                    width=width)

    parser = argparse.ArgumentParser(description=desc, epilog=epilog, formatter_class=formatter)
    parser.add_argument('--import-module',
                        '-i',
                        metavar='MODULE',
                        action='append',
                        help="Import the given python MODULE before setting up the simulation. "
                        "This is useful if the module contains user-defined subclasses. "
                        "Use python-style names like `numpy` without the .py ending.")
    parser.add_argument('--sim-class',
                        '-c',
                        default=None,
                        help="selects the Simulation (sub)class, "
                        "e.g. 'GroundStateSearch' (default) or 'RealTimeEvolution'.")
    parser.add_argument('--merge',
                        '-m',
                        default='error',
                        help="Selects how to merge conflicts in case of multiple yaml files. "
                        "Options are 'error', 'first' or 'last'.")
    parser.add_argument('parameters_file',
                        nargs='*',
                        help="Yaml (*.yml) file with the simulation parameters/options. "
                        "Multiple files get merged according to MERGE; "
                        "see tenpy.tools.misc.merge_recursive for details.")
    opt_help = textwrap.dedent("""\
        Allows overwriting some options from the yaml files.
        KEY can be recursive separated by `.`, e.g. ``algorithm_params.trunc_params.chi``.
        VALUE is initialized by python's ``eval(VALUE)`` with `np`, `scipy` and `tenpy` defined.
        Thus ``'1.2'`` and ``'2.*np.pi*1.j/6'`` or ``'np.linspace(0., 1., 6)'`` will work if you
        include the quotes on the command line to ensure that the VALUE is passed as a single
        argument.""")
    parser.add_argument('--option',
                        '-o',
                        nargs=2,
                        action='append',
                        metavar=('KEY', 'VALUE'),
                        help=opt_help)
    parser.add_argument('--version', '-v', action='version', version=__full_version__)
    return parser
