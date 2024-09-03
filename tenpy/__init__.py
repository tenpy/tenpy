"""TeNPy - a Python library for Tensor Network Algorithms

TeNPy is a library for algorithms working with tensor networks,
e.g., matrix product states and -operators,
designed to study the physics of strongly correlated quantum systems.
The code is intended to be accessible for newcomers
and yet powerful enough for day-to-day research.
"""
# Copyright (C) TeNPy Developers, GNU GPLv3
# This file marks this directory as a python package.

# Note: all external packages that are imported should be `del`-ed at the end of the file!
import logging

# main logger for tenpy
logger = logging.getLogger(__name__)

# load and provide sub packages on first input
# note that the order matters!
from . import tools
from . import linalg
from . import networks
from . import models
from . import algorithms
from . import simulations
from . import version  # needs to be after linalg!

# provide the more important functions and classes directly from the main namespace:
from .algorithms.dmrg_parallel import DMRGThreadPlusHC
from .algorithms.dmrg import SingleSiteDMRGEngine, TwoSiteDMRGEngine
from .algorithms.exact_diag import ExactDiag
from .algorithms.mpo_evolution import ExpMPOEvolution, TimeDependentExpMPOEvolution
from .algorithms.mps_common import VariationalCompression, VariationalApplyMPO, QRBasedVariationalApplyMPO
from .algorithms.network_contractor import ncon, contract
from .algorithms.purification import PurificationApplyMPO, PurificationTEBD, PurificationTEBD2
from .algorithms.tdvp import (SingleSiteTDVPEngine, TwoSiteTDVPEngine, TimeDependentSingleSiteTDVP,
                              TimeDependentTwoSiteTDVP)
from .algorithms.vumps import SingleSiteVUMPSEngine, TwoSiteVUMPSEngine
from .algorithms.tebd import TEBDEngine, QRBasedTEBDEngine, RandomUnitaryEvolution, TimeDependentTEBD
from .linalg.truncation import TruncationError, truncate, svd_theta, decompose_theta_qr_based
from .linalg.charges import ChargeInfo, LegCharge, LegPipe
from .linalg.krylov_based import Arnoldi, LanczosGroundState, LanczosEvolution, lanczos_arpack
from .linalg.np_conserved import (Array, zeros, ones, eye_like, diag,
                                  concatenate, grid_concat, grid_outer, detect_grid_outer_legcharge,
                                  detect_qtotal, detect_legcharge, trace, outer, inner, tensordot,
                                  svd, pinv, norm, eigh, eig, eigvalsh, eigvals, speigs, expm, qr)
from .models.lattice import (Lattice, TrivialLattice, SimpleLattice, MultiSpeciesLattice,
                             IrregularLattice, HelicalLattice, Chain, Ladder, NLegLadder, Square,
                             Triangular, Honeycomb, Kagome, get_lattice)
from .models.tf_ising import TFIModel, TFIChain
from .models.xxz_chain import XXZChain, XXZChain2
from .models.spins import SpinModel, SpinChain
from .models.spins_nnn import SpinChainNNN, SpinChainNNN2
from .models.fermions_spinless import FermionModel, FermionChain
from .models.tj_model import tJModel, tJChain
from .models.hofstadter import HofstadterBosons, HofstadterFermions
from .models.clock import ClockModel, ClockChain
from .models.hubbard import (BoseHubbardModel, BoseHubbardChain, FermiHubbardModel,
                             FermiHubbardChain, FermiHubbardModel2)
from .models.haldane import BosonicHaldaneModel, FermionicHaldaneModel
from .models.toric_code import ToricCode
from .models.aklt import AKLTChain
from .models.mixed_xk import (MixedXKLattice, MixedXKModel, SpinlessMixedXKSquare,
                              HubbardMixedXKSquare)
from .networks.site import (Site, GroupedSite, group_sites, SpinHalfSite, SpinSite, FermionSite,
                            SpinHalfFermionSite, SpinHalfHoleSite, BosonSite, ClockSite,
                            spin_half_species, kron)
from .networks.mps import (MPS, MPSEnvironment, TransferMatrix, InitialStateBuilder,
                           build_initial_state)
from .networks.mpo import MPO, MPOEnvironment, MPOTransferMatrix
from .networks.purification_mps import PurificationMPS
from .networks.uniform_mps import UniformMPS
from .networks.momentum_mps import MomentumMPS
from .simulations.simulation import (Simulation, Skip, init_simulation, run_simulation,
                                     init_simulation_from_checkpoint, resume_from_checkpoint,
                                     run_seq_simulations, estimate_simulation_RAM)
from .simulations.ground_state_search import (GroundStateSearch, OrthogonalExcitations,
                                              ExcitationInitialState)
from .simulations.time_evolution import RealTimeEvolution
from .simulations.measurement import (m_measurement_index, m_bond_dimension, m_bond_energies,
                                      m_simulation_parameter, m_energy_MPO, m_entropy,
                                      m_onsite_expectation_value, m_correlation_length,
                                      m_evolved_time)
from .tools.hdf5_io import save, load, save_to_hdf5, load_from_hdf5
from .tools.misc import (setup_logging, consistency_check, TenpyInconsistencyError,
                         TenpyInconsistencyWarning, BetaWarning)
from .tools.params import Config, asConfig, load_yaml_with_py_eval



#: hard-coded version string
__version__ = version.version

#: full version from git description, and numpy/scipy/python versions
__full_version__ = version.full_version

__all__ = [
    # subpackages
    'algorithms', 'linalg', 'models', 'networks', 'simulations', 'tools', 'version',
    # from tenpy.algorithms
    'DMRGThreadPlusHC', 'SingleSiteDMRGEngine', 'TwoSiteDMRGEngine', 'ExactDiag', 'ExpMPOEvolution',
    'TimeDependentExpMPOEvolution', 'VariationalCompression', 'VariationalApplyMPO', 'QRBasedVariationalApplyMPO', 'ncon',
    'contract', 'PurificationApplyMPO', 'PurificationTEBD', 'PurificationTEBD2',
    'SingleSiteTDVPEngine', 'TwoSiteTDVPEngine', 'TimeDependentSingleSiteTDVP',
    'TimeDependentTwoSiteTDVP', 'TEBDEngine', 'QRBasedTEBDEngine', 'RandomUnitaryEvolution',
    'TimeDependentTEBD', 'TruncationError', 'truncate', 'svd_theta', 'decompose_theta_qr_based', 'SingleSiteVUMPSEngine',
    'TwoSiteVUMPSEngine',
    # from tenpy.linalg
    'ChargeInfo', 'LegCharge', 'LegPipe', 'Arnoldi', 'LanczosGroundState', 'LanczosEvolution',
    'lanczos_arpack', 'Array', 'zeros', 'ones', 'eye_like', 'diag', 'concatenate', 'grid_concat',
    'grid_outer', 'detect_grid_outer_legcharge', 'detect_qtotal', 'detect_legcharge', 'trace',
    'outer', 'inner', 'tensordot', 'svd', 'pinv', 'norm', 'eigh', 'eig', 'eigvalsh', 'eigvals',
    'speigs', 'expm', 'qr',
    # from tenpy.models
    'Lattice', 'TrivialLattice', 'SimpleLattice', 'MultiSpeciesLattice', 'IrregularLattice',
    'HelicalLattice', 'Chain', 'Ladder', 'NLegLadder', 'Square', 'Triangular', 'Honeycomb',
    'Kagome', 'get_lattice', 'TFIModel', 'TFIChain', 'XXZChain', 'XXZChain2', 'SpinModel',
    'SpinChain', 'SpinChainNNN', 'SpinChainNNN2', 'FermionModel', 'FermionChain', 'tJModel',
    'tJChain', 'HofstadterBosons', 'HofstadterFermions', 'ClockModel', 'ClockChain',
    'BoseHubbardModel', 'BoseHubbardChain', 'FermiHubbardModel', 'FermiHubbardChain',
    'FermiHubbardModel2', 'BosonicHaldaneModel', 'FermionicHaldaneModel', 'ToricCode', 'AKLTChain',
    'MixedXKLattice', 'MixedXKModel', 'SpinlessMixedXKSquare', 'HubbardMixedXKSquare',
    # from tenpy.networks
    'Site', 'GroupedSite', 'group_sites', 'SpinHalfSite', 'SpinSite', 'FermionSite',
    'SpinHalfFermionSite', 'SpinHalfHoleSite', 'BosonSite', 'ClockSite', 'spin_half_species',
    'kron', 'MPS', 'MPSEnvironment', 'TransferMatrix', 'InitialStateBuilder', 'build_initial_state',
    'MPO', 'MPOEnvironment', 'MPOTransferMatrix', 'PurificationMPS', 'UniformMPS', 'MomentumMPS',
    # from tenpy.simulations
    'Simulation', 'Skip', 'init_simulation', 'run_simulation', 'init_simulation_from_checkpoint',
    'resume_from_checkpoint', 'run_seq_simulations', 'GroundStateSearch', 'OrthogonalExcitations',
    'ExcitationInitialState', 'RealTimeEvolution', 'm_measurement_index', 'm_bond_dimension',
    'm_bond_energies', 'm_simulation_parameter', 'm_energy_MPO', 'm_entropy',
    'm_onsite_expectation_value', 'm_correlation_length', 'm_evolved_time',
    # from tenpy.tools
    'save', 'load', 'save_to_hdf5', 'load_from_hdf5', 'setup_logging', 'consistency_check',
    'TenpyInconsistencyError', 'TenpyInconsistencyWarning', 'BetaWarning', 'Config', 'asConfig',
    'load_yaml_with_py_eval',
    # from tenpy.__init__, i.e. defined below
    'show_config', 'console_main',
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
    context = {'tenpy': sys.modules[__name__], 'np': np, 'scipy': scipy}
    if args.import_module:
        sys.path.insert(0, '.')
        for module_name in args.import_module:
            module = importlib.import_module(module_name)
            context[module_name] = module
    # load parameters_file
    options = {}
    if args.parameters_file:
        options_files = []
        for fn in args.parameters_file:
            options = load_yaml_with_py_eval(fn, context)
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
        options['simulation_class'] = args.sim_class
    if args.RAM:
        # exit immediately
        return estimate_simulation_RAM(suppress_non_RAM_output=True, unit='MB', **options)
    if 'sequential' not in options:
        return run_simulation(**options)
    else:
        return run_seq_simulations(**options)


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
    parser.add_argument('--RAM',
                        action="store_true",
                        help="Estimates the required RAM. "
                        "This argument does not execute any simulation, but just initializes it "
                        "to predict the necessary RAM in MB and then exits.")
    parser.add_argument('parameters_file',
                        nargs='*',
                        help="Yaml (*.yml) file with the simulation parameters/options. "
                        "We support an additional yaml tag !py_eval: VALUE that gets initialized "
                        "by python's ``eval(VALUE)`` with `np`, `scipy` and `tenpy` defined. "
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


# remove the imported libraries again. we do not want to expose them e.g. as tenpy.logging
del logging
