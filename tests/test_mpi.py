# Copyright 2021 TeNPy Developers, GNU GPLv3

import pytest
import tenpy

mpi_parallel = pytest.importorskip('pytest_easyMPI').mpi_parallel


@mpi_parallel(2)
@pytest.mark.slow
def test_parallel_dmrg(eps=1.e-8):
    from mpi4py import MPI
    import tenpy.simulations.mpi_parallel
    # if MPI.COMM_WORLD.rank == 1:
    #     raise ValueError("damn it!")

    sim_params = {
        'model_class': 'XXZChain2',  # XXZChain doesn't support explicit_plus_hc!
        'model_params': {
            'L': 14,
            'bc_MPS': 'finite',
            'explicit_plus_hc': True,
        },
        'initial_state_params': {
            'method': 'lat_product_state',
            'product_state': [['up'], ['down']],
            'verbose': 10,
        },
    }
    res = tenpy.run_simulation('ParallelDMRGSim', {'setup_logging': False}, **sim_params)
    if MPI.COMM_WORLD.rank != 0:
        assert res is None
        return
    assert abs(-6.0267246618621728 - res['energy']) < eps
