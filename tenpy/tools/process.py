"""Tools to read out total memory usage of the running process.

To limit BLAS/LAPACK threads (OpenMP / MKL), set environment variables *before*
starting Python, for example::

    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4

If PyTorch is loaded, prefer ``torch.set_num_threads(n)`` for runtime changes;
ctypes wrappers around ``omp_set_num_threads`` / ``mkl_set_num_threads`` are no
longer provided (they are unreliable once PyTorch owns MKL thread-local state).
"""
# Copyright (C) TeNPy Developers, Apache license

import sys
import warnings

__all__ = [
    'memory_usage',
]


def memory_usage():
    """Return memory usage of the running python process.

    You can ``pip install psutil`` if you get only ``-1.``.

    Returns
    -------
    mem : float
        Currently used memory in megabytes. ``-1.`` if no way to read out.

    """
    try:
        import resource  # linux-only

        unit_to_MB = 1024**2 if sys.platform == 'darwin' else 1024  # linux uses kB, but MacOS byte
        # see also issue #262
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / unit_to_MB
    except ImportError:
        pass
    try:
        import psutil

        proc = psutil.Process()
        return proc.memory_info().rss / 1024**2
    except ImportError:
        pass
    warnings.warn('No tool to determine memory_usage')
    return -1.0
