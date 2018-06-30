"""Tools to read out total memory usage and get/set the number of threads.

If your python is compiled against MKL (e.g. if you use *anaconda* as recommended in INSTALL),
it will by default use as many threads as CPU cores are available.
If you run a job on a cluster, you should limit this to the number of cores you reserved --
otherwise your colleagues might get angry...
A simple way to achieve this is to set a suitable enviornment variable before calling your
python program, e.g. on the linux bash ``export OMP_NUM_THREADS=4`` for 4 threads.
(MKL used OpenMP and thus respects its settings.)

Alternatively, this module provides :func:`omp_get_nthreads` and :func:`omp_set_nthreads`,
which give their best to get and set the number of threads at runtime,
while still being failsave if the shared OpenMP library is not found.  In the latter case,
you might also try the equivalent :func:`mkl_get_nthreads` and :func:`mkl_set_nthreads`.
"""
# Copyright 2018 TeNPy Developers

import warnings
import ctypes
from ctypes.util import find_library

__all__ = [
    'memory_usage', 'omp_get_nthreads', 'omp_set_nthreads', 'mkl_get_nthreads', 'mkl_set_nthreads'
]

_omp_lib = None
_mkl_lib = None


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
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        pass
    try:
        import psutil
        proc = psutil.Process()
        return proc.memory_info().rss / 1024**2
    except ImportError:
        pass
    warnings.warn("No tool to determine memory_usage")
    return -1.


def load_omp_library(libs=["libiomp5.so",
                           find_library("libiomp5md"),
                           find_library("gomp")],
                     verbose=True):
    """Tries to load openMP library.

    Parameters
    ----------
    libs :
        list of possible library names we should try to load (with ctypes.CDLL).
    verbose : bool
        wheter to print the name of the loaded library.

    Returns
    -------
    omp : CDLL | None
        OpenMP shared libary if found, otherwise None.
        Once it was sucessfully imported, no re-imports are tried.
    """
    global _omp_lib
    if _omp_lib is None:
        for l in libs:
            if l is None:
                continue
            try:
                _omp_lib = ctypes.CDLL(l)
                if verbose:
                    print("loaded " + l + " for omp")
                break
            except OSError:
                pass
    if _omp_lib is None:
        warnings.warn("OpenMP library not found: can get/set nthreads")
    return _omp_lib


def omp_get_nthreads():
    """wrapper around OpenMP ``get_max_threads``.

    Returns
    -------
    max_threads : int
        The maximum number of threads used by OpenMP (and thus MKL).
        ``-1`` if unable to read out.
    """
    omp = load_omp_library()
    if omp is not None:
        return omp.omp_get_max_threads()
    return -1


def omp_set_nthreads(n):
    """wrapper around OpenMP ``set_nthreads``.

    Parameters
    ----------
    n : int
        the number of threads to use

    Returns
    -------
    success : bool
        whether the shared library was found and set.
    """
    omp = load_omp_library()
    if omp is not None:
        omp.omp_set_num_threads(int(n))
        return True
    return False


def mkl_get_nthreads():
    """wrapper around MKL ``get_max_threads``.

    Returns
    -------
    max_threads : int
        The maximum number of threads used by MKL. ``-1`` if unable to read out.
    """
    try:
        import mkl  # available in conda MKL
        return mkl.get_max_threads()
    except ImportError:
        try:
            mkl_rt = ctypes.CDLL('libmkl_rt.so')
            return mkl_rt.mkl_get_max_threads()
        except OSError:
            warnings.warn("MKL library not found: can't get nthreads")
    return -1


def mkl_set_nthreads(n):
    """wrapper around MKL ``set_num_threads``.

    Parameters
    ----------
    n : int
        the number of threads to use

    Returns
    -------
    success : bool
        whether the shared library was found and set.
    """
    try:
        import mkl  # available in conda MKL
        mkl.set_num_threads(n)
        return True
    except ImportError:
        try:
            mkl_rt = ctypes.CDLL('libmkl_rt.so')
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(n)))
            return True
        except OSError:
            warnings.warn("MKL library not found: can't set nthreads")
    return False
