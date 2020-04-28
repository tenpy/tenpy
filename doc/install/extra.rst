Extra requirements
==================

We have some extra requirements that you don't need to install to use TeNPy, but that you might find usefull to work with.
TeNPy does not import the following libraries (at least not globally), but some functions might expect arguments
behaving like objects from these libraries.

.. note ::
    
    If you created a [conda]_ environment with ``conda env create -f environment.yml``, all the extra requirements below
    should already be installed :)
    (However, a ``pip install -r requirements.txt`` does not install them.)

Matplotlib
^^^^^^^^^^
The first extra requirement is the [matplotlib]_ plotting library.
Some functions expect a :class:`matplotlib.axes.Axes` instance as argument to plot some data for visualization.

Intel's Math Kernel Library (MKL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you want to run larger simulations, we recommend the use of Intel's MKL.
It ships with a Lapack library, and uses optimization for Intel CPUs.
Moreover, it uses parallelization of the LAPACK/BLAS routines, which makes execution much faster.
As of now, the library itself supports no other way of parallelization.

If you don't have a python version which is built against MKL, 
we recommend using [conda]_ or directly `intelpython <https://software.intel.com/en-us/distribution-for-python/get-started>`_.
Conda has the advantage that it allows to use different environments for different projects.
Both are available for Linux, Mac and Windows; note that you don't even need administrator rights to install it on linux.
Simply follow the (straight-forward) instructions of the web page for the installation.
After a successfull installation, if you run ``python`` interactively, the first output line should 
state the python version and contain ``Anaconda`` or ``Intel Corporation``, respectively.

If you have a working conda package manager, you can install the numpy build against mkl with::

    conda install mkl numpy scipy

.. note ::
    
    MKL uses different threads to parallelize various BLAS and LAPACK routines.
    If you run the code on a cluster, make sure that you specify the number of used cores/threads correctly.
    By default, MKL uses all the available CPUs, which might be in stark contrast than what you required from the
    cluster. The easiest way to set the used threads is using the environment variable `MKL_NUM_THREADS` (or `OMP_NUM_THREADS`).
    For a dynamic change of the used threads, you might want to look at :mod:`~tenpy.tools.process`.

HDF5 file format support
^^^^^^^^^^^^^^^^^^^^^^^^
We support exporting data to files in the [HDF5]_ format through the python interface of the 
`h5py <https://docs.h5py.org/en/stable/>` package, see :doc:`/intro/input_output` for more information.
However, that requires the installation of the HDF5 library and h5py.

YAML parameter files
^^^^^^^^^^^^^^^^^^^^
The :class:`tenpy.tools.params.Config` class supports reading and writing YAML files, which requires the package
`pyyaml`; ``pip install pyyaml``.

Tests
-----
To run the tests, you need to install `pytest <http://pytest.org>`_, which you can for example do with ``pip install pytest``.
For information how to run the tests, see :doc:`/install/test`.
