Installation instructions
=========================

Right now, this documentetion is only distributed with the code, so I guess you already have the code somehow.
Let's assume you got the code with ``git clone {url-to-gitrepo} $HOME/TenPyLight``.

If you don't want to contribute, you should checkout the latest stable release::

    cd $HOME/TenPyLight
    git tag   # this prints the available version tags
    git co v0.0.0  # or whatever is the lastest stable version


Minimal Requirements
--------------------
This code is (currently) pure Python 2.7.  with 
Beside the standard library, you will only need fairly recent versions of `NumPy <http://www.numpy.org>`_ and `SciPy <http://www.scipy.org>`_.
Beside the standard library you will only need few things listed below.

Some kind of LAPACK/BLAS library is a prerequisite for SciPy.
If you have numpy >= 1.11.1 and scipy >= 0.17.0, the code should run fine.

Including tenpy into PYTHONPATH
-------------------------------
The python source is in the directory `tenpy`. 
This folder `tenpy` should placed in (one of the folders of) the environment variable 
`$PYTHONPATH <http://docs.python.org/2/using/cmdline.html#envvar-PYTHONPATH>`_.
Whenever the path is set, you should be able to use the library from within python::

    >>> import tenpy
    >>> print tenpy.__full__version__

This should output you the current version of this library as well as the used python, numpy and scipy libraries.
(For reproducability, you might want to save this along with your data.)

On Linux, to set the PYTHONPATH permanently you can add the following line to your `~/.bashrc`::

    export PYTHONPATH=$HOME/path/to/TenPyLight

(If you have already a path in there, separate the paths with a colon ``:``.) 
You might need to start a new terminal session or relogin to reload the ``~/.bashrc``.

Recommendation for optimization
-------------------------------
If you want to run larger simulations, we recommend the use of Intels MKL.
It ships with a Lapack library, and uses optimization for Intel CPUs.
Moreover, it uses parallelization of the LAPACK/BLAS routines, which makes execution much faster.
Currently, the library itself supports no other way of parallelization.

If you don't have a python version which is built against MKL, 
we recommend using the `anaconda <https://www.continuum.io/downloads>`_ distribution, which ships with Intel MKL
and is available for Linux, Mac and Windows. Note that you don't need administrator rights to install it.
Simply follow the (straight-forward) instructions of the web page for the installation.
It installs everything needed into a single folder (by default ``$HOME/anaconda2``).
Note that on linux it may add a line to your ``.bashrc`` to add ``$HOME/andaconda2/bin`` to the ``$PATH`` environment
variable, thus changing the default ``python``. If you run ``python`` interactively, the first output line should 
state the python version (2.7.#) and contain ``Anaconda``.

Once you managed to install conda, ensure that you have the needed packages with::

    conda install mkl scipy numpy

Optional packages (see `Optional Requirements`_ below) can be installed with::

    conda install matplotlib sphinx numpydoc pip nose bottleneck
    pip install yapf    # (if you want to contribute)

.. note :
    MKL uses different threads to parallelize different BLAS and LAPACK routines.
    If you run the code on a cluster, make sure that you specify the number of used cores/threads correctly.
    By default, MKL uses all the available CPUs, which might be in stark contrast than what you required from the
    cluster. The easiest way to acchieve to set the used threads is using the environment variable `MKL_NUM_THREADS` (or `OMP_NUM_THREADS`).
    For a dynamic change of the used threads, you might want to look at :mod:`~tenpy.tools.process`.

For further optimization, look at :mod:`tenpy.tools.optimization`.


Checking the installation
-------------------------
As a first check of the installation you can try to run the python files in the `examples/` subfolder; all of them
should run without error.

You can also run the automated testsuite with `nose` to make sure everything works fine::

    cd $HOME/path/to/TenPyLight/tests
    nosetests

This should run some tests, hopefully indicated by a lot of dots ``.`` and 
no ``E`` or ``F``, which indicate successfully run tests, errors and failures, respectively.
In case of errors or failures it gives a detailed traceback and possibly some output of the test.
At least the stable releases should run these tests without any failures.

If you can run the examples but not the tests, you might want to check whether `nosetests` actually uses the correct
python version.

Optional Requirements
---------------------
In addition, some code uses `MatPlotLib <http://www.matplotlib.org>`_ for plotting and visualization.
However, this is not necessary for running the simulations.

For building the documentation, you need
`Sphinx <http://www.sphinx-doc.org>`_ and `numpydoc <http://pypi.python.org/pypi/numpydoc>`_.

If you plan to contribute to the code, you should use
`yapf <http://github.com/google/yapf>`_ and `nose <http://nose.readthedocs.io/en/latest/>`_.

If you have the python package manager ``pip``, all of these can be installed with (assuming some kind of linux) ::

    sudo pip install --upgrade numpy scipy      # the required libraries
    sudo pip install --upgrade matplotlib       # for plotting
    sudo pip install --upgrade Sphinx numpydoc  # for building html documentation
    sudo pip install --upgrade yapf             # python formater to unify the code style
    sudo pip install --upgrade nose             # nosetest: automated teseting to check if everything works as it should
    sudo pip install --upgrade bottleneck       # some optimization of numpy bottlenecks


.. note ::

   If you don't have super user rights (``sudo``), try ``pip install --upgrade --user [packagenames...]``
   instead to install the packages to your home directory.
   If you still run into troubles, you might want to check whether the `pip` you call corresponds to the python version
   you want to use.


Compilation of np_conserved
---------------------------
At the heart of the tenpy library is the module :mod:`tenpy.linalg.np_conseved`, which provides an Array class to exploit the
conservation of abelian charges. The data model of python is not ideal for the required book-keeping, thus
we have implemented the same np_conserved module in `Cython <http://cython.org>`_. 
This allows to compile (and thereby optimize) the corresponding python module, thereby speeding up the execution of the
code. While this might give a significant speed-up for code with small matrix dimensions, don't expect the same speed-up in
cases where most of the CPU-time is already spent in matrix dimensions (i.e. if the bond dimension of your MPS is huge).

To compile the code, you first need to install cython ::

    conda install cython                    # when using anaconda, or
    sudo pip install --upgrade cython       # when using pip

After that, go to the root directory of tenpy and simply run ::

    bash ./compile.sh

It is not required to separately download (and install) Intel MKL: the compilation just obtains the includes from numpy.
In other words, if your current numpy version uses MKL (as the one provided in anaconda), the compiled code will also use it.
