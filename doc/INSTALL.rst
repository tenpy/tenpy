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
This code is (currently) pure python 2.7. 
Beside the standard library you will need only a few things listed below.

Python 2.7 with fairly recent versions of `NumPy <http://www.numpy.org>`_ and `SciPy <http://www.scipy.org>`_.
Some kind of LAPACK/BLAS library is a prerequisite for SciPy.

Optional Requirements
---------------------
In addition, some code uses `MatPlotLib <http://www.matplotlib.org>`_ for plotting and visualization.
However, this is not necessary for running the simulations.

For building the documentation, you need
`Sphinx <http://www.sphinx-doc.org>`_ and `numpydoc <http://pypi.python.org/pypi/numpydoc>`_.
If you plan to contribute to the code, you should use
`yapf <http://github.com/google/yapf>`_ and `nose <http://nose.readthedocs.io/en/latest/>`_.

If you have ``pip``, all of these can be installed with ::

    sudo pip install --upgrade numpy scipy   # required
    sudo pip install --upgrade matplotlib    # for plotting
    sudo pip install --upgrade Sphinx numpydoc  # for building html documentation
    suod pip install --upgrade yapf          # python formater to unify the code style

If you want to run larger simulations, we recommend the use of Intels MKL.
It ships with a Lapack library, and uses optimization for Intel CPUs.
Moreover, it uses parallelization of the LAPACK/BLAS routines, which makes execution much faster.

Recommendation for optimization
-------------------------------
If you don't have a python version which is built against MKL, 
we recommend using the `anaconda <https://www.continuum.io/downloads>` distribution, which ships with Intel MKL
and is available for Linux, Mac and Windows. Note that you don't need administrator rights to install it.
Simply follow the (straight-forward) instructions of the web page for the installation.
It installs everything needed into a single folder (by default ``$HOME/anaconda2``).
Note that on linux it may add a line to your ``.bashrc`` to add ``$HOME/andaconda2/bin`` to the ``$PATH`` environment
variable, thus changing the default ``python``. If you run ``python`` interactively, the first output line should 
state the python version (2.7.#) and contain ``Anaconda``.

Once you managed to install conda, ensure that you have the needed packages with::

    conda install mkl scipy numpy bottleneck

The optional packages can be installed with::

    conda install matplotlib sphinx numpydoc pip nose
    pip install yapf    # (if you want to contribute)

.. note :
    MKL uses different threads to parallelize different BLAS and LAPACK routines.
    If you want to change that (for example because you run tenpy on a cluster),
    take a look at :mod:`tenpy/tools/process`.

Including tenpy into PYTHONPATH
-------------------------------

The python source is in the directory `tenpy`. 
This folder `tenpy` should placed in (one of the folders of) the environment variable 
`$PYTHONPATH <http://docs.python.org/2/using/cmdline.html#envvar-PYTHONPATH>`_.
Whenever the path is set, you should be able to use the library from within python::

    >>> import tenpy

On Linux, to set the PYTHONPATH permanently you can add the following line to your `~/.bashrc`::

    export PYTHONPATH=$HOME/TenPyLight

(If you have already a path in there, separate the paths with a colon ``:``.) 
You might need to start a new terminal session or relogin to reload the ``~/.bashrc``.

If you want to make sure everything works fine, you can also check the installation with `nose`::

    cd $HOME/TenPyLight/tenpy/tests
    nosetests

This should run some tests, hopefully indicated by a lot of dots ``.....`` and 
no ``E`` or ``F`` indicating errors and failures, respectively.
In case of failures it gives a detailed traceback and possibly some output of the test.
At least the stable releases should run these tests without failures.
