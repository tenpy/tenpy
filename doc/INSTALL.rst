Installation instructions
=========================

Minimal Requirements
--------------------
This code works with a minimal requirement of pure Python 3 
and somewhat recent versions of `NumPy <http://www.numpy.org>`_ and `SciPy <http://www.scipy.org>`_.
(If you have numpy >= 1.11.1 and scipy >= 0.17.0, the code should run fine.)

Getting the source
------------------

The following instructions are for (some kind of) Linux, and tested on Ubuntu. 
However, the code itself should work on other operating systems as well (in particular MacOS and Windows).

The offical repository is at https://github.com/tenpy/tenpy.git.
To get the latest version of the code, you can clone it with `Git <https://git-scm.com/>`_ using the following commands::

    git clone https://github.com/tenpy/tenpy.git $HOME/TeNPy
    cd $HOME/TeNPy

Adjust ``$HOME/TeNPy`` to the path wherever you want to save the library.

Optionally, if you don't want to contribute, you can checkout the latest stable release::

    git tag   # this prints the available version tags
    git checkout v0.3.0  # or whatever is the lastest stable version

.. note ::

    In case you don't have Git, you can download the repository as a ZIP archive.
    You can find under `releases <https://github.com/tenpy/tenpy/releases>`_,
    or the `latest development version <https://github.com/tenpy/tenpy/archive/master.zip>`_.


Minimal installation: Including tenpy into PYTHONPATH
-----------------------------------------------------
The python source is in the directory `tenpy/` of the repository.
This folder `tenpy/` should be placed in (one of the folders of) the environment variable 
`PYTHONPATH <http://docs.python.org/2/using/cmdline.html#envvar-PYTHONPATH>`_.
On Linux, you can simply do this with the following line in the terminal::

    export PYTHONPATH=$HOME/TeNPy

(If you have already a path in this variable, separate the paths with a colon ``:``.) 
However, if you enter this in the terminal, it will only be temporary for the terminal session where you entered it.
To make it permanently, you can add the above line to the file ``$HOME/.bashrc``.
You might need to restart the terminal session or need to relogin to force a reload of the ``~/.bashrc``.

Whenever the path is set, you should be able to use the library from within python::

    >>> import tenpy
    >>> print(tenpy.__version__)
    (0, 3, 0)
    >>> print(tenpy.__full_version__)
    v0.3.0-57-gdfacc70 using python 3.5.2 |Intel Corporation| (default, Feb 12 2017, 04:02:50)
    [GCC 4.8.2 20140120 (Red Hat 4.8.2-15)] with numpy 1.13.3, scipy 1.0.0

This should statement printes the current version of this TeNPy library as well as the versions of the used python, numpy and scipy libraries,
which might be different on your computer. It is a good idea to save this along with your data.

If you got a similar output as above: congratulations! You can now run the codes :)

Recommendation for optimization
-------------------------------
If you want to run larger simulations, we recommend the use of Intels MKL.
It ships with a Lapack library, and uses optimization for Intel CPUs.
Moreover, it uses parallelization of the LAPACK/BLAS routines, which makes execution much faster.
As of now, the library itself supports no other way of parallelization.

If you don't have a python version which is built against MKL, 
we recommend using the `anaconda <https://www.continuum.io/downloads>`_ distribution, which ships with Intel MKL,
or directly `intelpython <https://software.intel.com/en-us/distribution-for-python/get-started>`_.
Both are available for Linux, Mac and Windows; note that you don't even need administrator rights to install it on linux.
Simply follow the (straight-forward) instructions of the web page for the installation.
After a successfull installation, if you run ``python`` interactively, the first output line should 
state the python version and contain ``Anaconda`` or ``intelpython``, respectively.
On Linux, make sure that ``$PATH`` contains the ``bin/`` folder of the installation; you might want to adjust your
``~/.bashrc`` if necessary.

Once you managed to install conda, ensure that you have the needed packages with::

    conda install mkl scipy numpy

Optional packages (see `Optional Requirements`_ below) can be installed with::

    conda install matplotlib sphinx numpydoc pip nose bottleneck
    pip install yapf    # (if you want to contribute)


.. note ::

    MKL uses different threads to parallelize various BLAS and LAPACK routines.
    If you run the code on a cluster, make sure that you specify the number of used cores/threads correctly.
    By default, MKL uses all the available CPUs, which might be in stark contrast than what you required from the
    cluster. The easiest way to set the used threads is using the environment variable `MKL_NUM_THREADS` (or `OMP_NUM_THREADS`).
    For a dynamic change of the used threads, you might want to look at :mod:`~tenpy.tools.process`.


Compilation of np_conserved
---------------------------
At the heart of the TeNPy library is the module :mod:`tenpy.linalg.np_conseved`, which provides an Array class to exploit the
conservation of abelian charges. The data model of python is not ideal for the required book-keeping, thus
we have implemented the same np_conserved module in `Cython <http://cython.org>`_.
This allows to compile (and thereby optimize) the corresponding python module, thereby speeding up the execution of the
code. While this might give a significant speed-up for code with small matrix dimensions, don't expect the same speed-up in
cases where most of the CPU-time is already spent in matrix multiplications (i.e. if the bond dimension of your MPS is huge).

To compile the code, you first need to install `Cython <http://cython.org>`_ ::

    conda install cython                    # when using anaconda, or
    sudo pip install --upgrade cython       # when using pip

After that, go to the root directory of TeNPy (``$HOME/TeNPy``) and simply run ::

    bash ./compile.sh

It is not required to separately download (and install) Intel MKL: the compilation just obtains the includes from numpy.
In other words, if your current numpy version uses MKL (as the one provided by anaconda), the compiled TeNPy code will also use it.

.. note ::

    For further optimization options, look at :mod:`tenpy.tools.optimization`.


Checking the installation
-------------------------
As a first check of the installation you can try to run the python files in the `examples/` subfolder; all of them
should run without error.

You can also run the automated testsuite with `nose <http://nose.readthedocs.io/en/latest/>`_ to make sure everything works fine::

    cd $HOME/TeNPy/tests
    nosetests

This should run some tests, hopefully indicated by a lot of dots ``.`` and 
no ``E`` or ``F``, which indicate successfully run tests, errors and failures, respectively.
In case of errors or failures it gives a detailed traceback and possibly some output of the test.
At least the stable releases should run these tests without any failures.

If you can run the examples but not the tests, check whether `nosetests` actually uses the correct python version.


Optional Requirements
---------------------
Some code uses `MatPlotLib <http://www.matplotlib.org>`_ for plotting, e.g., to visualize a lattice.
However, having matplotlib is not necessary for running any of the algorithms: tenpy does not ``import matplotlib`` by default.

For building the documentation, you need `Sphinx <http://www.sphinx-doc.org>`_ and `numpydoc <http://pypi.python.org/pypi/numpydoc>`_.

If you ever plan to contribute to the code, you should use `yapf <http://github.com/google/yapf>`_ and `nose <http://nose.readthedocs.io/en/latest/>`_.

If you have the python package manager ``pip``, all of these can be installed with::

    sudo pip install --upgrade numpy scipy      # the required libraries
    sudo pip install --upgrade matplotlib       # for plotting
    sudo pip install --upgrade bottleneck       # some optimization of numpy bottlenecks
    sudo pip install --upgrade nose             # nosetest: automated teseting to check if everything works as it should
    sudo pip install --upgrade Sphinx numpydoc  # for building html documentation
    sudo pip install --upgrade yapf             # python formater to unify the code style


.. note ::

    If you don't have super user rights (``sudo``), try ``pip install --upgrade --user [packagenames...]``
    instead to install the packages to your home directory.
    If you still run into troubles, you might want to check whether the `pip` you call corresponds to the python version
    you want to use.
   
.. warning ::
    
    It might just be a temporary problem, but I found that the `pip` version of numpy is incompatible with 
    the python distribution of anaconda. 
    If you have installed the intelpython or anaconda distribution, use the `conda` packagemanager instead of `pip` for updating the packages whenever possible!
