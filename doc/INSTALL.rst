Installation instructions: from source
======================================

Minimal Requirements
--------------------
This code works with a minimal requirement of pure Python>=3.5 
and somewhat recent versions of `NumPy <http://www.numpy.org>`_ and `SciPy <http://www.scipy.org>`_.

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
    You can find it under `releases <https://github.com/tenpy/tenpy/releases>`_,
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
    /home/username/TeNPy/tenpy/tools/optimization.py:276: UserWarning: Couldn't load compiled cython code. Code will run a bit slower.
      warnings.warn("Couldn't load compiled cython code. Code will run a bit slower.")
    >>> tenpy.show_config()
    tenpy 0.4.0.dev0+7706003 (not compiled),
    git revision 77060034a9fa64d2c7c16b4211e130cf5b6f5272 using
    python 3.7.3 (default, Mar 27 2019, 22:11:17) 
    [GCC 7.3.0]
    numpy 1.16.3, scipy 1.2.1


:func:`tenpy.show_config` prints the current version of the used TeNPy library as well as the versions of the used python, numpy and scipy libraries,
which might be different on your computer. It is a good idea to save this data (given as string in ``tenpy.version.version_summary`` along with your data to allow to reproduce your results exactly.

If you got a similar output as above: congratulations! You can now run the codes :)


MKL and further packages
------------------------
If you want to run larger simulations, we recommend the use of Intel's MKL.
It ships with a Lapack library, and uses optimization for Intel CPUs.
Moreover, it uses parallelization of the LAPACK/BLAS routines, which makes execution much faster.
As of now, the library itself supports no other way of parallelization.

If you don't have a python version which is built against MKL, 
we recommend using the `anaconda <https://www.continuum.io/downloads>`_ distribution, which ships with Intel MKL,
or directly `intelpython <https://software.intel.com/en-us/distribution-for-python/get-started>`_.
Conda has the advantage that it allows to use different environments for different projects.
Both are available for Linux, Mac and Windows; note that you don't even need administrator rights to install it on linux.
Simply follow the (straight-forward) instructions of the web page for the installation.
After a successfull installation, if you run ``python`` interactively, the first output line should 
state the python version and contain ``Anaconda`` or ``Intel Corporation``, respectively.

If you have a working conda package manager, you can install the numpy build against mkl with::

    conda install mkl numpy scipy

If you prefer using a separete conda environment, you can also use the following code to install all the recommended
packages::

    conda env create -f environment.yml
    conda activate tenpy

.. note ::

    MKL uses different threads to parallelize various BLAS and LAPACK routines.
    If you run the code on a cluster, make sure that you specify the number of used cores/threads correctly.
    By default, MKL uses all the available CPUs, which might be in stark contrast than what you required from the
    cluster. The easiest way to set the used threads is using the environment variable `MKL_NUM_THREADS` (or `OMP_NUM_THREADS`).
    For a dynamic change of the used threads, you might want to look at :mod:`~tenpy.tools.process`.

Some code uses `MatPlotLib <http://www.matplotlib.org>`_ for plotting, e.g., to visualize a lattice.
However, having matplotlib is not necessary for running any of the algorithms: tenpy does not ``import matplotlib`` by default.
Further optional requirements are listed in the ``requirements*.txt`` files in the source repository.

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
    pip install --upgrade Cython            # when using pip

Moreover, you need a C++ compiler. 
For example, on Ubuntu you can install ``sudo apt-get install build_essential``,
or on Windows you can download MS Visual Studio 2015.

After that, go to the root directory of TeNPy (``$HOME/TeNPy``) and simply run ::

    bash ./compile.sh

Note that it is not required to separately download (and install) Intel MKL: the compilation just obtains the includes 
from numpy. In other words, if your current numpy version uses MKL (as the one provided by anaconda),
the compiled TeNPy code will also use it.

After a successful compilation, the warning that TeNPy was not compiled should go away::

    >>> import tenpy
    >>> tenpy.show_config()
    tenpy 0.4.0.dev0+b60bad3 (compiled from git rev. b60bad3243b7e54f549f4f7c1f074dc55bb54ba3),
    git revision b60bad3243b7e54f549f4f7c1f074dc55bb54ba3 using
    python 3.7.3 (default, Mar 27 2019, 22:11:17) 
    [GCC 7.3.0]
    numpy 1.16.3, scipy 1.2.1

.. note ::

    For further optimization options, look at :mod:`tenpy.tools.optimization`.

Checking the installation
-------------------------
As a first check of the installation you can try to run (one of) the python files in the `examples/` subfolder;
hopefully all of them should run without error.

You can also run the automated testsuite with `pytest <http://pytest.org>`_  (``pip install pytest``) to make sure everything works fine::

    cd $HOME/TeNPy/tests
    pytest

This should run some tests. In case of errors or failures it gives a detailed traceback and possibly some output of the test.
At least the stable releases should run these tests without any failures.

If you can run the examples but not the tests, check whether `pytest` actually uses the correct python version.

The test suite is also run automatically with `travis-ci <https://travis-ci.org>`_, results can be inspected at `here <https://travis-ci.org/tenpy/tenpy>`_.

Installation instructions: from packages
========================================

If you have the conda package manager, you can simply download the ``environment.yml`` file and create a new environment for tenpy with all the require packages::

    conda env create -f environment.yml
    conda activate tenpy
    
This will also install `pip <https://pip.pypa.io/en/stable/>`_. Alternatively, if you only have `pip`, install the
required packages with::
    
    pip install -r requirements.txt

.. note ::

    Make sure that the `pip` you call corresponds to the python version
    you want to use. (e.g. by using ``python -m pip`` instead of a simple ``pip``
    Also, you might need to use the arguement ``--user`` to install the packages to your home directory, 
    if you don't have ``sudo`` rights.

.. warning ::
    
    It might just be a temporary problem, but I found that the `pip` version of numpy is incompatible with 
    the python distribution of anaconda. 
    If you have installed the intelpython or anaconda distribution, use the `conda` packagemanager instead of `pip` for updating the packages whenever possible!


After that, you can install the latest *stable* TeNPy package (without downloading the source) from `PyPi
<https://pypi.org>` with::

    pip install physics-tenpy # note the different package name - 'tenpy' was taken!

To get the latest development version from the github master branch, you can use::

    pip install git+git://github.com/tenpy/tenpy.git

Finally, if you downloaded the source and want to **modify parts of the source**, you should install tenpy in
development version with ``-e``::

    cd $HOME/TeNPy # after downloading the source
    pip install --editable .

In all cases, you can uninstall tenpy with::

    pip uninstall physics-tenpy  # note the longer name!
   

Updating to a new version
=========================
**Before** you update, take a look at the :doc:`/changelog`, which lists the changes, fixes, and new stuff. 
Most importantly, it has a section on *backwards incompatible changes* (i.e., changes which may break your
existing code) along with information how to fix it. Of course, we try to avoid introducing such incompatible changes,
but sometimes, there's no way around them.

How to update depends a little bit on the way you installed TeNPy. Of course, you have always the option to just remove
the tenpy files and download the newest version, following the instructions above.

Alternatively, if you used ``git clone ...`` to download the repository, you can update to the newest version using `Git`.
First, briefly check that you didn't change anything you need to keep with ``git status``.
Then, do a ``git pull`` to download (and possibly merge) the newest commit from the repository.

.. note ::

    If some Cython file (ending in ``.pyx``) got renamed/removed (e.g., when updating from v0.3.0 to v0.4.0), 
    you first need to remove the corresponding binary files. 
    You can do so with the command ``bash cleanup.sh``.

    Furthermore, whenever one of the cython files (ending in ``.pyx``) changed, you need to re-compile it.
    To do that, simply call the command ``bash ./compile`` again.
    If you are unsure whether a cython file changed, compiling again doesn't hurt.

To summarize, you need to execute the folllowing bash commands in the repository::

    # 0) make a backup of the whole folder
    git status   # check the output whether you modified some files
    git pull 
    bash ./cleanup.sh  # (confirm with 'y')
    bash ./compile.sh
