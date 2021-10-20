Installation from source
------------------------

Minimal Requirements
^^^^^^^^^^^^^^^^^^^^
This code works with a minimal requirement of pure Python>=3.6
and somewhat recent versions of `NumPy <https://www.numpy.org>`_ and `SciPy <https://www.scipy.org>`_.

Getting the source
^^^^^^^^^^^^^^^^^^

The following instructions are for (some kind of) Linux, and tested on Ubuntu. 
However, the code itself should work on other operating systems as well (in particular MacOS and Windows).

The offical repository is at https://github.com/tenpy/tenpy.git.
To get the latest version of the code, you can clone it with [git]_ using the following commands::

    git clone https://github.com/tenpy/tenpy.git $HOME/TeNPy
    cd $HOME/TeNPy

.. note ::

    Adjust ``$HOME/TeNPy`` to the path wherever you want to save the library.

Optionally, if you don't want to contribute, you can checkout the latest stable release::

    git tag   # this prints the available version tags
    git checkout v0.3.0  # or whatever is the lastest stable version

.. note ::
    
    In case you don't have [git]_ installed, you can download the repository as a ZIP archive.
    You can find it under `releases <https://github.com/tenpy/tenpy/releases>`_,
    or the `latest development version <https://github.com/tenpy/tenpy/archive/main.zip>`_.


Minimal installation: Including tenpy into PYTHONPATH
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The python source is in the directory `tenpy/` of the repository.
This folder `tenpy/` should be placed in (one of the folders of) the environment variable 
`PYTHONPATH <https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH>`_.
On Linux, you can simply do this with the following line in the terminal::

    export PYTHONPATH=$HOME/TeNPy

(If you have already a path in this variable, separate the paths with a colon ``:``.) 
However, if you enter this in the terminal, it will only be temporary for the terminal session where you entered it.
To make it permanently, you can add the above line to the file ``$HOME/.bashrc``.
You might need to restart the terminal session or need to relogin to force a reload of the ``~/.bashrc``.

Whenever the path is set, you should be able to use the library from within python::

    >>> import tenpy
    /home/johannes/postdoc/2021-01-TenPy-with-MKL/TeNPy/tenpy/tools/optimization.py:308: UserWarning: Couldn't load compiled cython code. Code will run a bit slower.
    warnings.warn("Couldn't load compiled cython code. Code will run a bit slower.")
    >>> tenpy.show_config()
    tenpy 0.7.2.dev130+76c5b7f (not compiled),
    git revision 76c5b7fe46df3e2241d85c47cbced3400caad05a using
    python 3.9.1 | packaged by conda-forge | (default, Jan 10 2021, 02:55:42) 
    [GCC 9.3.0]
    numpy 1.19.5, scipy 1.6.0


:func:`tenpy.show_config` prints the current version of the used TeNPy library as well as the versions of the used python, numpy and scipy libraries,
which might be different on your computer. It is a good idea to save this data (given as string in ``tenpy.version.version_summary`` along with your data to allow to reproduce your results exactly.

If you got a similar output as above: congratulations! You can now run the codes :)


Compilation of np_conserved
^^^^^^^^^^^^^^^^^^^^^^^^^^^
At the heart of the TeNPy library is the module :mod:`tenpy.linalg.np_conseved`, which provides an Array class to exploit the
conservation of abelian charges. The data model of python is not ideal for the required book-keeping, thus
we have implemented the same np_conserved module in `Cython <https://cython.org>`_.
This allows to compile (and thereby optimize) the corresponding python module, thereby speeding up the execution of the
code. While this might give a significant speed-up for code with small matrix dimensions, don't expect the same speed-up in
cases where most of the CPU-time is already spent in matrix multiplications (i.e. if the bond dimension of your MPS is huge).

To compile the code, you first need to install `Cython <https://cython.org>`_ ::

    conda install cython                    # when using anaconda, or
    pip install --upgrade Cython            # when using pip

Moreover, you need a C++ compiler. 
For example, on Ubuntu you can install ``sudo apt-get install build_essential``,
or on Windows you can download MS Visual Studio 2015.
If you use anaconda, you can also use ``conda install -c conda-forge cxx-compiler``. 

After that, go to the root directory of TeNPy (``$HOME/TeNPy``) and simply run ::

    bash ./compile.sh

.. note ::

   There is no need to compile if you installed TeNPy directly with conda or pip.
   (You can verify this with `tenpy.show_config()` as illustrated below.)

Note that it is not required to separately download (and install) Intel MKL: the compilation just obtains the includes 
from numpy. In other words, if your current numpy version uses MKL (as the one provided by anaconda),
the compiled TeNPy code will also use it.

After a successful compilation, the warning that TeNPy was not compiled should go away::

    >>> import tenpy
    >>> tenpy.show_config()
    tenpy 0.7.2.dev130+76c5b7f (compiled without HAVE_MKL),
    git revision 76c5b7fe46df3e2241d85c47cbced3400caad05a using
    python 3.9.1 | packaged by conda-forge | (default, Jan 10 2021, 02:55:42) 
    [GCC 9.3.0]
    numpy 1.19.5, scipy 1.6.0

.. note ::
    
    For further optimization options, e.g. how to link against MKL, look at :doc:`/install/extra` and :mod:`tenpy.tools.optimization`.


Quick-setup of a development environment with conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can use the following bash commands to setup a new conda environment called `tenpy_dev` (call it whatever you
want!) and install TeNPy in there in a way which allows editing TeNPy's python code and still have it available everywhere in the conda environment::

    git clone https://github.com/tenpy/tenpy TeNPy
    cd TeNPy
    conda env create -f environment.yml -n tenpy_dev
    conda activate tenpy_dev
    pip install -e .
