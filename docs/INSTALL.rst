Installation instructions
=========================

How do I get set up? short summary
----------------------------------
Right now, this documentetion is only distributed with the code, so I guess you already have the code somehow.

Minimal Requirements:
Python 2.7 with fairly recent versions of `NumPy <http://www.numpy.org>`_ and `SciPy <http://www.scipy.org>`_.
Further `Cython <http://www.cython.org>`_ and some kind of LaPack library (including the headers).

In addition, some code uses `MatPlotLib <http://www.matplotlib.org>`_ for plotting and visualization.
However, this is not necessary for running simulations.
For building the documentation, you need
`Sphinx <http://www.sphinx-doc.org>`_ and `numpydoc <http://pypi.python.org/pypi/numpydoc>`_.
If you plan to contribute to the code, you should use
`yapf <http://github.com/google/yapf>`_ and `nose <http://nose.readthedocs.io/en/latest/>`_.

If you want to run larger simulations, we recommend the use of Intels MKL and icc.
It ships with a Lapack library, and uses optimization for Intel CPUs.
Moreover, it uses parallelization of the Lapack routines, which makes execution much faster...
However, that requires compiling python with the same icc, which is a bit complicated. 
Details can be found in :doc:`docs/install.rst <install>`

The python source is in the directory `tenpy`. 
This folder `tenpy` should placed in (one of the folders of) the environment variable 
`$PYTHONPATH <http://docs.python.org/2/using/cmdline.html#envvar-PYTHONPATH>`_.
For example, if you have this library in `$HOME/path/to/TenPy/`, you can add the following line in your `~/.bashrc`::

    export PYTHONPATH=$HOME/path/to/TenPy/

You can then include the library in python::

    >>> import tenpy


.. Note ::
    The remainder is a literal translation of TenPy's docs/setup.tex to rst. It might be outdated at some points.


How to compile TenPy against MKL
--------------------------------

These are instructions to get Roger and Mike's DMRG program, TenPy2, working on a computer. They also serve as instructions for installing any of the other programs on the following list, which are all required. There are some dependencies, so install these in the order given if you can. After the list, there are separate section for each of the items in the list.
- Intel MKL
- Python 2.7
- Numpy
- Scipy
- Cython
- slycot
- six
- matplotlib
- control
- TenPy2

It is a good idea to make a folder in your home directory, which I will call sources. When I say download something, I mean that you should put in in sources, and run everything in there.

IMPORTANT: you must use the SAME compiler (gcc or icc) for Python, numpy and scipy. MKL will install icc, and set the environment variable CC=icc. So if you want to install everything with gcc, you have to change this back. This guide will cover installing with icc.

Intel MKL
---------
These are linear algebra libraries. You could use different libraries, such as BLAS or ATLAS, instead of MKL, but I haven't done this. 


To install MKL, visit this 
`link <https://registrationcenter.intel.com/RegCenter/NComForm.aspx?ProductID=1540&pass=yes>`_

Download the MKL .tgz file and unzip its contents into `/sources`. Go into the directory and run `./install.sh`. There will probably be some warnings about our OS being unsupported (yay scientific linux!), but it works anyway. Follow the steps of the installer.

For MKL to work, a bunch of environment variables need to be set. We can set these manually, but then every time we logout (or switch to root), we will have to reset them. Instead, do the following to make it so all users have these environment variables set all the time. Add the following line to /etc/bashrc::

  source /opt/intel/bin/compilervars.sh intel64

Before moving on, it's a good idea to test that icc and ifort work. Make a simple c program and try to compile it with icc.

Python 2.7
----------
Python is likely already installed on your computer in some form. However it is a language which is update often and is not all that backwards-compatible. Scientific Linux ships with version 2.6, TenPy uses 2.7. The newest version is 3, but its not backwards compatible so we actually need 2.7. These instructions are for installing 2.7, assuming 2.6 is already installed. The same instructions should also work for 3, with the obvious changes made.

The first step is to get a source tarball of Python, which can be found `here <https://www.python.org/download/>`_
Download this to `~/sources` and unzip it. Now run the following commands::

    ./configure
    make
    make test

If you're trying to compile Python with Intel, make test probably failed. To get it working again there are a few steps. First, cd to Modules/zlib. Run::

    ./configure
    make
    make test

you shouldn't have any errors, now switch to root and run ::

    make install

Also, go into the file 
`Modules/_ctypes/libffi/src/x86/ffi64.c`
and after the includes add the line ::

    typedef struct {int64_t m[2];} __m128;

which fixes a bug in icc.

The next step is to go back to the Python root directory and rerun `./configure`. The configure script produces a `Makefile`, we now need to edit this Makefile (so if you ever rerun ./configure, you will also need to redo these changes). Change the following lines, in my Makefile they were lines 36 and 37::

    CC=             icc -pthread -fPIC -fp-model strict
    CXX=            icpc -pthread -fPIC -fp-model strict

Finally, add the following line to the start of `/Modules/\_ctypes/libffc/src/x86/ffi64.c` ::

    typedef struct {int64_t m[2];} __m128;


Run::

    make
    make test

There should be no errors. Finally switch to root and run ::

    make install

Python 2.7 is now installed.

Now by default on my system the command python still calls version 2.6, which is a pain. To fix this, we need to link the default python command with python 2.7. The command for this is ::

    ln -s /usr/local/bin/python2.7 /usr/local/bin/python

You may have to restart your shell to see the effect


Numpy
-----
We now want to install numpy, and tell it to use the MKL libraries. First download numpy and extract it into /sources. Then find the part of the site.cfg file in the numpy directory that looks like the following and edit it. There are two sections to edit, [DEFAULT] and [mkl]. The default section you should probably simply need to uncomment, as long as `/usr/local/lib` contains the stuff you would expect. For the [mkl] section, make it look like this::

    [mkl]
    libraries = lapack,f77blas,cblas,atlas
    library_dirs= /opt/intel/composerxe/mkl/lib/intel64:/opt/intel/composer_xe_2013_sp1.2.144/mkl/lib/intel64
    include_dirs=/opt/intel/include/:/opt/intel/include/intel64/:/opt/intel/mkl/include
    mkl_libs=mkl_rt
    lapack_libs=

Note that the numbers in the `composer\_xe\_` folder might change depending on which version you have, so check what the folder is actually called.
Then look in the file `numpy/distutils/intelccompiler.py`, and edit it to::

    self.cc_exe = 'icc -O3 -g -fPIC -fp-model strict -fomit-frame-pointer -openmp -xhost' 

Also edit `numpy/distutils/fcompiler/intel.py` to read::

    ifort -xhost -openmp -fp-model strict -fPIC

Install numpy by running the following command as root::

    python setup.py config --compiler=intelem build_clib --compiler=intelem build_ext --compiler=intelem install

Test numpy by doing the following in python::

    import numpy as np
    np.test('full')

Skipped and knownfail tests are ok, there should be no errors or failed tests. You will need to install the nose package to run these tests
Also make sure numpy is seeing mkl, if it isn't there is likely a mistake in the `site.cfg` file. ::

    import numpy as np
    np.show_config()

There should be stuff under the mkl entry. If there isn't check the `site.cfg`. For your changes to have any effect you will need to delete the `/build` directory in the numpy folder and then rerun the `setup.py` command.

SciPy
-----
Execute this as root:: 

    python setup.py config --compiler=intelem --fcompiler=intelem build_clib --compiler=intelem --fcompiler=intelem build_ext --compiler=intelem --fcompiler=intelmen install

And test this by opening python and trying ::

    import scipy
    scipy.test('full')

Cython
------
Change to the directory you downloaded and do as root::

    python setup.py install

slycot
------
Change to the directory you downloaded and do as root::

    python setup.py install

six
---
matplotlib
----------
Change to the directory you downloaded and do as root::

    python setup.py install

Likely it will bring up a list of packages, you need to install all the mandatory ones it says it doesn't have. In particular, you may need to install pyparsing, setuptools and dateutil. You will also need to do ::

    yum install libpng-devel

control
-------

git
---

TenPy2
------
To get the libraries, first install dropbox. Once you've got that installed, do ::

    git clone ~/Dropbox/TenPy2.git TenPy2

do ::

    export MKL_DIR=/opt/intel/composer_xe_2013.sp1.2.144/mkl
    ./compile.sh
