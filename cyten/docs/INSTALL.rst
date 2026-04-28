Installation instructions
=========================

.. todo ::
   This doesn't work yet, we still need to set this up, once we have the first (beta) release.
   Meanwhile, just do a `pip install .` from the top folder of the repo.

With the `conda package manager <https://docs.conda.io>`_ you can install python with::

    conda install --channel=conda-forge cyten  # TODO doesn't work yet

If you don't have conda, but you have `pip <https://pip.pypa.io>`_, you can::

    pip install cyten   # TODO doesn't work yet

Building from source
++++++++++++++++++++

To build cyten locally on your machine, install the following requirements
(currently only tested on standard linux distros like ubuntu - no Windows support yet, use WSL):

- C++ compiler with at least C++17 standard  (can be installed manually with `conda install -c conda-forge compilers` if needed)
- CMake, make
- boost library (only required for the intrusive pointer - can we get rid of that requirement?)
- Python >= 3.10, with numpy>=2.0, scipy and a few other python packages as listed `environment.yml`
- scikit-build

The easiest way to install all of those is to create a conda envrironment from the `environment.yml` to install all requirements
and then pip install the package (use `docs/environment.yml` if you plan to build the documentation as well):

```
conda env create -f environment.yml -n cyten
conda activate cyten
conda install -c conda-forge _openmp_mutex=*=*_llvm # on Linux/WSL only
conda install -c conda-forge llvm-openmp # on MacOS only
pip install -v .
```

If needed, you can add defines for the CMake build as options to pip, e.g. `pip install -v -C cmake.define.=ON .`.


For a debug build, you can even enable automatic rebuild upon python import:
```
pip install -v --no-build-isolation -C editable.rebuild=true -e .
```
