Installation with conda from conda-forge
========================================

We provide a package for the [conda]_ package manager in the `conda-forge` channel, so you can install TeNPy as::

    conda install --channel=conda-forge physics-tenpy


Following the recommondation of `conda-forge <https://conda-forge.org/docs/user/introduction.html>`_, you can also make
conda-forge the default channel as follows::

    conda config --add channels conda-forge
    conda config --set channel_priority strict

If you have done this, you don't need to specify the ``--channel=conda-forge`` explicitly.

.. note ::

    The `numpy` package provided by the `conda-forge` channel by default uses openblas on linux. 
    As outlined in the `conda forge docs <https://conda-forge.org/docs/maintainer/knowledge_base.html#switching-blas-implementation>`_, 
    you can switch to MKL using::

        conda install "libblas=*=*mkl"

.. warning ::

    If you use the `conda-forge` channe and don't pin BLAS to the MKL version as outlined in the above version,
    but nevertheless have mkl-devel installed during compilation of TeNPy, this can have *crazy* effects on the number
    of threads used: `numpy` will call openblas and open up ``$OMP_NUM_THREADS - 1`` new threads, 
    while MKL called from tenpy will open another ``$MKL_NUM_THREADS - 1`` threads, making it very hard to control the
    number of threads used!

Moreover, it is actually recommended to create a separate environment. 
To create a conda environment with the name `tenpy`, where the TeNPy package (called `physics-tenpy`) is installed::

    conda create --name tenpy --channel=conda-forge physics-tenpy

In that case, you need to activate the environment each time you want to use the package with::

    conda activate tenpy

The big advantage of this approach is that it allows multiple version of software to be installed in parallel, 
e.g., if one of your projects requires python>=3.8 and another one requires an old library which doesn't support that.
Further info can be found in the `conda documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.
