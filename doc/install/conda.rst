Installation with conda from conda-forge
========================================

We provide a package for the [conda]_ package manager in the `conda-forge` channel, so you can install TeNPy as::

    conda install --channel=conda-forge physics-tenpy

Following the recommondation of `conda-forge <https://conda-forge.org/docs/user/introduction.html>`_, you can also make
conda-forge the default channel as follows::

    conda config --add channels conda-forge
    conda config --set channel_priority strict

If you have done this, you don't need to specify the ``--channel=conda-forge`` explicitly.


Moreover, it is actually recommended to create a separate environment. 
To create a conda environment with the name `tenpy`, where the TeNPy package (called `physics-tenpy`) is installed::

    conda create --name tenpy --channel=conda-forge physics-tenpy

In that case, you need to activate the environment each time you want to use the package with::

    conda activate tenpy

The big advantage of this approach is that it allows multiple version of software to be installed in parallel, 
e.g., if one of your projects requires python>=3.8 and another one requires an old library which doesn't support that.
Further info can be found in the `conda documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.
