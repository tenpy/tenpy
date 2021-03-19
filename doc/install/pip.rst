Installation from PyPi with pip
===============================

Preparation: install requirements
---------------------------------

If you have the [conda]_ package manager from `anaconda <https://www.anaconda.com/distribution>`_, you can just download the 
`environment.yml <https://raw.githubusercontent.com/tenpy/tenpy/main/environment.yml>`_ file (using the `conda-forge`
channel, or the `environment_other.yml <https://raw.githubusercontent.com/tenpy/tenpy/main/environment_other.yml>`_ for all other channels) out of the repository
and create a new environment (called ``tenpy``, if you don't speficy another name) for TeNPy with all the required packages::

    conda env create -f environment.yml
    conda activate tenpy

Further information on conda environments can be found in the `conda documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_. Note that installing conda also installs a version of [pip]_.

Alternatively, if you only have [pip]_ (and not [conda]_), install the
required packages with the following command (after downloading the 
`requirements.txt <https://raw.githubusercontent.com/tenpy/tenpy/main/requirements.txt>`_ file from the repository)::

    pip install -r requirements.txt

.. note ::
    
    Make sure that the ``pip`` you call corresponds to the python version you want to use.
    (One way to ensure this is to use ``python -m pip`` instead of a simple ``pip``.)
    Also, you might need to use the argument ``--user`` to install the packages to your home directory, 
    if you don't have ``sudo`` rights. (Using ``--user`` with conda's ``pip`` is discouraged, though.)

.. warning ::
    
    It might just be a temporary problem, but I found that the `pip` version of numpy is incompatible with 
    the python distribution of anaconda. 
    If you have installed the intelpython or anaconda distribution, use the `conda` packagemanager instead of `pip` for updating the packages whenever possible!

Installing the latest stable TeNPy package
------------------------------------------

Now we are ready to install TeNPy. It should be as easy as (note the different package name - 'tenpy' was taken!) ::

    pip install physics-tenpy

.. note ::
    
    If the installation fails, don't give up yet. In the minimal version, tenpy requires only pure Python with
    somewhat up-to-date NumPy and SciPy. See :doc:`from_source`.

Installation of the latest version from Github
----------------------------------------------

To get the latest development version from the github main branch, you can use::

    pip install git+git://github.com/tenpy/tenpy.git

This should already have the lastest features described in :doc:`/changelog/latest`.
Disclaimer: this might sometimes be broken, although we do our best to keep to keep it stable as well.

Installation from the downloaded source folder
----------------------------------------------

Finally, if you downloaded the source and want to **modify parts of the source**, 
You can also install TeNPy with in
development version with ``--editable``::

    cd $HOME/tenpy # after downloading the source, got to the repository
    pip install --editable .

Uninstalling a pip-installed version
------------------------------------

In all of the above cases, you can uninstall tenpy with::

    pip uninstall physics-tenpy
