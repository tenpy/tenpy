Installation from PyPi with pip
===============================

If you have the conda package manager from `anaconda <https://www.anaconda.com/distribution>`_, you can just download the 
`environment.yml <https://raw.githubusercontent.com/tenpy/tenpy/master/environment.yml>`_ file out of the repository
and create a new environment for tenpy with all the required packages::

    conda env create -f environment.yml
    conda activate tenpy

Further information on conda environments can be found in the `conda documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_. Note that this will also install `pip <https://pip.pypa.io/en/stable/>`_. 

Alternatively, if you only have `pip` (and not `conda`), install the
required packages with the following command (after downloading the 
`requirements.txt <https://raw.githubusercontent.com/tenpy/tenpy/master/requirements.txt>`_ file from the repository)::

    pip install -r requirements.txt

.. note ::
    
    Make sure that the `pip` you call corresponds to the python version
    you want to use. (e.g. by using ``python -m pip`` instead of a simple ``pip``
    Also, you might need to use the argument ``--user`` to install the packages to your home directory, 
    if you don't have ``sudo`` rights. (Using ``--user`` with conda's ``pip`` is discouraged, though.)

.. warning ::
    
    It might just be a temporary problem, but I found that the `pip` version of numpy is incompatible with 
    the python distribution of anaconda. 
    If you have installed the intelpython or anaconda distribution, use the `conda` packagemanager instead of `pip` for updating the packages whenever possible!


After that, you can **install the latest *stable* TeNPy package** (without downloading the source) from 
`PyPi <https://pypi.org>`_ with::

    pip install physics-tenpy # note the different package name - 'tenpy' was taken!

.. note ::
    
    If the installation fails, don't give up yet. In the minimal version, tenpy requires only pure Python with
    somewhat up-to-date NumPy and SciPy. See :doc:`from_source`.

To get the latest development version from the github master branch, you can use::

    pip install git+git://github.com/tenpy/tenpy.git

Finally, if you downloaded the source and want to **modify parts of the source**, you should install tenpy in
development version with ``--editable``::

    cd $HOME/tenpy # after downloading the source, got to the repository
    pip install --editable .

In all cases, you can uninstall tenpy with::

    pip uninstall physics-tenpy  # note the longer name!


