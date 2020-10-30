Installation instructions
=========================

With the [conda]_ package manager you can install python with::

    conda install --channel=conda-forge physics-tenpy

More details and tricks in :doc:`/install/conda`.

If you don't have conda, but you have [pip]_, you can::

    pip install physics-tenpy

More details for this method can be found in :doc:`/install/pip`.

We also have a bunch of optional :doc:`/install/extra`, which you don't have to install to use TeNPy, but you might want to.

The method with the minimal requirements is to just download the source and adjust the `PYTHONPATH`, 
as described in :doc:`/install/from_source`. This is also the recommended way if you plan to modify parts of the source.

.. If you read this file in raw-text, look at the files in the doc/install/ subfolder
   of the repository for the different methods to install TeNPy.

.. toctree::
    
    install/conda
    install/pip
    install/updating
    install/from_source
    install/extra
    install/test
    install/license
