Bulding the documentation
=========================

You can use `Sphinx <https://www.sphinx-doc.org>`_ to generate the full documentation 
in various formats (including HTML or PDF) yourself, as described in the following.
First of all, I will assume that you installed TeNPy from source, see :doc:`/install/from_source`.

In addition to the packages required by TeNPy, you need the packages specified in ``/doc/requirements.txt``.
Assuming that you are in the top folder of the git repository and have `pip`, you can install `Sphinx`_ and further
requirements with::

    pip install -r doc/requirements.txt

.. note ::

    Plotting the inheritance graphs also requires `Graphviz <https://graphviz.org/>`_.
    If you have `conda`, installing it requires just ``conda install graphviz``.

Afterwards, simply go to the folder ``doc/`` and run the following command::

    make html

This should generate the html documentation in the folder `doc/sphinx_build/html`.
Open this folder (or to be precise: the file `index.html` in it) in your webbroser
and enjoy this and other documentation beautifully rendered, with cross links, math formulas
and even a search function.
Other output formats are available as other make targets, e.g., ``make latexpdf``.

.. note ::

   Building the documentation with sphinx requires loading the TeNPy modules.
   The `conf.py` adjusts the python `sys.path` to include the `/tenpy` folder from root directory of the git repository.
   It will not use the cython-compiled parts.
