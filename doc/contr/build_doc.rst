Bulding the documentation
=========================

You can use `Sphinx <https://www.sphinx-doc.org>`_ to generate the full documentation 
in various formats (including HTML or PDF) yourself, as described in the following.

First, I will assume that you downloaded the [TeNPySource]_ repository with::

    git clone --recursive https://github.com/tenpy/tenpy

This includes the [TeNPyNotebooks]_ as a git submodule; you might need to `git submodule update` if it is out of date.

Building the documentation requires a few more packages (including `Sphinx`_).
The recommended way is to create a separate conda environment for it with::

    conda env create -f doc/environment.yml  # make sure to use the file from the doc/ subfolder!
    conda activate tenpydoc

Alternatively, you can use `pip` and ``pip install -r doc/requirements.txt``, but this will not be able to install 
all dependencies: some packages like `Graphviz <https://graphviz.org/>`_ are not available from pip alone.

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
