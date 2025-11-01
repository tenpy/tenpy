Examples
========

Toy codes
---------

These "toy codes" are meant to give you a flavor of the different algorithms,
while keeping the codes as readable and simple as possible.
They can be found in a separate [TeNPyToycodes]_ repository https://github.com/tenpy/tenpy_notebooks in the folder ``tenpy_toycodes/``.
The only requirements to run them are Python 3, Numpy, and Scipy, (and jupyter for running the notebooks).
For reference, we include them here as examples.

.. toctree::
    :glob:

    toycode_stubs/*

It has the following tutorial notebooks.

.. toctree::
    :glob:
    :maxdepth: 1

    toycodes/*

Python scripts
--------------

These example scripts illustrate the very basic interface for calling TeNPy.
They are included in the [TeNPySource]_ repository in the folder ``examples/``,
we include them here in the documentation for reference.
You need to install TeNPy to call them (see :doc:`/INSTALL`), but you can copy them anywhere before execution.
(Some scripts include other files from the same folder, though; copy those as well.)

.. toctree::
    :glob:

    examples/*

A bit more elaborate examples from the subfolders in ``examples/*`` are included in this list:

.. toctree::
    :glob:

    examples/advanced/*
    examples/chern_insulators/*


YAML config examples
--------------------
We also have abunch of example config files that can be used for standard simulations, see :doc:`/intro/simulations`.

.. toctree::
    :glob:

    examples/yaml/*


Jupyter Notebooks
-----------------

This is a collection of [jupyter]_ notebooks from the [TeNPyNotebooks]_ repository.
You need to install TeNPy to execute them (see :doc:`/INSTALL`), but you can copy them anywhere before execution.
Note that some of them might take a while to run, as they contain more extensive examples.

.. toctree::
    :glob:
    :maxdepth: 1

    notebooks/*
