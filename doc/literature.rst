Literature and References
=========================

This is a (by far non-exhaustive) list of some references for the various ideas behind the code.
They can be cited like this:

- [TeNPyNotes]_ for TeNPy/software related sources
- :cite:`white1992` (lowercase first-author + year) for entries from `literature.bib`.

.. comment
    When you add something, please also add a reference to it, i.e., give a short comment in the top of the subsection.

TeNPy related sources
---------------------
[TeNPyNotes]_ are lecture notes, meant as an introduction to tensor networks (focusing on MPS), and introduced TeNPy to
the scientific community by giving examples how to call the algorithms in TeNPy.
[TeNPySource]_ is the location of the source code, and the place where you can report bugs.
We have split example notebooks into [TeNPyNotebooks]_ to keep the git history of the original repository clean.
[TeNPyDoc]_ is where the documentation is hosted online.
[TeNPyForum]_ is the place where you can ask questions and look for help when you are stuck with implementing something.

.. [TeNPyNotes]
    "Efficient numerical simulations with Tensor Networks: Tensor Network Python (TeNPy)"
    J. Hauschild, F. Pollmann, SciPost Phys. Lect. Notes 5 (2018), :arxiv:`1805.00055`, :doi:`10.21468/SciPostPhysLectNotes.5`
    also below as :cite:`hauschild2018a`.
.. [TeNPySource]
    https://github.com/tenpy/tenpy
.. [TeNPyNotebooks]
    Collection of example [jupyter]_ notebooks using TeNPy: https://github.com/tenpy/tenpy_notebooks
.. [TeNPyDoc]
    Online documentation, https://tenpy.readthedocs.io/
.. [TeNPyForum]
    Community forum for discussions, FAQ and announcements, https://tenpy.johannes-hauschild.de
.. [TeNPyProjectTemplate]
    Template git repository for custom projects with a simplified setup for running many simulations on a cluster,
    https://github.com/tenpy/project_template

Software-related
----------------
The following links are not physics-related, but are good to know if you want to work with TeNPy (or more generally Python).

.. [git]
    "git version control system", https://git-scm.com
    A software which we use to keep track of changes in the source code.

.. [conda]
    "conda package manger", https://docs.conda.io/en/latest/
    A package and environment management system that allows to easily install (multiple version of) various software,
    and in particular python packages like TeNPy.

.. [pip]
    "pip - the Python Package installer", https://pip.pypa.io/en/stable/
    Traditional way to handle installed python packages with ``pip install ...`` and ``pip uninstall ...`` on the command line.

.. [matplotlib]
    "Matplotlib", https://matplotlib.org/
    A Python 2D plotting library. Some TeNPy functions expect :class:`matplotlib.axes.Axes` as arguments to plot into.

.. [HDF5]
    "Hierarchical Data Format 5 (R)", https://portal.hdfgroup.org/display/HDF5/HDF5
    A file format and library for saving data (including metadata).
    We use it through the python interface of the `h5py <https://docs.h5py.org/en/stable/>`_ library, 
    see :doc:`/intro/input_output`.

.. [yaml]
    "YAML Ain't Markup Language", https://yaml.org
    A human-readable file format for configuration files.
    TeNpy (optionally) uses it through `pyyaml <https://pyyaml.org/>`_ for reading in simulation parameters, and in some
    places in the documentation to keep things more readable.

.. [jupyter]
    Jupyter notebooks, https://jupyter.org/
    An amazing interface for (python) notebooks which can contain both source code, text and outputs in a single file.
    They provide a good way to get started with python, we use them for examples.

General reading
---------------
:cite:`schollwoeck2011` is an extensive introduction to MPS, DMRG and TEBD with lots of details on the implementations, and a classic read, although a bit lengthy.
Our [TeNPyNotes]_ are a shorter summary of the important concepts, similar as :cite:`orus2014`.
:cite:`paeckel2019` is a very good, recent review focusing on time evolution with MPS.
The lecture notes of :cite:`eisert2013` explain the area law as motivation for tensor networks very well.
PEPS are for example reviewed in :cite:`verstraete2008`, :cite:`eisert2013` and :cite:`orus2014`.
:cite:`cirac2020` is a recent, broad review of MPS and MPS with a focus on analytical theorems.
:cite:`stoudenmire2012` reviews the use of DMRG for 2D systems.
:cite:`cirac2009` discusses the different groups of tensor network states.
:cite:`vanderstraeten2019` is a great review on tangent space methods for infinite, uniform MPS.


Algorithm developments
----------------------
:cite:`white1992,white1993` is the invention of DMRG, which started everything.
:cite:`vidal2004` introduced TEBD.
:cite:`white2005` and :cite:`hubig2015` solved problems for single-site DMRG.
:cite:`mcculloch2008` was a huge step forward to solve convergence problems for infinite DMRG.
:cite:`singh2010,singh2011` explain how to incorporate Symmetries.
:cite:`haegeman2011` introduced TDVP, again explained more accessible in :cite:`haegeman2016`.
:cite:`zaletel2015` is another standard method for time-evolution with long-range Hamiltonians.
:cite:`karrasch2013` gives some tricks to do finite-temperature simulations (DMRG), which is a bit extended in :cite:`hauschild2018a`.
:cite:`vidal2007` introduced MERA.
The scaling :math:`S=c/6 log(\chi)` at a 1D critical point is explained in :cite:`pollmann2009`.
:cite:`vanderstraeten2019` gives a very good introductin to infinite, uniform MPS.


References
----------

.. bibliography:: literature.bib
    :style: custom1
    :all:
