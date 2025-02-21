TeNPy: Tensor Network Python
----------------------------

.. image:: https://img.shields.io/github/last-commit/tenpy/tenpy
    :alt: GitHub last commit
    :target: https://github.com/tenpy/tenpy
.. image:: https://readthedocs.org/projects/tenpy/badge/?version=latest
    :alt: Documentation
    :target: https://tenpy.readthedocs.io/en/latest/
.. image:: https://github.com/tenpy/tenpy/actions/workflows/pytest.yml/badge.svg?branch=main
    :alt: Build
    :target: https://github.com/tenpy/tenpy/actions/
.. image:: https://img.shields.io/github/issues/tenpy/tenpy
    :alt: GitHub issues
    :target: https://github.com/tenpy/tenpy/issues
.. image:: https://img.shields.io/conda/vn/conda-forge/physics-tenpy?color=blue
    :alt: conda
    :target: https://anaconda.org/conda-forge/physics-tenpy
.. image:: https://img.shields.io/pypi/v/physics-tenpy?color=blue
    :alt: PyPi
    :target: https://pypi.org/project/physics-tenpy/
.. image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Jakob-Unfried/9e2e197d6a2e6e2c9440b2c0eda04d5c/raw/tenpy_coverage_badge.json
    :alt: Code Coverage
    :target: https://github.com/tenpy/tenpy/actions/workflows/code-coverage.yml


TeNPy (short for 'Tensor Network Python') is a Python library for the simulation of strongly correlated quantum systems with tensor networks.

The philosophy of this library is to get a new balance of a good readability and usability for new-comers, and at the same time powerful algorithms and fast development of new algorithms for experts.
For good readability, we include an extensive documentation next to the code, both in Python doc strings and separately as "user guides", as well as simple example codes and even toy codes, which just demonstrate various algorithms (like TEBD and DMRG) in ~100 lines per file.

How do I get set up?
--------------------
If you have the `conda` package manager, you can install the latest released version of TeNPy with::

    conda install --channel=conda-forge physics-tenpy

Further details and alternative methods can be found the file `doc/INSTALL.rst <https://tenpy.readthedocs.io/en/latest/INSTALL.html>`_.
The latest version of the source code can be obtained from https://github.com/tenpy/tenpy.

How to read the documentation
-----------------------------
The **documentation is available online** at https://tenpy.readthedocs.io/.
The documentation is roughly split in two parts: on one hand the full "reference" containing the documentation of all functions,
classes, methods, etc., and on the other hand the "user guide" containing some `introductions <https://tenpy.readthedocs.io/en/latest/introductions.html>`_ with additional explanations and `examples <https://tenpy.readthedocs.io/en/latest/examples.html>`_.

The documentation is based on Python's docstrings, and some additional ``*.rst`` files located in the folder `doc/` of the repository.
All documentation is formatted as `reStructuredText <http://www.sphinx-doc.org/en/stable/rest.html>`_,
which means it is quite readable in the source plain text, but can also be converted to other formats.
If you like it simple, you can just use interactive python `help()`, Python IDEs of your choice or jupyter notebooks, or just read the source.
Moreover, the documentation gets converted into HTML using `Sphinx <http://www.sphinx-doc.org>`_, and is made available online at https://tenpy.readthedocs.io/.
The big advantages of the (online) HTML documentation are a lot of cross-links between different functions, and even a search function.
If you prefer yet another format, you can try to build the documentation yourself, as described in ``doc/contr/build_doc.rst``.

Help - I looked at the documentation, but I don't understand how ...?
---------------------------------------------------------------------
We have set up a **community forum** at https://tenpy.johannes-hauschild.de/,
where you can post questions and hopefully find answers.
Once you got some experience with TeNPy, you might also be able to contribute to the community and answer some questions yourself ;-)
We also use this forum for official announcements, for example when we release a new version.

I found a bug
-------------
You might want to check the `github issues <https://github.com/tenpy/tenpy/issues>`_, if someone else already reported the same problem.
To report a new bug, just `open a new issue <https://github.com/tenpy/tenpy/issues/new>`_ on github.
If you already know how to fix it, you can just create a pull request :)
If you are not sure whether your problem is a bug or a feature, you can also ask for help in the `TeNPy forum <https://tenpy.johannes-hauschild.de/>`_.

Citing TeNPy
------------
When you use TeNPy for a work published in an academic journal, you can cite `this paper <https://dx.doi.org/10.21468/SciPostPhysCodeb.41>`_  to acknowledge the work put into the development of TeNPy.
(The license of TeNPy does not force you, however.)
For example, you could add the sentence ``"Calculations were performed using the TeNPy Library (version X.X.X)\cite{tenpy2024}."`` in the acknowledgements or in the main text.

The corresponding BibTex Entry would be the following

.. code-block:: text

    @Article{tenpy2024,
        title={{Tensor network Python (TeNPy) version 1}},
        author={Johannes Hauschild and Jakob Unfried and Sajant Anand and Bartholomew Andrews and Marcus Bintz and Umberto Borla and Stefan Divic and Markus Drescher and Jan Geiger and Martin Hefel and Kévin Hémery and Wilhelm Kadow and Jack Kemp and Nico Kirchner and Vincent S. Liu and Gunnar Möller and Daniel Parker and Michael Rader and Anton Romen and Samuel Scalet and Leon Schoonderwoerd and Maximilian Schulz and Tomohiro Soejima and Philipp Thoma and Yantao Wu and Philip Zechmann and Ludwig Zweng and Roger S. K. Mong and Michael P. Zaletel and Frank Pollmann},
        journal={SciPost Phys. Codebases},
        pages={41},
        year={2024},
        publisher={SciPost},
        doi={10.21468/SciPostPhysCodeb.41},
        url={https://scipost.org/10.21468/SciPostPhysCodeb.41},
    }

To keep us motivated, you can also include your work into the list of `papers using TeNPy <https://tenpy.readthedocs.io/en/latest/papers_using_tenpy.html>`_.


Acknowledgment
--------------
This work was funded by the U.S. Department of Energy, Office of Science, Office of Basic Energy Sciences, Materials Sciences and Engineering Division under Contract No. DE-AC02-05- CH11231 through the Scientific Discovery through Advanced Computing (SciDAC) program (KC23DAC Topological and Correlated Matter via Tensor Networks and Quantum Monte Carlo).

License
-------
Sinc TeNPy version 1.0.4, the code is licensed under Apache v2 given in the file ``LICENSE`` of the repository,
in the online documentation readable at https://tenpy.readthedocs.io/en/latest/install/license.html.
