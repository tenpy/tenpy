Checklist for a new release
===========================

Follow this checklist when creating a new release, i.e. updating the version number.

#. Check deprecated features, e.g. full-text search for ``.. deprecated ::``, ``DeprecationWarning``
   and ``FutureWarning``. They should mention at which release the deprecated implementations will
   be removed. If they are due, remove them.

#. Make sure that tenpy can be compiled (``bash ./compile.sh``) and installed (``pip install .``).
   Make sure that *all* tests run successfully.
   A convenient way to check this is to re-run the github.com actions on the latest commit.

#. Update the changelog and release notes.

   - Make sure all important changes are listed in ``doc/changelog/latest.rst``.
   - Rename that file to the next version number.
   - Add it to the toctree in ``doc/releases.rst``.
   - Create a new ``doc/changelog/latest.rst`` from the template.

#. Update ``tenpy/version.py``
  
   - Update the release number.
   - set ``released=True``.

#. Create the version commit and a tag::
    
     git commit -m "VERSION 0.42.1"
     git tag -s "v0.42.1"
    
   Change the version number appropriately.
   You should GPG sign the commit.

#. Reset the ``released=False`` flag in ``tenpy/version.py``.
   Commit and push these changes::
   
     git add tenpy/version.py
     git commit -m "reset released=False"
     git push
     git push origin v0.42.1  # also push the tag

#. Pushing a tag should cause a github action to publish to test.pypi.org.
   It executes ``.github/workflows/publish.yml``.
   Double check that everything looks ok there.

#. Perform a test installation.

   This is best done in a dedicated fresh environment.
   Since the dependencies (numpy, scipy) are not available on TestPyPI, we need to pre-install them.
   On live PyPI we could omit them here and ``pip install physics-tenpy`` would install them for us::
   
     conda create -n tenpytest python=3.12 pip numpy scipy h5py pyyaml pytest ipython
     conda activate tenpytest
     pip install -i https://test.pypi.org/simple/ physics-tenpy==0.42.1
   
   Make sure your working directory is *not* the root folder of a tenpy repo.
   That would cause the tenpy version from the repo to be in the path and probably take
   precedence over the tenpy version that we just installed.
   In e.g. an ``ipython`` console, check that tenpy can be imported without errors and warnings and
   check that you are importing the correct version::
   
     In [1] import tenpy as tp
     In [2]: print(tp.__file__)  # make sure the following path contains 'site-packages'.
     /Users/jakobunfried/anaconda3/envs/tenpytest/lib/python3.12/site-packages/tenpy/__init__.py
     In [3]: tp.show_config()  # check the version number
     tenpy 0.11.0.dev0+unknown (compiled without HAVE_MKL),
     git revision unknown using
     python 3.12.1 | packaged by Anaconda, Inc. | (main, Jan 19 2024, 09:45:58) [Clang 14.0.6 ]
     numpy 1.26.3, scipy 1.11.4
   
   Run the test suite. Again, ensure that you are not in the root directory of the repo::

      pytest path/to/local/repo/tests

#. Create a release with release notes on github.
   The release triggers the github action for uploading the package to PyPI.
   It executes ``.github/workflows/publish.yml``.

#. Wait for conda-forge bot to create a pull request in the `feedstock repo <https://github.com/conda-forge/physics-tenpy-feedstock>`_
   and merge it.


How to publish on PyPI
~~~~~~~~~~~~~~~~~~~~~~

Usually, PyPI publish should happen via github actions.
This is how you would do it manually

Assume you working directory is the project root, i.e. the folder which contains ``setup.py``.
Make sure you are running this in a recent python::

   python -m pip install --upgrade pip
   python -m pip install --upgrade setuptools wheel build twine

Then we build the package::

   python -m build .

This will write the build to ``dist/``.
We can check that the project page on PyPI will render correctly with::

   python -m twine check --strict dist/*

Then we can upload to test.pypi.org via::

   python -m twine upload -r testpypi dist/*

The twine command will prompt for test.pypi.org credentials.
Double check the project page on test.pypi.
Finally, we upload to the live PyPI::

   python -m twine upload dist/*
