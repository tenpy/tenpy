Checklist for a new release
===========================

Follow this checklist when creating a new release, i.e. updating the version number.

1. Make sure that tenpy can be compiled (``bash ./compile.sh``) and installed (``pip install .``).
   Make sure that *all* tests run successfully.
   A convenient way to check this is to re-run the github.com actions on the latest commit.

2. Update the changelog and release notes.

3. In the files ``setup.py`` and ``tenpy/version.py``, update the release number and set released=True.

4. Create the version commit and a tag::
    
     git commit -m "VERSION 0.42.1"
     git tag -s "v0.42.1"
    
   Change the version number appropriately. 
   You should GPG sign the commit.

5. Reset the released=False flag in ``setup.py`` and ``tenpy/version.py``.
   Commit and push these changes::
   
     git add setup.py tenpy/version.py
     git commit -m "reset released=False"
     git push
     git push origin v0.42.1 # also push the tag

6. Pushing a tag should cause a github action to publish to test.pypi.org.
   It executes ``.github/workflows/publish-test-pypi.yml``.
   Double check that everything looks ok there.
   Perform a test installation with ``pip install -i https://test.pypi.org/simple/ physics-tenpy``
   and check it by running the tests.

7. Create a release with release notes on github.
   The release triggers the github action for uploading the package to PyPI.
   It executes ``.github/workflows/publish-pypi.yml``.

8. Wait for conda-forge bot to create a pull request in the `feedstock repo <https://github.com/conda-forge/physics-tenpy-feedstock>`_
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

The twine command will prompt for a test.pypi.ort credentials.
Double check the project page on test.pypi.
Finally, we upload to the live PyPI::

   python -m twine upload dist/*
