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
   Commit the rest::
   
    git add setup.py tenpy/version.py
    git commit -m "reset released=False"
    git push
    git push origin v0.42.1 # also push the tag

6. Create release with release notes on github.
   The release triggers the github action for uploading the package to PyPI.
   It executes ``.github/workflows/publish-pypi.yml``.

7. Wait for conda-forge bot to create a pull request in the `feedstock repo <https://github.com/conda-forge/physics-tenpy-feedstock>`_
   and merge it.
