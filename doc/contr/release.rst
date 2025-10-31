Checklist for a new release
===========================

Follow this checklist when creating a new release, i.e. updating the version number.

#. Make a new branch. Jakob usually uses the ``release`` branch.
   This makes it so that you can rewrite history as needed, if publishing fails.
   See troubleshooting section below if you need that.
   It is probably a good idea to open a PR, to keep track.

#. Check deprecated features, e.g. full-text search for ``.. deprecated ::``, ``DeprecationWarning``
   and ``FutureWarning``. They should mention at which release the deprecated implementations will
   be removed. If they are due, remove them.

#. If you use git in your terminal, set a variable with the new version number.
   Then you can copy paste the git commands below::

     version="0.42.1"

#. Update the changelog and release notes.

   a. The "latest" contributions should have added entries in ``doc/changelog/latest/``.
      Starting a docbuild (``make html`` in the doc folder) pastes all of these entries into
      ``doc/changelog/_latest.rst``, even if aborted after a few seconds. This is a good starting
      point for the new release notes.
   #. Create a new changelog file by duplicating the ``doc/changelog/template.txt``.
      Name it ``vX.Y.Z.rst`` according to the new version number.
      Include all notes from the latest changelog, add a summary.
   #. Clean up the latest entries (add links, consistent past tense, ...).
      We usually do not thank or honor contributors here.
   #. Adjust the title and date
   #. Check the commit list on the main branch and make sure all important changes are covered.
   #. Add the new file to the toctree in ``doc/releases.rst``.
   #. Delete the outdated files in ``doc/changelog/latest``.
   #. Make sure to *not* delete the ``doc/changelog/latest/README`` though!
   #. Make sure to *not* accidentally commit changes to ``doc/changelog/_latest.rst``.
      It should be empty except for the header.

#. Update ``tenpy/version.py``

   a. Update the release number.
   #. Set ``released=True``.

#. Create the version commit and a tag::

     git commit -m "VERSION $version"
     git tag -s "v$version" -m "VERSION $version"

   You should GPG sign the commit.

#. Reset the ``released=False`` flag in ``tenpy/version.py``.
   Commit and push these changes::

     git add tenpy/version.py
     git commit -m "reset released=False"
     git push
     git push origin "v$version"  # also push the tag

#. Pushing a tag should cause a github action to publish to test.pypi.org.
   It executes ``.github/workflows/publish.yml``.
   It runs tests on all built wheels, so it takes a while (1 to 2 hours currently).
   Check the `project page <https://https://test.pypi.org/project/physics-tenpy>`_ and check
   the correct version number and that the page looks ok.

#. Merge the release branch into main.
   You should always *merge*, never rebase or squash this branch.
   This is to guarantee that the version commit retains the exact same state of all files.

#. Create a release with release notes on github, from the tag.
   You can use the same text body as in the header section of the changelog.
   The release triggers the github action for uploading the package to PyPI.
   It executes ``.github/workflows/publish.yml``.

#. Wait for conda-forge bot to create a pull request in the `feedstock repo <https://github.com/conda-forge/physics-tenpy-feedstock>`_
   and merge it.


Troubleshooting
~~~~~~~~~~~~~~~
The following is a loose collection of tips and pointers, in case something goes wrong during the
release steps outlined above.

- If you notice a problem quickly, cancel the publishing action on github.
  TestPyPI, as well as the live PyPI can not be modified once uploaded and accepts only one upload
  per unique version number. If you do not do this in time, subsequent uploads
  (with the same version number) will be rejected.

- If you accidentally published a bad version to live PyPI, consider their FAQ on how to communicate
  that and make a new version next (start over). If not, we can still fix things

- If you made the version commits in a separate branch, as suggested above,
  we can rewrite history safely (without messing up the main branch).
  Fix the problems, then interactively rebase to move the fix before the version commit.
  Force-push these changes to the remote release branch.
  Now we adjust the tag::

    git tag -d "v$version"                  # delete the old tag locally
    git push origin :refs/tags/"v$version"  # delete the old tag remotely
    git tag "v$version" <commitId>          # make a new tag locally
    git push origin "v$version"             # push the new local tag to the remote

- If there was already an upload to TestPyPI, the upload action triggered by pushing the tag will
  fail. Make sure the steps before uploading (building the wheels and checking them locally) run
  through without error.
  If you are reasonably sure that everything is ok still, you can ignore this failed action.
  If you want to check again, create a dummy branch, change the version number to something
  we will not use for actual releases, e.g. ``0.42.1-test`` and manually publish to TestPyPI

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
