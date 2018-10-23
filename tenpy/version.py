"""Access to version of this library.

Before updating a version, make sure nosetest runs *all* tests in test successfully!
To update a version, change `version` in this module and create a git tag ::

    git tag -a "v1.0.2"

Make shure to push your tag into the shared git repo with `git push origin [tagname]`.

This module provides the three variables `version`, `git_version` and `full_version`.
`version` is a tuple of three integers (major, minor, revision).
`git_version` is a string including information about the current git commit,
`full_version` includes the versions of the used python, numpy and scipy libraries.
"""
# Copyright 2018 TeNPy Developers

import numpy
import scipy
import sys
import subprocess
import os

__all__ = ["version", "git_version", "full_version"]

# hard-coded version for people without git...
version = (0, 3, 0)


def _get_git_version(file=__file__):
    """Get current library version from git"""
    lib_dir = os.path.dirname(os.path.abspath(file))
    try:
        return subprocess.check_output(['git', 'describe', '--always'],
                                       cwd=lib_dir, stderr=subprocess.STDOUT).decode().strip()
    except:
        pass
    return 'v' + '.'.join(map(str, version)) + ' git unknown'


# git descritpion of the version
git_version = _get_git_version()

# git version + numpy, scipy versions
full_version = "tenpy {0!s} using \npython {3!s}\nnumpy {1!s}, scipy {2!s}".format(
    git_version, numpy.version.full_version, scipy.version.full_version, sys.version)
