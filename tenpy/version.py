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
from .tools.optimization import have_cython_functions

__all__ = ["version", "short_version", "git_version", "full_version"]

# hard-coded version for people without git...
version = (0, 3, 0)
short_version = 'v' + '.'.join(map(str, version))


def _get_git_version(file=__file__):
    """Get current library version from git"""
    lib_dir = os.path.dirname(os.path.abspath(file))
    try:
        return subprocess.check_output(['git', 'describe', '--always'],
                                       cwd=lib_dir, stderr=subprocess.STDOUT).decode().strip()
    except:
        pass
    return short_version + ' (git unknown)'


# git descritpion of the version
git_version = _get_git_version()

# git version + numpy, scipy versions
full_version = ("tenpy {0!s}, have_cython_functions={1!s}, using \n"
                "python {2!s}\nnumpy {3!s}, scipy {4!s}").format(
                    git_version,
                    have_cython_functions,
                    sys.version,
                    numpy.version.full_version,
                    scipy.version.full_version)
