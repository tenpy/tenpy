"""Access to version of this library.

Before updating a version, make sure nosetest runs *all* tests in test successfully!
To update a version, change `version` in this module and create a git tag ::

    git tag -a "v1.0.2"

Make shure to push your tag into the shared git repo with `git push origin [tagname]`.
"""

import numpy
import scipy
import subprocess
import os

# hard-coded version
version = (0, 0, 0)


def git_version():
    """Get current library version from git"""
    lib_dir = os.path.dirname(__file__)
    try:
        return subprocess.check_output(['git', 'describe'], cwd=lib_dir).strip()
    except:
        pass
    return 'v' + '.'.join(map(str, version)) + ' git=?'

# git descritpion of the version
git_version = git_version()

# git version + numpy, scipy versions
full_version = "{0:s} using numpy {1:s}, scipy {2:s}".format(
    git_version, numpy.version.full_version, scipy.version.full_version)
