#!/usr/bin/env python
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

version = 'v0.0.0'


def git_version():
    """Get current library version from git"""
    lib_dir = os.path.dirname(__file__)
    try:
        return subprocess.check_output(['git', 'describe'], cwd=lib_dir).strip()
    except:
        pass
    return version

git_version = git_version()
full_version = ''.join([git_version, ', using',
                        'numpy ', numpy.version.full_version,
                        ', scipy ', scipy.version.fullversion])

__version__ = full_version
