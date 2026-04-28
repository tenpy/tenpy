"""Access to version of this library.

The version is provided in the standard python format ``major.minor.revision`` as string. Use
``pkg_resources.parse_version`` before comparing versions.

.. autodata :: version
.. autodata :: released
.. autodata :: short_version
.. autodata :: git_revision
.. autodata :: full_version
.. autodata :: version_summary
"""
# Copyright (C) TeNPy Developers, Apache license

import os
import subprocess
import sys

# hard-coded version for people without git...
#: current release version as a string
version = '2.0.0-alpha'

#: whether this is a released version or modified
released = False

#: same as version, but with 'v' in front
short_version = 'v' + version


def _get_git_revision(cwd=None):
    """Get revision hash from git.

    Parameters
    ----------
    cwd : str | None
        Directory contained in the git repository to be considered.
        ``None`` defaults to the top directory of the used cyten source code.

    Returns
    -------
    revision : str
        Revision hash of the git HEAD, i.e, the last git commit to which git compares everything.

    """
    if cwd is None:
        cwd = os.path.dirname(os.path.abspath(__file__))
    try:
        rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd, stderr=subprocess.STDOUT).decode().strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        # FileNotFound e.g if git is not installed or cwd doesn't exist
        # SubprocessError: git command failed for whatever reason
        rev = 'unknown'
    return rev


def _get_git_description():
    """Get number of commits since last git tag.

    If unknown, return 0
    """
    try:
        descr = (
            subprocess.check_output(
                ['git', 'describe', '--tags', '--long'],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.STDOUT,
            )
            .decode()
            .strip()
        )
    except Exception:
        return 0
    return int(descr.split('-')[1])


#: the hash of the last git commit (if available)
git_revision = _get_git_revision()


def _get_full_version():
    """Obtain version from git."""
    full_version = version
    if not released:
        full_version += f'.dev{_get_git_description():d}+{git_revision[:7]!s}'
    return full_version


#: if not released additional info with part of git revision
full_version = _get_full_version()


def _get_version_summary():
    import numpy
    import scipy

    summary = (
        'cyten {cyten_ver!s},\n'
        'git revision {git_rev!s} using\n'
        'python {python_ver!s}\n'
        'numpy {numpy_ver!s}, scipy {scipy_ver!s}'
    )
    summary = summary.format(
        cyten_ver=full_version,
        git_rev=git_revision,
        python_ver=sys.version,
        numpy_ver=numpy.version.full_version,
        scipy_ver=scipy.version.full_version,
    )
    return summary


#: summary of the cyten, python, numpy and scipy versions used
version_summary = _get_version_summary()
