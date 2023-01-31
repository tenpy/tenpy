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
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import sys
import subprocess
import os

__all__ = [
    "version", "released", "short_version", "git_revision", "full_version", "version_summary"
]

# hard-coded version for people without git...
#: current release version as a string
version = '0.10.0'

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
        ``None`` defaults to the top directory of the used tenpy source code.

    Returns
    -------
    revision : str
        Revision hash of the git HEAD, i.e, the last git commit to which git compares everything.
    """
    if cwd is None:
        cwd = os.path.dirname(os.path.abspath(__file__))
    try:
        rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                      cwd=cwd,
                                      stderr=subprocess.STDOUT).decode().strip()
    except:
        rev = "unknown"
    return rev


def _get_git_description():
    """Get number of commits since last git tag.

    If unknown, return 0
    """
    try:
        descr = subprocess.check_output(['git', 'describe', '--tags', '--long'],
                                        cwd=os.path.dirname(os.path.abspath(__file__)),
                                        stderr=subprocess.STDOUT).decode().strip()
    except:
        return 0
    return int(descr.split('-')[1])


#: the hash of the last git commit (if available)
git_revision = _get_git_revision()


def _get_full_version():
    """obtain version from git."""
    full_version = version
    if not released:
        full_version += '.dev{0:d}+{1!s}'.format(_get_git_description(), git_revision[:7])
    return full_version


#: if not released additional info with part of git revision
full_version = _get_full_version()


def _get_version_summary():
    from .tools.optimization import have_cython_functions, compiled_with_MKL
    import numpy
    import scipy
    import warnings

    try:
        from . import _version
        if _version.version != version:
            raise ValueError("Version changed since installation/compilation")
        if have_cython_functions:
            cython_info = "compiled"
            if compiled_with_MKL:
                cython_info = cython_info + " with HAVE_MKL"
            else:
                cython_info = cython_info + " without HAVE_MKL"
            if _version.git_revision != "unknown":
                if git_revision != "unknown" and _version.git_revision != git_revision:
                    warnings.warn("TeNPy is compiled from different git "
                                  "version than the current HEAD. Recompile!")
                    cython_info = cython_info + " from git rev. " + _version.git_revision
        else:
            cython_info = "not compiled"
    except ImportError:
        cython_info = "not compiled"
        if have_cython_functions:
            warnings.warn("Compiled, but tenpy/_version.py not available!")

    summary = ("tenpy {tenpy_ver!s} ({cython_info!s}),\n"
               "git revision {git_rev!s} using\n"
               "python {python_ver!s}\n"
               "numpy {numpy_ver!s}, scipy {scipy_ver!s}")
    summary = summary.format(tenpy_ver=full_version,
                             cython_info=cython_info,
                             git_rev=git_revision,
                             python_ver=sys.version,
                             numpy_ver=numpy.version.full_version,
                             scipy_ver=scipy.version.full_version)
    return summary


#: summary of the tenpy, python, numpy and scipy versions used
version_summary = _get_version_summary()
