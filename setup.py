# Copyright 2018-2020 TeNPy Developers, GNU GPLv3
from setuptools import setup, find_packages
from setuptools import Extension
import numpy
import sys
import os
import subprocess

if not sys.version_info >= (3, 5):
    print("ERROR: old python version, the script got called by\n" + sys.version)
    sys.exit(1)

# hardcode version for people without git

MAJOR = 0
MINOR = 7
MICRO = 2
RELEASED = True
VERSION = '{0:d}.{1:d}.{2:d}'.format(MAJOR, MINOR, MICRO)

#  Before updating a version, make sure that *all* tests run successfully!
#  To update to a new release:
#      # update changelog and release notes
#      # update the version in this module and in tenpy/version.py, set RELEASED=True
#      git commit -m "VERSION 0.1.2"
#      git tag -s "v0.1.2"  # (sign: requires GPG key)
#      bash ./compile.sh
#      pytest -m "not slow"  # run at least a quick test!
#      # python setup.py sdist  # create source package for PyPI, done by github action
#      # reset RELEASED = False in this module and tenpy/version.py, copy changelog template.
#      git commit -m "reset released=False"
#      git push
#      git push origin v0.1.2 # also push the tag
#      create release with release-notes on github
#      # python -m twine upload dist/physics-tenpy-0.1.2.tar.gz # done by github action
# or   # python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/physics-tenpy-0.1.2.tar.gz
#      # wait for conda-forge bot to create a pull request with the new version and merge it


def get_git_revision():
    """Get revision hash from git."""
    if not os.path.exists('.git'):
        rev = "unknown"
    else:
        try:
            rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                          stderr=subprocess.STDOUT).decode().strip()
        except:
            rev = "unknown"
    return rev


def get_git_description():
    """Get number of commits since last git tag.

    If unknown, return 0
    """
    if not os.path.exists('.git'):
        return 0
    try:
        descr = subprocess.check_output(['git', 'describe', '--tags', '--long'],
                                        stderr=subprocess.STDOUT).decode().strip()
    except:
        return 0
    return int(descr.split('-')[1])


def get_version_info():
    full_version = VERSION
    git_rev = get_git_revision()
    if not RELEASED:
        full_version += '.dev{0:d}+{1!s}'.format(get_git_description(), git_rev[:7])
    return full_version, git_rev


def write_version_py(full_version, git_rev, filename='tenpy/_version.py'):
    """Write the version during compilation to disc."""
    content = """\
# THIS FILE IS GENERATED FROM setup.py
# thus, it contains the version during compilation
# Copyright 2018-2020 TeNPy Developers, GNU GPLv3
version = '{version!s}'
short_version = 'v' + version
released = {released!s}
full_version = '{full_version!s}'
git_revision = '{git_rev!s}'
numpy_version = '{numpy_ver!s}'
cython_version = '{cython_ver!s}'
"""
    try:
        import Cython
        cython_ver = Cython.__version__
    except:
        cython_ver = "(not available)"
    content = content.format(version=VERSION,
                             full_version=full_version,
                             released=RELEASED,
                             git_rev=git_rev,
                             numpy_ver=numpy.version.full_version,
                             cython_ver=cython_ver)
    with open(filename, 'w') as f:
        f.write(content)
    # done


def read_requ_file(filename):
    with open(filename, 'r') as f:
        requ = f.readlines()
    return [l.strip() for l in requ if l.strip()]


def setup_cython_extension():
    try:
        from Cython.Build import cythonize
    except:
        return []
    # see tenpy/tools/optimization.py for details on "TENPY_OPTIMIZE"
    TENPY_OPTIMIZE = int(os.getenv('TENPY_OPTIMIZE', 1))
    include_dirs = [numpy.get_include()]
    libs = []
    lib_dirs = []

    extensions = [
        Extension("*", ["tenpy/linalg/*.pyx"],
                  include_dirs=include_dirs,
                  libraries=libs,
                  library_dirs=lib_dirs,
                  language='c++')
    ]

    comp_direct = {  # compiler_directives
        'language_level': 3,  # use python 3
        'embedsignature': True,  # write function signature in doc-strings
    }
    if TENPY_OPTIMIZE > 1:
        comp_direct['initializedcheck'] = False
        comp_direct['boundscheck'] = False
    if TENPY_OPTIMIZE < 1:
        comp_direct['profile'] = True
        comp_direct['linetrace'] = True

    # compile time flags (#DEF ...)
    comp_flags = {'TENPY_OPTIMIZE': TENPY_OPTIMIZE}

    ext_modules = cythonize(extensions,
                            compiler_directives=comp_direct,
                            compile_time_env=comp_flags)
    return ext_modules


def setup_package():
    # change directory to root path of the repository
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(src_path)

    # avoid warning from tenpy.tools.optimization.use_cython when we didn't compile yet.
    os.environ.setdefault("TENPY_NO_CYTHON", "true")

    full_version, git_rev = get_version_info()
    write_version_py(full_version, git_rev)

    ext_modules = setup_cython_extension()

    extras_require = {
        'extra': ['bottleneck', 'yapf==0.28.0', 'docformatter==1.3.1'],
        'io': ['h5py', 'pyyaml'],
        'plot': ['matplotlib>=2.0'],
        'test': ['pytest'],
    }
    extras_require['all'] = [r for requ in extras_require.values() for r in requ]

    setup(version=full_version,
          ext_modules=ext_modules,
          install_requires=read_requ_file('requirements.txt'),
          extras_require=extras_require)


if __name__ == "__main__":
    setup_package()
