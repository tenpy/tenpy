# Copyright 2018 TeNPy Developers
from setuptools import setup, find_packages
from setuptools import Extension
from Cython.Build import cythonize
import numpy
import sys
import os
import subprocess


if not sys.version_info >= (3, 5):
    print("ERROR: old python version, the script got called by\n" + sys.version)
    sys.exit(1)


# hardcode version for people without git

MAJOR = 0
MINOR = 3
MICRO = 0
RELEASED = False
VERSION = '{0:d}.{1:d}.{2:d}'.format(MAJOR, MINOR, MICRO)

#  Before updating a version, make sure that *all* tests run successfully!
#  To update:
#      # update the version in this module and in tenpy/version.py, set RELEASED=True
#      git commit -m "VERSION 0.1.2"
#      git tag -a "v0.1.2"
#      # run tests!
#      # reset RELEASED = False in this module"
#      git commit -m "reset RELEASED to False"
#      git push
#      git push origin v0.1.2 # also push the tag


def get_git_revision():
    """get revision hash from git"""
    if not os.path.exists('.git'):
        rev = "unknown"
    else:
        try:
            rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                          stderr=subprocess.STDOUT).decode().strip()
            descr = subprocess.check_output(['git', 'describe', '--always'],
                                            stderr=subprocess.STDOUT).decode().strip()
        except:
            rev = "unknown"
    if rev != "unknown":
        if not descr.startswith("v" + VERSION):
            raise ValueError("Hard-coded version doesn't fit to git version")
    return rev


def get_version_info():
    full_version = VERSION
    git_rev = get_git_revision()
    if not RELEASED:
        full_version += '.dev0+' + git_rev[:7]
    return full_version, git_rev


def write_version_py(full_version, git_rev, filename='tenpy/_version.py'):
    """write the version during compilation to disc"""
    content = """
# THIS FILE IS GENERATED FROM setup.py
# thus, it contains the version during compilation
version = '{version!s}'
short_version = 'v' + version
released = {released!s}
full_version = '{full_version!s}'
git_revision = '{git_rev!s}'
numpy_version = '{numpy_ver!s}'
cython_version = '{cython_ver!s}'
"""
    import Cython
    content = content.format(version=VERSION,
                             full_version=full_version,
                             released=RELEASED,
                             git_rev=git_rev,
                             numpy_ver=numpy.version.full_version,
                             cython_ver=Cython.__version__)
    with open(filename, 'w') as f:
        f.write(content)
    # done


def setup_cython_extension():
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

    full_version, git_rev = get_version_info()
    write_version_py(full_version, git_rev)

    ext_modules = setup_cython_extension()

    setup(version=full_version, ext_modules=ext_modules)

if __name__ == "__main__":
    setup_package()
