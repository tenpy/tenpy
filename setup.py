# Copyright 2018 TeNPy Developers
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import sys
import os

if not sys.version_info >= (3, 0):
    print("ERROR: old python version, the script got called by\n" + sys.version)
    sys.exit(1)

# options
# see tenpy/tools/optimization.py for details on "TENPY_OPTIMIZE"
TENPY_OPTIMIZE = int(os.getenv('TENPY_OPTIMIZE', 0))

# setup
include_dirs = [numpy.get_include()]
libs = []
lib_dirs = []

extensions = [
    Extension(
        "*", ["tenpy/linalg/*.pyx"],
        include_dirs=include_dirs,
        libraries=libs,
        library_dirs=lib_dirs)
]

comp_direct = {  # compiler_directives
    'language_level': 3,  # use python 3
    'embedsignature': True,  # write function signature in doc-strings
}
if TENPY_OPTIMIZE > 0:
    comp_direct['initializedcheck'] = False
    comp_direct['boundscheck'] = False
if TENPY_OPTIMIZE < 0:
    comp_direct['profile'] = True
    comp_direct['linetrace'] = True

# compile time flags (#DEF ...)
comp_flags = {'TENPY_OPTIMIZE': TENPY_OPTIMIZE}

ext_modules = cythonize(extensions, compiler_directives=comp_direct, compile_time_env=comp_flags)
setup(name="tenpy", ext_modules=ext_modules, packages=['tenpy'])
