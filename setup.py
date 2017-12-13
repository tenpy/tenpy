#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

include_dirs = [numpy.get_include()]
libs = []
lib_dirs = []

modules = ['charges', 'np_conserved']
extensions = [
    Extension(
        "tenpy.linalg.{m!s}".format(m=m),
        ["tenpy/linalg/{m!s}.pyx".format(m=m)],
        include_dirs=include_dirs,
        libraries=libs,
        library_dirs=lib_dirs) for m in modules
]

# compiler_directives
comp_direct = {
    'language_level': 3,  # use python 3
    'profile': True,  # Change this if you want to profile code.
}
comp_direct['linetrace'] = comp_direct['profile']

# compile time flags (#DEF ...)
comp_flags = {}

ext_modules = cythonize(extensions, compiler_directives=comp_direct, compile_time_env=comp_flags)
setup(name="TenPyLight", ext_modules=ext_modules)
