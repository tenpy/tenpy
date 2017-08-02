#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

include_dirs = [numpy.get_include()]
libs = []
lib_dirs = []

extensions = [
    Extension("charges", ["charges.pyx"],
              include_dirs=include_dirs, libraries=libs, library_dirs=lib_dirs),
    Extension("np_conserved", ["np_conserved.pyx"],
              include_dirs=include_dirs, libraries=libs, library_dirs=lib_dirs),
]
setup(
    name = "TenPyLight",
    ext_modules = cythonize(extensions),
)
