"""
TenPy - a Python library for Tensor Networks
============================================

TenPy is a library for working with tensor networks,
e.g., matrix product states and - operators
designed to study the physics of strongly correlated quantum systems.
The code is intended to be easy readable for students new to the field,
and yet powerful enough for day-to-day research.

The library provides:

- various predefined models with standard hamiltonians of the field, e.g., the XXZ spin chain
- various algorithms, e.g., DMRG, TEBD, MPO time evolution
- the necessary tools to perform this, including
  an array class to handle charge conservation for abelian symmetries

TenPy was developed over many years by various people.
Over the years, it developed into a powerful library with many tools;
yet, for people new to the library, the big number of tools made it hard
to focus on the essential parts.
This fork tries getting back to a simple version of TenPy.
"""
# This file marks this directory as a python package.

from . import version

# hard-coded tuple of versions
__version__ = version.version
# full version from git description, and numpy/scipy versions
__full_version__ = version.full_version

# TODO: set __all__, include some parts of the library?
# __all__ = ["algorithms", "models", "mps", "tools"]
