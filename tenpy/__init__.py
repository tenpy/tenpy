"""TenPyLight - a light version of TenPy

Over the years, TenPy has developed into a powerful library with many tools;
yet it is hard to get used to it when new to the topic.
The original intention was to have a light-weight (and yet)
easy to use python library for tensor networks (most notably MPS & MPO).

This fork of TenPy tries getting back to a simple version of TenPy.
"""
# This file marks this directory as a python package.

from . import version

__version__ = version.version
__full_version__ = version.full_version

# TODO: set __all__, include some parts of the library?
# __all__ = ["algorithms", "models", "mps", "tools"]
