"""Check for consistencies."""
# Copyright 2019-2021 TeNPy Developers, GNU GPLv3

import tenpy
import types
import os
import re


def test_all(check_module=tenpy):
    """Recursively check that `__all__` of a module contains only valid entries.

    In each *.py file under tenpy/, there should be an __all__,
    """
    _file_ = check_module.__file__
    _name_ = check_module.__name__
    _package_ = check_module.__package__
    if not hasattr(check_module, '__all__'):
        raise AssertionError("module {0} has no line __all__ = [...]".format(_name_))
    _all_ = check_module.__all__

    print("test __all__ of", _name_)
    # find entries in __all__ but not in the module
    nonexistent = [n for n in _all_ if not hasattr(check_module, n)]
    if len(nonexistent) > 0:
        raise AssertionError("found entries {0!s} in __all__ but not in module {1}".format(
            nonexistent, _name_))

    # find objects in the module, which are not listed in __all__ (although they should be)
    for n in dir(check_module):
        if n[0] == '_' or n in _all_:  # private or listed in __all__
            continue
        obj = getattr(check_module, n)
        if getattr(obj, "__module__", None) == _name_:
            # got a class or function defined in the module
            raise AssertionError("object {0!r} defined in {1} but not in __all__".format(
                obj, _name_))
        if _name_ == "tenpy.models":
            # HACK: submodules of models (like xxz_chain.py) are not imported by default,
            # but by the other tests. They can be ignored here.
            continue
        if hasattr(obj, "__package__") and obj.__name__.startswith(_name_):
            # imported submodule
            raise AssertionError("Module {0!r} imported in {1} but not listed in __all__".format(
                obj.__name__, _name_))

    # recurse into submodules
    submodules = [getattr(check_module, n, None) for n in _all_]
    for m in submodules:
        if isinstance(m, types.ModuleType) and m.__name__.startswith('tenpy'):
            test_all(m)


def get_python_files(top):
    """return list of all python files recursively in a `top` directory."""
    python_files = []
    for dirpath, dirnames, filenames in os.walk(top):
        if '__pycache__' in dirnames:
            del dirnames[dirnames.index('__pycache__')]
        for fn in filenames:
            if fn.endswith('.py') and fn != '_npc_helper.py':
                # exlude _npc_helper.py generated in the egg by ``python setup.py install``
                python_files.append(os.path.join(dirpath, fn))
    return python_files


def test_copyright():
    tenpy_files = get_python_files(os.path.dirname(tenpy.__file__))
    #  to check also files in examples/ and toycodes/ etc, if you have the full repository,
    #  you can use the following:
    # tenpy_files = get_python_files(os.path.dirname(os.path.dirname(tenpy.__file__)))
    #  (but this doesn't work for the pip-installed tenpy, so you can only do it temporary!)
    regex = re.compile(r'#\s[Cc]opyright 20[0-9\-]+\s+(TeNPy|tenpy) [dD]evelopers, GNU GPLv3')
    for fn in tenpy_files:
        with open(fn, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    match = regex.match(line)
                    if match is not None:
                        break
            else:  # no break
                raise AssertionError("No/wrong copyright notice in {0!s}".format(fn))
