"""Check for consistencies."""
# Copyright 2019 TeNPy Developers

import tenpy
import types


def test_all(check_module=tenpy):
    """Recursively check that `__all__` of a module contains only valid entries.

    In each *.py file under tenpy/, there should be an __all__, """
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
        if hasattr(obj, "__package__") and getattr(obj, "__name__").startswith(_name_):
            # imported submodule
            print(_package_)
            raise AssertionError("Module {0!r} imported in {1} but not listed in __all__".format(
                obj.__name__, _name_))

    # recurse into submodules
    submodules = [getattr(check_module, n, None) for n in _all_]
    for m in submodules:
        if isinstance(m, types.ModuleType):
            test_all(m)
