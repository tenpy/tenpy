"""Check if subpackages define proper __all__."""
# Copyright 2019-2023 TeNPy Developers, GNU GPLv3

import types
import tenpy


dunder_all_exceptions = {
    'tenpy.linalg.tensors': ['T', 'ElementwiseData']
}


def test_dunder_all(check_module=tenpy):
    """Recursively check that `__all__` of a module contains only valid entries and no duplicates.
    """
    print(f'Checking module {check_module.__name__}')
    _all_ = getattr(check_module, '__all__', None)
    _name_ = check_module.__name__
    assert _all_ is not None, f'missing __all__'

    nonexistent = [n for n in _all_ if not hasattr(check_module, n)]
    assert len(nonexistent) == 0, f'entries found in __all__ but not in module: {", ".join(nonexistent)}'

    duplicates = [n for i, n in enumerate(_all_) if n in _all_[:i]]
    assert len(duplicates) == 0, f'duplicates in __all__ : {", ".join(duplicates)}'

    # find objects in the module, which are not listed in __all__ (although they should be)
    missing = []
    for n in dir(check_module):
        if n[0] == '_' or n in _all_:  # private or listed in __all__
            continue
        if n in dunder_all_exceptions.get(_name_, []):
            continue
        obj = getattr(check_module, n)
        if getattr(obj, "__module__", None) == _name_:
            # got a class or function defined in the module
            missing.append(n)
    if missing:
        print('Objects missing from __all__:')
        print('\n'.join(f'  {n}' for n in missing))
        raise AssertionError

    # resurse into submodules
    for n in _all_:
        m = getattr(check_module, n, None)
        if isinstance(m, types.ModuleType) and m.__name__.startswith('tenpy'):
            test_dunder_all(m)
