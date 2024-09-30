"""Check if subpackages define proper __all__."""
# Copyright (C) TeNPy Developers, Apache license
import types

import tenpy


def test_duplicate_free_all(check_module=tenpy):
    """Recursively check that `__all__` of a module contains only valid entries and no duplicates.
    """
    print(f'Checking module {check_module.__name__}')
    _all_ = getattr(check_module, '__all__', None)
    assert _all_ is not None, f'missing __all__'

    nonexistent = [n for n in _all_ if not hasattr(check_module, n)]
    assert len(nonexistent) == 0, f'entries found in __all__ but not in module: {", ".join(nonexistent)}'

    duplicates = [n for i, n in enumerate(_all_) if n in _all_[:i]]
    assert len(duplicates) == 0, f'duplicates in __all__ : {", ".join(duplicates)}'

    # resurse into submodules
    for n in _all_:
        m = getattr(check_module, n, None)
        if isinstance(m, types.ModuleType) and m.__name__.startswith('tenpy'):
            test_duplicate_free_all(m)
