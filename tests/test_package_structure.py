"""Tests if packages and modules define proper ``__all__``."""
# Copyright (C) TeNPy Developers, GNU GPLv3

import types
import tenpy


# dunder_all_exceptions[check_module] is a list of exceptions for check_dunder_all_recursive(check_module)
# that do not need to appear in __all__, eventhough the ruleset of check_dunder_all_recursive says
# they should.
dunder_all_exceptions = {
    'tenpy.linalg': ['dummy_config', 'misc'],  # TODO these are temporary, remove exception when they are gone
    'tenpy.linalg.backends': ['abelian', 'abstract_backend', 'array_api', 'backend_factory',
                              'no_symmetry', 'fusion_tree_backend', 'numpy', 'torch'],
    'tenpy.linalg.tensors': ['T', 'ElementwiseData', 'elementwise_function'],
}


def check_dunder_all_recursive(check_module, error_lines: list[str] = None):
    """Recursively check that `__all__` of a module contains only valid entries and no duplicates.

    Returns
    -------
    error_lines : list of str
        If non-empty, the test should fail and the list contains lines to be printed as an
        explanation
    """
    if error_lines is None:
        error_lines = []
    
    print(f'Checking module {check_module.__name__}')
    is_init_py = '__init__.py' in check_module.__file__
    _all_ = getattr(check_module, '__all__', None)
    _name_ = check_module.__name__
    assert _all_ is not None, f'missing {_name_}.__all__'

    nonexistent = [n for n in _all_ if not hasattr(check_module, n)]
    if nonexistent:
        error_lines.append(f'Objects from {_name_}.__all__ missing in the module:')
        error_lines.extend(['\n'.join(f'  {n}' for n in nonexistent)])

    duplicates = [n for i, n in enumerate(_all_) if n in _all_[:i]]
    if duplicates:
        error_lines.append(f'Duplicates in {_name_}.__all__:')
        error_lines.extend(['\n'.join(f'  {n}' for n in duplicates)])

    # find objects in the module, which are not listed in __all__ (although they should be)
    missing_objects = []
    for n in dir(check_module):
        if n[0] == '_' or n in _all_:  # private or listed in __all__
            continue
        if n in dunder_all_exceptions.get(_name_, []):
            continue
        obj = getattr(check_module, n)
        is_module = isinstance(obj, types.ModuleType)
        is_from_tenpy = getattr(obj, '__name__', '').startswith('tenpy')
        if is_init_py and is_module and is_from_tenpy:
            print(getattr(obj, '__name__', ''))
            # in __init__.py files, imported modules should be exposed.
            missing_objects.append(n)
        if getattr(obj, "__module__", None) == _name_:
            # got a class or function defined in the module (and not just imported from elsewhere)
            missing_objects.append(n)

    if missing_objects:
        error_lines.append(f'Objects missing from {_name_}.__all__:')
        error_lines.extend([', '.join(f"'{n}'" for n in missing_objects)])

    # recurse into submodules
    for n in _all_:
        m = getattr(check_module, n, None)
        if isinstance(m, types.ModuleType) and m.__name__.startswith('tenpy'):
            check_dunder_all_recursive(m, error_lines=error_lines)
    return error_lines


def test_dunder_all(max_print_lines=100):
    lines = check_dunder_all_recursive(tenpy)
    if lines:
        print()
        print('\n'.join(lines[:max_print_lines]))
        if len(lines) > max_print_lines:
            print(f'... and {len(lines) - max_print_lines} more lines')
        raise AssertionError
