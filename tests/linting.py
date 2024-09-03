"""Perform linting checks.

These are checks for coding guidelines, best practices etc.
The code may still run fine even if these checks fail.
We therefore consider them part of a linting routine and do *not* call them from pytest.
"""

from __future__ import annotations
import types
import tenpy
import os


def main(max_print_lines=30):
    """Called when this script is called, e.g. via `python linting.py`"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', action='store_true', help='quiet')
    args = parser.parse_args()
    quiet = args.q

    if not quiet:
        print('Checking __all__ attributes')
    lines = check_dunder_all_recursive(tenpy, quiet=quiet)
    if lines:
        msg = '\n'.join(lines[:max_print_lines])
        if len(lines) > max_print_lines:
            msg += f'... and {len(lines) - max_print_lines} more lines'
        assert False, msg
    if not quiet:
        print('Checking copyright notices')
    check_copyright_notice()
    print('Custom linting checks passed.')


# dunder_all_exceptions[check_module] is a list of exceptions for check_dunder_all_recursive(check_module)
# that do not need to appear in __all__, eventhough the ruleset of check_dunder_all_recursive says
# they should.
dunder_all_exceptions = {
    'tenpy.linalg': ['dummy_config', 'misc'],  # TODO these are temporary, remove exception when they are gone
    'tenpy.linalg.backends': ['abelian', 'abstract_backend', 'array_api', 'backend_factory',
                              'no_symmetry', 'fusion_tree_backend', 'numpy', 'torch'],
    'tenpy.linalg.tensors': ['T', 'ElementwiseData', 'elementwise_function'],
}
copyright_exceptions = [
    'tenpy/linalg/generate_clebsch_gordan'
]


def check_dunder_all_recursive(check_module, error_lines: list[str] = None, quiet: bool = False):
    """Recursively check that `__all__` of a module contains only valid entries and no duplicates.

    Returns
    -------
    error_lines : list of str
        If non-empty, the test should fail and the list contains lines to be printed as an
        explanation
    """
    if error_lines is None:
        error_lines = []

    if not quiet:
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
            check_dunder_all_recursive(m, error_lines=error_lines, quiet=quiet)
    return error_lines


def get_python_files(top):
    """return list of all python files recursively in a `top` directory."""
    python_files = []
    for dirpath, dirnames, filenames in os.walk(top):
        if '__pycache__' in dirnames:
            del dirnames[dirnames.index('__pycache__')]
        for fn in filenames:
            if fn.endswith('.py') and fn != '_npc_helper.py':
                # exclude _npc_helper.py generated in the egg by ``python setup.py install``
                python_files.append(os.path.join(dirpath, fn))
    return python_files


def check_copyright_notice(max_lines=100):
    tenpy_files = get_python_files(os.path.dirname(tenpy.__file__))
    #  to check also files in examples/ and toycodes/ etc, if you have the full repository,
    #  you can use the following:
    # tenpy_files = get_python_files(os.path.dirname(os.path.dirname(tenpy.__file__)))
    #  (but this doesn't work for the pip-installed tenpy, so you can only do it temporary!)
    wrong_files = []
    for fn in tenpy_files:
        with open(fn, 'r') as f:
            for line in f:
                if line.startswith('# Copyright (C) TeNPy Developers, GNU GPLv3'):
                    break
            else:  # no break
                is_exception = False
                for e in copyright_exceptions:
                    if e in fn:
                        is_exception = True
                if not is_exception:
                    wrong_files.append(fn)
    if wrong_files:
        print('No / wrong copyright notices in the following files:')
        for f in wrong_files[:max_lines]:
            print(f)


if __name__ == '__main__':
    main()
