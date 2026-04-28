"""Perform linting checks.

These are checks for coding guidelines, best practices etc.
The code may still run fine even if these checks fail.
We therefore consider them part of a linting routine and do *not* call them from pytest.
"""
# Copyright (C) TeNPy Developers, Apache license

import os

import cyten


def main():
    """Called when this script is called, e.g. via `python linting.py`"""
    print('Custom Linting:')
    print('  Checking copyright notices')
    check_copyright_notice()
    print('Custom Linting passed.')


def get_python_files(top):
    """return list of all python files in a directory. Recursive."""
    python_files = []
    for dirpath, dirnames, filenames in os.walk(top):
        if '__pycache__' in dirnames:
            del dirnames[dirnames.index('__pycache__')]
        for fn in filenames:
            if fn.endswith('.py') and fn != '_npc_helper.py':
                # exclude _npc_helper.py generated in the egg by ``python setup.py install``
                python_files.append(os.path.join(dirpath, fn))
    return python_files


def check_copyright_notice():
    expected_notice = '# Copyright (C) TeNPy Developers, Apache license'
    cyten_files = get_python_files(os.path.dirname(cyten.__file__))
    for fn in cyten_files:
        with open(fn, 'r') as f:
            for line in f:
                if line.startswith(expected_notice):
                    break
            else:  # no break
                raise AssertionError(f'No/wrong copyright notice in {fn}.')


if __name__ == '__main__':
    main()
