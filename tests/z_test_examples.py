"""test whether the examples can at least be imported without problems.

The files are only imported, so please protect example code from running with standard ``if
__name__ == "__main__": ...`` clauses, if you want to demonstrate an interactive code.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import sys
import os
import importlib
import pytest
import warnings

# get directory where the examples can be found
examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
toycodes_dir = os.path.join(os.path.dirname(__file__), '..', 'toycodes')

exclude = ["__pycache__"]


def import_file(filename, dir):
    """Import the module given by `filename`.

    Since the examples are not protected  by ``if __name__ == "__main__": ...``,
    they run immediately at import. Thus an ``import filename`` (where filename is the actual name,
    not a string) executes the example.

    Parameters
    ----------
    filename : str
        the name of the file (without the '.py' ending) to import.
    """
    old_sys_path = sys.path[:]
    if dir not in sys.path:
        sys.path[:0] = [dir]  # add the directory to sys.path
    # to make sure the examples are found first with ``import``.
    print("importing file ", os.path.join(dir, filename))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # disable warngings temporarily
            mod = importlib.import_module(filename)
    finally:
        sys.path[:] = old_sys_path
    return mod


@pytest.mark.example  # allow to skip the examples with ``$> pytest -m "not example"``
@pytest.mark.slow
def test_examples_import():
    for fn in sorted(os.listdir(examples_dir)):
        if fn in exclude:
            continue
        if fn[-3:] == '.py':
            import_file(fn[:-3], examples_dir)


@pytest.mark.example  # allow to skip the examples with ``$> pytest -m "not example"``
@pytest.mark.slow
def test_toycodes_import():
    for fn in sorted(os.listdir(toycodes_dir)):
        if fn[-3:] == '.py' and fn not in exclude:
            import_file(fn[:-3], toycodes_dir)
