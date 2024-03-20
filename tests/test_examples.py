"""A test for the examples.

The files are only imported, so please protect example code from running with standard ``if
__name__ == "__main__": ...`` clauses, if you want to demonstrate an interactive code, which
should not be executed as part of the tests.
"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import sys
import os
import pytest

# get directory where the examples can be found
examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')

exclude = ["__pycache__"]

try:
    examples = [fn for fn in os.listdir(examples_dir) if fn[-3:] == '.py' and fn not in exclude]
except FileNotFoundError:
    # examples are not contained in source distro, so they may not be accessible,
    # e.g. when conda tests its build
    examples = []


@pytest.mark.example  # allow to skip the examples with ``$> pytest -m "not example"``
@pytest.mark.slow
@pytest.mark.parametrize('filename', examples)
@pytest.mark.filterwarnings('ignore')
def test_examples_import(filename):
    assert filename[-3:] == '.py'
    old_sys_path = sys.path[:]
    if examples_dir not in sys.path:
        sys.path[:0] = [examples_dir]  # add the directory to sys.path
    try:
        with open(os.path.join(examples_dir, filename)) as f:
            script = f.read()
        scope = {}
        exec(script, scope, scope)
    finally:
        sys.path[:] = old_sys_path
