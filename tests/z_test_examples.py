"""test whether the examples can at least be imported without problems.

The files are only imported, so please protect example code from running with standard ``if
__name__ == "__main__": ...`` clauses, if you want to demonstrate an interactive code.
"""
# Copyright 2018-2024 TeNPy Developers, GNU GPLv3

import sys
import os
import importlib
import pytest
import warnings

# get directory where the examples can be found
examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
toycodes_path = os.path.join(os.path.dirname(__file__), '..', 'toycodes')
toycodes_dir = os.path.join(toycodes_path, 'tenpy_toycodes')

exclude = ["__pycache__"]

examples = [fn for fn in os.listdir(examples_dir) if fn[-3:] == '.py' and fn not in exclude]
if os.path.exists(toycodes_dir):
    toycodes = [fn for fn in os.listdir(toycodes_dir) if fn[-3:] == '.py' and fn not in exclude]
else:
    toycodes = []


@pytest.mark.example  # allow to skip the examples with ``$> pytest -m "not example"``
@pytest.mark.slow
@pytest.mark.parametrize('filename', examples)
def test_examples_import(filename):
    assert filename[-3:] == '.py'
    old_sys_path = sys.path[:]
    if examples_dir not in sys.path:
        sys.path[:0] = [examples_dir]  # add the directory to sys.path
    try:
        with open(os.path.join(examples_dir, filename)) as f:
            script = f.read()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # disable warngings temporarily
            scope = {}
            exec(script, scope, scope)
    finally:
        sys.path[:] = old_sys_path


@pytest.mark.example  # allow to skip the examples with ``$> pytest -m "not example"``
@pytest.mark.slow
@pytest.mark.parametrize('filename', toycodes)
def test_toycodes_import(filename):
    assert filename[-3:] == '.py'
    old_sys_path = sys.path[:]
    if toycodes_path not in sys.path:
        sys.path[:0] = [toycodes_path]  # add the directory to sys.path
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # disable warngings temporarily
            importlib.import_module('tenpy_toycodes.' + filename[:-3])
    finally:
        sys.path[:] = old_sys_path

