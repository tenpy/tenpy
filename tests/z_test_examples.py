"""test whether the examples run without problems.

The `test_examples` (in combination with `nose`)
runs *all* the 'examples/*.py' (except files listed in `exclude`).
However, the files are only imported, so you can protect example code from running with
``if __name__ == "__main__": ... `` clauses, if you want to demonstrate an interactive code.
"""
# Copyright 2018 TeNPy Developers

import sys
import os
import importlib
from nose.plugins.attrib import attr
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


@attr('example')  # allow to skip the examples with ``$> nosetest -a '!example'``
@attr('slow')
def test_examples():
    for fn in sorted(os.listdir(examples_dir)):
        if fn in exclude:
            continue
        if fn[-3:] == '.py':
            yield import_file, fn[:-3], examples_dir


@attr('example')
@attr('slow')
def test_toycodes():
    for fn in sorted(os.listdir(toycodes_dir)):
        if fn[-3:] == '.py' and fn not in exclude:
            yield import_file, fn[:-3], toycodes_dir


if __name__ == "__main__":
    for f, fn, dir in test_examples():
        f(fn, dir)
    for f, fn, dir in test_toycodes():
        f(fn, dir)
