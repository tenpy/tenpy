"""test whether the examples run without problems.

The `test_examples` (in combination with `nose`)
runs *all* the 'examples/*.py' (except files listed in `exclude`).
However, the files are only imported, so you can protect example code from running with
``if __name__ == "__main__": ... `` clauses, if you want to demonstrate an interactive code.
"""

import sys
import os
import importlib
from nose.plugins.attrib import attr

# get directory where the examples can be found
ex_dir = os.path.join(os.path.dirname(__file__), '../examples')

exclude = []


def run_example(filename='npc_intro'):
    """Import the module given by `filename`.

    Since the examples are not protected  by ``if __name__ == "__main__": ...``,
    they run immediately at import. Thus an ``import filename`` (where filename is the actual name,
    not a string) executes the example. This function takes `filename` as string and uses
    `importlib` to do something like ``eval("import " + filename)``.

    Paramters
    ---------
    filename : str
        the name of the file (without the '.py' ending) to import.
    """
    if ex_dir not in sys.path:
        sys.path[:0] = [ex_dir]  # add the directory to sys.path
    # to make sure the examples are found first with ``import``.
    print "running example file ", filename
    mod = importlib.import_module(filename)
    print "finished example"
    if sys.path[0] == ex_dir:  # restore original sys.path
        sys.path = sys.path[1:]
    return mod


@attr('example')  # allow to skip the examples with ``$> nosetest -a '!example'``
@attr('slow')
def test_examples():
    """test generator to run all *.py files in ``ex_dir``.

    Generator, which yields ``function, arguments...``.
    `nosetests` then runs one test for each of them like ``function(arguments...)``
    """
    for fn in os.listdir(ex_dir):
        if fn[-3:] == '.py' and fn not in exclude:
            yield run_example, fn[:-3]
