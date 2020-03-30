"""A test for tenpy.tools.params."""
# Copyright 2018-2020 TeNPy Developers, GNU GPLv3

import warnings
from tenpy.tools.params import Parameters


def example_function(example_pars, keys=['a', 'b']):
    """example function using a parameter dictionary."""
    for default, k in enumerate(keys):
        p_k = example_pars.get(k, default)
        print("read out parameter {p_k!r}".format(p_k=p_k))


def test_parameters():
    pars = Parameters(dict(), "Test empty")
    example_function(pars)
    pars = Parameters(dict(a=None, b=2.5, c="dict-style", d="non-used"), "Test parameters")
    assert pars['c'] == "dict-style"
    example_function(pars)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        pars = Parameters(dict(miss_spelled=1.23), "Test unused")
        example_function(pars)
        unused = pars.unused
        assert len(unused) == 1
        assert len(w) == 1
