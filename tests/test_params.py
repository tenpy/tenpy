"""A test for tenpy.tools.params."""
# Copyright 2018-2020 TeNPy Developers, GNU GPLv3

import warnings
from tenpy.tools.params import Config, asConfig


def example_function(example_pars, keys=['a', 'b', 'c']):
    """example function using a parameter dictionary."""
    for default, k in enumerate(keys):
        p_k = example_pars.get(k, default)
        print("read out parameter {p_k!r}".format(p_k=p_k))


def test_parameters():
    pars = Config(dict(), "Test empty")
    example_function(pars)
    pars = dict(a=None, b=2.5, d="dict-style", e="non-used", sub=dict(x=10, y=20), verbose=1)
    config = asConfig(pars, "Test parameters")
    example_function(config)
    assert config['d'] == "dict-style"
    sub = config.subconfig("sub")
    sub.setdefault('z', 30)
    example_function(sub)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert len(config.unused) == 1
        del config  # catch the warning for 'e'
        assert len(w) == 1
        sub.deprecated_alias('y', 'y_new')
        assert len(w) == 2

        assert len(sub.unused) == 2
        del sub  # catch warnings for 'x', y'
        assert len(w) == 3
