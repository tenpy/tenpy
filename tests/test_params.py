"""A test for tenpy.tools.params."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import warnings
from tenpy.tools.params import Config, asConfig
import copy


def example_function(example_pars, keys=['a', 'b', 'c']):
    """example function using a parameter dictionary."""
    for default, k in enumerate(keys):
        p_k = example_pars.get(k, default)
        print("read out parameter {k!r} = {p_k!r}".format(k=k, p_k=p_k))


def test_parameters():
    pars = Config(dict(), "Test empty")
    example_function(pars)
    pars = dict(
        a=None,
        b=2.5,
        d="dict-style access",
        e="non-used",
        sub=dict(x=10, y=20),
    )
    pars_copy = copy.deepcopy(pars)
    config = asConfig(pars, "Test parameters")
    example_function(config)
    assert config['d'] == "dict-style access"  # reads out d
    pars_copy['c'] = 2
    assert config.as_dict() == pars_copy
    sub = config.subconfig("sub")
    sub.setdefault('z', 30)
    pars_copy['sub']['z'] = 30
    assert config.as_dict() == pars_copy
    example_function(sub)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert len(config.unused) == 1
        del config  # catch the warning for 'e'
        del pars
        # assert len(w) == 1
        sub.deprecated_alias('y', 'y_new')
        # assert len(w) == 2
        assert len(sub.unused) == 2
        sub.__del__()
        assert len(w) == 3
        sub.touch('x', 'y_new')  # avoid warnings when deconstructed outside of the catch_warning
