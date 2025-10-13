"""A test for tenpy.tools.params."""
# Copyright (C) TeNPy Developers, Apache license

from tenpy.tools.params import Config, asConfig
import copy
import pytest


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

    # test .get(..., expect_types) argument
    _ = config.get('a', 4, NotADirectoryError)  # value of None always passes
    _ = config.get('a', 12, [NotADirectoryError, dict])  # value of None always passes
    _ = config.get('b', 5 ,'real')
    with pytest.warns(UserWarning, match='Invalid type for key'):
        _ = config.get('b', 5, int)
    _ = config.get('uses_default_value', 5.3 ,'real')
    with pytest.warns(UserWarning, match='Invalid type for key'):
        _ = config.get('uses_default_value', 5.3, int)
    _ = config.get('b', 5, [float, dict])
    with pytest.warns(UserWarning, match='Invalid type for key'):
        _ = config.get('b', 5, [int, dict])

    # test warnings on deletion
    assert len(config.unused) == 1
    # the match is a bit ugly, since brackets have special meaning in regex.
    # the expected message is
    #  "unused option ['e'] for config Test parameters"
    with pytest.warns(UserWarning, match=r"unused option \['e'\] for config Test parameters"):
        del config
    del pars
    with pytest.warns(FutureWarning, match="Deprecated option in 'sub': 'y' renamed to 'y_new'"):
        sub.deprecated_alias('y', 'y_new')
    assert len(sub.unused) == 2
    with pytest.warns(UserWarning, match=r"unused options for config sub:\n\['x', 'y_new'\]"):
        sub.__del__()

