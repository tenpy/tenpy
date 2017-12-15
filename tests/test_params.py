"""A test for tenpy.tools.params"""

import warnings
from tenpy.tools import params
import nose.tools as nst


def example_function(example_pars, keys=['a', 'b']):
    """example function using a parameter dictionary"""
    for default, k in enumerate(keys):
        p_k = params.get_parameter(example_pars, k, default, "testing")
        print("read out parameter {p_k!r}".format(p_k=p_k))


def test_paramters():
    pars = dict()
    example_function(pars)
    pars = dict(a=None, b=2.5, c="non-used")
    example_function(pars)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        pars = dict(miss_spelled=1.23)
        example_function(pars)
        unused = params.unused_parameters(pars, "testing")
        nst.eq_(len(unused), 1)
        nst.eq_(len(w), 1)
