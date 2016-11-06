"""Miscellaneous tools, somewhat random mix yet often helpful.

.. todo :
    move random stuff from tools/math here....
"""

from __future__ import division

import numpy as np

try:
    """you can ``pip install bottleneck`` to use this framework for fast NaN processing"""
    import bottleneck as bn
    has_bottleneck = True
except:
    has_bottleneck = False

all = ['to_iterable', 'anynan', 'argsort']


def to_iterable(a):
    """If `a` is a not iterable or a string, return ``[a]``, else return ``a``."""
    if type(a) == str:
        return [a]
    try:
        iter(a)
    except TypeError:
        return [a]
    else:
        return a

if has_bottleneck:
    anynan = bn.anynan
else:
    def anynan(a):
        """check whether any entry of a ndarray `a` is 'NaN'"""
        return np.isnan(np.sum(a))  # still faster than 'np.isnan(a).any()'


def argsort(a, sort=None, **kwargs):
    """wrapper around np.argsort to allow sorting ascending/descending and by magnitude.

    Parameters
    ----------
    a : array_like
        the array to sort
    sort : {'m>', 'm<', '>', '<', ``None``}
        Specify how the arguments should be sorted.

        ========  ===========================
        `sort`    order
        ========  ===========================
        'm>'      Largest magnitude first
        'm<'      Smallest magnitude first
        '>'       Largest algebraic first
        '<'       Smallest algebraic first
        ``None``  numpy default: same as '<'
        ========  ===========================
    **kwargs :
        further keyword arguments given directly to ``numpy.argsort``.

    Returns
    -------
    index_array : ndarray, int
        same shape as `a`, such that ``a[index_array]`` is sorted in the specified way.
    """
    if sort is not None:
        if sort == 'm<':
            a = np.abs(a)
        elif sort == 'm>':
            a = -np.abs(a)
        elif sort == '>':
            a = -a
        elif sort != '<':
            raise ValueError("unknown sort option " + repr(sort))
    return np.argsort(a, **kwargs)
