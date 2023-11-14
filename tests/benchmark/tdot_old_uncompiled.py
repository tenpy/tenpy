"""To be used in the `-m` argument of benchmark.py."""
# Copyright 2023 TeNPy Developers, GNU GPLv3

# Note: this module is *exactly* the same as tdot_old.py, except for which module is imported as otp

try:
    import old_tenpy_uncompiled as otp  # type: ignore
except ModuleNotFoundError:
    print('This benchmark expects you to have an uncompiled version of tenpy v0.10 in your '
          '$PYTHONPATH under the name "old_tenpy_uncompiled".')
    raise

import tdot_tenpy
from misc import convert_Tensor_to_Array
    

def setup_benchmark(**kwargs):
    assert kwargs.get('block_backend', 'numpy') == 'numpy'
    a, b, legs1, legs2 = tdot_tenpy.setup_benchmark(**kwargs)
    a = convert_Tensor_to_Array(a, old_tenpy=otp)
    b = convert_Tensor_to_Array(b, old_tenpy=otp, chinfo=a.chinfo)
    return a, b, legs1, legs2


def benchmark(data):
    a, b, l1, l2 = data
    _ = otp.tensordot(a, b, (l1, l2))