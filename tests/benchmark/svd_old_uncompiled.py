"""To be used in the `-m` argument of benchmark.py."""
# Copyright 2023 TeNPy Developers, GNU GPLv3

# Note: this module is *exactly* the same as svd_old.py, except for which module is imported as otp

try:
    import old_tenpy_uncompiled as otp  # type: ignore
except ModuleNotFoundError:
    print('This benchmark expects you to have an uncompiled version of tenpy v0.10 in your '
          '$PYTHONPATH under the name "old_tenpy_uncompiled".')
    raise

import svd_tenpy
from misc import convert_Tensor_to_Array


def setup_benchmark(**kwargs):
    assert kwargs.get('block_backend', 'numpy') == 'numpy'
    a, u_legs, vh_legs = svd_tenpy.setup_benchmark(**kwargs)
    a = convert_Tensor_to_Array(a, old_tenpy=otp)
    return a, u_legs, vh_legs


def benchmark(data):
    # v2.0 svd includes combining legs before the "matrix svd" and splitting them after.
    # so the v0.x version should include those steps too.
    a, u_legs, vh_legs = data
    if len(a._data) == 0:
        # size=1 sometimes has no blocks and causes an error without this speecial case
        return
    a = a.combine_legs([u_legs, vh_legs])
    u, s, vh = otp.linalg.np_conserved.svd(a)
    u.split_legs(0)
    vh.split_legs(-1)
