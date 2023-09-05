"""To be used in the `-m` argument of benchmark.py."""
# Copyright 2023 TeNPy Developers, GNU GPLv3

try:
    import old_tenpy as otp
except ModuleNotFoundError:
    print('This benchmark expects you to have a compiled version of tenpy v0.10 in your '
          '$PYTHONPATH under the name "old_tenpy".')
    raise
import tenpy as tp
import svd_tenpy
import tdot_old


def setup_benchmark(**kwargs):
    assert kwargs.get('block_backend', 'numpy') == 'numpy'
    a, q_legs, r_legs = svd_tenpy.setup_benchmark(**kwargs)
    a = tdot_old.convert_Tensor_to_Array(a)
    return a, q_legs, r_legs


def benchmark(data):
    a, q_legs, r_legs = data
    a = a.combine_legs([q_legs, r_legs])
    q, r = otp.linalg.np_conserved.qr(a)
    q = q.split_legs(0)
    r = r.split_legs(-1)
