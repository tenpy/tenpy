"""To be used in the `-m` argument of benchmark.py."""
# Copyright 2023 TeNPy Developers, GNU GPLv3

try:
    import old_tenpy as otp
except ModuleNotFoundError:
    print('This benchmark expects you to have a compiled version of tenpy v0.10 in your '
          '$PYTHONPATH under the name "old_tenpy".')
    raise
import tenpy as tp
import numpy as np
import tdot_tenpy


def get_qmod(sym):
    if isinstance(sym, tp.linalg.symmetries.groups.ProductSymmetry):
        qmod = []
        for s in sym.factors:
            qmod.extend(get_qmod(s))
        return qmod
    if isinstance(sym, tp.linalg.symmetries.groups.U1Symmetry):
        return [1]
    if isinstance(sym, tp.linalg.symmetries.groups.ZNSymmetry):
        return [sym.N]
    raise NotImplementedError


def convert_Tensor_to_Array(a: tp.linalg.tensors.Tensor, chinfo=None):
    if isinstance(a.backend, tp.linalg.backends.no_symmetry.AbstractNoSymmetryBackend):
        return otp.linalg.np_conserved.Array.from_ndarray_trivial(a.data)
    if not isinstance(a.backend, tp.linalg.backends.abelian.AbstractAbelianBackend):
        raise NotImplementedError
    # can assume abelian backend from now on
    
    if chinfo is None:
        chinfo = otp.linalg.charges.ChargeInfo(get_qmod(a.symmetry))
    legs = []
    for leg in a.legs:
        slices = np.insert(np.cumsum(leg._sorted_multiplicities), 0, 0)
        assert slices.shape == (leg.num_sectors + 1,)
        charges = leg._non_dual_sorted_sectors
        assert charges.shape == (leg.num_sectors, leg.symmetry.sector_ind_len)
        qconj = -1 if leg.is_dual else +1
        l = otp.linalg.charges.LegCharge(chinfo, slices, charges, qconj)
        legs.append(l)
    res = otp.linalg.np_conserved.Array(legs)
    res._data = a.data.blocks
    res._qdata = a.data.block_inds
    res._qdata_sorted = True
    res.test_sanity()
    return res
    

def setup_benchmark(**kwargs):
    assert kwargs.get('block_backend', 'numpy') == 'numpy'
    a, b, legs1, legs2 = tdot_tenpy.setup_benchmark(**kwargs)
    a = convert_Tensor_to_Array(a)
    b = convert_Tensor_to_Array(b, b.chinfo)
    return a, b, legs1, legs2


def benchmark(data):
    a, b, l1, l2 = data
    _ = otp.tensordot(a, b, (l1, l2))
