"""A collection of tests for tenpy.linalg.lanczos"""
from __future__ import division

import tenpy.linalg.np_conserved as npc
import numpy as np
import numpy.testing as npt

from test_charges import gen_random_legcharge
from tenpy.linalg import lanczos

ch = npc.ChargeInfo([2])


def test_gramschmidt(n=30, k=5, tol=1.e-15):
    leg = gen_random_legcharge(ch, n)
    vecs_old = [npc.Array.from_func(np.random.random, [leg], shape_kw='size') for i in range(k)]
    vecs_new, _ = lanczos.gram_schmidt(vecs_old, rcond=0., verbose=1)
    assert all([v == w for v, w in zip(vecs_new, vecs_old)])
    vecs_new, _ = lanczos.gram_schmidt(vecs_old, rcond=tol, verbose=1)
    vecs = [v.to_ndarray() for v in vecs_new]
    ovs = np.zeros((k, k))
    for i, v in enumerate(vecs):
        for j, w in enumerate(vecs):
            ovs[i, j] = np.inner(v.conj(), w)
    print ovs
    assert(np.linalg.norm(ovs - np.eye(k)) < 2*n*k*k*tol)
