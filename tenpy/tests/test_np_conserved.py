""" A collection of tests for TEBD with a spin chain """

import numpy as np
import scipy as sp
import scipy.linalg
from tenpy.algorithms.linalg import np_conserved as npc
from tenpy.algorithms.linalg import npc_helper
from tenpy.algorithms.linalg import LA_tools
import functools
import itertools
import timeit
import time
import random
import sys
from tenpy.tools.string import joinstr

Q_p = np.array([-1, -1, 1, 1]).reshape((-1, 1))

#chi = 61, 62

#Not sorted
Q_0 = np.array([-7, -5, -5, -5, -5, -5, -5, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 7]).reshape((-1, 1))
#Sorted
Q_1 = np.array([-6, -6, -6, -4, -4, -4, 4, 4, -4, -4, -4, -4, -2, -2, -2, -2, -2, -2, -2, -2, -2,
                -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, -2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6]).reshape((-1, 1))

################################################################################


def test_lanczos():
    print "------ lanczos eigh -------"

    Q = np.sort(np.concatenate((Q_0, Q_0)), axis=None).reshape((-1, 1))
    l = len(Q)
    rTh = sp.linalg.toeplitz(np.arange(l))
    iTh = sp.linalg.toeplitz(np.arange(l), -np.arange(l)) * 1j
    perm, rTh = npc.array.from_ndarray_flat(rTh, [Q, Q], q_conj=[1, -1])
    perm, iTh = npc.array.from_ndarray_flat(iTh, [Q, Q], q_conj=[1, -1])
    rTh.check_sanity()
    iTh.check_sanity()

    for Th in [rTh, rTh + iTh]:
        ndTh = Th.to_ndarray()
        t0 = time.time()
        A, B = npc.eigh(Th)

        print "Time with conservation", time.time() - t0
        t0 = time.time()
        ndA, ndB = np.linalg.eigh(ndTh)
        print "Time without conservation", time.time() - t0
        A = np.sort(A)
        perm = np.argsort(ndA)
        ndA = ndA[perm]
        ndB = ndB[:, perm]
        eigs_err = np.linalg.norm(A - ndA)
        print "npc.eigh err:", eigs_err
        assert abs(eigs_err) < 5e-13

        Es = []
        Vs = []
        for q in np.sort(Th.q_ind[0][:, 2]):  #all charge sectors
            H = LA_tools.lin_op(Th)
            LANCZOS_PAR = {'N_min': 5, 'N_max': 25, 'e_tol': 1 * 10**(-15), 'cache_v': 4}
            v0 = npc.array.from_npfunc(
                np.random.random, [npc.q_ind_from_q_flat(Q)], q_conj=[1], charge=[q])
            psinorm = v0.norm()

            Th.check_sanity()
            v0 *= (1. / psinorm)
            v0.check_sanity()
            #print "-"*20
            #t0 = time.time()
            for i in range(3):
                #E, v0, N = LA_tools.eigh(H, v0, LANCZOS_PAR)
                E, v0, N = LA_tools.lanczos(H, v0, LANCZOS_PAR)
                #print E, v0.norm(), N
            #print time.time() - t0
            Vs.append(v0)
            Es.append(E)

        #NOTE: I have checked that first four eigs are from different charge sectors, so this works.

        Es = np.array(Es)
        perm = np.argsort(Es)
        Es = Es[perm]
        Vs = [Vs[i] for i in perm]
        print "Lanczos E | eigh E:", Es[0:4], A[0:4]
        eigs_err = np.linalg.norm(A[0:4] - Es[0:4])
        print "Lanczos E err:", eigs_err
        assert abs(eigs_err) < 5e-13

        print "Lanczos Eigenvector Overlaps:",
        eigvec_err = 0.
        for i in range(4):
            e = 1. - np.abs(np.vdot(ndB[:, i], Vs[i].to_ndarray()))
            eigvec_err += e
            print e,
        print
        print "Lanczos V err:", eigvec_err
        assert abs(eigvec_err) < 1e-13


def test_np_conserved():
    partitions = [1, 2, 3]
    chi = sum(partitions)
    q_ind = np.zeros((len(partitions), 3), int)
    q_ind[0, 0] = 0
    for i in range(len(partitions)):
        q_ind[i, 1] = q_ind[i, 0] + partitions[i]
        if i + 1 < len(partitions): q_ind[i + 1, 0] = q_ind[i, 1]
        q_ind[i, 2] = i
    print q_ind, "= single q_ind"
    a = np.arange(chi**2).reshape((chi, chi))
    print a, "= full a"

    for totalq in range(2 * len(partitions) - 1):
        print "Total charge sector:", totalq
        ac = npc.array.from_ndarray(a, [q_ind] * 2, charge=totalq)
        #ac.itranspose()
        for i in range(len(ac.q_dat)):
            print "Charges:", ac.q_dat[i, :].transpose(), "\n", ac.dat[i]
        newa = ac.to_ndarray()
        print newa, "= new a"
    #print "Diff:", np.linalg.norm(newa - a)


def test_reshape():

    d = len(Q_p)
    c1 = len(Q_1)
    c0 = len(Q_0)
    rep = 10

    print "------ combine_legs, split_legs ------"
    print "Theta construction:"
    B0 = npc.array.from_npfunc(
        np.random.standard_normal, npc.q_ind_from_q_flat([Q_p, Q_1, Q_0]), q_conj=[1, -1, 1])
    B1 = npc.array.from_npfunc(
        np.random.standard_normal, npc.q_ind_from_q_flat([Q_p, Q_0, Q_1]), q_conj=[1, -1, 1])

    t0 = time.time()
    for i in range(rep):
        Th = npc.tensordot(B0, B1, axes=((2, ), (1, )), timing=False)
    print "tensordot time with conservation", (time.time() - t0) / rep

    Th.check_sanity()
    ndB0 = B0.to_ndarray()
    ndB1 = B1.to_ndarray()
    ndTh = Th.to_ndarray()
    t0 = time.time()
    for i in range(rep):
        ndTh2 = np.tensordot(ndB0, ndB1, axes=((2, ), (1, )))
    print "tensordot time without conservation", (time.time() - t0) / rep

    tensordot_err = np.linalg.norm(ndTh2 - ndTh)
    print "tensordot error", tensordot_err
    assert (tensordot_err) < 7e-14

    sca_mul_err = np.linalg.norm((2. * B0 + B0) - ndB0 * 3.)
    print "Sca mul err", sca_mul_err
    assert (sca_mul_err) < 1e-16
    print "norm", npc.inner(2 * B0, B0)

    #print "--------- reshape + SVD -----------"
    t0 = time.time()
    print "Making Left Pipe"
    pipe1 = Th.make_pipe([0, 1], qt_conj=-1)
    print "Making Right Pipe"
    pipe2 = Th.make_pipe([2, 3])
    print "Make pipes: ", time.time() - t0

    t0 = time.time()
    Th_comb = Th.combine_legs([[0, 1], [2, 3]], pipes=[pipe1, pipe2], timing=True)

    print "Time combine legs with conservation", time.time() - t0
    Th_comb.check_sanity()

    # check svd
    U, S, V = npc.svd(Th_comb, full_matrices=False, compute_uv=True, cutoff=None)
    print "Time reshape and SVD1 with conservation", time.time() - t0

    t0 = time.time()
    ndTh = ndTh.reshape((d * c1, -1))
    print "Time reshape without conservation", time.time() - t0

    ndU, ndS, ndV = np.linalg.svd(ndTh, full_matrices=False, compute_uv=True)
    print "Time reshape and SVD1 without conservation", time.time() - t0

    print "S err:", np.linalg.norm(np.sort(S) - np.sort(ndS))

    Th_dressed = Th_comb.copy()
    t0 = time.time()
    Th_resplit = Th_comb.split_legs([0, 1], [pipe1, pipe2], verbose=0)
    print "Time split legs:", time.time() - t0
    Th_resplit.check_sanity()

    combine_resplit_err = np.linalg.norm((Th.to_ndarray() - Th_resplit.to_ndarray()).reshape(-1))
    print "Resplitting error:", combine_resplit_err
    assert abs(combine_resplit_err) < 1e-16


def test_svd():
    print "--------- svd ---------"
    Q = np.sort(np.concatenate((Q_0)), axis=None).reshape((-1, 1))
    #	Q = np.sort(np.concatenate((Q_0, Q_1, -Q_0, -Q_1)), axis = None).reshape((-1, 1))
    Th = npc.array.from_npfunc(
        np.random.standard_normal, npc.q_ind_from_q_flat([Q, Q]), q_conj=[1, -1])
    Th = Th / Th.norm()
    Th.check_sanity()
    print "Th:", Th.sparse_stats()
    print "Theta SVD:"
    t0 = time.time()
    U, S, V = npc.svd(Th, full_matrices=False, compute_uv=True, cutoff=None)
    U.check_sanity()
    V.check_sanity()
    print "Time with conservation", time.time() - t0
    ndTh = Th.to_ndarray()
    t0 = time.time()
    nd2U, nd2S, nd2V = np.linalg.svd(ndTh, full_matrices=False, compute_uv=True)
    print "Time without conservation", time.time() - t0

    s_err = np.linalg.norm(np.sort(S) - np.sort(nd2S))
    print "S err:", s_err
    assert abs(s_err) < 2e-15

    USV = npc.tensordot(U.scale_axis(S), V, axes=((1, ), (0, )))
    USV.check_sanity()
    USV.sort_q_dat()
    Th.sort_q_dat()

    res = USV - Th
    print "USV - Th err:", res.norm()
    assert abs(res.norm()) < 1e-14

    print "Testing .iproject()"
    for c in np.arange(0, 0.1, .025):
        cut = S > c
        U.iproject(cut, 1)
        U.check_sanity()
        S = S[cut]
        V.iproject(cut, 0)
        V.check_sanity()

        USV2 = npc.tensordot(U.scale_axis(S), V, axes=((1, ), (0, )))

        a = npc.tensordot(USV2.transpose([1, 0]).conj(), USV2, [[1], [0]])
        diff = USV - USV2
        diff.check_sanity()

        print "Norms (should agree)", np.linalg.norm(S), USV2.norm(), "diff =", np.linalg.norm(
            S) - USV2.norm()
        assert abs(np.linalg.norm(S) - USV2.norm()) < 1e-15
        #print a.to_ndarray()


def test_pinv():
    print "--------- pinv ----------"
    q_flatL = np.array([0, 0, 1, 3, 1, 1]).reshape((6, 1))
    q_flatR = np.array([0, 1, 0, 4, 3]).reshape((5, 1))
    a = np.random.random((6, 5))
    perm, a = npc.array.from_ndarray_flat(
        a, [q_flatL, q_flatR], q_conj=[1, -1], charge=[0], sort=False)

    b = npc.pinv(a)

    pinv_err = np.linalg.norm(b.to_ndarray() - np.linalg.pinv(a.to_ndarray()))
    print "p_inv error:", pinv_err
    assert np.linalg.norm(b.to_ndarray() - np.linalg.pinv(a.to_ndarray())) < 8e-15


def test_eigh():
    print "--------- eigh ----------"
    Q = np.concatenate((Q_0, Q_0, Q_1, Q_1)).reshape((-1, 1))
    ndTh = np.random.standard_normal((len(Q), len(Q)))
    ndTh = ndTh + ndTh.transpose()
    perm, Th = npc.array.from_ndarray_flat(ndTh, [Q, Q], sort=False, q_conj=[1, -1])
    ndTh = Th.to_ndarray()
    Th.check_sanity()
    print "Th:", Th.sparse_stats()
    t0 = time.time()
    A, B = npc.eigh(Th)
    print "Time with conservation", time.time() - t0
    t0 = time.time()
    ndA, ndB = np.linalg.eigh(ndTh)
    print "Time without conservation", time.time() - t0

    eigs_err_norm = np.linalg.norm(np.sort(A) - np.sort(ndA))
    print "Eigs (fractured) err:", eigs_err_norm
    assert eigs_err_norm < 1e-13

    res = npc.tensordot(B, B.conj().scale_axis(A, axis=1), axes=[1, 1])
    decomp_err = npc.norm(res - Th) / npc.norm(Th)
    print "|T - U w U^d|", decomp_err
    assert decomp_err < 1e-13


def test_BLAS():
    print "--------- blas (iaxpy, iscal, inner, two_norm) ----------"
    Q = np.concatenate((Q_0, Q_0, Q_1, Q_1)).reshape((-1, 1))

    ndTh1 = np.random.standard_normal((len(Q), len(Q)))
    perm, Th1 = npc.array.from_ndarray_flat(ndTh1, [Q, Q], sort=False, q_conj=[1, -1])
    ndTh1 = Th1.to_ndarray()
    ndTh2 = np.random.standard_normal((len(Q), len(Q)))
    perm, Th2 = npc.array.from_ndarray_flat(ndTh2, [Q, Q], sort=False, q_conj=[1, -1])
    ndTh2 = Th2.to_ndarray()

    npc_helper.iscal(0.2, Th1)
    ndTh1 *= 0.2
    err = np.linalg.norm(Th1.to_ndarray() - ndTh1)
    print "iscal err", err
    assert err < 1e-15

    npc_helper.iaxpy(0.2, Th1, Th2)
    ndTh2 = 0.2 * ndTh1 + ndTh2
    err = np.linalg.norm(Th1.to_ndarray() - ndTh1)
    print "iaxpy err", err
    assert err < 1e-15

    err = np.abs(npc_helper.inner(Th1, Th2, do_conj=True) - np.vdot(ndTh1, ndTh2))
    print "inner err", err
    assert err < 1e-13

    err = np.abs(npc_helper.two_norm(Th1) - np.linalg.norm(ndTh1))
    print "two_norm err", err
    assert err < 1e-13


################################################################################
def repeat_ndarray(a, factors):
    if len(factors) > 0:
        return np.array([factors[i] * a for i in range(len(factors))])
    else:
        return np.array([]).reshape((len(a), 0))


def rand_distinct_int(a, b, n):
    """ returns n distinct integers from a to b inclusive """
    if n < 0: raise ValueError
    if n > b - a + 1: raise ValueError
    return np.sort((np.random.random_integers(a, b - n + 1, size=n))) + np.arange(n)


def rand_partitions(a, b, n):
    """ returns an array length n+1 (if possible) """
    if b - a <= n:
        return np.array(range(a, b + 1))
    else:
        return np.concatenate(([a], rand_distinct_int(a + 1, b - 1, n - 1), [b]))


def rand_permutation(n):
    perm = range(n)
    random.shuffle(perm)
    return perm


def rand_q_ind(dim, n_qsector):
    """ generates 1 q_ind for one leg,
		dim is the size of the leg, n_qsector is the number of charge sectors (may be a list) """
    if np.isscalar(n_qsector):
        n_qsector = np.array([n_qsector])
    elif type(n_qsector) != np.ndarray:
        n_qsector = np.array(n_qsector)
    part = rand_partitions(0, dim, np.prod(np.array(n_qsector)))
    q_ind = np.zeros((len(part) - 1, len(n_qsector) + 2), int)
    q_ind[:, 0] = part[:-1]
    q_ind[:, 1] = part[1:]
    q_combos = [a for a in itertools.product(*[range(-(nq / 2), nq / 2 + 1) for nq in n_qsector])]
    qs = np.array(q_combos)[rand_distinct_int(0, len(q_combos) - 1, len(part) - 1), :]
    #print rand_distinct_int(0, len(q_combos)-1, len(part)-1)
    q_ind[:, 2:] = qs
    return q_ind


def generate_q_ind_pair(d_ao,
                        d_bo,
                        d_in,
                        rand_var=0,
                        rand_seed=0,
                        size=3,
                        n_qsector=[2, 2],
                        q_mul=[1],
                        size_ao=None,
                        size_bo=None,
                        size_in=None,
                        verbose=0):
    """
		Returns:
			q_ind_a,q_ind_b
			The q_inds in a and b are copies, not linked to each other
		"""
    random.seed(rand_seed)
    np.random.seed(rand_seed)

    if np.isscalar(n_qsector):
        n_qsector = [n_qsector]
    if verbose > 0:
        print "n_qsector:", n_qsector

    if size_ao is None: size_ao = size
    if size_bo is None: size_bo = size
    if size_in is None: size_in = size
    shape_ao = np.random.random_integers(size_ao, size_ao + rand_var, size=d_ao)
    shape_bo = np.random.random_integers(size_bo, size_bo + rand_var, size=d_bo)
    shape_in = np.random.random_integers(size_in, size_in + rand_var, size=d_in)
    if verbose > 0:
        print shape_ao, shape_bo, shape_in

    q_ind_ao = [rand_q_ind(d, n_qsector) for d in shape_ao]
    q_ind_bo = [rand_q_ind(d, n_qsector) for d in shape_bo]
    q_ind_ai = [rand_q_ind(d, n_qsector) for d in shape_in]
    q_ind_bi = [q_ind.copy() for q_ind in q_ind_ai]
    for i in range(d_in):
        q_ind_bi[i][:, 2:] *= -1

#	for q_ind in q_ind_ao: print q_ind, "= q_ind (ao)"
#	for q_ind in q_ind_bo: print q_ind, "= q_ind (bo)"
#	for q_ind in q_ind_ai: print q_ind, "= q_ind (ai)"
#	for q_ind in q_ind_bi: print q_ind, "= q_ind (bi)"
    return (q_ind_ao + q_ind_ai), (q_ind_bi + q_ind_bo)


def check_tensordot_basic(para, verbose=1, rep=1):
    """ para is a dictionary
		"""

    # defaults
    num_q = 0
    seed = 0
    select_frac = 1.
    dtype = float
    tookNP = tookNPC = 0.
    # parameters
    if para.has_key('num_q'): num_q = para['num_q']
    d_ao, d_bo, d_contract = para['d']
    if para.has_key('seed'): seed = para['seed']
    if para.has_key('select_frac'): select_frac = para['select_frac']
    if para.has_key('dtype'): dtype = para['dtype']

    if d_ao + d_contract <= 0 or d_bo + d_contract <= 0: return 0, 0
    q_ind_a, q_ind_b = generate_q_ind_pair(
        d_ao, d_bo, d_contract, size=20, n_qsector=[2] * num_q, rand_var=3, rand_seed=seed)
    axesa = range(d_ao, d_ao + d_contract)
    axesb = range(d_contract)

    # randomize the axes in a and b
    perma = rand_permutation(d_ao + d_contract)
    rev_perma = np.argsort(perma)
    permb = rand_permutation(d_bo + d_contract)
    rev_permb = np.argsort(permb)
    q_ind_a = [q_ind_a[i] for i in perma]
    q_ind_b = [q_ind_b[i] for i in permb]
    axesa = [rev_perma[i] for i in axesa]
    axesb = [rev_permb[i] for i in axesb]

    shape_a = npc.shape_from_q_ind(q_ind_a)
    total_el_a = np.prod(np.array(shape_a))
    shape_b = npc.shape_from_q_ind(q_ind_b)
    total_el_b = np.prod(np.array(shape_b))

    # flat check
    a_trivial = npc.array.from_ndarray_trivial(np.arange(total_el_a, dtype=dtype))
    a_trivial.check_sanity()
    b_trivial = npc.array.from_ndarray_trivial(np.arange(total_el_b, dtype=dtype))
    b_trivial.check_sanity()

    if verbose > 1:
        if verbose > 2:
            print "a", shape_a, "* b", shape_b
            print q_ind_a, "= q_ind_a"
            print q_ind_b, "= q_ind_b"
    a = npc.array.from_ndarray(np.arange(total_el_a, dtype=dtype).reshape(shape_a), q_ind_a)
    b = npc.array.from_ndarray(np.arange(total_el_b, dtype=dtype).reshape(shape_b), q_ind_b)
    if num_q > 0 and select_frac < 1.:
        a_subset = rand_distinct_int(0, len(a.dat) - 1, int(len(a.dat) * select_frac))
        b_subset = rand_distinct_int(0, len(b.dat) - 1, int(len(b.dat) * select_frac))
        if len(a.q_dat) > 0: a.q_dat = a.q_dat[a_subset, :]
        if len(b.q_dat) > 0: b.q_dat = b.q_dat[b_subset, :]
        a.dat = [a.dat[i] for i in a_subset]
        b.dat = [b.dat[i] for i in b_subset]
    a.check_sanity()
    b.check_sanity()
    nd_a = a.to_ndarray()
    nd_b = b.to_ndarray()
    npc.tensordot_compat(a, b, axes=(axesa, axesb))
    t0 = time.time()
    for i in range(rep):
        c = npc.tensordot(a, b, axes=(axesa, axesb), verbose=verbose - 2, timing=False)
    tookNPC += time.time() - t0

    a.check_sanity()
    b.check_sanity()
    error_a = np.linalg.norm((nd_a - a.to_ndarray()).reshape(-1))
    error_b = np.linalg.norm((nd_b - b.to_ndarray()).reshape(-1))
    if d_ao + d_bo > 0:
        c.check_sanity()
        if verbose > 2:
            print 'a',
            a.print_q_dat(print_norm=True)
            print 'b',
            b.print_q_dat(print_norm=True)
            print 'c',
            c.print_q_dat(print_norm=True)
        nd_c = c.to_ndarray()
        t0 = time.time()
        for i in range(rep):
            np_tensordot_c = np.tensordot(nd_a, nd_b, axes=(axesa, axesb))
        tookNP += time.time() - t0
        error_c = np.linalg.norm((np_tensordot_c - nd_c).reshape(-1))
    else:  # d_ao + d_bo == 0
        if verbose > 2: print "c =", c
        t0 = time.time()
        for i in range(rep):
            np_tensordot_c = np.tensordot(nd_a, nd_b, axes=[axesa, axesb])[()]
        tookNP += time.time() - t0
        error_c = abs(np_tensordot_c - c)

    if verbose > 0:
        print "num_q:", num_q, ", dim ao, bo, contract:", d_ao, d_bo, d_contract, "\t",
        if verbose > 1:
            print
            print "norms(A,B,C) =", np.array([a.norm(), b.norm(), npc.norm(c)])
        print "A,B,C err:", error_a, error_b, error_c, "\t",
        if verbose > 1: print
        print "a", shape_a, "* b", shape_b, ", axes=" + str(axesa) + "," + str(axesb), "= c",
        if d_ao + d_bo > 0:
            print c.shape
        else:
            print "()"
        if verbose > 1: print
    assert abs(error_a) < 1E-13
    assert abs(error_b) < 1E-13
    assert abs(error_c) < 1E-13

    return tookNP / rep, tookNPC / rep


def tensordot_timing(para, verbose=1, rep=1, do_np=False):
    """ para is a dictionary
		"""
    # defaults
    num_q = 0
    seed = 0
    select_frac = 1.
    tookNP = tookNPC = 0.
    size = 20
    type = np.float
    # parameters
    if para.has_key('num_q'): num_q = para['num_q']
    d_ao, d_bo, d_contract = para['d']
    if para.has_key('seed'): seed = para['seed']
    if para.has_key('size'): size = para['size']
    if para.has_key('type'): type = para['type']
    if para.has_key('select_frac'): select_frac = para['select_frac']

    if d_ao + d_contract <= 0 or d_bo + d_contract <= 0: return 0, 0
    q_ind_a, q_ind_b = generate_q_ind_pair(
        d_ao, d_bo, d_contract, size=size, n_qsector=[2] * num_q, rand_var=3, rand_seed=seed)
    axesa = range(d_ao, d_ao + d_contract)
    axesb = range(d_contract)

    # randomize the axes in a and b
    perma = rand_permutation(d_ao + d_contract)
    rev_perma = np.argsort(perma)
    permb = rand_permutation(d_bo + d_contract)
    rev_permb = np.argsort(permb)
    q_ind_a = [q_ind_a[i] for i in perma]
    q_ind_b = [q_ind_b[i] for i in permb]
    axesa = [rev_perma[i] for i in axesa]
    axesb = [rev_permb[i] for i in axesb]

    shape_a = npc.shape_from_q_ind(q_ind_a)
    total_el_a = np.prod(np.array(shape_a))
    shape_b = npc.shape_from_q_ind(q_ind_b)
    total_el_b = np.prod(np.array(shape_b))

    #a = npc.array.test_array(q_ind_a, rand_seed=seed)
    #b = npc.array.test_array(q_ind_b, rand_seed=seed)
    #a = npc.array.test_array(q_ind_a, rand_seed='ones')
    #b = npc.array.test_array(q_ind_b, rand_seed='ones')
    a = npc.array.from_ndarray(
        np.sqrt(np.arange(
            total_el_a, dtype=np.float)).reshape(shape_a), q_ind_a)
    b = npc.array.from_ndarray(
        1. / np.sqrt(np.arange(
            1, total_el_b + 1, dtype=np.float)).reshape(shape_b), q_ind_b)
    #print len(a.dat), len(b.dat)
    a.ipurge_zeros()
    b.ipurge_zeros()

    #a.print_sparse_stats()
    #b.print_sparse_stats()

    if num_q > 0 and select_frac < 1.:
        a_subset = rand_distinct_int(0, len(a.dat) - 1, int(len(a.dat) * select_frac))
        b_subset = rand_distinct_int(0, len(b.dat) - 1, int(len(b.dat) * select_frac))
        if len(a.q_dat) > 0: a.q_dat = a.q_dat[a_subset, :]
        if len(b.q_dat) > 0: b.q_dat = b.q_dat[b_subset, :]
        a.dat = [a.dat[i] for i in a_subset]
        b.dat = [b.dat[i] for i in b_subset]

    if do_np:
        nd_a = a.to_ndarray()
        nd_b = b.to_ndarray()

    t0 = time.time()
    for i in range(rep):
        c = npc.tensordot(a, b, axes=(axesa, axesb), verbose=verbose - 2, timing=False)
    tookNPC += time.time() - t0

    if do_np:
        if d_ao + d_bo > 0:
            nd_c = c.to_ndarray()
            t0 = time.time()
            for i in range(rep):
                np_tensordot_c = np.tensordot(nd_a, nd_b, axes=(axesa, axesb))
            tookNP += time.time() - t0
            error_c = np.linalg.norm((np_tensordot_c - nd_c).reshape(-1))
        else:  # d_ao + d_bo == 0
            t0 = time.time()
            for i in range(rep):
                np_tensordot_c = np.tensordot(nd_a, nd_b, axes=[axesa, axesb])[()]
            tookNP += time.time() - t0
            error_c = abs(np_tensordot_c - c)
    else:
        error_c = 0.
    if verbose > 0:
        print "num_q:", num_q, ", dim ao, bo, contract:", d_ao, d_bo, d_contract, "\t",
        if verbose > 1: print
        print "a", shape_a, "* b", shape_b, ", axes=" + str(axesa) + "," + str(axesb), "= c",
        if d_ao + d_bo > 0:
            print c.shape
        else:
            print "()"
        if verbose > 1: print

    if np.abs(error_c) > 1E-12:
        print "error_c", error_c

    assert abs(error_c) < 1E-10
    t0 = time.time()
    for i in range(rep):
        pass
    loopt = time.time() - t0
    return (tookNP - loopt) / rep, (tookNPC - loopt) / rep


def check_tensordot_0len(para, verbose=1):
    """ para is a dictionary
		"""

    # defaults
    num_q = 1
    seed = 0

    # parameters
    if para.has_key('num_q'): num_q = para['num_q']
    d_ao, d_bo, d_contract = para['d']
    if para.has_key('seed'): seed = para['seed']

    if d_ao + d_contract <= 0 or d_bo + d_contract <= 0: return
    q_ind_a, q_ind_b = generate_q_ind_pair(
        d_ao, d_bo, d_contract, size=15, n_qsector=[2] * num_q, rand_var=10, rand_seed=seed)
    cutoff = 1
    while True:
        pickzeros = np.random.randint(0, 3, size=d_ao + d_bo + d_contract)
        if np.any(pickzeros > cutoff): break
    for l in range(d_ao):
        if pickzeros[l] > cutoff: q_ind_a[l] = np.empty((0, 2 + num_q), int)
    for l in range(d_bo):
        if pickzeros[l + d_ao] > cutoff: q_ind_b[l + d_contract] = np.empty((0, 2 + num_q), int)
    for l in range(d_contract):
        if pickzeros[l + d_ao + d_bo] > cutoff:
            q_ind_a[l + d_ao] = np.empty((0, 2 + num_q), int)
            q_ind_b[l] = np.empty((0, 2 + num_q), int)

    # randomize the axes in a and b
    axesa = range(d_ao, d_ao + d_contract)
    axesb = range(d_contract)

    perma = rand_permutation(d_ao + d_contract)
    rev_perma = np.argsort(perma)
    permb = rand_permutation(d_bo + d_contract)
    rev_permb = np.argsort(permb)
    q_ind_a = [q_ind_a[i] for i in perma]
    q_ind_b = [q_ind_b[i] for i in permb]
    axesa = [rev_perma[i] for i in axesa]
    axesb = [rev_permb[i] for i in axesb]

    shape_a = npc.shape_from_q_ind(q_ind_a)
    shape_b = npc.shape_from_q_ind(q_ind_b)

    if verbose > 0:
        print "dim ao, bo, contract:", d_ao, d_bo, d_contract, "\t",
        if verbose > 1:
            print
            print "\tpickzeros: %s %s" % (pickzeros[:d_ao] + pickzeros[d_ao + d_bo:],
                                          pickzeros[d_ao:])
            print joinstr(["\tq_ind_a:"] + map(str, q_ind_a), delim=" ")
            print joinstr(["\tq_ind_b:"] + map(str, q_ind_b), delim=" ")
        print "a %s * b %s, axes=%s,%s  -->  c" % (shape_a, shape_b, axesa, axesb),

    if np.all(np.array(shape_a) > 0) and np.all(np.array(shape_b) > 0):
        raise ValueError, "test_np_conserved error: shapes a and b should have a zero: %s %s" % (
            shape_a, shape_b)
    a = npc.zeros(q_ind_a, dtype=int)
    b = npc.array.from_ndarray(np.zeros(shape_b), q_ind_b)
    a.check_sanity(suppress_warning=True)
    b.check_sanity(suppress_warning=True)
    nd_a = a.to_ndarray()
    nd_b = b.to_ndarray()

    npc.tensordot_compat(a, b, axes=(axesa, axesb), suppress_warning=True)
    c = npc.tensordot(a, b, axes=(axesa, axesb), verbose=verbose - 2)
    a.check_sanity(suppress_warning=True)
    b.check_sanity(suppress_warning=True)
    if d_ao + d_bo > 0:
        c.check_sanity(suppress_warning=True)
        if verbose > 0: print c.shape
        shape_c = np.array([a.shape[i] for i in range(a.rank) if i not in axesa] + [b.shape[
            i] for i in range(b.rank) if i not in axesb])
        if not np.array_equiv(np.array(c.shape), shape_c):
            raise ValueError, "c.shape mismatch: %s != %s" % (c.shape, shape_c)
        if c.norm() != 0: raise ValueError, "c.norm = %s != 0" % c.norm()
    else:
        if c != 0: raise ValueError, "c = %s != 0" % (c, )
        if verbose > 0: print


def test_tensordot_2(verbose=0, timing=False):
    print "------ test_tensordot_2 ------"
    q_mul = np.array([1, -2])
    num_q = len(q_mul)
    q_ind_a, q_ind_b = generate_q_ind_pair(1, 0, 3, n_qsector=3, rand_var=1, q_mul=q_mul)

    q_flat1 = repeat_ndarray(np.array([1, 0, -1]), q_mul).transpose()
    q_flat2 = repeat_ndarray(np.array([1, 0, 0, -1]), q_mul).transpose()
    q_flat3 = repeat_ndarray(np.array([-2, -1, 0, 1, 2]), q_mul).transpose()
    q_flat4 = repeat_ndarray(np.array([-1, -1, 1, 1]), q_mul).transpose()
    q_flat5 = repeat_ndarray(np.array([-2, -2, 0, 0, 2, 2]), q_mul).transpose()

    perm, a = npc.array.from_ndarray_flat(
        np.arange(
            3 * 4 * 5 * 4, dtype=float).reshape((3, 4, 5, 4)),
        [q_flat1, q_flat2, q_flat3, q_flat4],
        charge=[0] * num_q,
        sort=False)
    a.check_sanity()
    perm, b = npc.array.from_ndarray_flat(
        np.ones(
            (5, 4, 6), dtype=float), [-q_flat3, -q_flat4, q_flat5],
        charge=[0] * num_q,
        sort=False)
    b.check_sanity()
    nd_a = a.to_ndarray()
    #print nd_a, "= a"
    #a.print_q_dat()
    nd_b = b.to_ndarray()

    #a.print_sparse_stats()
    #b.print_sparse_stats()
    c = npc.tensordot(a, b, axes=2, verbose=0)
    a.check_sanity()
    b.check_sanity()
    c.check_sanity()
    nd_c = c.to_ndarray()
    error_a = np.linalg.norm((nd_a - a.to_ndarray()).reshape(-1))
    error_b = np.linalg.norm((nd_b - b.to_ndarray()).reshape(-1))
    error_c = np.linalg.norm((np.tensordot(nd_a, nd_b, axes=2) - nd_c).reshape(-1))

    if verbose:
        print "A,B,C err:",
        print error_a, error_b, error_c

    #for i in range(len(c.q_dat)): print c.q_dat[i].tolist()
    c.sort_q_dat()
    #for i in range(len(c.q_dat)): print c.q_dat[i].tolist()
    c.check_sanity()

################################################################################


def run_tensordot_timing(verbose, do_np=False):
    print "------ tensordot_timing ------"

    for num_q in range(0, 3):
        print "num_q:", num_q
        for size in range(5, 60, 5):
            total_NP = 0.
            total_NPC = 0.
            for d_ao, d_bo, d_contract, seed in itertools.product(
                    range(2, 3), range(1, 2), range(2, 3), range(3)):
                para = {
                    'num_q': num_q,
                    'd': (d_ao, d_bo, d_contract),
                    'seed': seed,
                    'size': size,
                    'type': np.float
                }
                tookNP, tookNPC = tensordot_timing(para, verbose=verbose, rep=3, do_np=do_np)
                total_NP += tookNP
                total_NPC += tookNPC
                #print num_q, size, tookNP/tookNPC

            if do_np:
                print "Size, NP , NPC", size, total_NP / (size**3.), total_NPC / (size**3.)
            else:
                print "NQ S", num_q, size, total_NPC / (size**0.)
        print


def test_tensordot_basic():
    for num_q, d_ao, d_bo, d_contract, seed in itertools.product(
            range(0, 3), range(0, 3), range(0, 3), range(0, 3), range(3)):
        para = {
            'num_q': num_q,
            'd': (d_ao, d_bo, d_contract),
            'seed': seed,
        }
        yield check_tensordot_basic, para


def run_tensordot_basic(verbose):
    print "------ test_tensordot_basic ------"
    for num_q, d_ao, d_bo, d_contract, seed in itertools.product(
            range(0, 3), range(0, 3), range(0, 3), range(0, 3), range(3)):
        para = {
            'num_q': num_q,
            'd': (d_ao, d_bo, d_contract),
            'seed': seed,
        }
        check_tensordot_basic(para, verbose=verbose)
    #for test in test_tensordot_basic():
    #	test[0](test[1])


def testtensordot_0len():
    for d_ao, d_bo, d_contract, seed in itertools.product(
            range(0, 3), range(0, 3), range(0, 3), range(3)):
        para = {
            'd': (d_ao, d_bo, d_contract),
            'seed': seed,
        }
        yield check_tensordot_0len, para


def run_tensordot_0len(verbose):
    print "------ test_tensordot_0len ------"
    for d_ao, d_bo, d_contract, seed in itertools.product(
            range(0, 3), range(0, 3), range(0, 3), range(2)):
        para = {
            'd': (d_ao, d_bo, d_contract),
            'seed': seed,
        }
        check_tensordot_0len(para, verbose=verbose)

################################################################################
print "======================================== from_npfunc demo ========================================"
np.set_printoptions(linewidth=2000, precision=3, threshold=400)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0:
        run_tensordot_timing(verbose=0, do_np=int(args[0]))
    else:
        test_BLAS()
        test_lanczos()
        test_eigh()
        test_reshape()
        test_svd()
        test_pinv()
        test_tensordot_2(verbose=0)
        run_tensordot_basic(verbose=1)
        run_tensordot_0len(verbose=1)
