"""A collection of tests for tenpy.tools submodules."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import logging
import numpy as np
import numpy.testing as npt
import itertools as it
import tenpy
from tenpy import tools
import warnings
import pytest
import os.path
import sys


def test_inverse_permutation(N=10):
    x = np.random.random(N)
    p = np.arange(N)
    np.random.shuffle(p)
    xnew = x[p]
    pinv = tools.misc.inverse_permutation(p)
    npt.assert_equal(x, xnew[pinv])
    npt.assert_equal(pinv[p], np.arange(N))
    npt.assert_equal(p[pinv], np.arange(N))
    pinv2 = tools.misc.inverse_permutation(tuple(p))
    npt.assert_equal(pinv, pinv2)


def test_argsort():
    x = [1., -1., 1.5, -1.5, 2.j, -2.j]
    npt.assert_equal(tools.misc.argsort(x, 'LM'), [4, 5, 2, 3, 0, 1])
    npt.assert_equal(tools.misc.argsort(x, 'SM'), [0, 1, 2, 3, 4, 5])
    npt.assert_equal(tools.misc.argsort(x, 'LR'), [2, 0, 4, 5, 1, 3])


def test_speigs():
    x = np.array([1., -1.2, 1.5, -1.8, 2.j, -2.2j])
    tol_NULP = len(x)**3
    x_LM = x[tools.misc.argsort(x, 'm>')]
    x_SM = x[tools.misc.argsort(x, 'SM')]
    A = np.diag(x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # disable warngings temporarily
        for k in range(4, 9):
            print(k)
            W, V = tools.math.speigs(A, k, which='LM')
            W = W[tools.misc.argsort(W, 'LM')]
            print(W, x_LM[:k])
            npt.assert_array_almost_equal_nulp(W, x_LM[:k], tol_NULP)
            W, V = tools.math.speigs(A, k, which='SM')
            W = W[tools.misc.argsort(W, 'SM')]
            print(W, x_SM[:k])
            npt.assert_array_almost_equal_nulp(W, x_SM[:k], tol_NULP)


def test_matvec_to_array():
    A_orig = np.random.random([5, 5]) + 1.j * np.random.random([5, 5])

    class A_matvec:
        def __init__(self, A):
            self.A = A
            self.shape = A.shape
            self.dtype = A.dtype

        def matvec(self, v):
            return np.dot(self.A, v)

    A_reg = tools.math.matvec_to_array(A_matvec(A_orig))
    npt.assert_array_almost_equal(A_orig, A_reg, 14)


def test_perm_sign():
    res = [tools.math.perm_sign(u) for u in it.permutations(range(4))]
    check = [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1]
    npt.assert_equal(res, check)


def test_qr_li():
    cutoff = 1.e-10
    for shape in [(5, 4), (4, 5)]:
        print('shape =', shape)
        A = np.arange(20).reshape(shape)  # linearly dependent: only two rows/columns independent
        A[3, :] = np.random.random() * (cutoff / 100)  # nearly linear dependent
        q, r = tools.math.qr_li(A)
        assert np.linalg.norm(r - np.triu(r)) == 0.
        qdq = q.T.conj().dot(q)
        assert np.linalg.norm(qdq - np.eye(len(qdq))) < 1.e-13
        assert np.linalg.norm(q.dot(r) - A) < cutoff * 20
        r, q = tools.math.rq_li(A)
        assert np.linalg.norm(r - np.triu(r, r.shape[1] - r.shape[0])) == 0.
        qqd = q.dot(q.T.conj())
        assert np.linalg.norm(qqd - np.eye(len(qqd))) < 1.e-13
        assert np.linalg.norm(r.dot(q) - A) < cutoff * 20


def test_memory_usage():
    tools.process.memory_usage()


def test_omp(n=2):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # disable warngings temporarily
        if tools.process.omp_set_nthreads(n):
            nthreads = tools.process.omp_get_nthreads()
            print(nthreads)
            assert (nthreads == n)
        else:
            print("test_omp failed to import the OpenMP libaray.")


def test_mkl(n=2):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # disable warngings temporarily
        if tools.process.mkl_set_nthreads(n):
            nthreads = tools.process.mkl_get_nthreads()
            print(nthreads)
            assert (nthreads == n)
        else:
            print("test_mkl failed to import the shared MKL libaray.")


def test_group_by_degeneracy():
    group_by_degeneracy = tools.misc.group_by_degeneracy
    #    0     1       2    3       4  5    6
    E = [2., 2.4, 1.9999, 1.8, 2.3999, 5, 1.8]
    k = [0,    1,      2,   2,      1, 2,   1]  # yapf: disable
    g = group_by_degeneracy(E)
    assert g == [(0, ), (1, ), (2, ), (3, 6), (4, ), (5, )]
    g = group_by_degeneracy(E, cutoff=0.01)
    assert g == [(0, 2), (1, 4), (3, 6), (5, )]
    g = group_by_degeneracy(E, k, cutoff=0.01)
    assert g == [(0, ), (1, 4), (2, ), (3, ), (5, ), (6, )]


def test_optimization():
    level_now = tools.optimization.get_level()
    level_change = "none" if level_now == 1 else "default"
    level_change = tools.optimization.OptimizationFlag[level_change]
    assert tools.optimization.get_level() == level_now
    assert tools.optimization.get_level() != level_change
    with tools.optimization.temporary_level(level_change):
        assert tools.optimization.get_level() == level_change
    assert tools.optimization.get_level() == level_now


def test_events():
    noted = []
    counters = []
    event_counter = [0]

    ev1 = tools.events.EventHandler("event_name, expected_event_counter")

    @ev1.connect
    def note_event(event_name, expected_event_counter):
        noted.append(event_name)
        counters.append(expected_event_counter)

    def increase_counter(event_name, expected_event_counter):
        print("callback from event ", event_name)
        event_counter[0] += 1

    def check_event_counter_before(event_name, expected_event_counter):
        assert expected_event_counter == event_counter[0]

    def check_event_counter_after(event_name, expected_event_counter):
        assert expected_event_counter + 1 == event_counter[0]

    ev2 = tools.events.EventHandler("event_name, expected_event_counter")
    ev2.connect(note_event, 0)
    note_id = ev2.id_of_last_connected
    for ev in [ev1, ev2]:
        ev.connect(check_event_counter_before, 2)  # called before `increase_counter`
        ev.connect(check_event_counter_after, -1)  # called after `increase_counter`
    for ev in [ev1, ev2]:
        ev.connect(increase_counter, 1)  # high priority
    print("start events")
    ev1.emit("a", 0)
    ev2.emit("b", 1)
    ev2.emit("c", 2)
    ev2.disconnect(note_id)
    ev2.emit("d", 3)
    ev1.emit("e", 4)
    print("after calls")
    assert event_counter[0] == 5
    assert tuple(counters) == (0, 1, 2, 4)  # disconnected event 2 note before 3
    assert tuple(noted) == ("a", "b", "c", "e")


def three_exp(x):
    lam = np.array([0.9, 0.4, 0.2])
    pref = np.array([0.01, 0.4, 20])
    return tools.fit.sum_of_exp(lam, pref, x)


def screened_coulomb(x):
    return np.exp(-0.1 * x) / x**2


def test_approximate_sum_of_exp(N=100):
    x = np.arange(1, N + 1)
    for n, f, max_err in [(3, three_exp, 1.e-13), (5, three_exp, 1.e-13), (2, three_exp, 0.04),
                          (1, three_exp, 0.1), (4, screened_coulomb, 7.e-4)]:
        lam, pref = tools.fit.fit_with_sum_of_exp(f, n=n, N=N)
        err = np.sum(np.abs(f(x) - tools.fit.sum_of_exp(lam, pref, x)))
        print(n, f.__name__, err)
        assert err < max_err


def test_find_subclass():
    BaseCls = tenpy.models.lattice.Lattice
    SimpleLattice = tenpy.models.lattice.SimpleLattice  # direct sublcass of Lattice
    Square = tenpy.models.lattice.Square  # sublcass of SimpleLattice -> recursion necessary

    with pytest.raises(ValueError):
        tools.misc.find_subclass(BaseCls, 'UnknownSubclass')
    simple_found = tools.misc.find_subclass(BaseCls, 'SimpleLattice')
    assert simple_found is SimpleLattice
    square_found = tools.misc.find_subclass(BaseCls, 'Square')
    assert square_found is Square


def test_get_set_recursive():
    data = {'some': {'nested': {'data': 123, 'other': 456}, 'parts': 789}}
    assert tools.misc.get_recursive(data, 'some.nested.data') == 123
    assert tools.misc.get_recursive(data, '.some.nested.data') == 123
    tools.misc.set_recursive(data, 'some.nested.data', 321)
    assert tools.misc.get_recursive(data, 'some:nested:data', ':') == 321
    tools.misc.set_recursive(data, ':some:parts', 987, ':')
    assert tools.misc.get_recursive(data, 'some.parts') == 987
    flat_data = tools.misc.flatten(data)
    assert flat_data == {'some.nested.data': 321, 'some.nested.other': 456, 'some.parts': 987}


@pytest.mark.skip(reason="interferes with pytest logging setup")
def test_logging_setup(tmp_path, capsys):
    import logging.config
    logger = logging.getLogger('tenpy.test_logging')
    root = logging.getLogger()
    output_filename = tmp_path / 'output.pkl'
    logging_params = {
        'to_stdout': 'INFO',
        'to_file': 'WARNING',
        'skip_setup': False,
    }
    tools.misc.setup_logging(logging_params, output_filename)

    test_message = "test %s message 12345"
    logger.info(test_message, 'info')
    logger.warning(test_message, 'warning')

    # clean up loggers -> close file handlers (?)
    logging.config.dictConfig({'version': 1, 'disable_existing_loggers': False})

    assert os.path.exists(tmp_path / 'output.log')
    with open(tmp_path / 'output.log', 'r') as f:
        file_text = f.read()
    assert test_message % 'warning' in file_text
    assert test_message % 'info' not in file_text  # should have filtered that out

    capture = capsys.readouterr()
    stdout_text = capture.out
    assert test_message % 'warning' in stdout_text
    assert test_message % 'info' in stdout_text
