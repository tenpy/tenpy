"""A collection of tests for tenpy.tools submodules."""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
import numpy.testing as npt
import itertools as it
import tenpy
from tenpy import tools
import pytest
import os.path


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
    npt.assert_equal(tools.misc.argsort(x, 'LM', kind='stable'), [4, 5, 2, 3, 0, 1])
    npt.assert_equal(tools.misc.argsort(x, 'SM', kind='stable'), [0, 1, 2, 3, 4, 5])
    npt.assert_equal(tools.misc.argsort(x, 'LR', kind='stable'), [2, 0, 4, 5, 1, 3])


def test_speigs():
    x = np.array([1., -1.2, 1.5, -1.8, 2.j, -2.2j])
    tol_NULP = len(x)**3
    x_LM = x[tools.misc.argsort(x, 'm>')]
    x_SM = x[tools.misc.argsort(x, 'SM')]
    A = np.diag(x)

    with pytest.warns(UserWarning, match='trimming speigs k to smaller matrix dimension d'):
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


@pytest.mark.filterwarnings('ignore')
def test_omp(n=2):
    if tools.process.omp_set_nthreads(n):
        nthreads = tools.process.omp_get_nthreads()
        print(nthreads)
        assert (nthreads == n)
    else:
        print("test_omp failed to import the OpenMP libaray.")


@pytest.mark.filterwarnings('ignore')
def test_mkl(n=2):
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


def test_merge_recursive():
    data1 = {'some': {'nested': {'data': 123, 'other': 456},
                      'conflict': 'first'},
             'only': 1}
    data2 = {'some': {'different': {'x': 234, 'y': 567},
                      'conflict': 'second'},
             'extra': 2}
    data3 = {'some': {'yet another': {'a': 1, 'b': 2},
                      'conflict': 'third'},
             'foo': 3}

    with pytest.raises(ValueError) as excinfo:
        merged = tools.misc.merge_recursive(data1, data2, data3)
    assert "'some':'conflict'" in str(excinfo.value)
    merged_first = tools.misc.merge_recursive(data1, data2, data3, conflict='first')
    expected_merged = {
        'some': {'nested': {'data': 123, 'other': 456},
                 'conflict': 'first',
                 'different': {'x': 234, 'y': 567},
                 'yet another': {'a': 1, 'b': 2},
                 },
        'only': 1,
        'extra': 2,
        'foo': 3,
    }
    assert merged_first == expected_merged
    expected_merged['some']['conflict'] = 'third'
    merged_last = tools.misc.merge_recursive(data1, data2, data3, conflict='last')
    assert merged_last == expected_merged


@pytest.mark.skip(reason="interferes with pytest logging setup")
def test_logging_setup(tmp_path):
    from contextlib import redirect_stdout
    import logging.config
    logger = logging.getLogger('tenpy.test_logging')
    root = logging.getLogger()
    logging_params = {
        'output_filename': tmp_path / 'output.pkl',
        'to_stdout': 'INFO',
        'to_file': 'WARNING',
        'skip_setup': False,
    }


    with open(tmp_path / 'stdout.txt', 'w') as stdout:
        with redirect_stdout(stdout):
            # example logging code
            tools.misc.setup_logging(**logging_params)

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

    with open(tmp_path / 'stdout.txt', 'r') as stdout:
        stdout_text = stdout.read()
    assert test_message % 'warning' in stdout_text
    assert test_message % 'info' in stdout_text
    print("test_logging_setup() finished without errors")


def test_convert_memory_units():
    assert tools.misc.convert_memory_units(12.5*1024, 'bytes', 'bytes') == (12.5 * 1024, 'bytes')
    assert tools.misc.convert_memory_units(12.5*1024, 'KB', 'MB') == (12.5, 'MB')
    assert tools.misc.convert_memory_units(12.5*1024, 'MB', 'KB') == (12.5 * 1024**2, 'KB')
    assert tools.misc.convert_memory_units(12.5*1024, 'MB', None) == (12.5, 'GB')


def test_setup_logging():
    tenpy.tools.misc.setup_logging(to_stdout="INFO", skip_setup=False)


def test_fixes_consistency_check_IrregularLattice():
    # tests if the bug reported at https://tenpy.johannes-hauschild.de/viewtopic.php?t=757 is fixed.

    class SpinModel_triangular_finite(tenpy.models.CouplingMPOModel):
        def init_sites(self, model_params):
            return tenpy.SpinHalfSite(conserve=None, sort_charge=True)

        def init_terms(self, model_params):
            self.add_onsite(-1, 0, 'Sz')

        def init_lattice(self, model_params):
            regular_lat = tenpy.Triangular(5, 5, self.init_sites(model_params))
            return tenpy.IrregularLattice(regular_lat, remove=[(0, 0, 0)])

    model = SpinModel_triangular_finite({})

    ad_hoc_fix = False
    if ad_hoc_fix:
        model.lat.N_sites_per_ring = 1

    psi = tenpy.MPS.from_lat_product_state(model.lat, [[['up']]])
    dmrg_params = dict(max_sweeps=2)
    _ = tenpy.algorithms.dmrg.run(psi, model, dmrg_params)


def test_failing_consistency_check():
    # compare function fails due to complex not being comparable with <=
    tools.misc.consistency_check(1.2 + 1.j, {}, 'a', 1., '<')
    # this should raise a logger error and print the taceback there,
    # but the program should continue

if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp_path:
        test_logging_setup(Path(tmp_path))
