#!/usr/bin/env python3

import pickle
import numpy as np
import os
import numpy.testing as npt
import tenpy
import tenpy.linalg.np_conserved as npc
import warnings
import pytest

datadir = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.isdir(datadir):
    os.mkdir(datadir)


@pytest.mark.parametrize('fn', [fn for fn in os.listdir(datadir) if fn.endswith('.pkl')])
def test_pickle_import_old_version(fn):
    print("import ", fn)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        with open(os.path.join(datadir, fn), 'rb') as f:
            data = pickle.load(f)
    assert isinstance(data, dict)
    print(list(data.keys()))
    for k, v in data.items():
        v.test_sanity()
        if k == 'Sz':
            Sz = np.array([[0.5, 0.], [0., -0.5]])
            npt.assert_equal(v.to_ndarray(), Sz)
        elif k == 'trivial_array':
            npt.assert_equal(v.to_ndarray(), np.arange(20).reshape([4, 5]))


def test_pickle_export():
    s = tenpy.networks.site.SpinHalfSite()
    data = {
        'SpinHalfSite': s,
        'trivial_array': npc.Array.from_ndarray_trivial(np.arange(20).reshape([4, 5])),
        'Sz': s.Sz
    }
    dump = pickle.dumps(data)
    # save to file or compare to existing file
    fn = "pickled_from_tenpy_{0}.pkl".format(tenpy.version.short_version)
    filename = os.path.join(datadir, fn)
    if os.path.exists(filename):
        print("don't pickle dump to file: existing", fn)
        dump = pickle.dumps(filename, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # create a pickle file
        print("dump to file data/", fn, sep='')
        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    test_pickle_export()
    for f, fn in test_data_import():
        f(fn)
