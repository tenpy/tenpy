"""Shared pytest configuration for the tenpy tests."""

# Copyright (C) TeNPy Developers, Apache license
from pathlib import Path

import lasyten as lasy
import pytest

_SUN_DATA_PATH = Path(__file__).resolve().parents[1] / 'lasyten_repo' / 'external' / 'SUN_symbols'


@pytest.fixture(scope='session', autouse=True)
def _sun_data_path():
    """Point lasyten to the bundled SU(N) data, independent of env vars / user config (if the data exists).

    ``SpinSite(conserve='SU2')`` uses the Clebsch-Gordan-data based ``SUN(N=2)`` symmetry, which reads
    its data from the lasyten option ``su_n_data_path``. The data is the ``lasyten_repo/external/SUN_symbols``
    submodule (``git submodule update --init --recursive``). If it is not available, the configured
    path (env var ``LASYTEN_SU_N_DATA_PATH``, ``.lasytenconfig.yaml``, ...) is left untouched.
    """
    if not any(_SUN_DATA_PATH.glob('*.hdf5')):
        yield
        return
    with lasy.temporary_options(su_n_data_path=str(_SUN_DATA_PATH)):
        yield
