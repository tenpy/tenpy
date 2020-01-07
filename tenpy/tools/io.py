"""Tools to save and load data (from TeNPy) to disk."""
# Copyright 2020 TeNPy Developers, GNU GPLv3

import pickle
import gzip

__all__ = ['dump', 'load']


def dump(data, filename, mode='wb'):
    """Save `data` to file with given `filename`.

    This function guesses the type of the file from the filename ending.
    Supported endings:

    ======== ===============================
    ending   description
    ======== ===============================
    .pkl     Pickle without compression
    -------- -------------------------------
    .pklz    Pickle with gzip compression.
    ======== ===============================

    Parameters
    ----------
    filename : str
        The name of the file where to save the data.
    mode : str
        File mode for opening the file. ``'w'`` for write, ``'a'`` for append,
        ``'b'`` for binary (required for pickle). See :func:`open` for more details.
    """
    filename = str(filename)
    if filename.endswith('.pkl'):
        with open(filename, mode) as f:
            pickle.dump(data, f)
    elif filename.endswith('.pklz'):
        with gzip.open(filename, mode) as f:
            pickle.dump(data, f)
    else:
        raise ValueError("Don't recognise file ending of " + repr(filename))


def load(filename):
    """Load data from file with given `filename`.

    Guess the type of the file from the filename ending, see :func:`dump` for possible endings.

    Parameters
    ----------
    filename : str
        The name of the file to load.

    Returns
    -------
    data : obj
        The object loaded from the file.
    """
    filename = str(filename)
    if filename.endswith('.pkl'):
        with open(filename, mode) as f:
            data = pickle.load(f, 'rb')
    elif filename.endswith('.pklz'):
        with gzip.open(filename, mode) as f:
            data = pickle.load(f, 'rb')
    else:
        raise ValueError("Don't recognise file ending of " + repr(filename))
    return data
