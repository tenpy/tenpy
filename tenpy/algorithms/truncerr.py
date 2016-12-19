"""truncation errors for TEBD.
"""


class TruncationError(object):
    """Class representing a truncation error.

    .. warning:
        For imaginary time evolution, this is *not* the error you are interested in!

    Examples
    --------
    >>> TE = TruncationError()
    >>> TE += tebd.update(...)
    """
    def __init__(self, TE=0., norm=1.):
        assert(TE >= 0.)
        self.TE = TE/norm
        self.NE = (1.-TE)/norm

    def __add__(self, other):
        res = TruncationError()
        res.TE = self.TE + other.TE
        res.NE = self.NE * other.NE
        return res

# TODO: im update:
# return TruncationError(sum(si^2 thrown away), sum(all si^2))
