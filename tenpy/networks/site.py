"""Defines a class describing the local physical Hilbert space.

.. todo ::
    implement more standard sites, e.g. fermions, bosons, hard-core-bosons, ...
"""
from __future__ import division

import numpy as np

from ..linalg import np_conserved as npc


class Site(object):
    """Collects necessary information about a single local site of a lattice.

    It has the on-site operators as attributes, e.g. ``self.Id`` is the identy.

    (This class is implemented in :mod:`tenpy.networks.site` but also imported in
    :mod:`tenpy.models.lattice` for convenience.)

    Parameters
    ----------
    leg : :class:`npc.LegCharge`
        Charges of the physical states, to be used for the physical leg of MPS & co).
    state_labels : None | list of str
        Optinally a label for each local basis states. ``None`` entries are ignored / not set.
    **site_ops :
        Additional keyword arguments of the form ``name=op`` given to :meth:`add_op`.
        The identity operator 'Id' is always included.

    Attributes
    ----------
    dim
    onsite_ops
    leg : :class:`npc.LegCharge`
        Charges of the local basis states.
    state_labels : {str: int}
        (Optional) labels for the local basis states.
    opnames : set
        Labels of all onsite operators (i.e. ``self.op`` exists if ``'op'`` in ``self.opnames``).
    ops : :class:`npc.Array`
        Onsite operators are added directly as attributes to self.
        For example after ``self.add_op('Sz', Sz)`` you can use ``self.Sz`` for the `Sz` operator.

    Examples
    --------
    The following generates a site for spin-1/2 with Sz conservation.
    Note that ``Sx = (Sp + Sm)/2`` violates Sz conservation and is thus not a valid
    on-site operator.

    >>> chinfo = npc.ChargeInfo([1], ['Sz'])
    >>> ch = npc.LegCharge.from_qflat(chinfo, [1, -1])
    >>> Sp = [[0, 1.], [0, 0]]
    >>> Sm = [[0, 0], [1., 0]]
    >>> Sz = [[0.5, 0], [0, -0.5]]
    >>> site = Site(ch, ['up', 'down'], Splus=Sp, Sminus=Sm, Sz=Sz)
    >>> print site.Splus.to_ndarray()
    array([[ 0.,  1.],
           [ 0.,  0.]])

    .. todo ::
        Problem: what if we later want to remove the charges / add new charges?!?
        Some onsite op's might not be compatible with charges, although the resulting
        Hamiltonian might be?
    .. todo ::
        add option to sort by charges and save the resulting permutation.
    .. todo ::
        need clever way to handle Jordan-Wigner strings for fermions...
    """
    def __init__(self, charges, state_labels=None, **site_ops):
        self.leg = charges
        self.state_labels = dict()
        if state_labels is not None:
            for i, v in enumerate(state_labels):
                self.state_labels[str(v)] = i
        self.opnames = set()
        self.add_op('Id', npc.diag(1., self.leg))
        for name, op in site_ops.iteritems():
            self.add_op(name, op)
        self.test_sanity()

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        for lab, ind in self.state_labels.iteritems():
            if type(lab) != str:
                raise ValueError("wrong type of state label")
            if not 0 <= ind < self.dim:
                raise ValueError("index of state label out of bounds")
        for name in self.opnames:
            if not hasattr(self, name):
                raise ValueError("missing onsite operator " + name)
        for op in self.onsite_ops.values():
            if op.rank != 2:
                raise ValueError("only rank-2 onsite operators allowed")
            op.legs[0].test_equal(self.leg)
            op.legs[1].test_contractible(self.leg)
            op.test_sanity()

    @property
    def dim(self):
        """Dimension of the local Hilbert space"""
        return self.leg.ind_len

    @property
    def onsite_ops(self):
        """dictionary of on-site operators for iteration.

        (single operators are accessible as attributes.)"""
        return dict([(name, getattr(self, name)) for name in sorted(self.opnames)])

    def add_op(self, name, op):
        """add one or multiple on-site operators

        Parameters
        ----------
        name : str
            A valid python variable name, used to label the operator.
            The name under which `op` is added as attribute to self.
        op : np.ndarray | npc.Array
            A matrix acting on the local hilbert space representing the local operator.
            Dense numpy arrays are automatically converted to :class:`npc.Array`.
            LegCharges have to be [leg, leg.conj()].
        """
        name = str(name)
        if name in self.opnames:
            raise ValueError("operator with that name already existent: " + name)
        if hasattr(self, name):
            raise ValueError("Site already has that attribute name: " + name)
        if not isinstance(op, npc.Array):
            op = np.asarray(op)
            if op.shape != (self.dim, self.dim):
                raise ValueError("wrong shape of on-site operator")
            # try to convert op into npc.Array
            op = npc.Array.from_ndarray(op, self.leg.chinfo, [self.leg, self.leg.conj()])
        if op.rank != 2:
            raise ValueError("only rank-2 on-site operators allowed")
        op.legs[0].test_equal(self.leg)
        op.legs[1].test_contractible(self.leg)
        op.test_sanity()
        setattr(self, name, op)
        self.opnames.add(name)

    def get_state_index(self, label):
        """return index of a basis state from its label.

        Parameters
        ----------
        label : int | string
            eather the index directly or a label (string) set before.

        Returns
        -------
        state_index : int
            the index of the basis state associated with the label.
        """
        res = self.state_labels.get(label, label)
        try:
            res = int(res)
        except:
            raise KeyError("label not found: " + repr(label))
        return res

    def get_op(self, name):
        """return operator of given name."""
        return getattr(self, name)

# ------------------------------------------------------------------------------
# functions for generating the most common local sites.


def spin_half_site(conserve='Sz'):
    """Generate spin-1/2 site.

    Local states are spin-up (0) and spin-down (1).
    Local operators are spin-1/2 operators, e.g. ``Sz = [[0.5, 0.], [0., -0.5]]``.

    ============== ====  ======================
    `conserve`     qmod  onsite operators
    ============== ====  ======================
    ``'Sz'``       [1]   ``Id, Sz, Sp, Sm``
    ``'parity'``   [2]   additional ``Sx, Sy``
    ``None``       []    all above
    ============== ====  ======================

    Parameters
    ----------
    conserve : str
        Defines what is conserved, see table above.


    Returns
    -------
    site : class:`Site`
        Spin-1/2 site with `leg`, `leg.chinfo` and onsite operators.

    """
    if conserve not in ['Sz', 'parity', None]:
        raise ValueError("invalid 'conserve': " + repr(conserve))
    Sx = [[0., 0.5], [0.5, 0.]]
    Sy = [[0., -0.5j], [+0.5j, 0.]]
    Sz = [[0.5, 0.], [0., -0.5]]
    Sp = [[0., 1.], [0., 0.]]  # == Sx + i Sy
    Sm = [[0., 0.], [1., 0.]]  # == Sx - i Sy
    ops = dict(Sp=Sp, Sm=Sm, Sz=Sz)
    if conserve == 'Sz':
        chinfo = npc.ChargeInfo([1], ['2*Sz'])
        leg = npc.LegCharge.from_qflat(chinfo, [1, -1])
    else:
        ops.update(Sx=Sx, Sy=Sy)
        if conserve == 'parity':
            chinfo = npc.ChargeInfo([2], ['parity'])
            leg = npc.LegCharge.from_qflat(chinfo, [1, 0])  # [1, -1] would need ``qmod=[4]``...
        else:
            leg = npc.LegCharge.from_trivial(2)
    site = Site(leg, ['up', 'down'], **ops)
    return site

def boson_site(Nc=2, conserve='N'):
    raise NotImplementedError()
