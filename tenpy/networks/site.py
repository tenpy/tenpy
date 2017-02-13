"""Defines a class describing the local physical Hilbert space.

.. todo ::
    implement more standard sites, e.g. spin-full fermions, spin-S, ...
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
        All onsite operators have labels ``'p', 'p*'``.

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
        Use it where helpful, e.g. for boson_site(conserve=parity) and co.
    """

    def __init__(self, charges, state_labels=None, **site_ops):
        self.leg = charges
        self.state_labels = dict()
        if state_labels is not None:
            for i, v in enumerate(state_labels):
                if v is not None:
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
        """Dictionary of on-site operators for iteration.

        (single operators are accessible as attributes.)"""
        return dict([(name, getattr(self, name)) for name in sorted(self.opnames)])

    def add_op(self, name, op):
        """Add one on-site operators

        Parameters
        ----------
        name : str
            A valid python variable name, used to label the operator.
            The name under which `op` is added as attribute to self.
        op : np.ndarray | npc.Array
            A matrix acting on the local hilbert space representing the local operator.
            Dense numpy arrays are automatically converted to :class:`npc.Array`.
            LegCharges have to be [leg, leg.conj()].
            We set labels ``'p', 'p*'``.
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
            op = npc.Array.from_ndarray(op, [self.leg, self.leg.conj()])
        if op.rank != 2:
            raise ValueError("only rank-2 on-site operators allowed")
        op.legs[0].test_equal(self.leg)
        op.legs[1].test_contractible(self.leg)
        op.test_sanity()
        op.set_leg_labels(['p', 'p*'])
        setattr(self, name, op)
        self.opnames.add(name)

    def rename_op(self, old_name, new_name):
        """Rename an added operator.

        Parameters
        ----------
        old_name : str
            The old name of the operator.
        new_name : str
            The new name of the operator.
        """
        if old_name == new_name:
            return
        op = getattr(self, old_name)
        setattr(self, new_name, op)
        delattr(self, old_name)

    def state_index(self, label):
        """Return index of a basis state from its label.

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

    def state_indices(self, labels):
        """Same as :meth:`state_index`, but for multiple labels."""
        return [self.state_index(lbl) for lbl in labels]

    def get_op(self, name):
        """Return operator of given name."""
        return getattr(self, name)

# ------------------------------------------------------------------------------
# functions for generating the most common local sites.


def spin_half_site(conserve='Sz'):
    """Generate spin-1/2 site.

    Local states are ``up``(0) and ``down``(1).
    Local operators are the usual spin-1/2 operators, e.g. ``Sz = [[0.5, 0.], [0., -0.5]]``,
    ``Sx = 0.5*sigma_x`` for the Pauli matrix `sigma_x`.

    ============== ====  ==========================
    `conserve`     qmod  onsite operators
    ============== ====  ==========================
    ``'Sz'``       [1]   ``Id, Sz, Sp, Sm``
    ``'parity'``   [2]   ``Id, Sz, Sp, Sm, Sx, Sy``
    ``None``       []    ``Id, Sz, Sp, Sm, Sx, Sy``
    ============== ====  ==========================

    ==============  ================================================
    operator        description
    ==============  ================================================
    ``Id``          identity :math:`\mathbb{1}`
    ``Sx, Sy, Sz``  spin components :math:`S^{x,y,z}`,
                    equal to half the Pauli matrices.
    ``Sp, Sm``      spin flips :math:`S^{\pm} = S^{x} \pm i S^{y}`
    ==============  ================================================

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
        raise ValueError("invalid `conserve`: " + repr(conserve))
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


def fermion_site(conserve='N', filling=0.5):
    r"""Create a :class:`Site` for spin-less fermions.

    Local states are ``empty`` and ``occupied``.
    Local operators can be built from creation operators.

    .. warning :
        Using the Jordan-Wigner string (``JW``) is crucial to get correct results,
        otherwise you just describe hardcore bosons!

    ============== ====  ==========================
    `conserve`     qmod  onsite operators
    ============== ====  ==========================
    ``'N'``        [1]   ``Id, JW, C, Cd, N``
    ``'parity'``   [2]   ``Id, JW, C, Cd, N``
    ``None``       []    ``Id, JW, C, Cd, N``
    ============== ====  ==========================

    ==============  ========================================
    operator        description
    ==============  ========================================
    ``Id``          identity :math:`\mathbb{1}`
    ``JW``          Sign for the Jordan-Wigner string.
    ``C``           Annihilation operator :math:`c`
    ``Cd``          Creation operator :math:`c^\dagger`
    ``N``           Number operator :math:`n= c^\dagger c`
    ``dN``          :math:`\delta n := n - filling`
    ``dNdN``        :math:`(\delta n)^2`
    ==============  ========================================

    Parameters
    ----------
    conserve : str
        Defines what is conserved, see table above.
    filling : float
        Average filling. Used to define ``dN``.

    Returns
    -------
    site : class:`Site`
        (Spin-less) fermion site with `leg`, `leg.chinfo` and onsite operators.

    .. todo ::
        Write userguide for Fermions describing Jordan-Wigner-trafo/-string...
    """
    if conserve not in ['N', 'parity', None]:
        raise ValueError("invalid `conserve`: " + repr(conserve))
    JW = np.array([[0., 0.], [0., 1.]])
    C = np.array([[0., 1.], [0., 0.]])
    Cd = np.array([[0., 0.], [1., 0.]])
    N = np.array([[0., 0.], [0., 1.]])
    dN = np.array([[-filling, 0.], [0., 1. - filling]])
    dNdN = dN**2  # (element wise power is fine since dN is diagonal)
    ops = dict(JW=JW, C=C, Cd=Cd, N=N, dN=dN, dNdN=dNdN)
    if conserve == 'N':
        chinfo = npc.ChargeInfo([1], ['N'])
        leg = npc.LegCharge.from_qflat(chinfo, [0, 1])
    elif conserve == 'parity':
        chinfo = npc.ChargeInfo([2], ['parity'])
        leg = npc.LegCharge.from_qflat(chinfo, [0, 1])
    else:
        leg = npc.LegCharge.from_trivial(2)
    return Site(leg, ['empty', 'occupied'], **ops)


def boson_site(Nmax=1, conserve='N', filling=0.):
    r"""Create a :class:`Site` for up to `Nmax` bosons.

    Local states are ``vac, 1, 2, ... , Nc``.
    (Exception: for parity conservation, we sort as ``vac, 2, 4, ..., 1, 3, 5, ...``.)
    Local operators can be built from creation operators.


    ============== ====  ==================================
    `conserve`     qmod  onsite operators
    ============== ====  ==================================
    ``'N'``        [1]   ``Id, B, Bd, N, NN, dN, dNdN, P``
    ``'parity'``   [2]   ``Id, B, Bd, N, NN, dN, dNdN, P``
    ``None``       []    ``Id, B, Bd, N, NN, dN, dNdN, P``
    ============== ====  ==================================

    ==============  ========================================
    operator        description
    ==============  ========================================
    ``Id``          identity :math:`\mathbb{1}`
    ``B``           Annihilation operator :math:`b`
    ``Bd``          Creation operator :math:`b^\dagger`
    ``N``           Number operator :math:`N= b^\dagger b`
    ``NN``          :math:`N^2`
    ``dN``          :math:`\delta N := N - filling`
    ``dNdN``        :math:`(\delta N)^2`
    ``P``           Parity :math:`Id - 2 (N \mod 2)`.
    ==============  ========================================

    Parameters
    ----------
    Nmax : int
        Cutoff defining the maximum number of bosons per site.
        The default ``Nmax=1`` describes hard-core bosons.
    conserve : str
        Defines what is conserved, see table above.
    filling : float
        Average filling. Used to define ``dN``.

    Returns
    -------
    site : class:`Site`
        Bosonic site with `leg`, `leg.chinfo` and onsite operators.
    """
    if conserve not in ['N', 'parity', None]:
        raise ValueError("invalid `conserve`: " + repr(conserve))
    dim = Nmax + 1
    if dim < 2:
        raise ValueError("local dimension should be larger than 1....")
    B = np.zeros([dim, dim], dtype=np.float)  # destruction/annihilation operator
    for n in xrange(1, dim):
        B[n - 1, n] = np.sqrt(n)
    Bd = np.transpose(B)  # .conj() wouldn't do anything
    # Note: np.dot(Bd, B) has numerical roundoff errors of eps~=4.4e-16.
    Ndiag = np.arange(dim, dtype=np.float)
    N = np.diag(Ndiag)
    NN = np.diag(Ndiag**2)
    dN = np.diag(Ndiag - filling)
    dNdN = np.diag((Ndiag - filling)**2)
    P = np.diag(1. - 2. * np.mod(Ndiag, 2))
    ops = dict(B=B, Bd=Bd, N=N, NN=NN, dN=dN, dNdN=dNdN, P=P)
    if conserve == 'N':
        chinfo = npc.ChargeInfo([1], ['N'])
        leg = npc.LegCharge.from_qflat(chinfo, range(dim))
    elif conserve == 'parity':
        chinfo = npc.ChargeInfo([2], ['parity'])
        leg_unsorted = npc.LegCharge.from_qflat(chinfo, [i % 2 for i in range(dim)])
        # sort by charges
        perm_qind, leg = leg_unsorted.sort()
        perm_flat = leg_unsorted.perm_flat_from_perm_qind(perm_qind)
        # permute operators accordingly
        for opname in ops:
            ops[opname] = ops[opname][np.ix_(perm_flat, perm_flat)]
    else:
        leg = npc.LegCharge.from_trivial(dim)
    return Site(leg, ['vac'] + [str(n) for n in range(1, dim)], **ops)
