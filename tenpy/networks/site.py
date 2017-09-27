"""Defines a class describing the local physical Hilbert space.

"""
from __future__ import division

import numpy as np

from ..linalg import np_conserved as npc

__all__ = ['Site', 'SpinHalfSite', 'FermionSite', 'SpinHalfFermionSite', 'BosonSite']


class Site(object):
    """Collects necessary information about a single local site of a lattice.

    It has the on-site operators as attributes, e.g. ``self.Id`` is the identy.

    (This class is implemented in :mod:`tenpy.networks.site` but also imported in
    :mod:`tenpy.models.lattice` for convenience.)

    Parameters
    ----------
    leg : :class:`~tenpy.linalg.charges.LegCharge`
        Charges of the physical states, to be used for the physical leg of MPS & co).
    state_labels : None | list of str
        Optionally a label for each local basis states. ``None`` entries are ignored / not set.
    **site_ops :
        Additional keyword arguments of the form ``name=op`` given to :meth:`add_op`.
        The identity operator 'Id' is automatically included.

    Attributes
    ----------
    dim
    onsite_ops
    leg : :class:`~tenpy.linalg.charges.LegCharge`
        Charges of the local basis states.
    state_labels : {str: int}
        (Optional) labels for the local basis states.
    opnames : set
        Labels of all onsite operators (i.e. ``self.op`` exists if ``'op'`` in ``self.opnames``).
    ops : :class:`~tenpy.linalg.charges.Array`
        Onsite operators are added directly as attributes to self.
        For example after ``self.add_op('Sz', Sz)`` you can use ``self.Sz`` for the `Sz` operator.
        All onsite operators have labels ``'p', 'p*'``.
    perm : 1D array
        Index permutation of the physical leg compared to `conserve=None`.

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
    """

    def __init__(self, leg, state_labels=None, **site_ops):
        self.leg = leg
        self.state_labels = dict()
        if state_labels is not None:
            for i, v in enumerate(state_labels):
                if v is not None:
                    self.state_labels[str(v)] = i
        self.opnames = set()
        self.add_op('Id', npc.diag(1., self.leg))
        for name, op in site_ops.iteritems():
            self.add_op(name, op)
        if not hasattr(self, 'perm'):  # default permutation for the local states
            self.perm = np.arange(self.dim)
        self.test_sanity()

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        for lab, ind in self.state_labels.iteritems():
            if not isinstance(lab, str):
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
        op : np.ndarray | :class:`~tenpy.linalg.charges.Array`
            A matrix acting on the local hilbert space representing the local operator.
            Dense numpy arrays are automatically converted to
            :class:`~tenpy.linalg.np_conserved.Array`.
            LegCharges have to be ``[leg, leg.conj()]``.
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
        op.iset_leg_labels(['p', 'p*'])
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
        if new_name in self.opnames:
            raise ValueError("new_name already exists")
        op = getattr(self, old_name)
        setattr(self, new_name, op)
        delattr(self, old_name)
        del self.opnames[old_name]
        self.opnames.add(new_name)

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
        """Return operator of given name.

        Parameters
        ----------
        name : str
            The name of the operator to be returned.
            In case of multiple operator names separated by whitespace,
            we multiply them together to a single on-site operator
            (with the one on the right acting first).

        Returns
        -------
        op : :class:`~tenpy.linalg.np_conserved`
            The operator given by `name`, with labels ``'p', 'p*'``.
        """
        names = name.split()
        op = getattr(self, names[0])
        for name2 in names[1:]:
            op2 = getattr(self, name2)
            op = npc.tensordot(op, op2, axes=['p*', 'p'])
        return op

    def __repr__(self):
        """Debug representation of self"""
        return "<Site, d={dim:d}, ops={ops!r}>".format(dim=self.dim, ops=self.opnames)


# ------------------------------------------------------------------------------
# The most common local sites.


class SpinHalfSite(Site):
    """Spin-1/2 site.

    Local states are ``up``(0) and ``down``(1).
    Local operators are the usual spin-1/2 operators, e.g. ``Sz = [[0.5, 0.], [0., -0.5]]``,
    ``Sx = 0.5*sigma_x`` for the Pauli matrix `sigma_x`.

    ==============  ================================================
    operator        description
    ==============  ================================================
    ``Id``          Identity :math:`\mathbb{1}`
    ``Sx, Sy, Sz``  Spin components :math:`S^{x,y,z}`,
                    equal to half the Pauli matrices.
    ``Sp, Sm``      Spin flips :math:`S^{\pm} = S^{x} \pm i S^{y}`
    ==============  ================================================

    ============== ====  ============================
    `conserve`     qmod  *excluded* onsite operators
    ============== ====  ============================
    ``'Sz'``       [1]   ``Sx, Sy``
    ``'parity'``   [2]   --
    ``None``       []    --
    ============== ====  ============================

    Parameters
    ----------
    conserve : str
        Defines what is conserved, see table above.

    Attributes
    ----------
    conserve : str
        Defines what is conserved, see table above.
    """

    def __init__(self, conserve='Sz'):
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
                leg = npc.LegCharge.from_qflat(chinfo, [1, 0])  # ([1, -1] would need ``qmod=[4]``)
            else:
                leg = npc.LegCharge.from_trivial(2)
        self.conserve = conserve
        super(SpinHalfSite, self).__init__(leg, ['up', 'down'], **ops)

    def __repr__(self):
        """Debug representation of self"""
        return "SpinHalfSite({c!r})".format(c=self.conserve)


class SpinSite(Site):
    """General Spin S site.

    There are `2S+1` local states range from ``down`` (0)  to ``up`` (2S+1),
    corresponding to ``Sz=-S, -S+1, ..., S-1, S``.
    Local operators are the spin-S operators,
    e.g. ``Sz = [[0.5, 0.], [0., -0.5]]``,
    ``Sx = 0.5*sigma_x`` for the Pauli matrix `sigma_x`.

    ==============  ================================================
    operator        description
    ==============  ================================================
    ``Id``          Identity :math:`\mathbb{1}`
    ``Sx, Sy, Sz``  Spin components :math:`S^{x,y,z}`,
                    equal to half the Pauli matrices.
    ``Sp, Sm``      Spin flips :math:`S^{\pm} = S^{x} \pm i S^{y}`
    ==============  ================================================

    ============== ====  ============================
    `conserve`     qmod  *excluded* onsite operators
    ============== ====  ============================
    ``'Sz'``       [1]   ``Sx, Sy``
    ``'parity'``   [2]   --
    ``None``       []    --
    ============== ====  ============================

    Parameters
    ----------
    conserve : str
        Defines what is conserved, see table above.

    Attributes
    ----------
    S : {0.5, 1, 1.5, 2, ...}
        The 2S+1 states range from m = -S, -S+1, ... +S.
    conserve : str
        Defines what is conserved, see table above.
    """

    def __init__(self, S=0.5, conserve='Sz'):
        if conserve not in ['Sz', 'parity', None]:
            raise ValueError("invalid `conserve`: " + repr(conserve))
        self.S = S = float(S)
        d = 2*S+1
        if d <= 1:
            raise ValueError("negative S?")
        if np.rint(d) != d:
            raise ValueError("S is not half-integer or integer")
        d = int(d)
        Sz_diag = -S + np.arange(d)
        Sz = np.diag(Sz_diag)
        Sp = np.zeros([d, d])
        for n in np.arange(d-1):
            # Sp |m> =sqrt( S(S+1)-m(m+1)) |m+1>
            m = n - S
            Sp[n+1, n] = np.sqrt(S*(S+1) - m*(m+1))
        Sm = np.transpose(Sp)
        # Sp = Sx + i Sy, Sm = Sx - i Sy
        Sx = (Sp + Sm) * 0.5
        Sy = (Sm - Sp) * 0.5j
        ops = dict(Sp=Sp, Sm=Sm, Sz=Sz)
        if conserve == 'Sz':
            chinfo = npc.ChargeInfo([1], ['2*Sz'])
            leg = npc.LegCharge.from_qflat(chinfo, np.array(2*Sz_diag, dtype=np.int))
        else:
            ops.update(Sx=Sx, Sy=Sy)
            if conserve == 'parity':
                chinfo = npc.ChargeInfo([2], ['parity'])
                leg = npc.LegCharge.from_qflat(chinfo, np.mod(np.arange(d), 2))
            else:
                leg = npc.LegCharge.from_trivial(d)
        self.conserve = conserve
        names = [None]*d
        names[0] = 'down'
        names[-1] = 'up'
        super(SpinSite, self).__init__(leg, names, **ops)

    def __repr__(self):
        """Debug representation of self"""
        return "SpinSite(S={S!s}, {c!r})".format(S=self.S, c=self.conserve)


class FermionSite(Site):
    r"""Create a :class:`Site` for spin-less fermions.

    Local states are ``empty`` and ``full``.

    .. warning ::
        Using the Jordan-Wigner string (``JW``) is crucial to get correct results,
        otherwise you just describe hardcore bosons!

    ==============  ========================================
    operator        description
    ==============  ========================================
    ``Id``          Identity :math:`\mathbb{1}`
    ``JW``          Sign for the Jordan-Wigner string.
    ``C``           Annihilation operator :math:`c`
    ``Cd``          Creation operator :math:`c^\dagger`
    ``N``           Number operator :math:`n= c^\dagger c`
    ``dN``          :math:`\delta n := n - filling`
    ``dNdN``        :math:`(\delta n)^2`
    ==============  ========================================

    ============== ====  ===============================
    `conserve`     qmod  *exluded* onsite operators
    ============== ====  ===============================
    ``'N'``        [1]   --
    ``'parity'``   [2]   --
    ``None``       []    --
    ============== ====  ===============================

    .. todo ::
        Write userguide for Fermions describing Jordan-Wigner-trafo/-string...
        Handle Jordan-Wigner strings correctly in Coupling-model!

    Parameters
    ----------
    conserve : str
        Defines what is conserved, see table above.
    filling : float
        Average filling. Used to define ``dN``.

    Attributes
    ----------
    conserve : str
        Defines what is conserved, see table above.
    filling : float
        Average filling. Used to define ``dN``.
    """

    def __init__(self, conserve='N', filling=0.5):
        if conserve not in ['N', 'parity', None]:
            raise ValueError("invalid `conserve`: " + repr(conserve))
        JW = np.array([[1., 0.], [0., -1.]])
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
        self.conserve = conserve
        self.filling = filling
        super(FermionSite, self).__init__(leg, ['empty', 'full'], **ops)

    def __repr__(self):
        """Debug representation of self"""
        return "FermionSite({c!r}, {f:f})".format(c=self.conserve, f=self.filling)


class SpinHalfFermionSite(Site):
    r"""Create a :class:`Site` for spinful (spin-1/2) fermions.

    Local states are:
         ``empty``  (vacuum),
         ``up``     (one spin-up electron),
         ``down``   (one spin-down electron), and
         ``full``   (both electrons)

    Local operators can be built from creation operators.

    .. warning ::
        Using the Jordan-Wigner string (``JW``) in the correct way is crucial to get correct
        results, otherwise you just describe hardcore bosons!

    ==============  =============================================================================
    operator        description
    ==============  =============================================================================
    ``Id``          Identity :math:`\mathbb{1}`
    ``JWu``         Sign for the Jordan-Wigner string :math:`(-1)^{n_{\uparrow}}`
    ``JWd``         Sign for the Jordan-Wigner string :math:`(-1)^{n_{\downarrow}}`
    ``Cu``          Annihilation operator spin-up :math:`c_{\uparrow}`
    ``Cud``         Creation operator spin-up :math:`c_{\uparrow}^\dagger`
    ``Cd``          Annihilation operator spin-down :math:`c_{\downarrow}`
    ``Cdd``         Creation operator spin-down :math:`c_{\downarrow}^\dagger`
    ``Nu``          Number operator :math:`n_{\uparrow}= c^{\dagger}_{\uparrow} c_{\uparrow}`
    ``Nd``          Number operator :math:`n_{\downarrow}= c^\dagger_{\downarrow} c_{\downarrow}`
    ``NuNd``        Dotted number operators :math:`n_{\uparrow} n_{\downarrow}`
    ``Ntot``        Total number operator :math:`n_t= n_{\uparrow} + n_{\downarrow}`
    ``dN``          Total number operator compared to the filling :math:`\Delta n = n_t-filling`
    ``Sx, Sy, Sz``  Spin operators :math:`S^{x,y,z}`, in particular
                    :math:`S^z = \frac{1}{2}( n_\uparrow - n_\downarrow )`
    ``Sp, Sm``      Spin flips :math:`S^{\pm} = S^{x} \pm i S^{y}`,
                    e.g. :math:`S^{+} = c_\uparrow^\dagger c_\downarrow`
    ==============  =============================================================================

    The spin operators are defined as :math:`S^\gamma =
    (c_{\uparrow}^\dagger, c_{\downarrow}^\dagger) \sigma^\gamma (c_{\uparrow}, c_{\downarrow})^T`,
    where :math:`\sigma^\gamma` are spin-1/2 matrices (i.e. half the pauli matrices).

    ============= ============= ======= =======================================
    `cons_N`      `cons_Sz`     qmod    *excluded* onsite operators
    ============= ============= ======= =======================================
    ``'N'``       ``'Sz'``      [1, 1]  ``Sx, Sy``
    ``'N'``       ``'parity'``  [1, 2]  --
    ``'N'``       ``None``      [1]     --
    ``'parity'``  ``'Sz'``      [2, 1]  ``Sx, Sy``
    ``'parity'``  ``'parity'``  [2, 2]  --
    ``'parity'``  ``None``      [2]     --
    ``None``      ``'Sz'``      [1]     ``Sx, Sy``
    ``None``      ``'parity'``  [2]     --
    ``None``      ``None``      []      --
    ============= ============= ======= =======================================

    .. todo ::
        (Inherited from FermionSite)
        Write userguide for Fermions describing Jordan-Wigner-trafo/-string...
        Handle Jordan-Wigner strings correctly in Coupling-model!

    .. todo ::
        Check if Jordan-Wigner strings for 4x4 operators are correct.


    Parameters
    ----------
    cons_N : ``'N', 'parity' | None``
        Whether particle number is conserved, c.f. table above.
    cons_Sz : ``'Sz', 'parity' | None``
        Whether spin is conserved, c.f. table above.
    filling : float
        Average filling. Used to define ``dN``.

    Attributes
    ----------
    conserve : str | ``None``
        Defines what is conserved, see table above.
    filling : float
        Average filling. Used to define ``dN``.  """
    def __init__(self, cons_N='N', cons_Sz='Sz', filling=1.):
        if cons_N not in ['N', 'parity', None]:
            raise ValueError("invalid `cons_N`: " + repr(cons_N))
        if cons_Sz not in ['Sz', 'parity', None]:
            raise ValueError("invalid `cons_Sz`: " + repr(cons_Sz))
        d = 4
        states = ['empty', 'up', 'down', 'full']
        # 0) Build the operators.
        Nu_diag = np.array([0., 1., 0., 1.], dtype=np.float)
        Nd_diag = np.array([0., 0., 1., 1.], dtype=np.float)
        Nu = np.diag(Nu_diag)
        Nd = np.diag(Nd_diag)
        Ntot = np.diag(Nu_diag + Nd_diag)
        dN = np.diag(Nu_diag + Nd_diag - filling)
        NuNd = np.diag(Nu_diag * Nd_diag)
        JWu = np.diag(1. - 2 * Nu_diag)     # (-1)^Nu
        JWd = np.diag(1. - 2 * Nd_diag)     # (-1)^Nd
        JW = JWu * JWd                      # (-1)^{Nu+Nd}

        Cu = np.zeros((d, d))
        Cu[0, 1] = Cu[2, 3] = 1
        # For spin-down annihilation operator: include a Jordan-Wigner string JWu
        # this ensures that Cud.Cd = - Cd.Cud
        # c.f. the chapter on the Jordan-Wigner trafo in the userguide
        Cd_noJW = np.zeros((d, d))
        Cd_noJW[0, 2] = Cd_noJW[1, 3] = 1
        Cd = np.dot(JWu, Cd_noJW)           # (don't do this for spin-up...)
        Cud = np.transpose(Cu)
        Cdd = np.transpose(Cd)

        # spin operators are defined as  (Cud, Cdd) S^gamma (Cu, Cd)^T,
        # where S^gamma is the 2x2 matrix for spin-half
        Sz = np.diag(0.5 * (Nu_diag - Nd_diag))
        Sp = np.dot(Cud, Cd)
        Sm = np.dot(Cdd, Cu)
        Sx = 0.5*(Sp + Sm)
        Sy = -0.5j*(Sp - Sm)

        ops = dict(JW=JW, JWu=JWu, JWd=JWd,
                   Cu=Cu, Cud=Cud, Cd=Cd, Cdd=Cdd,
                   Nu=Nu, Nd=Nd, Ntot=Ntot, NuNd=NuNd, dN=dN,
                   Sx=Sx, Sy=Sy, Sz=Sz, Sp=Sp, Sm=Sm)  # yapf: disable

        # handle charges
        qmod = []
        qnames = []
        charges = []
        if cons_N == 'N':
            qnames.append('N')
            qmod.append(1)
            charges.append([0, 1, 1, 2])
        elif cons_N == 'parity':
            qnames.append('N')
            qmod.append(2)
            charges.append([0, 1, 1, 0])
        if cons_Sz == 'Sz':
            qnames.append('Sz')
            qmod.append(1)
            charges.append([0, 1, -1, 0])
            del ops['Sx']
            del ops['Sy']
        elif cons_Sz == 'parity':
            qnames.append('Sz')
            qmod.append(4)    # difference between up and down is 2!
            charges.append([0, 1, 3, 0])  # == [0, 1, -1, 0] mod 4
            # chosen s.t. Cu, Cd have well-defined charges!

        if len(qmod) == 0:
            leg = npc.LegCharge.from_trivial(d)
        else:
            if len(qmod) == 1:
                charges = charges[0]
            else:  # len(charges) == 2: need to transpose
                charges = [[q1, q2] for q1, q2 in zip(charges[0], charges[1])]
            chinfo = npc.ChargeInfo(qmod, qnames)
            leg_unsorted = npc.LegCharge.from_qflat(chinfo, charges)
            # sort by charges
            perm_qind, leg = leg_unsorted.sort()
            perm_flat = leg_unsorted.perm_flat_from_perm_qind(perm_qind)
            self.perm = perm_flat
            # permute operators accordingly
            for opname in ops:
                ops[opname] = ops[opname][np.ix_(perm_flat, perm_flat)]
            # and the states
            states = [states[i] for i in perm_flat]
        self.cons_N = cons_N
        self.cons_Sz = cons_Sz
        super(SpinHalfFermionSite, self).__init__(leg, states, **ops)

    def __repr__(self):
        """Debug representation of self"""
        return "SpinHalfFermionSite({c!r})".format(c=self.conserve)


class BosonSite(Site):
    r"""Create a :class:`Site` for up to `Nmax` bosons.

    Local states are ``vac, 1, 2, ... , Nc``.
    (Exception: for parity conservation, we sort as ``vac, 2, 4, ..., 1, 3, 5, ...``.)

    ==============  ========================================
    operator        description
    ==============  ========================================
    ``Id``          Identity :math:`\mathbb{1}`
    ``B``           Annihilation operator :math:`b`
    ``Bd``          Creation operator :math:`b^\dagger`
    ``N``           Number operator :math:`n= b^\dagger b`
    ``NN``          :math:`n^2`
    ``dN``          :math:`\delta n := n - filling`
    ``dNdN``        :math:`(\delta n)^2`
    ``P``           Parity :math:`Id - 2 (n \mod 2)`.
    ==============  ========================================

    ============== ====  ==================================
    `conserve`     qmod  *excluded* onsite operators
    ============== ====  ==================================
    ``'N'``        [1]   --
    ``'parity'``   [2]   --
    ``None``       []    --
    ============== ====  ==================================

    Parameters
    ----------
    Nmax : int
        Cutoff defining the maximum number of bosons per site.
        The default ``Nmax=1`` describes hard-core bosons.
    conserve : str
        Defines what is conserved, see table above.
    filling : float
        Average filling. Used to define ``dN``.

    Attributes
    ----------
    conserve : str
        Defines what is conserved, see table above.
    filling : float
        Average filling. Used to define ``dN``.
    """

    def __init__(self, Nmax=1, conserve='N', filling=0.):
        if conserve not in ['N', 'parity', None]:
            raise ValueError("invalid `conserve`: " + repr(conserve))
        dim = Nmax + 1
        states = ['vac'] + [str(n) for n in range(1, dim)]
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
            self.perm = perm_flat
            # permute operators accordingly
            for opname in ops:
                ops[opname] = ops[opname][np.ix_(perm_flat, perm_flat)]
            # and the states
            states = [states[i] for i in perm_flat]
        else:
            leg = npc.LegCharge.from_trivial(dim)
        self.Nmax = Nmax
        self.conserve = conserve
        self.filling = filling
        super(BosonSite, self).__init__(leg, states, **ops)

    def __repr__(self):
        """Debug representation of self"""
        return "BosonSite({N:d}, {c!r}, {f:f})".format(
            N=self.Nmax, c=self.conserve, f=self.filling)
