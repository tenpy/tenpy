"""Defines a class describing the local physical Hilbert space.

The :class:`Site` is the prototype, read it's docstring.
"""
# Copyright 2018 TeNPy Developers

import numpy as np
import itertools
import copy

from ..linalg import np_conserved as npc
from ..tools.misc import inverse_permutation

__all__ = [
    'Site', 'GroupedSite', 'group_sites', 'multi_sites_combine_charges', 'SpinHalfSite',
    'SpinSite', 'FermionSite', 'SpinHalfFermionSite', 'BosonSite'
]


class Site:
    """Collects necessary information about a single local site of a lattice.

    This class defines what the local basis states are: it provides the :attr:`leg`
    defining the charges of the physical leg for this site.
    Moreover, it stores (local) on-site operators, which are directly available as attribute,
    e.g., ``self.Sz`` is the Sz operator for the :class:`SpinSite`.
    Alternatively, operators can be obained with :meth:`get_op`.
    The operator names ``Id`` and ``JW`` are reserved for the identy and Jordan-Wigner strings.

    .. warning ::
        The order of the local basis can change depending on the charge conservation!
        This is a *necessary* feature since we need to sort the basis by charges for efficiency.
        We use the :attr:`state_labels` and :attr:`perm` to keep track of these permutations.

    Parameters
    ----------
    leg : :class:`~tenpy.linalg.charges.LegCharge`
        Charges of the physical states, to be used for the physical leg of MPS.
    state_labels : None | list of str
        Optionally a label for each local basis states. ``None`` entries are ignored / not set.
    **site_ops :
        Additional keyword arguments of the form ``name=op`` given to :meth:`add_op`.
        The identity operator ``'Id'`` is automatically included.
        If no ``'JW'`` for the Jordan-Wigner string is given,
        ``'JW'`` is set as an alias to ``'Id'``.

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
        Note that :meth:`get_op` allows arbitrary concatenations of them.
    need_JW_string : set
        Labels of all onsite operators that need a Jordan-Wigner string.
    ops : :class:`~tenpy.linalg.np_conserved.Array`
        Onsite operators are added directly as attributes to self.
        For example after ``self.add_op('Sz', Sz)`` you can use ``self.Sz`` for the `Sz` operator.
        All onsite operators have labels ``'p', 'p*'``.
    perm : 1D array
        Index permutation of the physical leg compared to `conserve=None`,
        i.e. ``OP_conserved = OP_nonconserved[np.ix_(perm,perm)]`` and
        ``perm[state_labels_conserved["some_state"]] == state_labels_nonconserved["some_state"]``.
    JW_exponent : 1D array
        Exponents of the ``'JW'`` operator, such that
        ``self.JW.to_ndarray() = np.diag(np.exp(1.j*np.pi* JW_exponent))``

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
    >>> print(site.Splus.to_ndarray())
    array([[ 0.,  1.],
           [ 0.,  0.]])
    >>> print(site.get_op('Sminus').to_ndarray())
    array([[ 0.,  0.],
           [ 1.,  0.]])
    >>> print(site.get_op('Splus Sminus').to_ndarray())
    array([[ 1.,  0.],
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
        self.need_JW_string = set(['JW'])
        self.add_op('Id', npc.diag(1., self.leg))
        for name, op in site_ops.items():
            self.add_op(name, op)
        if not hasattr(self, 'perm'):  # default permutation for the local states
            self.perm = np.arange(self.dim)
        if 'JW' not in self.opnames:
            # include trivial `JW` to allow combinations
            # of bosonic and fermionic sites in an MPS
            self.add_op('JW', self.Id)
        self.test_sanity()

    def change_charge(self, new_leg_charge=None, permute=None):
        """Change the charges of the site (in place).

        Parameters
        ----------
        new_leg_charge : :class:`LegCharge` | None
            The new charges to be used. If ``None``, use trivial charges.
        permute : ndarray | None
            The permuation applied to the physical leg,
            which gets used to adjust :attr:`state_labels` and :attr:`perm`.
            If you sorted the previous leg with ``perm_qind, new_leg_charge = leg.sort()``,
            use ``leg.perm_flat_from_perm_qind(perm_qind)``.
            Ignored if ``None``.
        """
        if new_leg_charge is None:
            new_leg_charge = npc.LegCharge.from_trivial(self.dim)
        self.leg = new_leg_charge
        if permute is not None:
            permute = np.asarray(permute, dtype=np.intp)
            inv_perm = inverse_permutation(permute)
            self.perm = self.perm[permute]
            state_labels = self.state_labels.copy()
            for label in state_labels:
                self.state_labels[label] = inv_perm[state_labels[label]]
        for opname in self.opnames.copy():
            op = self.get_op(opname).to_ndarray()
            need_JW = opname in self.need_JW_string
            self.remove_op(opname)
            if permute is not None:
                op = op[np.ix_(permute, permute)]
            self.add_op(opname, op, need_JW)
        # done

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        for lab, ind in self.state_labels.items():
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
        for op in self.need_JW_string:
            assert op in self.opnames
        np.testing.assert_array_almost_equal(np.diag(np.exp(1.j * np.pi * self.JW_exponent)),
                                             self.JW.to_ndarray(), 15)

    @property
    def dim(self):
        """Dimension of the local Hilbert space"""
        return self.leg.ind_len

    @property
    def onsite_ops(self):
        """Dictionary of on-site operators for iteration.

        (single operators are accessible as attributes.)"""
        return dict([(name, getattr(self, name)) for name in sorted(self.opnames)])

    def add_op(self, name, op, need_JW=False):
        """Add one on-site operators

        Parameters
        ----------
        name : str
            A valid python variable name, used to label the operator.
            The name under which `op` is added as attribute to self.
        op : np.ndarray | :class:`~tenpy.linalg.np_conserved.Array`
            A matrix acting on the local hilbert space representing the local operator.
            Dense numpy arrays are automatically converted to
            :class:`~tenpy.linalg.np_conserved.Array`.
            LegCharges have to be ``[leg, leg.conj()]``.
            We set labels ``'p', 'p*'``.
        need_JW : bool
            Wheter the operator needs a Jordan-Wigner string.
            If ``True``, the function adds `name` to :attr:`need_JW_string`.
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
        if need_JW:
            self.need_JW_string.add(name)
        if name == 'JW':
            self.JW_exponent = np.angle(np.real_if_close(np.diag(op.to_ndarray()))) / np.pi

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
        need_JW = old_name in self.need_JW_string
        self.remove_op(old_name)
        setattr(self, new_name, op)
        self.opnames.add(new_name)
        if need_JW:
            self.need_JW_string.add(new_name)
        if new_name == 'JW':
            self.JW_exponent = np.real_if_close(np.angle(np.diag(op.to_ndarray())) / np.pi)

    def remove_op(self, name):
        """Remove an added operator.

        Parameters
        ----------
        name : str
            The name of the operator to be removed.
        """
        self.opnames.remove(name)
        delattr(self, name)
        self.need_JW_string.discard(name)

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
        except ValueError:
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
            If name already was an npc Array, it's directly returned.
        """
        names = name.split()
        op = getattr(self, names[0])
        for name2 in names[1:]:
            op2 = getattr(self, name2)
            op = npc.tensordot(op, op2, axes=['p*', 'p'])
        return op

    def op_needs_JW(self, name):
        """Wheter an (composite) onsite operator needs a Jordan-Wigner string.

        Parameters
        ----------
        name : str
            The name of the operator, as in :meth:`get_op`.

        Returns
        -------
        needs_JW : bool
            Wheter the operator needs a Jordan-Wigner string, judging from :attr:`need_JW_string`.
        """
        names = name.split()
        need_JW = bool(names[0] in self.need_JW_string)
        for op in names[1:]:
            if op in self.need_JW_string:
                need_JW = not need_JW  # == need_JW xor (op in self.need_JW_string)
        return need_JW

    def valid_opname(self, name):
        """Check wheter 'name' labels a valid onsite-operator.

        Parameters
        ----------
        name : str
            Label for the operator. Can be multiple operator(labels) separated by whitespace,
            indicating that they should  be multiplied together.

        Returns
        -------
        valid : bool
            ``True`` if `name` is a valid argument to :meth:`get_op`.
        """
        for name2 in name.split():
            if name2 not in self.opnames:
                return False
        return True

    def __repr__(self):
        """Debug representation of self"""
        return "<Site, d={dim:d}, ops={ops!r}>".format(dim=self.dim, ops=self.opnames)


class GroupedSite(Site):
    """Group two or more :class:`Site` into a larger one.

    A typical use-case is that you want a NearestNeigborModel for TEBD although you have
    next-nearest neighbor interactions: you just double your local Hilbertspace to consist of
    two original sites.
    Note that this is a 'hack' at the cost of other things (e.g., measurements of 'local'
    operators) getting more complicated/computationally expensive.

    If the individual sites indicate fermionic operators (with entries in `needs_JW_string`),
    we construct the new on-site oerators of `site1` to include the JW string of `site0`,
    i.e., we use the Kronecker product of ``[JW, op]`` instead of ``[Id, op]`` if necessary
    (but always ``[op, Id]``).
    In that way the onsite operators of this DoubleSite automatically fulfill the
    expected commutation relations. See also :doc:`/intro_JordanWigner`.

    Parameters
    ----------
    sites : list of :class:`Site`
        The individual sites being grouped together. Copied before use if ``charges!='same'``.
    labels :
        Include the Kronecker product of the each onsite operator `op` on ``sites[i]`` and
        identities on other sites with the name ``opname+labels[i]``.
        Similarly, set state labels for ``' '.join(state[i]+'_'+labels[i])``.
        Defaults to ``[str(i) for i in range(n_sites)]``, which for example grouping two SpinSites
        gives operators name like ``"Sz0"`` and sites labels like ``'up_0 down_1'``.
    charges : ``'same' | 'drop' | 'independent'``
        How to handle charges, defaults to 'same'.
        ``'same'`` means that all `sites` have the same `ChargeInfo`, and the total charge
        is the sum of the charges on the individual `sites`.
        ``'independent'`` means that the `sites` have possibly different `ChargeInfo`,
        and the charges are conserved separately, i.e., we have `n_sites` conserved charges.
        For ``'drop'``, we drop any charges, such that the remaining legcharges are trivial.

    Attributes
    ----------
    n_sites : int
        The number of sites grouped together, i.e. ``len(sites)``.
    sites : list of :class:`Site`
        The sites grouped together into self.
    labels: list of str
        The labels using which the single-site operators are added during construction.
    """

    def __init__(self, sites, labels=None, charges='same'):
        self.n_sites = n_sites = len(sites)
        self.sites = sites
        self.labels = labels
        assert n_sites > 0
        if labels is None:
            labels = [str(i) for i in range(n_sites)]
        if charges == 'same':
            pass  # nothing to do
        elif charges == 'drop':
            legs = [npc.LegCharge.from_drop_charge(sites[0].leg)]
            chinfo = legs[0].chinfo
            for site in sites[1:]:
                legs.append(npc.LegCharge.from_drop_charge(sites[0].leg, chargeinfo=chinfo))
        elif charges == 'independent':
            # charges are separately conserved
            legs = []
            for i in range(n_sites):
                d = sites[i].dim
                # trivial charges
                legs_triv = [npc.LegCharge.from_trivial(d, s.leg.chinfo) for s in sites]
                legs_triv[i] = sites[i].leg  # except on site i
                chinfo = None if i == 0 else legs[0].chinfo
                leg = npc.LegCharge.from_add_charge(legs_triv, chinfo)  # combine the charges
                legs.append(leg)
        else:
            raise ValueError("Unknown option for `charges`: " + repr(charges))
        if charges != 'same':
            sites = [copy.copy(s) for s in sites]  # avoid modifying the existing sites.
            # sort legs
            for i in range(n_sites):
                perm_qind, leg_s = legs[i].sort()
                sites[i].change_charge(leg_s, legs[i].perm_flat_from_perm_qind(perm_qind))
        chinfo = sites[0].leg.chinfo
        for s in sites[1:]:
            assert s.leg.chinfo == chinfo  # check for compatibility
        legs = [s.leg for s in sites]
        pipe = npc.LegPipe(legs)
        self.leg = pipe  # needed in kroneckerproduct
        JW_all = self.kroneckerproduct([s.JW for s in sites])

        # initialize Site
        Site.__init__(self, pipe, None, JW=JW_all)

        # set state labels
        for states_labels in itertools.product(*[s.state_labels.items() for s in sites]):
            inds = [v for k, v in states_labels]  # values of the dictionaries
            ind_pipe = pipe.map_incoming_flat(inds)
            label = ' '.join([st + '_' + lbl for (st, idx), lbl in zip(states_labels, labels)])
            self.state_labels[label] = ind_pipe
        # add remaining operators
        Ids = [s.Id for s in sites]
        JW_Ids = Ids[:]  # in the following loop equivalent to [JW, JW, ... , Id, Id, ...]
        for i in range(n_sites):
            site = sites[i]
            for opname, op in site.onsite_ops.items():
                if opname == 'Id':
                    continue
                need_JW = opname in site.need_JW_string
                ops = JW_Ids if need_JW else Ids
                ops[i] = op
                self.add_op(opname + labels[i], self.kroneckerproduct(ops), need_JW)
                Ids[i] = site.Id
                JW_Ids[i] = site.JW
        # done

    def kroneckerproduct(self, ops):
        r"""Return the Kronecker product :math:`op0 \otimes op1` of local operators.

        Parameters
        ----------
        ops : list of :class:`~tenpy.linalg.np_conserved.Array`
            One operator (or operator name) on each of the ungrouped sites.
            Each operator should have labels ``['p', 'p*']``.

        Returns
        -------
        prod : :class:`~tenpy.linalg.np_conserved.Array`
            Kronecker product :math:`ops[0] \otimes ops[1] \otimes \cdots`,
            with labels ``['p', 'p*']``.
        """
        sites = self.sites
        op = ops[0].transpose(['p', 'p*'])
        for op2 in ops[1:]:
            op = npc.outer(op, op2.transpose(['p', 'p*']))
        combine = [list(range(0, 2 * self.n_sites - 1, 2)), list(range(1, 2 * self.n_sites, 2))]
        pipe = self.leg
        op = op.combine_legs(combine, pipes=[pipe, pipe.conj()])
        return op.iset_leg_labels(['p', 'p*'])


def group_sites(sites, n=2, labels=None, charges='same'):
    """Given a list of sites, group each `n` sites together.

    Parameters
    ----------
    sites : list of :class:`Site`
        The sites to be grouped together.
    n : int
        We group each `n` consecutive sites from `sites` together in a :class:`GroupedSite`.
    labels, charges :
        See :class:`GroupedSites`.

    Returns
    -------
    grouped_sites : list of :class:`GroupedSite`
        The grouped sites. Has length ``(len(sites)-1)//n + 1``.
    """
    grouped_sites = []
    if labels is None:
        labels = [str(i) for i in range(n)]
    for i in range(0, len(sites), n):
        group = sites[i:i + n]
        s = GroupedSite(group, labels[:len(group)], charges)
        grouped_sites.append(s)
    return grouped_sites


def multi_sites_combine_charges(sites, same_charges=[]):
    """Adjust the charges of the given sites (in place) such that they can be used together.

    When we want to contract tensors corresponding to different :class:`Site` instances,
    these sites need to share a single :class:`~tenpy.linalg.charges.ChargeInfo`.
    This function adjusts the charges of these sites such that they can be used together.

    Parameters
    ----------
    sites : list of :class:`Site`
        The sites to be combined. Modified **in place**.
    same_charges : ``[[(int, int|str), (int, int|str), ...], ...]``
        Defines which charges actually are the same, i.e. their quantum numbers are added up.
        Each charge is specified by a tuple ``(s, i)= (int, int|str)``, where `s` gives the index
        of the site within ``sites`` and `i` the index or name of the charge in the
        :class:`~tenpy.linalg.charges.ChargeInfo` of this site.

    Returns
    -------
    perms : list of ndarray
        For each site the permutation performed on the physical leg to sort by charges.

    Examples
    --------
    >>> ferm = SpinHalfFermionSite(cons_N='N', cons_Sz='Sz')
    >>> spin = SpinSite(1.0, 'Sz')
    >>> ferm.leg.chinfo is spin.leg.chinfo
    False
    >>> print(spin.leg)
    +1
    0 [[-1]
    1  [ 1]]
    2
    >>> multi_sites_combine_charges([ferm, spin], same_charges=[[(0, 'Sz'), (1, 0)]])
    [array([0, 1, 2, 3]), array([0, 1])]
    >>> # no permutations where needed
    >>> ferm.leg.chinfo is spin.leg.chinfo
    True
    >> ferm.leg.chinfo.names
    ['N', 'Sz']
    >>> print(spin.leg)
    +1
    0 [[ 0 -1]
    1  [ 0  1]]
    2
    """
    # parse same_charges argument
    same_charges = list(same_charges)  # need to modify elements...
    same_charges_flat = []
    for j in range(len(same_charges)):
        same_charges_j = []
        for s, i in same_charges[j]:
            if isinstance(i, str):  # map string to ints
                i = sites[s].leg.chinfo.names.index(i)
            i = int(i)  # should be integer now...
            same_charges_j.append((s, i))
            same_charges_flat.append((s, i))
        same_charges[j] = same_charges_j
    if len(same_charges_flat) != len(set(same_charges_flat)):
        raise ValueError("Can't have duplicates in same_charges!")
    # find out which charges we keep
    keep_charges = []  # list of (s, i) which appear in the new ChargeInfo
    map_charges = {}  # dict (s, i)->(s,i): those not appearing in keep_charges to the one in it
    for s, site in enumerate(sites):
        for i in range(site.leg.chinfo.qnumber):
            keep_charges.append((s, i))  # first all, remove some below
    for same_charges_j in same_charges:
        s0, i0 = same_charges_j[0]
        for s, i in same_charges_j[1:]:
            idx = keep_charges.index((s, i))
            del keep_charges[idx]
            map_charges[(s, i)] = (s0, i0)
    # define common ChargeInfo class
    qnumber = len(keep_charges)
    names = [sites[s].leg.chinfo.names[i] for (s, i) in keep_charges]
    mod = [sites[s].leg.chinfo.mod[i] for (s, i) in keep_charges]
    chinfo = npc.ChargeInfo(mod, names)
    # now define the new legs and update the charges of the sites
    perms = []
    for s, site in enumerate(sites):
        old_qflat = site.leg.to_qflat()
        new_qflat = np.zeros((site.leg.ind_len, qnumber), old_qflat.dtype)
        for old_i in range(site.leg.chinfo.qnumber):
            if (s, old_i) in map_charges:
                new_i = keep_charges.index(map_charges[(s, old_i)])
            else:
                new_i = keep_charges.index((s, old_i))
            new_qflat[:, new_i] = old_qflat[:, old_i]
        # other charges are 0 = trivial
        leg_unsorted = npc.LegCharge.from_qflat(chinfo, new_qflat, site.leg.qconj)
        perm_qind, leg = leg_unsorted.sort()
        perm_flat = leg_unsorted.perm_flat_from_perm_qind(perm_qind)
        perms.append(perm_flat)
        site.change_charge(leg, perm_flat)
    return perms


# ------------------------------------------------------------------------------
# The most common local sites.


class SpinHalfSite(Site):
    r"""Spin-1/2 site.

    Local states are ``up`` (0) and ``down`` (1).
    Local operators are the usual spin-1/2 operators, e.g. ``Sz = [[0.5, 0.], [0., -0.5]]``,
    ``Sx = 0.5*sigma_x`` for the Pauli matrix `sigma_x`.

    =========================== ================================================
    operator                    description
    =========================== ================================================
    ``Id, JW``                  Identity :math:`\mathbb{1}`
    ``Sx, Sy, Sz``              Spin components :math:`S^{x,y,z}`,
                                equal to half the Pauli matrices.
    ``Sigmax, Sigmay, Sigmaz``  Pauli matrices :math:`\sigma^{x,y,z}`
    ``Sp, Sm``                  Spin flips :math:`S^{\pm} = S^{x} \pm i S^{y}`
    =========================== ================================================

    ============== ====  ============================
    `conserve`     qmod  *excluded* onsite operators
    ============== ====  ============================
    ``'Sz'``       [1]   ``Sx, Sy, Sigmax, Sigmay``
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
        Site.__init__(self, leg, ['up', 'down'], **ops)
        # further alias for state labels
        self.state_labels['-0.5'] = self.state_labels['down']
        self.state_labels['0.5'] = self.state_labels['up']
        # Add Pauli matrices
        if conserve != 'Sz':
            self.add_op('Sigmax', 2. * self.Sx)
            self.add_op('Sigmay', 2. * self.Sy)
        self.add_op('Sigmaz', 2. * self.Sz)

    def __repr__(self):
        """Debug representation of self"""
        return "SpinHalfSite({c!r})".format(c=self.conserve)


class SpinSite(Site):
    r"""General Spin S site.

    There are `2S+1` local states range from ``down`` (0)  to ``up`` (2S+1),
    corresponding to ``Sz=-S, -S+1, ..., S-1, S``.
    Local operators are the spin-S operators,
    e.g. ``Sz = [[0.5, 0.], [0., -0.5]]``,
    ``Sx = 0.5*sigma_x`` for the Pauli matrix `sigma_x`.

    ==============  ================================================
    operator        description
    ==============  ================================================
    ``Id, JW``      Identity :math:`\mathbb{1}`
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
        d = 2 * S + 1
        if d <= 1:
            raise ValueError("negative S?")
        if np.rint(d) != d:
            raise ValueError("S is not half-integer or integer")
        d = int(d)
        Sz_diag = -S + np.arange(d)
        Sz = np.diag(Sz_diag)
        Sp = np.zeros([d, d])
        for n in np.arange(d - 1):
            # Sp |m> =sqrt( S(S+1)-m(m+1)) |m+1>
            m = n - S
            Sp[n + 1, n] = np.sqrt(S * (S + 1) - m * (m + 1))
        Sm = np.transpose(Sp)
        # Sp = Sx + i Sy, Sm = Sx - i Sy
        Sx = (Sp + Sm) * 0.5
        Sy = (Sm - Sp) * 0.5j
        # Note: For S=1/2, Sy might look wrong compared to the Pauli matrix or SpinHalfSite.
        # Don't worry, I'm 99.99% sure it's correct (J. Hauschild)
        # The reason it looks wrong is simply that this class orders the states as ['down', 'up'],
        # while the usual spin-1/2 convention is ['up', 'down'], as you can also see if you look
        # at the Sz entries...
        # (The commutation relations are checked explicitly in `tests/test_site.py`)
        ops = dict(Sp=Sp, Sm=Sm, Sz=Sz)
        if conserve == 'Sz':
            chinfo = npc.ChargeInfo([1], ['2*Sz'])
            leg = npc.LegCharge.from_qflat(chinfo, np.array(2 * Sz_diag, dtype=np.int))
        else:
            ops.update(Sx=Sx, Sy=Sy)
            if conserve == 'parity':
                chinfo = npc.ChargeInfo([2], ['parity'])
                leg = npc.LegCharge.from_qflat(chinfo, np.mod(np.arange(d), 2))
            else:
                leg = npc.LegCharge.from_trivial(d)
        self.conserve = conserve
        names = [str(i) for i in np.arange(-S, S + 1, 1.)]
        Site.__init__(self, leg, names, **ops)
        self.state_labels['down'] = self.state_labels[names[0]]
        self.state_labels['up'] = self.state_labels[names[-1]]

    def __repr__(self):
        """Debug representation of self"""
        return "SpinSite(S={S!s}, {c!r})".format(S=self.S, c=self.conserve)


class FermionSite(Site):
    r"""Create a :class:`Site` for spin-less fermions.

    Local states are ``empty`` and ``full``.

    .. warning ::
        Using the Jordan-Wigner string (``JW``) is crucial to get correct results,
        otherwise you just describe hardcore bosons!
        Further details in :doc:`/intro_JordanWigner`.

    ==============  ===================================================================
    operator        description
    ==============  ===================================================================
    ``Id``          Identity :math:`\mathbb{1}`
    ``JW``          Sign for the Jordan-Wigner string.
    ``C``           Annihilation operator :math:`c` (up to 'JW'-string left of it)
    ``Cd``          Creation operator :math:`c^\dagger` (up to 'JW'-string left of it)
    ``N``           Number operator :math:`n= c^\dagger c`
    ``dN``          :math:`\delta n := n - filling`
    ``dNdN``        :math:`(\delta n)^2`
    ==============  ===================================================================

    ============== ====  ===============================
    `conserve`     qmod  *exluded* onsite operators
    ============== ====  ===============================
    ``'N'``        [1]   --
    ``'parity'``   [2]   --
    ``None``       []    --
    ============== ====  ===============================

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
        Site.__init__(self, leg, ['empty', 'full'], **ops)
        # specify fermionic operators
        self.need_JW_string |= set(['C', 'Cd', 'JW'])

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
    ``JW``          Sign for the Jordan-Wigner string :math:`(-1)^{n_{\uparrow}+n_{\downarrow}}`
    ``JWu``         Partial sign for the Jordan-Wigner string :math:`(-1)^{n_{\uparrow}}`
    ``JWd``         Partial sign for the Jordan-Wigner string :math:`(-1)^{n_{\downarrow}}`
    ``Cu``          Annihilation operator spin-up :math:`c_{\uparrow}`
                    (up to 'JW'-string on sites left of it).
    ``Cdu``         Creation operator spin-up :math:`c^\dagger_{\uparrow}`
                    (up to 'JW'-string on sites left of it).
    ``Cd``          Annihilation operator spin-down :math:`c_{\downarrow}`
                    (up to 'JW'-string on sites left of it).
                    Includes ``JWu`` such that it anti-commutes onsite with ``Cu, Cdu``.
    ``Cdd``         Creation operator spin-down :math:`c^\dagger_{\downarrow}`
                    (up to 'JW'-string on sites left of it).
                    Includes ``JWu`` such that it anti-commutes onsite with ``Cu, Cdu``.
    ``Nu``          Number operator :math:`n_{\uparrow}= c^\dagger_{\uparrow} c_{\uparrow}`
    ``Nd``          Number operator :math:`n_{\downarrow}= c^\dagger_{\downarrow} c_{\downarrow}`
    ``NuNd``        Dotted number operators :math:`n_{\uparrow} n_{\downarrow}`
    ``Ntot``        Total number operator :math:`n_t= n_{\uparrow} + n_{\downarrow}`
    ``dN``          Total number operator compared to the filling :math:`\Delta n = n_t-filling`
    ``Sx, Sy, Sz``  Spin operators :math:`S^{x,y,z}`, in particular
                    :math:`S^z = \frac{1}{2}( n_\uparrow - n_\downarrow )`
    ``Sp, Sm``      Spin flips :math:`S^{\pm} = S^{x} \pm i S^{y}`,
                    e.g. :math:`S^{+} = c^\dagger_\uparrow c_\downarrow`
    ==============  =============================================================================

    The spin operators are defined as :math:`S^\gamma =
    (c^\dagger_{\uparrow}, c^\dagger_{\downarrow}) \sigma^\gamma (c_{\uparrow}, c_{\downarrow})^T`,
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
        Check if Jordan-Wigner strings for 4x4 operators are correct.

    Parameters
    ----------
    cons_N : ``'N' | 'parity' | None``
        Whether particle number is conserved, c.f. table above.
    cons_Sz : ``'Sz' | 'parity' | None``
        Whether spin is conserved, c.f. table above.
    filling : float
        Average filling. Used to define ``dN``.

    Attributes
    ----------
    cons_N : ``'N' | 'parity' | None``
        Whether particle number is conserved, c.f. table above.
    cons_Sz : ``'Sz' | 'parity' | None``
        Whether spin is conserved, c.f. table above.
    filling : float
        Average filling. Used to define ``dN``.
    """

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
        JWu = np.diag(1. - 2 * Nu_diag)  # (-1)^Nu
        JWd = np.diag(1. - 2 * Nd_diag)  # (-1)^Nd
        JW = JWu * JWd  # (-1)^{Nu+Nd}

        Cu = np.zeros((d, d))
        Cu[0, 1] = Cu[2, 3] = 1
        Cdu = np.transpose(Cu)
        # For spin-down annihilation operator: include a Jordan-Wigner string JWu
        # this ensures that Cdu.Cd = - Cd.Cdu
        # c.f. the chapter on the Jordan-Wigner trafo in the userguide
        Cd_noJW = np.zeros((d, d))
        Cd_noJW[0, 2] = Cd_noJW[1, 3] = 1
        Cd = np.dot(JWu, Cd_noJW)  # (don't do this for spin-up...)
        Cdd = np.transpose(Cd)

        # spin operators are defined as  (Cdu, Cdd) S^gamma (Cu, Cd)^T,
        # where S^gamma is the 2x2 matrix for spin-half
        Sz = np.diag(0.5 * (Nu_diag - Nd_diag))
        Sp = np.dot(Cdu, Cd)
        Sm = np.dot(Cdd, Cu)
        Sx = 0.5 * (Sp + Sm)
        Sy = -0.5j * (Sp - Sm)

        ops = dict(JW=JW, JWu=JWu, JWd=JWd,
                   Cu=Cu, Cdu=Cdu, Cd=Cd, Cdd=Cdd,
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
            qmod.append(4)  # difference between up and down is 2!
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
        self.filling = filling
        Site.__init__(self, leg, states, **ops)
        # specify fermionic operators
        self.need_JW_string |= set(['Cu', 'Cdu', 'Cd', 'Cdd', 'JWu', 'JWd', 'JW'])

    def __repr__(self):
        """Debug representation of self"""
        return "SpinHalfFermionSite({cN!r}, {cS!r}, {f:f})".format(cN=self.cons_N,
                                                                   cS=self.cons_Sz,
                                                                   f=self.filling)


class BosonSite(Site):
    r"""Create a :class:`Site` for up to `Nmax` bosons.

    Local states are ``vac, 1, 2, ... , Nc``.
    (Exception: for parity conservation, we sort as ``vac, 2, 4, ..., 1, 3, 5, ...``.)

    ==============  ========================================
    operator        description
    ==============  ========================================
    ``Id, JW``      Identity :math:`\mathbb{1}`
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
        states = [str(n) for n in range(0, dim)]
        if dim < 2:
            raise ValueError("local dimension should be larger than 1....")
        B = np.zeros([dim, dim], dtype=np.float)  # destruction/annihilation operator
        for n in range(1, dim):
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
        Site.__init__(self, leg, states, **ops)
        self.state_labels['vac'] = self.state_labels['0']  # alias

    def __repr__(self):
        """Debug representation of self"""
        return "BosonSite({N:d}, {c!r}, {f:f})".format(N=self.Nmax,
                                                       c=self.conserve,
                                                       f=self.filling)
