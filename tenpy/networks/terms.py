"""Classes to store a collection of operator names and sites they act on, together with prefactors.

This modules collects classes which are not strictly speaking tensor networks but represent "terms"
acting on them. Each term is given by a collection of (onsite) operator names and indices of the
sites it acts on. Moreover, we associate a `strength` to each term, which corresponds to the
prefactor when specifying e.g. a Hamiltonian.
"""
# Copyright 2019-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import warnings
import itertools

from ..linalg import np_conserved as npc
from ..tools.misc import add_with_None_0
from ..tools.hdf5_io import Hdf5Exportable

__all__ = [
    'TermList', 'OnsiteTerms', 'CouplingTerms', 'MultiCouplingTerms', 'ExponentiallyDecayingTerms',
    'order_combine_term'
]


class TermList(Hdf5Exportable):
    r"""A list of terms (=operator names and sites they act on) and associated strengths.

    A representation of terms, similar as :class:`OnsiteTerms`, :class:`CouplingTerms`
    and :class:`MultiCouplingTerms`.

    This class does not store operator strings between the sites.
    Jordan-Wigner strings of fermions are added during conversion to (Multi)CouplingTerms.

    .. warning ::

        Since this class does **not** store the operator string between the sites,
        conversion from :class:`CouplingTerms` or :class:`MultiCouplingTerms`
        to :class:`TermList` is lossy!

    Parameters
    ----------
    terms : list of list of (str, int)
        List of terms where each `term` is a list of tuples ``(opname, i)``
        of an operator name and a site `i` it acts on.
        For Fermions, the order is the order in the mathematic sense, i.e., the right-most/last
        operator in the list acts last.
    strength : (list of) float/complex
        For each term in `terms` an associated prefactor or strength.
        A single number holds for all terms equally.

    Attributes
    ----------
    terms : list of list of (str, int)
        List of terms where each `term` is a tuple ``(opname, i)`` of an operator name and a site
        `i` it acts on.
    strength : 1D ndarray
        For each term in `terms` an associated prefactor or strength.

    Examples
    --------

    .. testsetup :: TermList

        from tenpy.networks.terms import TermList

    For fermions, the term :math:`0.5(c^\dagger_0 c_2 + h.c.) + 1.3 * n_1` can be represented by:

    .. doctest :: TermList

        >>> t = TermList([[('Cd', 0), ('C', 2)], [('Cd', 2), ('C', 0)], [('N', 1)]],
        ...              [0.5,                   0.5,                   1.3])
        >>> print(t)
        0.50000 * Cd_0 C_2 +
        0.50000 * Cd_2 C_0 +
        1.30000 * N_1

    If you have a :class:`~tenpy.models.lattice.Lattice`, you might also want to specify
    the location of the operators by lattice indices insted of MPS indices.
    For example, you can obtain the nearest-neighbor density terms
    **without double counting each pair**) on a :class:`~tenpy.models.lattice.TriangularLattice`:

    .. doctest :: TermList

        >>> lat = tenpy.models.lattice.Triangular(6, 6, None, bc_MPS='infinite', bc='periodic')
        >>> t2_terms = [[('N', [0, 0, u1]), ('N', [dx[0], dx[1], u2])]
        ...             for (u1, u2, dx) in lat.pairs['nearest_neighbors']]
        >>> t2 = TermList.from_lattice_locations(lat, t2_terms)
        >>> print(t2)
        1.00000 * N_0 N_6 +
        1.00000 * N_0 N_-5 +
        1.00000 * N_0 N_5

    The negative index -5 here indicates a tensor left of the current MPS unit cell.
    """
    def __init__(self, terms, strength=1.):
        self.terms = list(terms)
        self.strength = np.array(strength)
        if self.strength.ndim == 0:
            self.strength = np.ones([len(self.terms)]) * self.strength
        if (len(self.terms), ) != self.strength.shape:
            raise ValueError("different length of terms and strength")

    @classmethod
    def from_lattice_locations(cls, lattice, terms, strength=1., shift=None):
        """Initialize from a list of terms given in lattice indices instead of MPS indices.

        Parameters
        ----------
        lattice : :class:`~tenpy.models.lattice.Lattice`
            The underlying lattice to be used for conversion, e.g. `M.lat` from a
            :class:`~tenpy.models.model.Model`.
        terms : list of list of (str, tuple)
            List of terms, where each `term` is a tuple ``(opname, lat_idx)`` with
            `lat_idx` itself being a tuple ``(x, y, u)`` (for a 2D lattice) of the lattice
            corrdinates.
        strengths : (list of) float/complex
            For each term in `terms` an associated prefactor or strength.
            A single number holds for all terms equally.
        shift : None | tuple of int
            Overall shift added to all lattice coordinates `lat_idx` in `terms` before conversion.
            None defaults to no shift.

        Returns
        -------
        term_list : :class:`TermList`
            Representation of the terms.
        """
        converted_terms = []
        if shift is None:
            shift = np.zeros(lattice.dim + 1, np.intp)
        else:
            shift = np.array(shift, np.intp)
            if len(shift) != lattice.dim + 1:
                raise ValueError("wrong length of `shift`: " + repr(shift))
        for term in terms:
            new_term = [(op, lattice.lat2mps_idx(shift + idx)) for (op, idx) in term]
            converted_terms.append(new_term)
        return cls(converted_terms, strength)

    def to_OnsiteTerms_CouplingTerms(self, sites):
        """Convert to :class:`OnsiteTerms` and :class:`CouplingTerms`

        Performs Jordan-Wigner transformation for fermionic operators.

        Parameters
        ----------
        sites : list of :class:`~tenpy.networks.site.Site`
            Defines the local Hilbert space for each site.
            Used to check whether the operators need Jordan-Wigner strings.
            The length is used as `L` for the `onsite_terms` and `coupling_terms`.

        Returns
        -------
        onsite_terms : :class:`OnsiteTerms`
            Onsite terms.
        coupling_terms :  :class:`CouplingTerms` | :class:`MultiCouplingTerms`
            Coupling terms. If `self` contains terms involving more than two operators, a
            :class:`MultiCouplingTerms` instance, otherwise just :class:`CouplingTerms`.
        """
        L = len(sites)
        ot = OnsiteTerms(L)
        self.order_combine(sites)  # general terms might act multiple times on the same sites
        if any(len(t) > 2 for t in self.terms):
            ct = MultiCouplingTerms(L)
        else:
            ct = CouplingTerms(L)
        for term, strength in self:
            if len(term) == 1:
                op, i = term[0]
                ot.add_onsite_term(strength, i, op)
            elif len(term) == 2:
                args = ct.coupling_term_handle_JW(strength, term, sites)
                ct.add_coupling_term(*args)
            elif len(term) > 2:
                args = ct.multi_coupling_term_handle_JW(strength, term, sites)
                ct.add_multi_coupling_term(*args)
            else:
                raise ValueError("term without entry!?")
        return ot, ct

    def __iter__(self):
        """Iterate over ``zip(self.terms, self.strength)``."""
        return zip(self.terms, self.strength)

    def __add__(self, other):
        if isinstance(other, TermList):
            return TermList(self.terms + other.terms,
                            np.concatenate((self.strength, other.strength)))
        return NotImplemented

    def __mul__(self, other):
        return TermList(self.terms, self.strength * other)

    def __str__(self):
        res = []
        for term, strength in self:
            term_str = ' '.join(['{op!s}_{i:d}'.format(op=op, i=i) for op, i in term])
            res.append('{s:.5f} * {t}'.format(s=strength, t=term_str))
        return ' +\n'.join(res)

    def order_combine(self, sites):
        """Order and combine operators in each term.

        Parameters
        ----------
        sites : list of :class:`~tenpy.networks.site.Site`
            Defines the local Hilbert space for each site.
            Used to check whether the operators anticommute
            (= whether they need Jordan-Wigner strings) and for multiplication rules.

        See also
        --------
        order_and_combine_term : does it for a single term.
        """
        for idx, term in enumerate(self.terms):
            self.terms[idx], overall_sign = order_combine_term(term, sites)
            self.strength[idx] *= overall_sign
        # TODO: could sort terms and combine duplicates

    def limits(self):
        """Return the left-most site and right-most site any operator acts on."""
        all_i = []
        for term in self.terms:
            all_i.extend([i for _, i in term])
        return min(all_i), max(all_i)

    def shift(self, i0):
        """Return a copy where `i0` is added to all indices `i` in :attr:`terms`."""
        shifted_terms = [[(op, i + i0) for op, i in term] for term in self.terms]
        return TermList(shifted_terms, self.strength)

    def max_range(self):
        res = 0
        for term in self.terms:
            ijkl = [i for op, i in term]
            res = max(res, max(ijkl) - min(ijkl))
        return res


def order_combine_term(term, sites):
    """Combine operators in a term to one terms per site.

    Takes in a term of operators and sites they acts on, commutes operators to order them by site
    and combines operators acting on the same site with
    :meth:`~tenpy.networks.site.Site.multiply_op_names`.

    Parameters
    ----------
    term : a list of (opname_i, i) tuples
        Represents a product of onsite operators with site indices `i` they act on.
        Needs not to be ordered and can have multiple entries acting on the same site.
    sites : list of :class:`~tenpy.networks.site.Site`
        Defines the local Hilbert space for each site.
        Used to check whether the operators anticommute
        (= whether they need Jordan-Wigner strings) and for multiplication rules.

    Returns
    -------
    combined_term :
        Equivalent to `term` but with at most one operator per site.
    overall_sign : +1 | -1 | 0
        Comes from the (anti-)commutation relations.
        When the operators in `term` are multiplied from left to right, and
        then multiplied by `overall_sign`, the result is the same operator
        as the product of `combined_term` from left to right.
    """
    # Group all operators that are on the same site and get the corresponding sign
    L = len(sites)
    N = len(term)
    overall_sign = 1
    terms_commute = [(op, i, sites[i % L].op_needs_JW(op)) for op, i in term]
    # perform bubble sort on terms_commute and keep track of the sign
    if N > 100:  # bubblesort is O(N^2), assume that N is small
        # N = 1000 takes ~1s, so 100 should be fine...
        warnings.warn("not intended for large number of operators.")
    for s_max in range(N - 1, 0, -1):
        for s in range(s_max):
            t1, t2 = terms_commute[s:s + 2]
            if t1[1] > t2[1]:  # t1 right of t2 -> swap
                terms_commute[s] = t2
                terms_commute[s + 1] = t1
                if t1[2] and t2[2]:
                    overall_sign = -overall_sign
    # combine terms on same site
    term = []
    for i, same_site_terms in itertools.groupby(terms_commute, lambda t: t[1]):
        ops = [t[0] for t in same_site_terms]
        op = sites[i % L].multiply_op_names(ops)
        term.append((op, i))
    return term, overall_sign


class OnsiteTerms(Hdf5Exportable):
    """Operator names, site indices and strengths representing onsite terms.

    Represents a sum of onsite terms where the operators are only given by their name (in the form
    of a string). What the operator represents is later given by a list of
    :class:`~tenpy.networks.site.Site` with :meth:`~tenpy.networks.site.Site.get_op`.

    Parameters
    ----------
    L : int
        Number of sites.

    Attributes
    ----------
    L : int
        Number of sites.
    onsite_terms : list of dict
        Filled by meth:`add_onsite_term`.
        For each index `i` a dictionary ``{'opname': strength}`` defining the onsite terms.
    """
    def __init__(self, L):
        assert L > 0
        self.L = L
        self.onsite_terms = [dict() for _ in range(L)]

    def max_range(self):
        """Maximum range of the terms.

        In this case ``0``.
        """
        return 0

    def add_onsite_term(self, strength, i, op):
        """Add a onsite term on a given MPS site.

        Parameters
        ----------
        strength : float
            The strength of the term.
        i : int
            The MPS index of the site on which the operator acts.
            We require ``0 <= i < L``.
        op : str
            Name of the involved operator.
        """
        term = self.onsite_terms[i]
        term[op] = term.get(op, 0) + strength

    def add_to_graph(self, graph):
        """Add terms from :attr:`onsite_terms` to an MPOGraph.

        Parameters
        ----------
        graph : :class:`~tenpy.networks.mpo.MPOGraph`
            The graph into which the terms from :attr:`onsite_terms` should be added.
        """
        assert self.L == graph.L
        for i, terms in enumerate(self.onsite_terms):
            for opname, strength in terms.items():
                graph.add(i, 'IdL', 'IdR', opname, strength)

    def to_Arrays(self, sites):
        """Convert the :attr:`onsite_terms` into a list of np_conserved Arrays.

        Parameters
        ----------
        sites : list of :class:`~tenpy.networks.site.Site`
            Defines the local Hilbert space for each site.
            Used to translate the operator names into :class:`~tenpy.linalg.np_conserved.Array`.

        Returns
        -------
        onsite_arrays : list of :class:`~tenpy.linalg.np_conserved.Array`
            Onsite terms represented by `self`. Entry `i` of the list lives on ``sites[i]``.
        """
        if len(sites) != self.L:
            raise ValueError("Incompatible length")
        res = []
        for site, terms in zip(sites, self.onsite_terms):
            H = None
            for opname, strength in terms.items():
                H = add_with_None_0(H, strength * site.get_op(opname))
                # Note: H can change from None to npc Array and can change dtype
            res.append(H)
        return res

    def remove_zeros(self, tol_zero=1.e-15):
        """Remove entries close to 0 from :attr:`onsite_terms`.

        Parameters
        ----------
        tol_zero : float
            Entries in :attr:`onsite_terms` with `strength` < `tol_zero` are considered to be
            zero and removed.
        """
        for term in self.onsite_terms:
            for op in list(term.keys()):
                if abs(term[op]) < tol_zero:
                    del term[op]
        # done

    def add_to_nn_bond_Arrays(self, H_bond, sites, finite, distribute=(0.5, 0.5)):
        """Add :attr:`self.onsite_terms` into nearest-neighbor bond arrays.

        Parameters
        ----------
        H_bond : list of {:class:`~tenpy.linalg.np_conserved.Array` | None}
            The :attr:`coupling_terms` rewritten as ``sum_i H_bond[i]`` for MPS indices ``i``.
            ``H_bond[i]`` acts on sites ``(i-1, i)``, ``None`` represents 0.
            Legs of each ``H_bond[i]`` are ``['p0', 'p0*', 'p1', 'p1*']``.
            Modified *in place*.
        sites : list of :class:`~tenpy.networks.site.Site`
            Defines the local Hilbert space for each site.
            Used to translate the operator names into :class:`~tenpy.linalg.np_conserved.Array`.
        distribute : (float, float)
            How to split the onsite terms (in the bulk) into the bond terms to the left
            (``distribute[0]``) and right (``distribute[1]``).
        finite : bool
            Boundary conditions of the MPS, :attr:`MPS.finite`.
            If finite, we distribute the onsite term of the
        """
        dist_L, dist_R = distribute
        if dist_L + dist_R != 1.:
            raise ValueError("sum of `distribute` not 1!")
        N_sites = self.L
        H_onsite = self.to_Arrays(sites)
        for j in range(N_sites):
            H_j = H_onsite[j]
            if H_j is None:
                continue
            if finite and j == 0:
                dist_L, dist_R = 0., 1.
            elif finite and j == N_sites - 1:
                dist_L, dist_R = 1., 0.
            else:
                dist_L, dist_R = distribute
            if dist_L != 0.:
                i = (j - 1) % N_sites
                Id_i = sites[i].Id
                H_bond[j] = add_with_None_0(H_bond[j], dist_L * npc.outer(Id_i, H_j))
            if dist_R != 0.:
                k = (j + 1) % N_sites
                Id_k = sites[k].Id
                H_bond[k] = add_with_None_0(H_bond[k], dist_R * npc.outer(H_j, Id_k))
        for H in H_bond:
            if H is not None:
                H.iset_leg_labels(['p0', 'p0*', 'p1', 'p1*'])
        # done

    def to_TermList(self):
        """Convert :attr:`onsite_terms` into a :class:`TermList`.

        Returns
        -------
        term_list : :class:`TermList`
            Representation of the terms as a list of terms.
        """
        terms = []
        strength = []
        for i, terms_i in enumerate(self.onsite_terms):
            for opname in sorted(terms_i):
                terms.append([(opname, i)])
                strength.append(terms_i[opname])
        return TermList(terms, strength)

    def __iadd__(self, other):
        if not isinstance(other, OnsiteTerms):
            return NotImplemented  # unknown type of other
        if other.L != self.L:
            raise ValueError("incompatible lengths")
        for self_t, other_t in zip(self.onsite_terms, other.onsite_terms):
            for key, value in other_t.items():
                self_t[key] = self_t.get(key, 0.) + value
        return self

    def _test_terms(self, sites):
        """Check that all given operators exist in the `sites`."""
        for site, terms in zip(sites, self.onsite_terms):
            for opname, strength in terms.items():
                if not site.valid_opname(opname):
                    raise ValueError("Operator {op!r} not in site".format(op=opname))


class CouplingTerms(Hdf5Exportable):
    """Operator names, site indices and strengths representing two-site coupling terms.

    Parameters
    ----------
    L : int
        Number of sites.

    Attributes
    ----------
    L : int
        Number of sites.
    coupling_terms : dict of dict
        Filled by :meth:`add_coupling_term`.
        Nested dictionaries of the form
        ``{i: {('opname_i', 'opname_string'): {j: {'opname_j': strength}}}}``.
        Note that always ``i < j``, but entries with ``j >= L`` are allowed for
        ``bc_MPS == 'infinite'``, in which case they indicate couplings between different
        iMPS unit cells.
    """
    def __init__(self, L):
        assert L > 0
        self.L = L
        self.coupling_terms = dict()

    def max_range(self):
        """Determine the maximal range in :attr:`coupling_terms`.

        Returns
        -------
        max_range : int
            The maximum of ``j - i`` for the `i`, `j` occuring in a term of :attr:`coupling_terms`.
        """
        max_range = 0
        for i, d1 in self.coupling_terms.items():
            for d2 in d1.values():
                j_max = max(d2.keys())
                max_range = max(max_range, j_max - i)
        return max_range

    def add_coupling_term(self, strength, i, j, op_i, op_j, op_string='Id'):
        """Add a two-site coupling term on given MPS sites.

        Parameters
        ----------
        strength : float
            The strength of the coupling term.
        i, j : int
            The MPS indices of the two sites on which the operator acts.
            We require ``0 <= i < N_sites``  and ``i < j``, i.e., `op_i` acts "left" of `op_j`.
            If j >= N_sites, it indicates couplings between unit cells of an infinite MPS.
        op1, op2 : str
            Names of the involved operators.
        op_string : str
            The operator to be inserted between `i` and `j`.
        """
        if not 0 <= i < self.L:
            raise ValueError("We need 0 <= i < N_sites, got i={i:d}".format(i=i))
        if not i < j:
            raise ValueError("need i < j")
        d1 = self.coupling_terms.setdefault(i, dict())
        # form of d1: ``{('opname_i', 'opname_string'): {j: {'opname_j': current_strength}}}``
        d2 = d1.setdefault((op_i, op_string), dict())
        d3 = d2.setdefault(j, dict())
        d3[op_j] = d3.get(op_j, 0) + strength

    def coupling_term_handle_JW(self, strength, term, sites, op_string=None):
        """Helping function to call before :meth:`add_coupling_term`.

        Parameters
        ----------
        strength : float
            The strength of the coupling term.
        term : [(str, int), (str, int)]
            List of two tuples ``[(op_i, i), (op_j, j)]`` where `i` is the MPS index of the site
            the operator named `op_i` acts on; we require `i < j`.
        sites : list of :class:`~tenpy.networks.site.Site`
            Defines the local Hilbert space for each site.
            Used to check whether the operators need Jordan-Wigner strings.
        op_string : None | str
            Operator name to be used as operator string *between* the operators, or ``None`` if the
            Jordan Wigner string should be figured out.

            .. warning ::

                ``None`` figures out for each segment between the operators, whether a
                Jordan-Wigner string is needed.
                This is different from a plain ``'JW'``, which just applies a string on
                each segment!

        Returns
        -------
        strength, i, j, op_i, op_j, op_string:
            Arguments for :meth:`MultiCouplingTerms.add_multi_coupling_term` such that the added
            term corresponds to the parameters of this function.
        """
        L = self.L
        (op_i, i), (op_j, j) = term
        site_i = sites[i % L]
        site_j = sites[j % L]
        need_JW_i = site_i.op_needs_JW(op_i)
        need_JW_j = site_j.op_needs_JW(op_j)
        if op_string is None:
            if need_JW_i and need_JW_j:
                op_string = 'JW'
            elif need_JW_i or need_JW_j:
                raise ValueError("Only one of the operators needs a Jordan-Wigner string?!")
            else:
                op_string = 'Id'
        if op_string == 'JW':
            op_i = site_i.multiply_op_names([op_i, op_string])
        return strength, i, j, op_i, op_j, op_string

    def plot_coupling_terms(self,
                            ax,
                            lat,
                            style_map='default',
                            common_style={'linestyle': '--'},
                            text=None,
                            text_pos=0.4):
        """"Plot coupling terms into a given lattice.

        This function plots the :attr:`coupling_terms`

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which we should plot.
        lat : :class:`~tenpy.models.lattice.Lattice`
            The lattice for plotting the couplings, most probably the ``M.lat`` of the
            corresponding model ``M``, see :attr:`~tenpy.models.model.Model.lat`.
        style_map : function | None
            Function which get's called with arguments ``i, j, op_i, op_string, op_j, strength``
            for each two-site coupling and should return a keyword-dictionary with the desired
            plot-style for this coupling.
            By default (``None``), the `linewidth` is given by the absolute value of `strength`,
            and the linecolor depends on the phase of `strength` (using the `hsv` colormap).
        common_style : dict
            Common style, which overwrites values of the dictionary returned by style_map.
            A ``'label'`` is only used for the first plotted line.
        text: format_string | None
            If not ``None``, we add text labeling the couplings in the plot.
            Available keywords are ``i, j, op_i, op_string, op_j, strength`` as well as
            ``strength_abs, strength_angle, strength_real``.
        text_pos : float
            Specify where to put the text on the line between `i` (0.0) and `j` (1.0),
            e.g. `0.5` is exactly in the middle between `i` and `j`.

        See also
        --------
        tenpy.models.lattice.Lattice.plot_sites : plot the sites of the lattice.
        """
        pos = lat.position(lat.order)  # row `i` gives position where to plot site `i`
        N_sites = lat.N_sites
        x_y = np.zeros((2, 2))  # columns=x,y, rows=i,j
        if style_map == 'default':
            import matplotlib
            from matplotlib.cm import hsv
            from matplotlib.colors import Normalize
            norm_angle = Normalize(vmin=-np.pi, vmax=np.pi)

            def style_map(i, j, op_i, op_string, op_j, strength):
                """define the plot style for a given coupling."""
                key = (op_i, op_string, op_j)
                style = {}
                style['linewidth'] = np.abs(strength) * matplotlib.rcParams['lines.linewidth']
                style['color'] = hsv(norm_angle(np.angle(strength)))
                return style

        text_pos = np.array([1. - text_pos, text_pos], np.float_)
        for i in sorted(self.coupling_terms.keys()):
            d1 = self.coupling_terms[i]
            x_y[0, :] = pos[i]
            for (op_i, op_string) in sorted(d1.keys()):
                d2 = d1[(op_i, op_string)]
                for j in sorted(d2.keys()):
                    d3 = d2[j]
                    shift = j - j % N_sites
                    if shift == 0:
                        x_y[1, :] = pos[j]
                    else:
                        lat_idx_j = np.array(lat.order[j % N_sites])
                        lat_idx_j[0] += (shift // N_sites) * lat.N_rings
                        x_y[1, :] = lat.position(lat_idx_j)
                    for op_j in sorted(d3.keys()):
                        if isinstance(op_j, tuple):
                            continue  # multi-site coupling!
                        strength = d3[op_j]
                        if style_map:
                            style = style_map(i, j, op_i, op_string, op_j, strength)
                        else:
                            style = {}
                        style.update(common_style)
                        ax.plot(x_y[:, 0], x_y[:, 1], **style)
                        if 'label' in common_style:
                            common_style = common_style.copy()
                            del common_style['label']
                        if text:
                            annotate = text.format(i=i,
                                                   j=j,
                                                   op_i=op_i,
                                                   op_string=op_string,
                                                   op_j=op_j,
                                                   strength=strength,
                                                   strength_abs=np.abs(strength),
                                                   strength_real=np.real(strength),
                                                   strength_angle=np.angle(strength))
                            loc = np.dot(x_y.T, text_pos)
                            ax.text(loc[0], loc[1], annotate)
        # done

    def add_to_graph(self, graph):
        """Add terms from :attr:`coupling_terms` to an MPOGraph.

        Parameters
        ----------
        graph : :class:`~tenpy.networks.mpo.MPOGraph`
            The graph into which the terms from :attr:`coupling_terms` should be added.
        """
        assert self.L == graph.L
        # structure of coupling terms:
        # {i: {('opname_i', 'opname_string'): {j: {'opname_j': strength}}}}
        for i, d1 in self.coupling_terms.items():
            for (opname_i, op_string), d2 in d1.items():
                label = ("left", i, opname_i, op_string)
                graph.add(i, 'IdL', label, opname_i, 1., skip_existing=True)
                for j, d3 in d2.items():
                    label_j = graph.add_string_left_to_right(i, j, label, op_string)
                    for opname_j, strength in d3.items():
                        graph.add(j, label_j, 'IdR', opname_j, strength)
        # done

    def to_nn_bond_Arrays(self, sites):
        """Convert the :attr:`coupling_terms` into Arrays on nearest neighbor bonds.

        Parameters
        ----------
        sites : list of :class:`~tenpy.networks.site.Site`
            Defines the local Hilbert space for each site.
            Used to translate the operator names into :class:`~tenpy.linalg.np_conserved.Array`.

        Returns
        -------
        H_bond : list of {:class:`~tenpy.linalg.np_conserved.Array` | None}
            The :attr:`coupling_terms` rewritten as ``sum_i H_bond[i]`` for MPS indices ``i``.
            ``H_bond[i]`` acts on sites ``(i-1, i)``, ``None`` represents 0.
            Legs of each ``H_bond[i]`` are ``['p0', 'p0*', 'p1', 'p1*']``.
        """
        L = self.L
        if len(sites) != L:
            raise ValueError("incompatible length")
        H_bond = [None] * L
        term_list = self.to_TermList()
        for term, strength in zip(term_list.terms, term_list.strength):
            assert len(term) == 2
            (op_i, i), (op_j, j) = term
            if i + 1 != j:
                raise ValueError("not nearest neighbor")
            j = j % L
            site_i = sites[i]
            site_j = sites[j]
            H_add = strength * npc.outer(site_i.get_op(op_i), site_j.get_op(op_j))
            # TeNPy convention: H_bond[j] is acting on sites (j-1, j), i.e. bond left of site j
            H_bond[j] = add_with_None_0(H_bond[j], H_add)
        for H in H_bond:
            if H is not None:
                H.iset_leg_labels(['p0', 'p0*', 'p1', 'p1*'])
        return H_bond

    def remove_zeros(self, tol_zero=1.e-15):
        """Remove entries close to 0 from :attr:`coupling_terms`.

        Parameters
        ----------
        tol_zero : float
            Entries in :attr:`coupling_terms` with `strength` < `tol_zero` are considered to be
            zero and removed.
        """
        for d1 in self.coupling_terms.values():
            # d1 = ``{('opname_i', 'opname_string'): {j: {'opname_j': strength}}}``
            for op_i_op_str, d2 in list(d1.items()):
                for j, d3 in list(d2.items()):
                    for op_j, st in list(d3.items()):
                        if abs(st) < tol_zero:
                            del d3[op_j]
                    if len(d3) == 0:
                        del d2[j]
                if len(d2) == 0:
                    del d1[op_i_op_str]
        # done

    def to_TermList(self):
        """Convert :attr:`onsite_terms` into a :class:`TermList`.

        Returns
        -------
        term_list : :class:`TermList`
            Representation of the terms as a list of terms.
        """
        terms = []
        strength = []
        d0 = self.coupling_terms
        for i in sorted(d0):
            d1 = d0[i]
            for (opname_i, op_str) in sorted(d1):
                d2 = d1[(opname_i, op_str)]
                for j in sorted(d2):
                    d3 = d2[j]
                    for opname_j in sorted(d3):
                        terms.append([(opname_i, i), (opname_j, j)])
                        strength.append(d3[opname_j])
        return TermList(terms, strength)

    def __iadd__(self, other):
        if not isinstance(other, CouplingTerms):
            return NotImplemented  # unknown type of other
        if isinstance(other, MultiCouplingTerms):
            raise ValueError("Can't add MultiCouplingTerms into CouplingTerms")
        if other.L != self.L:
            raise ValueError("incompatible lengths")
        # {i: {('opname_i', 'opname_string'): {j: {'opname_j': strength}}}}
        for i, other_d1 in other.coupling_terms.items():
            self_d1 = self.coupling_terms.setdefault(i, dict())
            for opname_i_string, other_d2 in other_d1.items():
                self_d2 = self_d1.setdefault(opname_i_string, dict())
                for j, other_d3 in other_d2.items():
                    self_d3 = self_d2.setdefault(j, dict())
                    for opname_j, strength in other_d3.items():
                        self_d3[opname_j] = self_d3.get(opname_j, 0.) + strength
        return self

    def _test_terms(self, sites):
        """Check the format of self.coupling_terms."""
        L = self.L
        for i, d1 in self.coupling_terms.items():
            site_i = sites[i]
            for (op_i, opstring), d2 in d1.items():
                if not site_i.valid_opname(op_i):
                    raise ValueError("Operator {op!r} not in site".format(op=op_i))
                for j, d3 in d2.items():
                    if not i < j:
                        raise ValueError("wrong order of indices in coupling terms")
                    for op_j in d3.keys():
                        if not sites[j % L].valid_opname(op_j):
                            raise ValueError("Operator {op!r} not in site".format(op=op_j))
        # done


class MultiCouplingTerms(CouplingTerms):
    """Operator names, site indices and strengths representing general `M`-site coupling terms.

    Generalizes the :attr:`coupling_terms` of :class:`CouplingTerms` to `M`-site couplings.
    The structure of the nested dictionary :attr:`coupling_terms` is similar, but we allow
    an arbitrary recursion depth of the dictionary and build the
    :class:`~tenpy.networks.mpo.MPOGraph` simultaneously from the left and right.
    Doing this from both sides simultaneously is necessary to keep the scaling of the MPO bond
    dimension optimial, which we try to acchieve by reusing states of the MPO graph as much as
    possible as follows.

    Each individual term gets split up at site `switchLR` (default: center of the coupling).
    To the left of `switchLR`, we only use "almost-left-canonical" states in the MPO graph,
    i.e., states where only the basic set of operators defined by the sites were applied with
    prefactor 1. Similarly, "almost-right-canonical" states are inserted into the MPO graph
    coming from the right. At site `switchLR`, we connect an almost-left- to an
    almost-right-canonical state in the MPO graph with the overall `strength` prefactor for the
    term. Since the almost-left/right-canonical states in the MPO graph represent well-defined
    strings of operators that were applied, they can be re-used for different terms.
    For example, ``A_i B_j C_k`` and ``A_i B_{j+1} C_k`` with ``i < j < j+1 < k`` can re-use
    the MPO graph states left of site `j` and right of `j+1` if `switchLR` is `j` or `j+1`.

    Parameters
    ----------
    L : int
        Number of sites.

    Attributes
    ----------
    L : int
        Number of sites.
    connections : list of (int, str, int, float)
        One entry ``(switchLR, op_switch, shift, strength)`` for each (multi) coupling term added,
        with `op_switch` being the operator acting on site `switchLR`, and `shift` defined below.
    terms_left, terms_right : dict of dict
        Nested dictionaries of the following form for left and right::

            terms_left = {
                ijkl[0]:
                    {(ops_ijkl[0], op_string[0]):
                        {ijkl[1]:
                            {(ops_ijkl[1], op_string[1]):
                                ...
                                    {ijkl[n]:
                                        {(ops_ijkl[n],  op_string[n]):
                                            {-1: [counter, ...]}
                                    }   }
                        }   }
               }   }
            terms_right = {
                ijkl[-1] - shift:
                    {(ops_ijkl[-1], op_string[-1]):
                        {ijkl[-2] - shift:
                            {(ops_ijkl[-2], op_string[-2]):
                                ...
                                    {ijkl[m] - shift:
                                        {(ops_ijkl[m],  op_string[m]):
                                            {L + 1: [counter, ...]}
                                    }   }
                        }   }
                }   }

        Here, `n` and `m` (and hence the nesting depth) are determined by `switchLR`, namely
        `n` is the largest index such that ``ijkl[n] < switchLR``, similarly `m` is the smallest
        index such that ``switchLR < ijkl[m]``.
        If there is no such index, we directly insert the ``-1: [counter, ...]`` part
        at the first level of `terms_left` or `terms_right`, respectively.
        The `counter` provides the index to ``connections`` for the given term.
        The `shift` for the `terms_right` is a multiple of `L` such that
        ``0 <= ijkl[-1] - shift < L``.
    """
    def __init__(self, L):
        assert L > 0
        self.L = L
        self.terms_left = dict()
        self.terms_right = dict()
        # start the counter at 1 to distinguish site indices in `terms_left/right` from counters.
        self.connections = [None]
        self._max_range = 0
        self._connect_left = -1
        self._connect_right = self.L + 1

    def max_range(self):
        """Determine the maximal range in :attr:`coupling_terms`.

        Returns
        -------
        max_range : int
            The maximum of ``j - i`` for the `i`, `j` occuring in a term of :attr:`coupling_terms`.
        """
        return self._max_range

    def add_multi_coupling_term(self, strength, ijkl, ops_ijkl, op_string="Id",
                                switchLR='middle_i'):
        """Add a multi-site coupling term.

        Parameters
        ----------
        strength : float
            The strength of the coupling term.
        ijkl : list of int
            The MPS indices of the sites on which the operators acts. With ``i, j, k, ... = ijkl``,
            we require that they are ordered ascending, ``i < j < k < ...`` and
            that ``0 <= i < N_sites``.
            Inidces >= N_sites indicate couplings between different unit cells of an infinite MPS.
        ops_ijkl : list of str
            Names of the involved operators on sites ``i, j, k, ...``.
        op_string : (list of) str
            Names of the operator to be inserted between the operators,
            e.g., ``op_string[0]`` is inserted between `i` and `j`.
            A single name holds for all in-between segments.
        switchLR: int | ``'middle_i'`` | ``'middle_op'``
            The site where we switch from building the coupling from the left to building the
            coupling from the right for an efficient MPO representation.
            This has implications for the final MPO bond dimension, but the optimal value depends
            on what other terms there are in the Hamiltonian. We therefore provide a few
            heurisitic choices that can be given as the following strings.

            middle_i :
                The overall middle index ``(ijkl[0] + ijkl[-1] + 1) // 2``, the default choice.
                Somewhat reasonable if we don't know anything about the terms,
                but often not optimal
            middle_op :
                The index of the middle operator ``ijkl[len(ijkl) // 2]``.
                This is for example a good choice if you have combinations
                ``A[i] B[i+1] C[i+j] + A[i] B[i+j-1] C[i+j]`` for large `j` and various `B`.
        """
        L = self.L
        if len(ijkl) < 2:
            raise ValueError("Need to act on at least 2 sites. Use onsite terms!")
        if isinstance(op_string, str):
            op_string = [op_string] * (len(ijkl) - 1)
        assert len(ijkl) == len(ops_ijkl) == len(op_string) + 1
        for i, j in zip(ijkl, ijkl[1:]):
            if not i < j:
                raise ValueError("Need i < j < k < ...")
        if switchLR is None:
            switchLR = 'middle_i'
        if isinstance(switchLR, str):
            if switchLR == 'middle_i':
                switchLR = (ijkl[0] + ijkl[-1] + 1) // 2
            elif switchLR == 'middle_op':
                switchLR = ijkl[len(ijkl) // 2]
        assert 0 <= ijkl[0] <= switchLR <= ijkl[-1] and ijkl[0] < L
        # find op_switch
        for n, i in enumerate(ijkl):
            if switchLR == i:
                op_switch = ops_ijkl[n]
                break
            elif switchLR < i:
                op_switch = op_string[n - 1]
                break
        else:  # no break
            assert False # can't happen, since switchLR <= ijkl[-1]

        d0L, d0R = self.terms_left, self.terms_right
        #add left terms
        for i, op, op_str in zip(ijkl, ops_ijkl, op_string):
            if i >= switchLR:
                break
            d1L = d0L.setdefault(i, dict())
            d0L = d1L.setdefault((op, op_str), dict())
        counters_left = d0L.setdefault(self._connect_left, [])
        #add right terms
        shift = ijkl[-1] - (ijkl[-1] % L)
        for i, op, op_str in zip(reversed(ijkl), reversed(ops_ijkl), reversed(op_string)):
            if i <= switchLR:
                break
            d1R = d0R.setdefault(i - shift, dict())
            d0R = d1R.setdefault((op, op_str), dict())
        counters_right = d0R.setdefault(self._connect_right, [])
        new_connection = (switchLR, op_switch, shift, strength)
        self._insert_connection(counters_left, counters_right, new_connection)
        self._max_range = max(ijkl[-1] - ijkl[0], self._max_range)

    def _insert_connection(self, counters_left, counters_right, new_connection):
        # check whether we already got that exact same term before
        existing_counter = None
        for c in counters_left:
            if self.connections[c][:3] == new_connection[:3] and c in counters_right:
                existing_counter = c
                break
        if existing_counter is not None:
            # if yes, just update strength of that term
            updated_strength = self.connections[existing_counter][3] + new_connection[3]
            self.connections[existing_counter] = new_connection[:3] + (updated_strength,)
        else:
            # otherwise append new entry to self.connections
            counter = len(self.connections)
            counters_left.append(counter)
            counters_right.append(counter)
            self.connections.append(new_connection)

    def multi_coupling_term_handle_JW(self, strength, term, sites, op_string=None):
        """Helping function to call before :meth:`add_multi_coupling_term`.

        Handle/figure out Jordan-Wigner strings if needed.

        Parameters
        ----------
        strength : float
            The strength of the term.
        term : list of (str, int)
            List of tuples ``(op_i, i)`` where `i` is the MPS index of the site the operator
            named `op_i` acts on.
            We **require** the operators to be sorted (strictly ascending) by sites.
            If necessary, call :func:`order_combine_term` beforehand.
        sites : list of :class:`~tenpy.networks.site.Site`
            Defines the local Hilbert space for each site.
            Used to check whether the operators need Jordan-Wigner strings.
        op_string : None | str
            Operator name to be used as operator string *between* the operators, or ``None`` if the
            Jordan Wigner string should be figured out.

            .. warning ::
                ``None`` figures out for each segment between the operators, whether a
                Jordan-Wigner string is needed.
                This is different from a plain ``'JW'``, which just applies a string on
                each segment!

        Returns
        -------
        strength, ijkl, ops_ijkl, op_string :
            Arguments for :meth:`MultiCouplingTerms.add_multi_coupling_term` such that the added
            term corresponds to the parameters of this function.
        """
        L = self.L
        number_ops = len(term)
        if number_ops < 2:
            raise ValueError("got onsite term instead of coupling")
        if op_string == 'JW':
            warnings.warn("op_string='JW' is probably not what you want!")
        ops = [t[0] for t in term]
        ijkl = [t[1] for t in term]
        assert all([i < j for i, j in zip(ijkl, ijkl[1:])])  # ascending?
        op_needs_JW = [sites[i % L].op_needs_JW(op) for op, i in term]
        if not any(op_needs_JW):
            op_string = 'Id'
        # shift ijkl such that first site is inside unit cell
        i0 = ijkl[0]
        if not 0 <= i0 < L:  # ensure this condition with a shift
            shift = i0 % L - i0
            ijkl = [i + shift for i in ijkl]
        if op_string is not None:
            # simpler case
            new_op_str = [op_string] * (number_ops - 1)
        else:
            # handle Jordan-Wigner transformation
            new_op_str = []  # new_op_string[x] is right of ops[x]
            JW_right = False  # right of site -1 : no JW string: even number
            for x in range(number_ops):
                if op_needs_JW[x]:
                    JW_right = not JW_right  # switch on the right
                if JW_right:
                    new_op_str.append('JW')
                    # need also 'JW' on current site
                    ops[x] = sites[ijkl[x] % L].multiply_op_names([ops[x], 'JW'])
                else:
                    new_op_str.append('Id')
            if JW_right:
                raise ValueError("odd number of Jordan Wigner strings")
            new_op_str.pop()  # created one entry too much
        return strength, ijkl, ops, new_op_str

    def add_coupling_term(self, strength, i, j, op_i, op_j, op_string='Id', switchLR=None):
        """Add a two-site coupling term on given MPS sites.

        Parameters
        ----------
        strength : float
            The strength of the coupling term.
        i, j : int
            The MPS indices of the two sites on which the operator acts.
            We require ``0 <= i < N_sites``  and ``i < j``, i.e., `op_i` acts "left" of `op_j`.
            If j >= N_sites, it indicates couplings between unit cells of an infinite MPS.
        op1, op2 : str
            Names of the involved operators.
        op_string : str
            The operator to be inserted between `i` and `j`.
        """
        if not 0 <= i < self.L:
            raise ValueError("We need 0 <= i < N_sites, got i={i:d}".format(i=i))
        if not i < j:
            raise ValueError("need i < j")
        ijkl = [i, j]
        ops_ijkl = [op_i, op_j]
        self.add_multi_coupling_term(strength, ijkl, ops_ijkl, op_string, switchLR)

    def add_to_graph(self, graph):
        """Add terms represented by `self` to an MPOGraph.

        Parameters
        ----------
        graph : :class:`~tenpy.networks.mpo.MPOGraph`
            The graph into which the terms from :attr:`coupling_terms` should be added.
        """
        assert self.L == graph.L
        keys_left = self._insert_to_graph(graph, True)
        keys_right = self._insert_to_graph(graph, False)
        for keyL, keyR, connection in zip(keys_left, keys_right, self.connections):
            if connection is None:
                assert keyL is None and keyR is None
                continue
            switchLR, op_switch, shift, strength = connection
            graph.add(switchLR, keyL, keyR, op_switch, strength)
        # done

    def _insert_to_graph(self, graph, from_left):
        all_keys = [None] * len(self.connections)
        d0 = self.terms_left if from_left else self.terms_right
        _connect = self._connect_left if from_left else self._connect_right
        for i, d1 in d0.items():
            if i == _connect:
                for c in d1:
                    all_keys[c] = 'IdL' if from_left else 'IdR'
            else:
                for (op_i, op_string_ij), d2 in d1.items():
                    if from_left:
                        key_from_i = ("left", i, op_i, op_string_ij)
                        graph.add(i, 'IdL', key_from_i, op_i, 1., skip_existing=True)
                    else:
                        key_from_i = ("right", i, op_i, op_string_ij)
                        graph.add(i, key_from_i, 'IdR', op_i, 1., skip_existing=True)
                    self._insert_to_graph_rec(graph, all_keys, d2, i, op_string_ij, key_from_i, from_left)
        return all_keys

    def _insert_to_graph_rec(self, graph, all_keys, d2, i, op_string_ij, key_from_i, from_left):
        """recursive part for :meth:`_insert_to_graph`"""
        _connect = self._connect_left if from_left else self._connect_right
        for j, d3 in d2.items():
            if j == _connect:
                for c in d3:
                    switchLR, _, shift, _ = self.connections[c]
                    if from_left:
                        key_to_switch = graph.add_string_left_to_right(i, switchLR, key_from_i, op_string_ij)
                    else:
                        key_to_switch = graph.add_string_right_to_left(i, switchLR-shift, key_from_i, op_string_ij)
                    all_keys[c] = key_to_switch
            else:
                if from_left:
                    key_to_j = graph.add_string_left_to_right(i, j, key_from_i, op_string_ij)
                else:
                    key_to_j = graph.add_string_right_to_left(i, j, key_from_i, op_string_ij)
                for (op_j, op_string_jk), d4 in d3.items():
                    if from_left:
                        key_from_j = key_to_j + (j, op_j, op_string_jk)
                        graph.add(j, key_to_j, key_from_j, op_j, 1., skip_existing=True)
                    else:
                        key_from_j = key_to_j + (j, op_j, op_string_jk)
                        graph.add(j, key_from_j, key_to_j, op_j, 1., skip_existing=True)
                    self._insert_to_graph_rec(graph, all_keys, d4, j, op_string_jk, key_from_j, from_left)

    def remove_zeros(self, tol_zero=1.e-15):
        """Remove entries close to 0 from :attr:`coupling_terms`.

        Parameters
        ----------
        tol_zero : float
            Entries in :attr:`terms_left` and :attr:`terms_right`
            with `strength` < `tol_zero` are considered to be zero and removed.
        """
        assert self._max_range is not None
        for c, connection in enumerate(self.connections):
            if connection is None:
                continue
            if abs(connection[3]) < tol_zero:
                self.connections[c] = None
                self._max_range = None
        if self._max_range is None:  # removed a term
            self._remove_empty_terms(self.terms_left, self._connect_left)
            self._remove_empty_terms(self.terms_right, self._connect_right)
            self._max_range = self.to_TermList().max_range()

    def _remove_empty_terms(self, d0, _connect):
        for i, d1 in d0.copy().items():  # shallow dict copy: might modify d0!
            if i == _connect:
                d0[i] = d1 = [c for c in d1 if self.connections[c] is not None]
            else:
                for ops, d2 in d1.copy().items():
                    self._remove_empty_terms(d2, _connect)
                    if len(d2) == 0:
                        del d1[ops]
            if len(d1) == 0:
                del d0[i]
        # done

    def to_TermList(self):
        """Convert :attr:`coupling_terms` into a :class:`TermList`.
        Returns
        -------
        term_list : :class:`TermList`
            Representation of the terms as a list of terms.
        """
        term_list_left = self._fill_term_list(self.terms_left, self._connect_left)
        term_list_right = self._fill_term_list(self.terms_right, self._connect_right)
        strengths = []
        terms = []
        for tL, tR, c in zip(term_list_left, term_list_right, self.connections):
            if c is None:
                assert tL is None and tR is None
                continue
            switchLR, op_switch, shift, strength = c
            term = []
            op_str = ""
            if tL is not None:
                for i, op_i, op_str in tL:
                    term.append((op_i, i))
            if op_switch != op_str:
                term.append((op_switch, switchLR))
            if tR is not None:
                for i, op_i, op_str in reversed(tR):
                    term.append((op_i, i + shift))
            terms.append(term)
            strengths.append(strength)
        return TermList(terms, strengths)

    def _fill_term_list(self, d0, _connect, term_list=None, term_part=()):
        if term_list is None:
            term_list = [None] * len(self.connections)
        for i, d1 in d0.items():
            if i == _connect:
                for c in d1:
                    term_list[c] = term_part
            else:
                for (op_i, op_str), d2 in d1.items():
                    self._fill_term_list(d2, _connect, term_list, term_part + ((i, op_i, op_str),))
        return term_list

    # to_nn_bond_Arrays: uses `to_TermList`, so same as CouplingTerms.to_nn_bond_Arrays

    def __iadd__(self, other):
        if not isinstance(other, CouplingTerms):
            return NotImplemented  # unknown type of other
        if other.L != self.L:
            raise ValueError("incompatible lengths")
        if not isinstance(other, MultiCouplingTerms):
            #transform coupling to multi coupling
            for i, d0 in other.coupling_terms.items():
                for (op_i, op_str), d1 in d0.items():
                    for j, d2 in d1.items():
                        for op_j, strength in d2.items():
                            ijkl = [i, j]
                            ops_ijkl = [op_i, op_j]
                            self.add_multi_coupling_term(strength, ijkl, ops_ijkl, op_str)
        else:  # add multi coupling
            self._iadd_multi_coupling(other)
            self._max_range = max(self._max_range, other._max_range)
        return self

    def _iadd_multi_coupling(self, other):
        # mix of self.to_TermList() and self.add_multi_coupling_term
        term_list_left = other._fill_term_list(other.terms_left, self._connect_left)
        term_list_right = other._fill_term_list(other.terms_right, self._connect_right)
        for tL, tR, c in zip(term_list_left, term_list_right, other.connections):
            if c is None:
                assert tL is None and tR is None
                continue
            d0L, d0R = self.terms_left, self.terms_right
            for i, op, op_str in tL:
                d1L = d0L.setdefault(i, dict())
                d0L = d1L.setdefault((op, op_str), dict())
            counters_left = d0L.setdefault(self._connect_left, [])
            for i, op, op_str in tR:
                d1R = d0R.setdefault(i, dict())
                d0R = d1R.setdefault((op, op_str), dict())
            counters_right = d0R.setdefault(self._connect_right, [])
            self._insert_connection(counters_left, counters_right, c)

    def _test_terms(self, sites):
        self._test_terms_recursive(sites, self.terms_left, self._connect_left)
        self._test_terms_recursive(sites, self.terms_right, self._connect_right)
        for c in self.connections:
            if c is None:
                continue
            switchLR, op_switch, _, _ = c
            site = sites[switchLR % self.L]
            if not site.valid_opname(op_switch):
                raise ValueError(f"Operator {op_switch!r} not in {site!r}")

    def _test_terms_recursive(self, sites, d0, _connect):
        N_sites = len(sites)
        assert N_sites == self.L
        for i, d1 in d0.items():
            if i == _connect:
                continue
            site_i = sites[i % N_sites]
            for (op_i, op_string_ij), d2 in d1.items():
                if not site_i.valid_opname(op_i):
                    raise ValueError(f"Operator {op_i!r} not in {site_i!r}")
                self._test_terms_recursive(sites, d2, _connect)


class ExponentiallyDecayingTerms(Hdf5Exportable):
    r"""Represent a sum of exponentially decaying (long-range) couplings.

    MPOs can represent translation invariant, exponentially decaying long-range terms of the
    following form with a single extra index of the virtual bonds:

    .. math ::
        sum_{i < j} lambda^{|i-j|} A_i B_j

    For 2D cylinders (or ladders), we need a slight generalization of this, where the operators
    act only on a subset of the sites in each unit cell, given by a 1D array `subsites`:

    .. math ::
        strength sum_{i < j} lambda^{|i-j|} A_{subsites[i]} B_{subsites[j]}

    Note that we still have ``|i-j|``, such that this will give uniformly decaying interactions,
    independent of the way the MPS winds through the 2D lattice, as long as `subsites` is sorted.
    An easy example would be a ladder, where we want the long-range interactions on the first rung
    only, ``subsites = lat.mps_idx_fix_u(u=0)``, see :meth:`~tenpy.models.lattice.mps_idx_fix_u`.

    Parameters
    ----------
    L : int
        Number of sites.

    Attributes
    ----------
    L : int
        Number of sites.
    exp_decaying_terms : list of tuples
        Each tuple ``(strength, opname_i, opname_j, lambda, subsites, opname_string)`` represents
        one of the terms as described above; see :meth:`add_exponentially_decaying_coupling` for
        more details.
    """
    def __init__(self, L):
        assert L > 0
        self.L = L
        self.exp_decaying_terms = []

    def add_exponentially_decaying_coupling(self,
                                            strength,
                                            lambda_,
                                            op_i,
                                            op_j,
                                            subsites=None,
                                            op_string='Id'):
        """Add an exponentially decaying long-range coupling.

        .. math ::
            strength sum_{i < j} lambda^{|i-j|} A_{subsites[i]} B_{subsites[j]}

        Where the operator `A` is given by `op_i`, and `B` is given by `op_j`.
        Note that the sum over i,j is long-range, for infinite systems beyond the MPS unit cell.

        Parameters
        ----------
        strength : float
            Overall prefactor.
        lambda_ : float
            Decay-rate
        op_i, op_j : string
            Names for the operators.
        subsites : None | 1D array
            Selects a subset of sites within the MPS unit cell on which the operators act.
            Needs to be sorted. ``None`` selects all sites.
        op_string : string
            The operator to be inserted between `A` and `B`; for Fermions this should be ``"JW"``.
        """
        if subsites is None:
            subsites = np.arange(self.L)
        else:
            subsites = np.array(subsites)
            if len(subsites) > 1 and np.any(subsites[1:] < subsites[:-1]):
                raise ValueError("subsites needs to be sorted; choose a different MPS ordering!")
            assert subsites[0] >= 0
            assert subsites[-1] < self.L
        self.exp_decaying_terms.append((strength, lambda_, op_i, op_j, subsites, op_string))

    def add_to_graph(self, graph, key="exp-decay"):
        """Add terms from :attr:`onsite_terms` to an MPOGraph.

        Parameters
        ----------
        graph : :class:`~tenpy.networks.mpo.MPOGraph`
            The graph into which the terms from :attr:`exp_decaying_terms` should be added.
        key : str
            Key to distinguish from other `states` in the :class:`~tenpy.networks.mpo.MPOGraph`.
            We find integers `key_nr` and use ``(key_nr, key)`` as `state` for the different
            entries in :attr:`exp_decaying_terms`.
        """
        assert self.L == graph.L
        # get set of states with `key` to find unique `key_nr` for each of the terms
        all_states = set()
        for states in graph.states:
            for label in states:
                try:
                    if label[1] == key:
                        all_states += label
                except:  # not a tuple / wrong types
                    pass
        key_nr = 1000  # start with high value such that they get added in the end of the MPO
        finite = (graph.bc == 'finite')

        for (strength, lambda_, op_i, op_j, subsites, op_string) in self.exp_decaying_terms:
            while (key_nr, key) in all_states:
                key_nr += 1
            label = (key_nr, key)
            all_states.add(label)
            in_subsites = np.zeros(self.L, dtype=np.bool_)
            in_subsites[subsites] = True
            first_subsite = subsites[0]
            last_subsite = subsites[-1]
            assert last_subsite < self.L
            if not finite:
                for i in range(self.L):
                    if in_subsites[i]:
                        graph.add(i, 'IdL', label, op_i, lambda_)
                        graph.add(i, label, label, op_string, lambda_)
                        graph.add(i, label, 'IdR', op_j, strength)
                    else:
                        graph.add(i, label, label, op_string, 1.)
            else:
                # first subsite
                graph.add(first_subsite, 'IdL', label, op_i, lambda_)
                for i in range(first_subsite + 1, last_subsite):
                    if in_subsites[i]:
                        graph.add(i, 'IdL', label, op_i, lambda_)
                        graph.add(i, label, label, op_string, lambda_)
                        graph.add(i, label, 'IdR', op_j, strength)
                    else:
                        graph.add(i, label, label, op_string, 1.)
                graph.add(last_subsite, label, 'IdR', op_j, strength)

        # done

    def to_TermList(self, cutoff=0.01, bc="finite"):
        """Convert self into a :class:`TermList`.

        Parameters
        ----------
        cutoff : float
            Drop terms where the overall prefactor is smaller then `cutoff`.
        bc : "finite" | "infinite"
            Boundary conditions to be used.

        Returns
        -------
        term_list : :class:`TermList`
            Representation of the terms as a list of terms.
            For "infinite" `bc`, only terms starting in the first MPS unit cell are included.
        """
        terms = []
        strengths = []
        L = self.L
        for term in self.exp_decaying_terms:
            strength, lambda_, op_i, op_j, subsites, op_string = term
            N = len(subsites)
            if bc == 'finite':
                for i2, i in enumerate(subsites):
                    for d, j in enumerate(subsites[i2:]):
                        if d == 0:
                            continue
                        pref = strength * lambda_**d
                        if abs(pref) < cutoff:
                            break
                        terms.append([(op_i, i), (op_j, j)])
                        strengths.append(pref)
            elif bc == 'infinite':
                for i2, i in enumerate(subsites):
                    for d in range(1, 1000):
                        j2 = i2 + d
                        j = subsites[j2 % N] + (j2 // N) * L
                        pref = strength * lambda_**d
                        if abs(pref) < cutoff:
                            break
                        terms.append([(op_i, i), (op_j, j)])
                        strengths.append(pref)
                    else:
                        raise ValueError("distance of 1000 not enough to reach precision cutoff")

            else:
                raise ValueError("unknown boundary conditions: " + repr(bc))
        return TermList(terms, strengths)

    def __iadd__(self, other):
        if not isinstance(other, ExponentiallyDecayingTerms):
            return NotImplemented  # unknown type of other
        if other.L != self.L:
            raise ValueError("incompatible lengths")
        self.exp_decaying_terms += other.exp_decaying_terms
        return self

    def max_range(self):
        """Maximum range of the couplings.

        In this case ``np.inf``.
        """
        return np.inf

    def _test_terms(self, sites):
        """Check the format of self.exp_decaying_terms."""
        L = self.L
        for term in self.exp_decaying_terms:
            strength, lambda_, op_i, op_j, subsites, op_string = term
            for i in subsites:
                for op in op_i, op_j:
                    if not sites[i].valid_opname(op):
                        raise ValueError("Operator {op!r} not in site {i:d}".format(op=op, i=i))
        # done
