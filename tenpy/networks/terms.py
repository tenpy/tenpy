"""Classes to store a collection of operator names and sites they act on, together with prefactors.

This modules collects classes which are not strictly speaking tensor networks but represent "terms"
acting on them. Each term is given by a collection of (onsite) operator names and indices of the
sites it acts on. Moreover, we associate a `strength` to each term, which corresponds to the
prefactor when specifying e.g. a Hamiltonian.
"""

import numpy as np
import warnings

from ..linalg import np_conserved as npc
from ..tools.misc import add_with_None_0

__all__ = ['TermList', 'OnsiteTerms', 'CouplingTerms', 'MultiCouplingTerms']


class TermList:
    """A list of terms (=operator names and sites they act on) and associated strengths.

    A representation of terms, similar as :class:`OnsiteTerms`, :class:`CouplingTerms`
    and :class:`MultiCouplingTerms`.

    .. warning :
        In contrast to the :class:`CouplingTerms` and :class:`MultiCouplingTerms`, this class
        does **not** store the operator string between the sites.
        Therefore, conversion from :class:`CouplingTerms` to :class:`TermList` is lossy!

    Parameters
    ----------
    terms : list of list of (str, int)
        List of terms where each `term` is a list of tuples ``(opname, i)``
        of an operator name and a site `i` it acts on.
        For Fermions, the order is the order in the mathematic sense, i.e., the right-most/last
        operator in the list acts last.
    strengths : list of float/complex
        For each term in `terms` an associated prefactor or strength (e.g. expectation value).

    Attributes
    ----------
    terms : list of list of (str, int)
        List of terms where each `term` is a tuple ``(opname, i)`` of an operator name and a site
        `i` it acts on.
    strengths : 1D ndarray
        For each term in `terms` an associated prefactor or strength (e.g. expectation value).
    """

    def __init__(self, terms, strength):
        self.terms = list(terms)
        self.strength = np.array(strength)
        if (len(self.terms), ) != self.strength.shape:
            raise ValueError("different length of terms and strength")

    def to_OnsiteTerms_CouplingTerms(self, sites):
        """Convert to :class:`OnsiteTerms` and :class:`CouplingTerms`

        Parameters
        ----------
        sites : list of {:class:`~tenpy.networks.site.Site` | None}
            Defines the local Hilbert space for each site.
            Used to check whether the operators need Jordan-Wigner strings.
            The length is used as `L` for the `onsite_terms` and `coupling_terms`.
            Use ``[None]*L`` if you don't want to check for Jordan-Wigner.

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
        if any(len(t) > 2 for t in self.terms):
            ct = MultiCouplingTerms(L)
        else:
            ct = CouplingTerms(L)
        for term, strength in self:
            if len(term) == 1:
                op, i = term[0]
                ot.add_onsite_term(strength, i, op)
            elif len(term) == 2:
                op_needs_JW = [(sites[i % L] is not None and sites[i % L].op_needs_JW(op))
                               for op, i in term]
                args = ct.coupling_term_handle_JW(term, op_needs_JW)
                ct.add_coupling_term(strength, *args)
            elif len(term) > 2:
                op_needs_JW = [(sites[i % L] is not None and sites[i % L].op_needs_JW(op))
                               for op, i in term]
                args = ct.multi_coupling_term_handle_JW(term, op_needs_JW)
                ct.add_multi_coupling_term(strength, *args)
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


class OnsiteTerms:
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
        for i, terms in enumerate(self.onsite_terms):
            for opname, strength in terms.items():
                graph.add(i, 'IdL', 'IdR', opname, strength)

    def to_Arrays(self, sites):
        """Convert the :attr:`onsite_terms` into a list of np_conserved Arrays.

        Parameters
        ----------
        sites : list of :class:`~tenpy.networks.site.Site`
            The sites for translating the operator names on the :attr:`L` sites into
            :class:`~tenpy.linalg.np_conserved.Array`.

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
            The sites for translating the operator names on the :attr:`L` sites into
            :class:`~tenpy.linalg.np_conserved.Array`.
        distribute : (float, float)
            How to split the onsite terms (in the bulk) into the bond terms to the left
            (``distribute[0]``) and right (``distribute[1]``).
        finite : bool
            Boundary conditions of the MPS, :attr:`MPS.finite`.
            If finite, we distribute the onsite term of the
        """
        dist_L, dist_R = distribute
        if dist_L + dist_R != 1.:
            warnings.warn("sum of `distribute` not 1!!!")
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


class CouplingTerms:
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

    def coupling_term_handle_JW(self, term, op_needs_JW, op_string=None):
        """Helping function to call before :meth:`add_multi_coupling_term`.

        Sort and groups the operators by sites they act on, such that the returned
        `ijkl` is strictly ascending, i.e. has entries `i < j < k < l`.
        Moreover, handle/figure out Jordan-Wigner strings if needed.

        Parameters
        ----------
        term : [(str, int), (str, int)]
            List of two tuples ``(op, i)`` where `i` is the MPS index of the site the operator
            named `op` acts on.
        op_needs_JW : list of bool
            For each entry in term whether the operator needs a JW string.
            Only used if `op_string` is None.
        op_string : None | str
            Operator name to be used as operator string *between* the operators, or ``None`` if the
            Jordan Wigner string should be figured out.

            .. warning :
                ``None`` figures out for each segment between the operators, whether a
                Jordan-Wigner string is needed.
                This is different from a plain ``'JW'``, which just applies a string on
                each segment!

        Returns
        -------
        i, j, op_i, op_j, op_string:
            Arguments for :meth:`MultiCouplingTerms.add_multi_coupling_term` such that the added
            term corresponds to the parameters of this function.
        """
        L = self.L
        (op_i, i), (op_j, j) = term
        if op_string is None:
            need_JW1, need_JW2 = op_needs_JW
            if need_JW1 and need_JW2:
                op_string = 'JW'
            elif need_JW1 or need_JW2:
                raise ValueError("Only one of the operators needs a Jordan-Wigner string?!")
            else:
                op_string = 'Id'
        swap = (j < i)
        if swap:
            op_i, i, op_j, j = op_j, j, op_i, i
        if op_string == 'JW':
            if swap:
                op_i = ' '.join([op_string, op_i])  # op_i=(original op_j) should act first
            else:
                op_i = ' '.join([op_i, op_string])  # op_j should act first
        return i, j, op_i, op_j, op_string

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
                """define the plot style for a given coupling"""
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
        # structure of coupling terms:
        # {i: {('opname_i', 'opname_string'): {j: {'opname_j': strength}}}}
        for i, d1 in self.coupling_terms.items():
            for (opname_i, op_string), d2 in d1.items():
                label = (i, opname_i, op_string)
                graph.add(i, 'IdL', label, opname_i, 1., skip_existing=True)
                for j, d3 in d2.items():
                    label_j = graph.add_string(i, j, label, op_string)
                    for opname_j, strength in d3.items():
                        graph.add(j, label_j, 'IdR', opname_j, strength)
        # done

    def to_nn_bond_Arrays(self, sites):
        """Convert the :attr:`coupling_terms` into Arrays on nearest neighbor bonds.

        Parameters
        ----------
        sites : list of :class:`~tenpy.networks.site.Site`
            The sites for translating the operator names on the :attr:`L` sites into
            :class:`~tenpy.linalg.np_conserved.Array`.

        Returns
        -------
        H_bond : list of {:class:`~tenpy.linalg.np_conserved.Array` | None}
            The :attr:`coupling_terms` rewritten as ``sum_i H_bond[i]`` for MPS indices ``i``.
            ``H_bond[i]`` acts on sites ``(i-1, i)``, ``None`` represents 0.
            Legs of each ``H_bond[i]`` are ``['p0', 'p0*', 'p1', 'p1*']``.
        """
        N_sites = self.L
        if len(sites) != N_sites:
            raise ValueError("incompatible length")
        H_bond = [None] * N_sites
        for i, d1 in self.coupling_terms.items():
            j = (i + 1) % N_sites
            site_i = sites[i]
            site_j = sites[j]
            H = H_bond[j]
            for (op1, op_str), d2 in d1.items():
                for j2, d3 in d2.items():
                    if isinstance(j2, tuple):
                        # This should only happen in a MultiSiteCoupling model
                        raise ValueError("MultiCouplingTerms: can't generate H_bond")
                    # i, j in coupling_terms are defined such that we expect j2 = i + 1
                    if j2 != i + 1:
                        msg = "Can't give nearest neighbor H_bond for long-range {i:d}-{j:d}"
                        raise ValueError(msg.format(i=i, j=j2))
                    for op2, strength in d3.items():
                        H_add = strength * npc.outer(site_i.get_op(op1), site_j.get_op(op2))
                        H = add_with_None_0(H, H_add)
            if H is not None:
                H.iset_leg_labels(['p0', 'p0*', 'p1', 'p1*'])
            H_bond[j] = H
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
        """Check the format of self.coupling_terms"""
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
    an arbitrary recursion depth of the dictionary.

    Parameters
    ----------
    L : int
        Number of sites.

    Attributes
    ----------
    L : int
        Number of sites.
    coupling_terms : dict of dict
        Nested dictionaries of the following form::

            {i: {('opname_i', 'opname_string_ij'):
                    {j: {('opname_j', 'opname_string_jk'):
                            {k: {('opname_k', 'opname_string_kl'):
                                ...
                                    {l: {'opname_l':
                                            strength
                                    }   }
                                ...
                            }   }
                    }   }
            }   }

        For a M-site coupling, this involves a nesting depth of ``2*M`` dictionaries.
        Note that always ``i < j < k < ... < l``, but entries with ``j,k,l >= L``
        are allowed for the case of ``bc_MPS == 'infinite'``, when they indicate couplings
        between different iMPS unit cells.

    """

    def add_multi_coupling_term(self, strength, ijkl, ops_ijkl, op_string="Id"):
        """Add a multi-site coupling term.

        Parameters
        ----------
        strength : float
            The strength of the coupling term.
        ijkl : list of int
            The MPS indices of the sites on which the operators acts. With `i, j, k, ... = ijkl`,
            we require that they are ordered ascending, ``i < j < k < ...`` and
            that ``0 <= i < N_sites``.
            Inidces >= N_sites indicate couplings between different unit cells of an infinite MPS.
        ops_ijkl : list of str
            Names of the involved operators on sites `i, j, k, ...`.
        op_string : (list of) str
            Names of the operator to be inserted between the operators,
            e.g., op_string[0] is inserted between `i` and `j`.
            A single name holds for all in-between segments.
        """
        if len(ijkl) < 2:
            raise ValueError("Need to act on at least 2 sites. Use onsite terms!")
        if isinstance(op_string, str):
            op_string = [op_string] * (len(ijkl) - 1)
        assert len(ijkl) == len(ops_ijkl) == len(op_string) + 1
        for i, j in zip(ijkl, ijkl[1:]):
            if not i < j:
                raise ValueError("Need i < j < k < ...")
        # create the nested structure
        # {ijkl[0]: {(ops_ijkl[0], op_string[0]):
        #            {ijkl[1]: {(ops_ijkl[1], op_string[1]):
        #                       ...
        #                           {ijkl[-1]: {ops_ijkl[-1]: strength}
        #            }         }
        # }         }
        d0 = self.coupling_terms
        for i, op, op_str in zip(ijkl, ops_ijkl, op_string):
            d1 = d0.setdefault(i, dict())
            d0 = d1.setdefault((op, op_str), dict())
        d1 = d0.setdefault(ijkl[-1], dict())
        op = ops_ijkl[-1]
        d1[op] = d1.get(op, 0) + strength

    def multi_coupling_term_handle_JW(self, term, op_needs_JW, op_string=None):
        """Helping function to call before :meth:`add_multi_coupling_term`.

        Sort and groups the operators by sites they act on, such that the returned
        `ijkl` is strictly ascending, i.e. has entries `i < j < k < l`.
        Moreover, handle/figure out Jordan-Wigner strings if needed.

        Parameters
        ----------
        term : list of (str, int)
            List of tuples ``(op, i)`` where `i` is the MPS index of the site the operator
            named `op` acts on.
        op_needs_JW : list of bool
            For each entry in term whether the operator needs a JW string.
            Only used if `op_string` is None.
        op_string : None | str
            Operator name to be used as operator string *between* the operators, or ``None`` if the
            Jordan Wigner string should be figured out.

            .. warning :
                ``None`` figures out for each segment between the operators, whether a
                Jordan-Wigner string is needed.
                This is different from a plain ``'JW'``, which just applies a string on
                each segment!

        Returns
        -------
        ijkl, ops_ijkl, op_string :
            Arguments for :meth:`MultiCouplingTerms.add_multi_coupling_term` such that the added term
            corresponds to the parameters of this function.
        """
        L = self.L
        number_ops = len(term)
        if number_ops < 2:
            raise ValueError("expect multi coupling")
        ops = [t[0] for t in term]
        ijkl = [t[1] for t in term]
        reorder = np.argsort(ijkl, kind='mergesort')  # need stable kind!!!
        i0 = ijkl[reorder[0]]
        if not 0 <= i0 < L:  # ensure this condition with a shift
            shift = i0 % L - i0
            ijkl = [i + shift for i in ijkl]
        # what we want to calculate:
        new_ijkl = []
        new_ops = []
        new_op_str = []  # new_op_string[x] is right of new_ops[x]
        # first make groups with strictly ``i < j < k < ... ``
        i0 = -1  # != the first i since -1 <  0 <= ijkl[:]
        grouped_reorder = []
        for x in reorder:
            i = ijkl[x]
            if i != i0:
                i0 = i
                new_ijkl.append(i)
                grouped_reorder.append([x])
            else:
                grouped_reorder[-1].append(x)
        if op_string is not None:
            # simpler case
            for group in grouped_reorder:
                new_ops.append(' '.join([ops[x] for x in group]))
                new_op_str.append(op_string)
            new_op_str.pop()  # remove last entry (created one too much)
        else:
            # more complicated: handle Jordan-Wigner
            for a, group in enumerate(grouped_reorder):
                right = [z for gr in grouped_reorder[a + 1:] for z in gr]
                onsite_ops = []
                need_JW_right = False
                JW_max = -1
                for x in group + [number_ops]:
                    JW_min, JW_max = JW_max, x
                    need_JW = (np.sum([op_needs_JW[z]
                                       for z in right if JW_min < z < JW_max]) % 2 == 1)
                    if need_JW:
                        onsite_ops.append('JW')
                        need_JW_right = not need_JW_right
                    if x != number_ops:
                        onsite_ops.append(ops[x])
                new_ops.append(' '.join(onsite_ops))
                op_str_right = 'JW' if need_JW_right else 'Id'
                new_op_str.append(op_str_right)
            new_op_str.pop()  # remove last entry (created one too much)
        return new_ijkl, new_ops, new_op_str

    def max_range(self):
        """Determine the maximal range in :attr:`coupling_terms`.

        Returns
        -------
        max_range : int
            The maximum of ``j - i`` for the `i`, `j` occuring in a term of :attr:`coupling_terms`.
        """
        max_range = 0
        for i, d1 in self.coupling_terms.items():
            max_range = max(max_range, self._max_range(i, d1))
        return max_range

    def add_to_graph(self, graph, _i=None, _d1=None, _label_left=None):
        """Add terms from :attr:`coupling_terms` to an MPOGraph.

        Parameters
        ----------
        graph : :class:`~tenpy.networks.mpo.MPOGraph`
            The graph into which the terms from :attr:`coupling_terms` should be added.
        _i, _d1, _label_left : None
            Should not be given; only needed for recursion.
        """
        # nested structure of coupling_terms:
        # d0 = {i: {('opname_i', 'opname_string_ij'): ... {l: {'opname_l': strength}}}
        if _i is None:  # beginning of recursion
            for i, d1 in self.coupling_terms.items():
                self.add_to_graph(graph, i, d1, 'IdL')
        else:
            for key, d2 in _d1.items():
                if isinstance(key, tuple):  # further nesting
                    op_i, op_string_ij = key
                    if isinstance(_label_left, str) and _label_left == 'IdL':
                        label = (_i, op_i, op_string_ij)
                    else:
                        label = _label_left + (_i, op_i, op_string_ij)
                    graph.add(_i, _label_left, label, op_i, 1., skip_existing=True)
                    for j, d3 in d2.items():
                        label_j = graph.add_string(_i, j, label, op_string_ij)
                        self.add_to_graph(graph, j, d3, label_j)
                else:  # maximal nesting reached: exit recursion
                    # i is actually the `l`
                    op_i, strength = key, d2
                    graph.add(_i, _label_left, 'IdR', op_i, strength)
        # done

    def remove_zeros(self, tol_zero=1.e-15, _d0=None):
        """Remove entries close to 0 from :attr:`coupling_terms`.

        Parameters
        ----------
        tol_zero : float
            Entries in :attr:`coupling_terms` with `strength` < `tol_zero` are considered to be
            zero and removed.
        _d0 : None
            Should not be given; only needed for recursion.
        """
        if _d0 is None:
            _d0 = self.coupling_terms
        # d0 = ``{i: {('opname_i', 'opname_string_ij'): ... {j: {'opname_j': strength}}}``
        for i, d1 in list(_d0.items()):
            for key, d2 in list(d1.items()):
                if isinstance(key, tuple):
                    self.remove_zeros(tol_zero, d2)  # recursive!
                    if len(d2) == 0:
                        del d1[key]
                else:
                    # key is opname_j, d2 is strength
                    if abs(d2) < tol_zero:
                        del d1[key]
            if len(d1) == 0:
                del _d0[i]
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
        self._to_TermList(terms, strength, [], self.coupling_terms)
        return TermList(terms, strength)

    def __iadd__(self, other):
        if not isinstance(other, CouplingTerms):
            return NotImplemented  # unknown type of other
        if other.L != self.L:
            raise ValueError("incompatible lengths")
        self._iadd_multi_coupling_terms(self.coupling_terms, other.coupling_terms)
        return self

    def _max_range(self, i, d1):
        max_range = 0
        for key, d2 in d1.items():
            if isinstance(key, tuple):
                # further couplings: d2 is dictionary
                j_max = max(d2.keys())
                max_range = max(max_range, j_max - i)
                for d3 in d2.values():
                    max_range = max(max_range, self._max_range(i, d3))
            # else: d2 is strength, last coupling reached
        return max_range

    def _to_TermList(self, terms, strength, term0, d0):
        for i in sorted(d0):
            d1 = d0[i]
            d1_keys_tuple = []
            d1_keys_str = []
            for key in d1.keys():
                if isinstance(key, tuple):
                    d1_keys_tuple.append(key)
                else:
                    d1_keys_str.append(key)
            for opname_i in sorted(d1_keys_str):
                # maximum recursion reached
                terms.append(term0 + [(opname_i, i)])
                strength.append(d1[opname_i])
            for key in sorted(d1_keys_tuple):
                (opname_i, op_str) = key
                term1 = term0 + [(opname_i, i)]
                d2 = d1[key]
                self._to_TermList(terms, strength, term1, d2)
        # done

    def _iadd_multi_coupling_terms(self, self_d0, other_d0):
        # {ijkl[0]: {(ops_ijkl[0], op_string[0]):
        #            {ijkl[1]: {(ops_ijkl[1], op_string[1]):
        #                       ...
        #                           {ijkl[-1]: {ops_ijkl[-1]: strength}
        #            }         }
        # }         }
        # d0 = ``{i: {('opname_i', 'opname_string_ij'): ... {j: {'opname_j': strength}}}``
        for i, other_d1 in other_d0.items():
            self_d1 = self_d0.setdefault(i, dict())
            for key, other_d2 in other_d1.items():
                if isinstance(key, tuple):  # further couplings
                    self_d2 = self_d1.setdefault(key, dict())
                    self._iadd_multi_coupling_terms(self_d2, other_d2)  # recursive!
                else:  # last term of the coupling
                    opname_j = key
                    strength = other_d2
                    self_d1[opname_j] = self_d1.get(opname_j, 0.) + strength
        # done

    def _test_terms(self, sites, d0=None):
        N_sites = len(sites)
        if d0 is None:
            d0 = self.coupling_terms
        for i, d1 in d0.items():
            site_i = sites[i % N_sites]
            for key, d2 in d1.items():
                if isinstance(key, tuple):  # further couplings
                    op_i, opstring_ij = key
                    if not site_i.valid_opname(op_i):
                        raise ValueError("Operator {op!r} not in site".format(op=op_i))
                    self._test_terms(sites, d2)  # recursive!
                else:  # last term of the coupling
                    op_i = key
                    if not site_i.valid_opname(op_i):
                        raise ValueError("Operator {op!r} not in site".format(op=op_i))
        # done
