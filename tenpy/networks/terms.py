"""Classes to store a collection of operator names and sites they act on, together with prefactors.

This modules collects classes which are not strictly speaking tensor networks but represent "terms"
acting on them. Each term is given by a collection of (onsite) operator names and indices of the
sites it acts on. Moreover, we associate a `strength` to each term, which corresponds to the
prefactor when specifying e.g. a Hamiltonian.
"""
# Copyright (C) TeNPy Developers, Apache license

import itertools
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Protocol

import cyten as ct
import numpy as np

from ..tools import Hdf5Exportable, add_with_None_0

if TYPE_CHECKING:
    from .mpo import MPOGraph


__all__ = [
    'TermList',
    'Terms',
    'OnsiteTerms',
    'CouplingTerms',
    'MultiCouplingTerms',
    'ExponentiallyDecayingTerms',
    'order_combine_term',
]


class DefinesCouplings(Protocol):
    """Placeholder, just for typehints, until we implement which class handles coupling names"""

    ...

    def coupling_n_sites(self, name: str) -> int: ...

    def decompose_coupling(self, name: str, *sites: ct.Site) -> list[tuple[complex, list[ct.SymmetricTensor]]]: ...

    def permute_coupling(self, name: str, perm: Sequence[int]) -> dict[str, float | complex]: ...

    def _get_switch_LR(self, name: str, *locs: int) -> int:
        assert self.coupling_n_sites(name) == len(locs)
        return len(locs) // 2

    def get_coupling(self, name: str, *sites: ct.Site) -> ct.Coupling: ...

    def has_coupling(self, name: str, *sites: ct.Site): ...  # raise ValueError

    def get_hc_coupling_name(self, name: str) -> str:
        ...
        


class Terms(Hdf5Exportable, metaclass=ABCMeta):
    """Base class for representations of terms that can be added to an MPO.

    TODO elaborate on classes and their differences.
    """

    @abstractmethod
    def max_range(self) -> int:
        """Maximum range of the terms in the MPS geometry.

        This is e.g. ``0`` for purely onsite or ``1`` for nearest-neighbor
        """
        ...

    @abstractmethod
    def add_to_graph(self, graph: MPOGraph):
        """Add the represented terms to an MPOGraph.

        Parameters
        ----------
        graph : :class:`~tenpy.networks.mpo.MPOGraph`
            The graph into which the terms from :attr:`onsite_terms` should be added.

        """

    @abstractmethod
    def test_terms(self, sites: list[ct.Site], model: DefinesCouplings):
        """Check that the terms are valid for a given model on given sites.

        Raises
        ------
        ValueError
            If something is inconsistent

        """
        ...

    @abstractmethod
    def to_TermList(self) -> TermList:
        """TODO: review termlist first"""
        ...

    @abstractmethod
    def remove_zeros(self, tol_zero: float = 1e-15, inplace: bool = True) -> Terms:
        """Remove terms with (almost) zero prefactor.

        Parameters
        ----------
        tol_zero : float
            Tolerance for considering prefactors zero
        inplace : bool
            Modify inplace (default) or work on a copy.

        Returns
        -------
        The modified instance.

        """
        ...


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

        >>> t = TermList([[('Cd', 0), ('C', 2)], [('Cd', 2), ('C', 0)], [('N', 1)]], [0.5, 0.5, 1.3])
        >>> print(t)
        0.50000 * Cd_0 C_2 +
        0.50000 * Cd_2 C_0 +
        1.30000 * N_1

    If you have a :class:`~tenpy.models.lattice.Lattice`, you might also want to specify
    the location of the operators by lattice indices instead of MPS indices.
    For example, you can obtain the nearest-neighbor density terms
    **without double counting each pair**) on a :class:`~tenpy.models.lattice.TriangularLattice`:

    .. doctest :: TermList

        >>> lat = tenpy.models.lattice.Triangular(6, 6, None, bc_MPS='infinite', bc='periodic')
        >>> t2_terms = [
        ...     [('N', [0, 0, u1]), ('N', [dx[0], dx[1], u2])] for (u1, u2, dx) in lat.pairs['nearest_neighbors']
        ... ]
        >>> t2 = TermList.from_lattice_locations(lat, t2_terms)
        >>> print(t2)
        1.00000 * N_0 N_6 +
        1.00000 * N_0 N_-5 +
        1.00000 * N_0 N_5

    The negative index -5 here indicates a tensor left of the current MPS unit cell.

    """

    def __init__(self, terms, strength=1.0):
        raise NotImplementedError  # Check what we even need it for...
        self.terms = list(terms)
        self.strength = np.array(strength)
        if self.strength.ndim == 0:
            self.strength = np.ones([len(self.terms)]) * self.strength
        if (len(self.terms),) != self.strength.shape:
            raise ValueError('different length of terms and strength')

    @classmethod
    def from_lattice_locations(cls, lattice, terms, strength=1.0, shift=None):
        """Initialize from a list of terms given in lattice indices instead of MPS indices.

        Parameters
        ----------
        lattice : :class:`~tenpy.models.lattice.Lattice`
            The underlying lattice to be used for conversion, e.g. `M.lat` from a
            :class:`~tenpy.models.model.Model`.
        terms : list of list of (str, tuple)
            List of terms, where each `term` is a tuple ``(opname, lat_idx)`` with
            `lat_idx` itself being a tuple ``(x, y, u)`` (for a 2D lattice) of the lattice
            coordinates.
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
                raise ValueError('wrong length of `shift`: ' + repr(shift))
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
                args = ct.coupling_term_handle_JW(strength, term, sites)  # TODO no longer exists!
                ct.add_coupling_term(*args)
            elif len(term) > 2:
                args = ct.multi_coupling_term_handle_JW(strength, term, sites)  # TODO no longer exists!
                ct.add_multi_coupling_term(*args)
            else:
                raise ValueError('term without entry!?')
        return ot, ct

    def __iter__(self):
        """Iterate over ``zip(self.terms, self.strength)``."""
        return zip(self.terms, self.strength)

    def __add__(self, other):
        if isinstance(other, TermList):
            return TermList(self.terms + other.terms, np.concatenate((self.strength, other.strength)))
        return NotImplemented

    def __mul__(self, other):
        return TermList(self.terms, self.strength * other)

    def __str__(self):
        res = []
        for term, strength in self:
            ops = []
            for op, i in term:
                if ' ' in op:
                    op = '[' + op + ']'
                ops.append(f'{op!s}_{i:d}')
            term_str = ' '.join(ops)
            res.append(f'{strength:.5f} * {term_str}')
        return ' +\n'.join(res)

    def order_combine(self, sites):
        """Order and combine operators in each term.

        Parameters
        ----------
        sites : list of :class:`~tenpy.networks.site.Site`
            Defines the local Hilbert space for each site.
            Used to check whether the operators anticommute
            (= whether they need Jordan-Wigner strings) and for multiplication rules.

        See Also
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
    # TODO probably does not fit in the new coupling framework that well anymore.
    #      for non-symmetric braids, we would at the very least need to define the chirality.
    raise NotImplementedError
    # Group all operators that are on the same site and get the corresponding sign
    L = len(sites)
    N = len(term)
    overall_sign = 1
    terms_commute = [(op, i, sites[i % L].op_needs_JW(op)) for op, i in term]
    # perform bubble sort on terms_commute and keep track of the sign
    if N > 100:  # bubblesort is O(N^2), assume that N is small
        # N = 1000 takes ~1s, so 100 should be fine...
        warnings.warn('not intended for large number of operators.')
    for s_max in range(N - 1, 0, -1):
        for s in range(s_max):
            t1, t2 = terms_commute[s : s + 2]
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


class OnsiteTerms(Terms):
    """Operator names, site indices and strengths representing onsite terms.

    Represents a sum of onsite terms where the operators are only given by their name
    (in the form of a string). What the operator represents is later given by a list of
    :class:`cyten.Site` via :meth:`~cyten.Site.get_op`.

    Parameters
    ----------
    L : int
        Number of sites.

    Attributes
    ----------
    L : int
        Number of sites.
    onsite_terms : list of {str: float | complex}
        Filled by meth:`add_onsite_term`.
        For each index `i` a dictionary ``{'opname': strength}`` defining the onsite terms.

    """

    def __init__(self, L):
        assert L > 0
        self.L = L
        self.onsite_terms: list[dict[str, float | complex]] = [dict() for _ in range(L)]

    def max_range(self):
        return 0

    def add_onsite_term(self, strength: float | complex, i: int, op: str):
        """Add an onsite term on a given MPS site.

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

    def add_to_graph(self, graph: MPOGraph):
        raise NotImplementedError  # TODO review MPOGraph first
        assert self.L == graph.L
        for i, terms in enumerate(self.onsite_terms):
            for opname, strength in terms.items():
                graph.add(i, 'IdL', 'IdR', opname, strength)
        if graph.max_range is not None:
            graph.max_range = max(graph.max_range, 1)

    def to_tensors(self, sites: list[ct.Site]) -> ct.SymmetricTensor | None:
        """Convert the represneted onsite terms into a list of tensors.

        Parameters
        ----------
        sites : list of :class:`cyten.Site`
            Defines the local Hilbert space for each site.
            Used to translate the operator names into tensors.

        Returns
        -------
        onsite_arrays : list of {:class:`~cyten.SymmetricTensor` | ``None``}
            Onsite terms represented by ``self``. Entry ``i`` of the list lives on ``sites[i]``.
            An entry ``None`` represents zero.

        """
        if len(sites) != self.L:
            raise ValueError('Incompatible length')
        res = []
        for site, terms in zip(sites, self.onsite_terms):
            H = None
            for opname, strength in terms.items():
                H = add_with_None_0(H, strength * site.get_op(opname))
                # Note: H can change from None to npc Array and can change dtype
            res.append(H)
        return res

    def remove_zeros(self, tol_zero: float = 1.0e-15, inplace: bool = True) -> OnsiteTerms:
        if not inplace:
            return deepcopy(self).remove_zeros(tol_zero)
        for term in self.onsite_terms:
            for op in list(term.keys()):
                if abs(term[op]) < tol_zero:
                    del term[op]
        return self

    def add_to_nn_bond_tensors(
        self,
        H_bond: list[ct.SymmetricTensor],
        sites: list[ct.Site],
        finite: bool,
        distribute: tuple[float, float] = (0.5, 0.5),
    ):
        """Add the represented terms to nearest-neighbor bond tensors, in-place.

        Parameters
        ----------
        H_bond : list of {:class:`~cyten.SymmetricTensor` | None}
            Bond terms representing a sum ``sum_i H_bond[i]``, where ``H_bond[i]`` acts on
            sites ``(i-1, i)``, ``None`` represents 0.
            The list in modified *in-place*, such that the :attr:`onsite_terms` represented by
            this instance are added to it.
            Legs of each ``H_bond[i]`` are ``['p0', 'p0*', 'p1', 'p1*']``.
        sites : list of :class:`cyten.Site`
            Defines the local Hilbert space for each site.
            Used to translate the operator names into tensors.
        distribute : (float, float)
            How to split the onsite terms (in the bulk) into the bond terms to the left
            (``distribute[0]``) and right (``distribute[1]``).
            At a *finite* boundary, this is ignored, since there is only one bond term to
            distribute over.
        finite : bool
            Boundary conditions of the MPS, :attr:`MPS.finite`.

        """
        dist_L, dist_R = distribute
        if dist_L + dist_R != 1.0:
            raise ValueError('sum of `distribute` not 1!')
        assert all(H is None or H.labels_are('p0', 'p1', 'p1*', 'p0*', planar=True) for H in H_bond)

        for j, H_j in enumerate(self.to_tensors(sites)):
            if H_j is None:
                continue
            if finite and j == 0:
                dist_L, dist_R = 0.0, 1.0
            elif finite and j == self.L - 1:
                dist_L, dist_R = 1.0, 0.0
            else:
                dist_L, dist_R = distribute
            if dist_L != 0.0:
                i = (j - 1) % self.L
                h = ct.outer(
                    sites[i].identity_tensor, H_j, relabel1={'p': 'p0', 'p*': 'p0*'}, relabel2={'p': 'p1', 'p*': 'p1*'}
                )
                H_bond[j] = add_with_None_0(H_bond[j], dist_L * h)
            if dist_R != 0.0:
                k = (j + 1) % self.L
                h = ct.outer(
                    H_j, sites[k].identity_tensor, relabel1={'p': 'p0', 'p*': 'p0*'}, relabel2={'p': 'p1', 'p*': 'p1*'}
                )
                H_bond[k] = add_with_None_0(H_bond[k], dist_R * h)
        assert all(H is None or H.labels_are('p0', 'p1', 'p1*', 'p0*', planar=True) for H in H_bond)

    def to_TermList(self):
        """Convert :attr:`onsite_terms` into a :class:`TermList`.

        Returns
        -------
        term_list : :class:`TermList`
            Representation of the terms as a list of terms.

        """
        raise NotImplementedError  # TODO review Termlist first!
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
            raise ValueError('incompatible lengths')
        for self_t, other_t in zip(self.onsite_terms, other.onsite_terms):
            for key, value in other_t.items():
                self_t[key] = self_t.get(key, 0.0) + value
        return self

    def test_terms(self, sites: list[ct.Site], model: DefinesCouplings):
        """Check that all given operators exist in the `sites`."""
        for site, terms in zip(sites, self.onsite_terms):
            for opname in terms:
                if not site.valid_opname(opname):
                    raise ValueError(f'Operator {opname!r} not in site')


class CouplingTerms(Terms):
    """Operator names, site indices and strengths representing two-site coupling terms.

    Parameters
    ----------
    L : int
        Number of sites.

    Attributes
    ----------
    L : int
        Number of sites.
    coupling_terms : {int: {}}
        Filled by :meth:`add_coupling_term`.
        Nested dictionaries such that ``coupling_terms[i][coupling_name][j] = strength`` describes
        a coupling with a string name ``coupling_name`` acting on sites ``i < j``.
        Note that always ``0 <= i < L``, but entries with ``j >= L`` are allowed for
        ``bc_MPS == 'infinite'``, in which case they indicate couplings between different
        iMPS unit cells.

    """

    def __init__(self, L):
        assert L > 0
        self.L = L
        self.coupling_terms: dict[int, dict[str, dict[int, float | complex]]] = dict()

    def max_range(self) -> int:
        max_range = 0
        for i, d1 in self.coupling_terms.items():
            for d2 in d1.values():
                j_max = max(d2.keys())
                max_range = max(max_range, j_max - i)
        return max_range

    def add_coupling_term(self, strength: float | complex, i: int, j: int, coupling_name: str):
        """Add a two-site coupling term on given MPS sites.

        Parameters
        ----------
        strength : float
            The strength of the coupling term.
        i, j : int
            The MPS indices of the two sites on which the operator acts.
            We require ``0 <= i < L``  and ``i < j``.
            If ``j >= L``, it indicates couplings between unit cells of an infinite MPS.
        coupling_name : str
            Names of the coupling.

        """
        if not 0 <= i < self.L:
            raise ValueError(f'We need 0 <= i < L, got i={i:d}')
        if not i < j:
            raise ValueError('need i < j')
        d1 = self.coupling_terms.setdefault(i, dict())
        # form of d1: ``{('opname_i', 'opname_string'): {j: {'opname_j': current_strength}}}``
        d2 = d1.setdefault(coupling_name, dict())
        d2[j] = d2.get(j, 0) + strength

    def plot_coupling_terms(
        self, ax, lat, style_map='default', common_style={'linestyle': '--'}, text=None, text_pos=0.4
    ):
        """Plot coupling terms into a given lattice.

        This function plots the :attr:`coupling_terms`

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which we should plot.
        lat : :class:`~tenpy.models.lattice.Lattice`
            The lattice for plotting the couplings, most probably the ``M.lat`` of the
            corresponding model ``M``, see :attr:`~tenpy.models.model.Model.lat`.
        style_map : function | None
            Function which get's called with arguments ``i, j, coupling_name, strength``
            for each two-site coupling and should return a keyword-dictionary with the desired
            plot-style for this coupling.
            By default (``None``), the `linewidth` is given by the absolute value of `strength`,
            and the linecolor depends on the phase of `strength` (using the `hsv` colormap).
        common_style : dict
            Common style, which overwrites values of the dictionary returned by style_map.
            A ``'label'`` is only used for the first plotted line.
        text: format_string | None
            If not ``None``, we add text labeling the couplings in the plot.
            Available keywords are ``i, j, coupling_name, strength`` as well as
            ``strength_abs, strength_angle, strength_real``.
        text_pos : float
            Specify where to put the text on the line between `i` (0.0) and `j` (1.0),
            e.g. `0.5` is exactly in the middle between `i` and `j`.

        See Also
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

            def style_map(i, j, coupling_name, strength):
                """Define the plot style for a given coupling."""
                style = {}
                style['linewidth'] = np.abs(strength) * matplotlib.rcParams['lines.linewidth']
                style['color'] = hsv(norm_angle(np.angle(strength)))
                return style

        text_pos = np.array([1.0 - text_pos, text_pos], np.float64)
        for i in sorted(self.coupling_terms.keys()):
            terms_i = self.coupling_terms[i]
            x_y[0, :] = pos[i]
            for coupling_name in sorted(terms_i.keys()):
                terms_i_name = terms_i[coupling_name]
                for j in sorted(terms_i_name.keys()):
                    strength = terms_i_name[j]
                    shift = j - j % N_sites
                    if shift == 0:
                        x_y[1, :] = pos[j]
                    else:
                        lat_idx_j = np.array(lat.order[j % N_sites])
                        lat_idx_j[0] += (shift // N_sites) * lat.N_rings
                        x_y[1, :] = lat.position(lat_idx_j)
                    if style_map:
                        style = style_map(i, j, coupling_name, strength)
                    else:
                        style = {}
                    style.update(common_style)
                    ax.plot(x_y[:, 0], x_y[:, 1], **style)
                    if 'label' in common_style:
                        common_style = common_style.copy()
                        del common_style['label']
                    if text:
                        annotate = text.format(
                            i=i,
                            j=j,
                            coupling_name=coupling_name,
                            strength=strength,
                            strength_abs=np.abs(strength),
                            strength_real=np.real(strength),
                            strength_angle=np.angle(strength),
                        )
                        loc = np.dot(x_y.T, text_pos)
                        ax.text(loc[0], loc[1], annotate)
        # done

    def add_to_graph(self, graph: MPOGraph):
        raise NotImplementedError  # TODO adapt MPOGraph first
        assert self.L == graph.L
        # structure of coupling terms:
        # {i: {('opname_i', 'opname_string'): {j: {'opname_j': strength}}}}
        # TODO consider using
        # for i, name, j, strength in self:
        for i, d1 in self.coupling_terms.items():
            for (opname_i, op_string), d2 in d1.items():
                label = ('left', i, opname_i, op_string)
                graph.add(i, 'IdL', label, opname_i, 1.0, skip_existing=True)
                for j, d3 in d2.items():
                    label_j = graph.add_string_left_to_right(i, j, label, op_string)
                    for opname_j, strength in d3.items():
                        graph.add(j, label_j, 'IdR', opname_j, strength)
        if graph.max_range is not None:
            graph.max_range = max(graph.max_range, self.max_range())

    def __iter__(self):
        for i, terms_i in self.coupling_terms.items():
            for n, terms_i_n in terms_i.items():
                for j, strength in terms_i_n.items():
                    yield i, n, j, strength

    def to_nn_bond_tensors(self, sites: list[ct.Site], model: DefinesCouplings) -> list[ct.SymmetricTensor]:
        """Convert the :attr:`coupling_terms` into tensors on nearest neighbor bonds.

        Parameters
        ----------
        sites : list of :class:`cyten.Site`
            Defines the local Hilbert space for each site.
        model:
            TODO remains to implement/define clearly. Some object that can translate coupling
            names to concrete tensors on the `sites`.

        Returns
        -------
        H_bond : list of {:class:`~cyten.SymmetricTensor` | None}
            The :attr:`coupling_terms` rewritten as ``sum_i H_bond[i]`` for MPS indices ``i``.
            ``H_bond[i]`` acts on sites ``(i-1, i)``, ``None`` represents 0.
            Legs of each ``H_bond[i]`` are ``['p0', 'p1', 'p1*', 'p0*']``.

        """
        L = self.L
        if len(sites) != L:
            raise ValueError('incompatible length')
        H_bond = [None] * L
        for i, coupling_name, j, strength in self:
            if j != i + 1:
                raise ValueError('not nearest neighbor')
            if not sites[i].symmetry.trivial_shift:
                raise NotImplementedError
            h = model.get_coupling(coupling_name, sites[i], sites[j % self.L])
            H_bond[j] = add_with_None_0(H_bond[j], strength * h.to_tensor())
        assert all(h.labels_are('p0', 'p1', 'p1*', 'p0*', planar=True))

    def remove_zeros(self, tol_zero: float = 1e-15, inplace: bool = True) -> CouplingTerms:
        if not inplace:
            return deepcopy(self).remove_zeros(tol_zero)
        # note: converting the dict.items() to list before iterating is needed,
        #       since we possibly pop entries off
        for i, terms_i in list(self.coupling_terms.items()):
            for n, terms_i_n in list(terms_i.items()):
                for j, strength in list(terms_i_n.items()):
                    if abs(strength) < tol_zero:
                        terms_i_n.pop(j)
                if len(terms_i_n) == 0:
                    terms_i.pop(n)
            if len(terms_i) == 0:
                self.coupling_terms.pop(i)
        return self

    def to_TermList(self):
        """Convert :attr:`onsite_terms` into a :class:`TermList`.

        Returns
        -------
        term_list : :class:`TermList`
            Representation of the terms as a list of terms.

        """
        raise NotImplementedError  # TODO adapt TermList first
        terms = []
        strength = []
        d0 = self.coupling_terms
        for i in sorted(d0):
            d1 = d0[i]
            for opname_i, op_str in sorted(d1):
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
        if other.L != self.L:
            raise ValueError('incompatible lengths')
        # {i: {('opname_i', 'opname_string'): {j: {'opname_j': strength}}}}
        for i, other_i in other.coupling_terms.items():
            self_i = self.coupling_terms.setdefault(i, dict())
            for name, other_i_n in other_i.items():
                self_i_n = self_i.setdefault(name, dict())
                for j, other_strength in other_i_n.items():
                    self_i_n[j] = self_i_n.get(j, 0.0) + other_strength
        return self

    def test_terms(self, sites: list[ct.Site], model: DefinesCouplings):
        L = self.L
        for i, name, j, strength in self:
            assert 0 <= i < L
            if not i < j:
                raise ValueError('wrong order of indices in coupling terms')
            model.test_coupling_name(name, sites[i], sites[j % L])
        return


class MultiCouplingTerms(Terms):
    """Coupling names, site indices and strengths representing general `M`-site coupling terms.

    Generalizes :class:`CouplingTerms` from `2`-site to `M`-site couplings.
    The structure of the nested dictionary :attr:`coupling_terms` is similar, but we allow
    an arbitrary recursion depth of the dictionary.

    Attributes
    ----------
    L : int
        Number of sites
    coupling_terms : nested dict
        Filled by :meth:`add_multi_coupling_term`
        Nested dicstionaries such that ``coupling_terms[i0][coupling_name, n_sites][i1]...[iN] = strength``
        describes a coupling with a string name ``coupling_name`` acting on ``N + 1`` sites
        ``i0 < i1 < ... < iN``.
        We must have ``0 <= i0 < L``, but the other site indices may extend beyond ``L`` for infinite
        bc, in which case they indicate couplings between different
        iMPS unit cells.

    """

    def __init__(self, L):
        assert L > 0
        self.L = L
        self.coupling_terms: dict[int, dict[str, dict[int, Any]]] = dict()
        self._max_range = 0

    def max_range(self):
        return self._max_range

    def add_multi_coupling_term(self, strength, ijkl, coupling_name: str):
        """Add a multi-site coupling term on given MPS sites.

        Parameters
        ----------
        strength : float
            The strength of the coupling term.
        ijkl : 1D array-like of int
            The MPS indices of the sites on which the operator acts.
            We require the first site within the unit cell ``0 <= ijkl[0] < L``, and that
            sites are sorted ``ijkl[1:] > ijkl[:-1]`` and ``i < j``, i.e., `op_i` acts "left" of `op_j`.
            For infinite bc, ``any(ijkl[1:] >= L)`` is allowed and indicates couplings between different
            unit cells.
        coupling_name : str
            Names of the coupling.

        """
        if len(ijkl) < 2:
            raise ValueError('Need to act on at least 2 sites. Use onsite terms!')
        for i, j in zip(ijkl, ijkl[1:]):
            if not i < j:
                raise ValueError('Need i < j < k < ...')
        if not (0 <= ijkl[0] < self.L):
            raise ValueError('Need 0 <= i < L')
        iL = ijkl[0]
        iR = ijkl[-1]
        d = self.coupling_terms.setdefault(iL, dict()).setdefault(coupling_name, dict())
        for i in ijkl[1:-1]:
            d = d.setdefault(i, dict())
        d[iR] = d.get(iR, 0) + strength
        self._max_range = max(self._max_range, iR - iL)

    def add_coupling_term(self, strength: float | complex, i: int, j: int, coupling_name: str):
        self.add_multi_coupling_term(strength, [i, j], coupling_name)

    def add_to_graph(self, graph: MPOGraph):
        raise NotImplementedError  # TODO adapt MPOGraph first / maybe do this via termlist only?

    def remove_zeros(self, tol_zero: float = 1e-15, inplace: bool = True) -> MultiCouplingTerms:
        if not inplace:
            return deepcopy(self).remove_zeros(tol_zero)
        for i, terms_i in list(self.coupling_terms.items()):
            for n, terms_i_n in list(terms_i.items()):
                _remove_zeros_recursive(terms_i_n, tol_zero)
                if len(terms_i_n) == 0:
                    terms_i.pop(n)
            if len(terms_i) == 0:
                self.coupling_terms.pop(i)
        return self

    def to_TermList(self):
        raise NotImplementedError('Adapt TermList fist!')

    def __iadd__(self, other):
        if not isinstance(other, (CouplingTerms, MultiCouplingTerms)):
            return NotImplemented
        if other.L != self.L:
            raise ValueError('incompatible lengths')
        if isinstance(other, CouplingTerms):
            for i, name, j, strength in other:
                self.add_coupling_term(strength, i, j, name)
            return self
        self._max_range = max(self._max_range, other._max_range)
        _iadd_dicts_recursive(self.coupling_terms, other.coupling_terms)

    def __iter__(self):
        for i, self_i in self.coupling_terms.items():
            for n, self_i_n in self_i.items():
                for ijkl, strength in _iter_dicts_recursive(self_i_n):
                    yield ijkl, n, strength

    def test_terms(self, sites: list[ct.Site], model: DefinesCouplings):
        L = self.L
        for ijkl, name, strength in self:
            assert 0 <= ijkl[0] < L
            assert all(i < j for i, j in zip(ijkl, ijkl[1:]))
            model.test_coupling_name(name, *(sites[i % L] for i in ijkl))
        return


def _remove_zeros_recursive(d, tol_zero):
    """Recursion for :meth:`MultiCouplingTerms.remove_zeros`.

    Remove entries from nested dict in-place.
    """
    for j, entry in list(d.items()):
        if isinstance(entry, dict):
            _remove_zeros_recursive(entry, tol_zero)
            if len(entry) == 0:
                d.pop(j)
        elif abs(entry) < tol_zero:
            d.pop(j)
    return


def _iadd_dicts_recursive(d1, d2):
    """Recursion for :meth:`MultiCouplingTerms.__iadd__`. Add entries from `d2` to `d1` in-place."""
    for key, val2 in d2.items():
        if isinstance(val2, dict):
            val1 = d1.setdefault(key, {})
            _iadd_dicts_recursive(val1, val2)
        else:
            d1[key] = d1.get(key, 0) + val2


def _iter_dicts_recursive(keys: list[int], dict_or_val: dict[int, complex] | complex):
    """Recursion for :meth:`MultiCouplingTerms.__iter__`.

    Yield bottom-level entries from nest dict `val` by appending keys to `idcs` and yielding
    ``idcs, name, val``
    """
    if isinstance(dict_or_val, dict):
        for key, val in dict_or_val.items():
            yield from _iter_dicts_recursive([*keys, key], val)
    else:
        yield keys, val


class ExponentiallyDecayingTerms(Terms):
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

    Suppose we want long-range (LR) couplings between all sites on a strip of width Ly. The
    `subsites` parameters allows us to implement LR couplings along each leg of the strip. We want
    LR couplings between sites that have both vertical and horizontal offset according to the
    Euclidean distance. To do this, we use `subsites_start`. For each site in
    `subsites_start`, we couple to all sites to the right of `subsites`. In our example, the sites
    in one leg can be coupled to all sites with larger index in another leg.

    .. math ::
        strength sum_{subsites_start[i] < subsites[j]} \
            lambda^{|j-argmin(subsites>subsites_start[i])|} A_{subsites_start[i]} B_{subsites[j]}

    Parameters
    ----------
    L : int
        Number of sites.

    Attributes
    ----------
    L : int
        Number of sites.
    exp_decaying_terms : list of tuples
        Each tuple ``(strength, opname_i, opname_j, lambda, subsites, subsites_start, opname_string)`` represents
        one of the terms as described above; see :meth:`add_exponentially_decaying_coupling` for
        more details.
    centered_terms : list of tuples
        Each tuple ``(strength, opname_i, opname_j, lambda_, subsites, opname_string)`` represents
        one of the centered terms as described in :meth:`add_centered_exponentially_decaying_term`.

    """

    def __init__(self, L):
        raise NotImplementedError  # TODO not ported yet
        assert L > 0
        self.L = L
        self.exp_decaying_terms = []
        self.centered_terms = []

    @property
    def is_empty(self):
        return len(self.exp_decaying_terms) == 0 and len(self.centered_terms) == 0

    def add_exponentially_decaying_coupling(
        self, strength, lambda_, op_i, op_j, subsites=None, subsites_start=None, op_string='Id'
    ):
        r"""Add an exponentially decaying long-range coupling.

        .. math ::
            strength \sum_{i} \sum_{j > i} \lambda^{|i-j|} A_i B_j

        Where the operator `A` is given by `op_i`, and `B` is given by `op_j`.
        Note that the sum over i,j is long-range, for infinite systems beyond the MPS unit cell.

        They can be generalized in several ways, see `lambda_`, `subsites`, `subsites_start`, as
        well as the notes below.

        Parameters
        ----------
        strength : float
            Overall prefactor.
        lambda_ : float | 1D array
            Decay-rate. Either a single number, applied uniformly or a sequence of length :attr:`L`.
        op_i, op_j : string
            Names for the operators.
        subsites, subsites_start : 1D array, optional
            Selects a subset of sites within the MPS unit cell on which the operators act. See docs
            in :class:`~tenpy.models.model.CouplingModel.add_exponentially_decaying_coupling`.
        op_string : string
            The operator to be inserted between `A` and `B`; for Fermions this should be ``"JW"``.

        """
        assert np.isscalar(lambda_) or len(lambda_) == self.L
        if subsites is None:
            subsites = np.arange(self.L)
        else:
            subsites = np.array(subsites)
            if len(subsites) > 1 and np.any(subsites[1:] < subsites[:-1]):
                raise ValueError('subsites needs to be sorted; choose a different MPS ordering!')
            assert subsites[0] >= 0
            assert subsites[-1] < self.L

        if subsites_start is None:
            subsites_start = subsites
        else:
            subsites_start = np.array(subsites_start)
            if len(subsites_start) > 1 and np.any(subsites_start[1:] < subsites_start[:-1]):
                raise ValueError('subsites needs to be sorted; choose a different MPS ordering!')
            assert subsites_start[0] >= 0
            assert subsites_start[-1] < self.L

        self.exp_decaying_terms.append((strength, lambda_, op_i, op_j, subsites, subsites_start, op_string))

    def add_centered_exponentially_decaying_term(self, strength, lambda_, op_i, op_j, i, subsites=None, op_string='Id'):
        """Add exponentially decaying terms centered around a single site.

        See :meth:`~tenpy.models.model.CouplingModel.add_centered_exponentially_decaying_term` for
        details.
        """
        assert -self.L <= i < self.L
        if i < 0:
            i = i + self.L
        assert np.isscalar(lambda_) or len(lambda_) == self.L
        if subsites is None:
            subsites = np.arange(self.L)
        else:
            subsites = np.array(subsites)
            if len(subsites) > 1 and np.any(subsites[1:] < subsites[:-1]):
                raise ValueError('subsites needs to be sorted; choose a different MPS ordering!')
            assert subsites[0] >= 0
            assert subsites[-1] < self.L
            assert i in subsites

        self.centered_terms.append((strength, lambda_, op_i, op_j, i, subsites, op_string))

    def add_to_graph(self, graph):
        # TODO this was a param before, but signatures got unified.
        #      is it ever used? maybe it can be a (class) attr?
        key = 'exp-decay'

        assert self.L == graph.L
        # get set of states with `key` to find unique `key_nr` for each of the terms
        all_states = set()
        for states in graph.states:
            for label in states:
                try:
                    if label[1] == key:
                        all_states += label
                except Exception:  # not a tuple / wrong types
                    pass
        key_nr = 1000  # start with high value such that they get added in the end of the MPO
        finite = graph.bc == 'finite'

        for strength, lambda_, op_i, op_j, subsites, subsites_start, op_string in self.exp_decaying_terms:
            if np.isscalar(lambda_):
                lambda_ = np.full(self.L, lambda_)
            while (key_nr, key) in all_states:
                key_nr += 1
            label = (key_nr, key)
            all_states.add(label)
            in_subsites = np.zeros(self.L, dtype=np.bool_)
            in_subsites[subsites] = True
            in_subsites_start = np.zeros(self.L, dtype=np.bool_)
            in_subsites_start[subsites_start] = True
            first_subsite = subsites_start[0]
            last_subsite = subsites[-1]
            assert first_subsite >= 0
            assert last_subsite < self.L
            if not finite:
                for i in range(self.L):
                    if in_subsites[i]:
                        graph.add(i, label, label, op_string, lambda_[i])
                        graph.add(i, label, 'IdR', op_j, strength)
                    if in_subsites_start[i]:
                        graph.add(i, 'IdL', label, op_i, lambda_[i])
                    if not in_subsites[i]:
                        graph.add(i, label, label, op_string, 1.0)
            else:
                if last_subsite > first_subsite:  # If not, there is no coupling to add.
                    # If lambda_ is not uniform and subsites_start != subsites, one needs to be very careful.
                    # Let jj be the index of the first subsite such that subsites > subsites>subsites_start[i].
                    # The interaction will be pref * lambda[subsites_start[i]] * lambda[jj:jj+r]
                    # up to desired interaction range r. So the first term is just pref * lambda[subsites_start[i]]
                    # since we only pick up a lambda on op_i and op_string. When we close a term,
                    # there is no lambda decay.

                    # first subsite
                    graph.add(first_subsite, 'IdL', label, op_i, lambda_[first_subsite])
                    for i in range(first_subsite + 1, last_subsite):
                        if in_subsites[i]:
                            graph.add(i, label, label, op_string, lambda_[i])
                            graph.add(i, label, 'IdR', op_j, strength)
                        if in_subsites_start[i]:
                            graph.add(i, 'IdL', label, op_i, lambda_[i])
                        # If not in subsites, we need to continue on with op_string.
                        # This is true even if we opened a site.
                        if not in_subsites[i]:
                            graph.add(i, label, label, op_string, 1.0)
                    graph.add(last_subsite, label, 'IdR', op_j, strength)

        for strength, lambda_, op_i, op_j, i, subsites, op_string in self.centered_terms:
            assert finite
            if np.isscalar(lambda_):
                lambda_ = np.full(self.L, lambda_)
            while (key_nr, key) in all_states:
                key_nr += 1
            label = (key_nr, key)
            all_states.add(label)
            in_subsites = np.zeros(self.L, dtype=np.bool_)
            in_subsites[subsites] = True
            first_subsite = subsites[0]
            last_subsite = subsites[-1]
            assert first_subsite >= 0
            assert last_subsite < self.L

            # terms with j < i
            if i != first_subsite:  # otherwise there are no j < i
                graph.add(first_subsite, 'IdL', label, op_j, strength)  # open op_j
                for j in range(first_subsite + 1, i):
                    if in_subsites[j]:
                        graph.add(j, 'IdL', label, op_j, strength)  # open op_j
                        graph.add(j, label, label, op_string, lambda_[j])  # continue op_j with lambda * op_string
                    else:
                        graph.add(j, label, label, op_string, 1.0)
                graph.add(i, label, 'IdR', op_i, lambda_[i])

            # terms with j > i
            if i != last_subsite:  # otherwise there are no j > i
                graph.add(i, 'IdL', label, op_i, lambda_[i])  # open op_i
                for j in range(i + 1, last_subsite):
                    if in_subsites[j]:
                        graph.add(j, label, label, op_string, lambda_[j])  # continue op_i with lambda * op_string
                        graph.add(j, label, 'IdR', op_j, strength)  # close op_i with op_j
                    else:
                        graph.add(j, label, label, op_string, 1.0)  # continue op_i with op_string
                graph.add(last_subsite, label, 'IdR', op_j, strength)  # close op_i with op_j

        if graph.max_range is not None:
            graph.max_range = np.inf

    def to_TermList(self, cutoff=0.01, bc='finite'):
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
            strength, lambda_, op_i, op_j, subsites, subsites_start, op_string = term
            if np.isscalar(lambda_):
                lambda_ = np.full(self.L, lambda_)
            N1 = len(subsites)
            if bc == 'finite':
                for i2, i in enumerate(subsites_start):
                    # Find index of first term in subsites larger than subsites_start[i2]
                    i3 = np.where(subsites > i)[0]
                    if len(i3):  # There is a site we couple to on the right
                        i3 = i3[0]  # Get index of first subsite > subsites_start[i2]
                        for d, j in enumerate(subsites[i3:]):
                            # First decay term is from subsites_start[i2].
                            # Then we go to subsites[i3]
                            pref = strength * np.prod(
                                [lambda_[subsites_start[i2]]] + list(lambda_[subsites[i3 : i3 + d]])
                            )
                            if abs(pref) < cutoff:
                                break
                            terms.append([(op_i, i), (op_j, j)])
                            strengths.append(pref)
            elif bc == 'infinite':
                for i2, i in enumerate(subsites_start):
                    # Get index of first subsite to the right of the start, inside the MPS unit cell
                    i3 = np.where(subsites > i)[0]
                    if len(i3) == 0:  # No subsites in the first unit MPS cell are to the right of i
                        i3 = N1  # Shift one unit cell over
                    else:
                        i3 = i3[0]
                    for d in range(0, 1000):  # run over subsites INDICES
                        j2 = i3 + d
                        j = subsites[j2 % N1] + (j2 // N1) * L
                        # See finite case for reasoning about decay constants
                        pref = strength * np.prod(
                            [lambda_[subsites_start[i2]]] + list(lambda_[subsites[np.arange(i3, j2) % N1]])
                        )
                        if abs(pref) < cutoff:
                            break
                        terms.append([(op_i, i), (op_j, j)])
                        strengths.append(pref)
                    else:
                        raise ValueError('distance of 1000 not enough to reach precision cutoff')

            else:
                raise ValueError('unknown boundary conditions: ' + repr(bc))

        for strength, lambda_, op_i, op_j, i, subsites, op_string in self.centered_terms:
            if np.isscalar(lambda_):
                lambda_ = np.full(self.L, lambda_)
            assert bc == 'finite'
            for j in subsites:
                if j == i:
                    continue
                if j < i:
                    prefactor_sites = subsites[(subsites > j) & (subsites <= i)]
                else:
                    prefactor_sites = subsites[(subsites >= i) & (subsites < j)]
                pref = strength * np.prod(lambda_[prefactor_sites])
                if abs(pref) < cutoff:
                    continue
                terms.append([(op_i, i), (op_j, j)])  # by definition, op_j acts first.
                strengths.append(pref)

        return TermList(terms, strengths)

    def __iadd__(self, other):
        if not isinstance(other, ExponentiallyDecayingTerms):
            return NotImplemented  # unknown type of other
        if other.L != self.L:
            raise ValueError('incompatible lengths')
        self.exp_decaying_terms += other.exp_decaying_terms
        self.centered_terms += other.centered_terms
        return self

    def max_range(self):
        return np.inf

    def test_terms(self, sites: list[ct.Site], model: DefinesCouplings):
        """Check the format of self.exp_decaying_terms."""
        for term in self.exp_decaying_terms:
            strength, lambda_, op_i, op_j, subsites, subsites_start, op_string = term
            for i in subsites_start:
                if not sites[i].valid_opname(op_i):
                    raise ValueError(f'Operator {op_i!r} not in site {i:d}')

            for j in subsites:
                if not sites[j].valid_opname(op_j):
                    raise ValueError(f'Operator {op_j!r} not in site {j:d}')

        for strength, lambda_, op_i, op_j, i, subsites, op_string in self.centered_terms:
            if not sites[i].valid_opname(op_i):
                raise ValueError(f'Operator {op_i=} not in site {i}')
            for j in subsites:
                if j == i:
                    continue
                if not sites[j].valid_opname(op_j):
                    raise ValueError(f'Operator {op_j=} not in site {j}')
                if not sites[j].valid_opname(op_string):
                    raise ValueError(f'Operator {op_string=} not in site {j}')
        # done
