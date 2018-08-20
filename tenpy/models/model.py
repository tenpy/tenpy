"""This module contains some base classes for a model.

A 'model' is supposed to represent a Hamiltonian in a generalized way.
Beside a :class:`~tenpy.models.lattice.Lattice` specifying the geometry and
underlying Hilbert space, it thus needs some way to represent the different terms
of the Hamiltonian.

Different algorithms require different representations of the Hamiltonian.
For example, if you only want to do DRMG, it is enough to specify the Hamiltonian
as an MPO with a :class:`MPOModel`.
On the other hand, TEBD needs the model to be 'nearest neighbor' and thus
a representation by nearest-neighbor terms.

The :class:`CouplingModel` is the attempt to generalize the representation of `H`
by explicitly specifying the couplings of onsite-terms, and providing functionality
for converting the specified couplings into an MPO or nearest-neighbor bonds.
This allows to quickly generate new model classes for a broad class of Hamiltonians.
For simplicity, the :class:`CouplingModel` is limited to interactions involving only two sites.
However, we also provide the :class:`MultiCouplingModel` to generate Models for Hamiltonians
involving couplings between multiple sites.

For other cases (e.g. exponentially decaying long-range interactions in 1D),
it might be simpler to just specify the MPO explicitly.

Of course, we also provide ways to transform a :class:`NearestNeighborModel` into a
:class:`MPOModel` and vice versa, as far as this is possible.

See also the introduction :doc:`../intro_model`.
"""
# Copyright 2018 TeNPy Developers

import numpy as np

from ..linalg import np_conserved as npc
from ..tools.misc import to_array
from ..networks import mpo  # used to construct the Hamiltonian as MPO

__all__ = ['CouplingModel', 'MultiCouplingModel', 'NearestNeighborModel', 'MPOModel']

_bc_coupling_choices = {'open': True, 'periodic': False}


class CouplingModel(object):
    """Base class for a general model of a Hamiltonian consisting of two-site couplings.

    In this class, the terms of the Hamiltonian are specified explicitly as onsite or coupling
    terms.

    Parameters
    ----------
    lattice : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the geometry and the local Hilbert space(s).
    bc_coupling : (iterable of) {``'open'`` | ``'periodic'`` | ``int``}
        Boundary conditions of the couplings in each direction of the lattice. Defines how the
        couplings are added in :meth:`add_coupling`. A single string holds for all directions.
        An integer `shift` means that we have periodic boundary conditions along this direction,
        but shift/tilt by ``lattice.basis[0] * shift`` (=cylinder axis for ``bc_MPS='infinite'``)
        when going around the boundary along this direction.

    Attributes
    ----------
    lat : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the geometry and the local Hilbert space(s).
    bc_coupling : bool ndarray
        Boundary conditions of the couplings in each direction of the lattice,
        translated into a bool array with the global `_bc_coupling_choices`.
    bc_shift : None | ndarray(int)
        The shift in x-direction when going around periodic boundaries in other directions.
    onsite_terms : list of dict
        Filled by :meth:`add_onsite`.
        For each MPS index `i` a dictionary ``{'opname': strength}`` defining the onsite terms.
    coupling_terms : dict of dict
        Filled by :meth:`add_coupling`.
        Nested dictionaries of the form
        ``{i: {('opname_i', 'opname_string'): {j: {'opname_j': strength}}}}``.
        Note that always ``i < j``, but entries with ``j >= lat.N_sites`` are allowed for
        ``lat.bc_MPS == 'infinite'``, in which case they indicate couplings between different
        iMPS unit cells.
    H_onsite : list of :class:`npc.Array`
        For each site (in MPS order) the onsite part of the Hamiltonian.
    """

    def __init__(self, lattice, bc_coupling='open'):
        self.lat = lattice
        global _bc_coupling_choices
        if bc_coupling in list(_bc_coupling_choices.keys()):
            bc_coupling = [_bc_coupling_choices[bc_coupling]] * self.lat.dim
            self.bc_shift = None
        else:
            self.bc_shift = np.zeros(self.lat.dim-1, np.int_)
            for i, bc in enumerate(bc_coupling):
                if isinstance(bc, int):
                    if i == 0:
                        raise ValueError("Invalid bc_coupling: first entry can't be a shift")
                    self.bc_shift[i-1] = bc
                    bc_coupling[i] = _bc_coupling_choices['periodic']
                else:
                    bc_coupling[i] = _bc_coupling_choices[bc]
            if not np.any(self.bc_shift != 0):
                self.bc_shift = None
        self.bc_coupling = np.array(bc_coupling)
        self.onsite_terms = [dict() for _ in range(self.lat.N_sites)]
        self.coupling_terms = dict()
        self.H_onsite = None
        CouplingModel.test_sanity(self)
        # like self.test_sanity(), but use the version defined below even for derived class

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        if self.bc_coupling.shape != (self.lat.dim, ):
            raise ValueError("Wrong len of bc_coupling")
        assert self.bc_coupling.dtype == np.bool
        assert int(_bc_coupling_choices['open']) == 1  # this is used explicitly
        assert int(_bc_coupling_choices['periodic']) == 0   # and this as well
        if self.bc_coupling[0] and self.lat.bc_MPS == 'infinite':
            raise ValueError("Need periodic boundary conditions along the x-direction "
                             "for 'infinite' `bc_MPS`")
        sites = self.lat.mps_sites()
        for site, terms in zip(sites, self.onsite_terms):
            for opname, strength in terms.items():
                if not site.valid_opname(opname):
                    raise ValueError("Operator {op!r} not in site".format(op=opname))
        self._test_coupling_terms()

    def add_onsite(self, strength, u, opname):
        """Add onsite terms to self.

        Adds a term :math:`\sum_{x_0, ..., x_{dim-1}} strength[x_0, ..., x_{dim-1}] * OP``,
        where the operator ``OP=lat.unit_cell[u].get_op(opname)``
        acts on the site given by a lattice index ``(x_0, ..., x_{dim-1}, u)``,
        to the represented Hamiltonian.

        The necessary terms are just added to :attr:`onsite_terms`; doesn't rebuild the MPO.

        Parameters
        ----------
        strength : scalar | array
            Prefactor of the onsite term. May vary spatially. If an array of smaller size
            is provided, it gets tiled to the required shape.
        u : int
            Picks a :class:`~tenpy.model.lattice.Site` ``lat.unit_cell[u]`` out of the unit cell.
        opname : str
            valid operator name of an onsite operator in ``lat.unit_cell[u]``.
        """
        strength = to_array(strength, self.lat.Ls)  # tile to lattice shape
        if not np.any(strength != 0.):
            return  # nothing to do: can even accept non-defined `opname`.
        if not self.lat.unit_cell[u].valid_opname(opname):
            raise ValueError("unknown onsite operator {0!r} for u={1:d}\n"
                             "{2!r}".format(opname, u, self.lat.unit_cell[u]))
        for i, i_lat in zip(*self.lat.mps_lat_idx_fix_u(u)):
            term = self.onsite_terms[i]
            term[opname] = term.get(opname, 0) + strength[tuple(i_lat)]

    def add_coupling(self,
                     strength,
                     u1,
                     op1,
                     u2,
                     op2,
                     dx,
                     op_string=None,
                     str_on_first=True,
                     raise_op2_left=False):
        r"""Add twosite coupling terms to the Hamiltonian.

        Represents couplings of the form
        :math:`\sum_{x_0, ..., x_{dim-1}} strength[loc(\vec{x})] * OP1 * OP2`, where
        ``OP1 := lat.unit_cell[u1].get_op(op1)`` acts on the site ``(x_0, ..., x_{dim-1}, u1)``,
        and ``OP2 := lat.unit_cell[u2].get_op(op2)`` acts on the site
        ``(x_0+dx[0], ..., x_{dim-1}+dx[dim-1], u2)``.
        For periodic boundary conditions (``bc_coupling[a] == False``)
        the index ``x_a`` is taken modulo ``lat.Ls[a]`` and runs through ``range(lat.Ls[a])``.
        For open boundary conditions, ``x_a`` is limited to ``0 <= x_a < Ls[a]`` and
        ``0 <= x_a+dx[a] < lat.Ls[a]``.
        The coupling `strength` may vary spatially, ``loc({x_i})`` indicates the lower left corner
        of the hypercube containing the involved sites :math:`\vec{x}` and
        :math:`\vec{x}+\vec{dx}`.

        The necessary terms are just added to :attr:`coupling_terms`; doesn't rebuild the MPO.

        Parameters
        ----------
        strength : scalar | array
            Prefactor of the coupling. May vary spatially (see above). If an arrow of smaller size
            is provided, it gets tiled to the required shape.
        u1 : int
            Picks the site ``lat.unit_cell[u1]`` for OP1.
        op1 : str
            Valid operator name of an onsite operator in ``lat.unit_cell[u1]`` for OP1.
        u2 : int
            Picks the site ``lat.unit_cell[u2]`` for OP2.
        op2 : str
            Valid operator name of an onsite operator in ``lat.unit_cell[u2]`` for OP2.
        dx : iterable of int
            Translation vector (of the unit cell) between OP1 and OP2.
            For a 1D lattice, a single int is also fine.
        op_string : str | None
            Name of an operator to be used between the OP1 and OP2 sites.
            Typical use case is the phase for a Jordan-Wigner transformation.
            The operator should be defined on all sites in the unit cell.
            If ``None``, auto-determine whether a Jordan-Wigner string is needed, using
            :meth:`~tenpy.networks.site.Site.op_needs_JW`.
        str_on_first : bool
            Wheter the provided `op_string` should also act on the first site.
            This option should be chosen as ``True`` for Jordan-Wigner strings.
            When handling Jordan-Wigner strings we need to extend the `op_string` to also act on
            the 'left', first site (in the sense of the MPS ordering of the sites given by the
            lattice). In this case, there is a well-defined ordering of the operators in the
            physical sense (i.e. which of `op1` or `op2` acts first on a given state).
            We follow the convention that `op2` acts first (in the physical sense),
            independent of the MPS ordering.
        raise_op2_left : bool
            Raise an error when `op2` appears left of `op1`
            (in the sense of the MPS ordering given by the lattice).
        """
        dx = np.array(dx, np.intp).reshape([self.lat.dim])
        shift_i_lat_strength, coupling_shape = self._coupling_shape(dx)
        strength = to_array(strength, coupling_shape)  # tile to correct shape
        if not np.any(strength != 0.):
            return  # nothing to do: can even accept non-defined onsite operators
        for op, u in [(op1, u1), (op2, u2)]:
            if not self.lat.unit_cell[u].valid_opname(op):
                raise ValueError("unknown onsite operator {0!r} for u={1:d}\n"
                                 "{2:!r}".format(op, u, self.lat.unit_cell[u]))
        if op_string is None:
            need_JW1 = self.lat.unit_cell[u1].op_needs_JW(op1)
            need_JW2 = self.lat.unit_cell[u1].op_needs_JW(op2)
            if need_JW1 and need_JW2:
                op_string = 'JW'
            elif need_JW1 or need_JW2:
                raise ValueError("Only one of the operators needs a Jordan-Wigner string?!")
            else:
                op_string = 'Id'
        for u in range(len(self.lat.unit_cell)):
            if not self.lat.unit_cell[u].valid_opname(op_string):
                raise ValueError("unknown onsite operator {0!r} for u={1:d}\n"
                                 "{2:!r}".format(op_string, u, self.lat.unit_cell[u]))
        if np.all(dx == 0) and u1 == u2:
            raise ValueError("Coupling shouldn't be onsite!")

        # prepare: figure out the necessary mps indices
        Ls = np.array(self.lat.Ls)
        N_sites = self.lat.N_sites
        mps_i, lat_i = self.lat.mps_lat_idx_fix_u(u1)
        lat_j_shifted = lat_i + dx
        lat_j = np.mod(lat_j_shifted, Ls) # assuming PBC
        if self.bc_shift is not None:
            lat_j[:, 0] += np.sum(((lat_j_shifted - lat_j) // Ls)[:, 1:] * self.bc_shift, axis=1)
        keep = np.all(
            np.logical_or(
                lat_j_shifted == lat_j,  # not accross the boundary
                np.logical_not(self.bc_coupling)),  # direction has PBC
            axis=1)
        mps_i = mps_i[keep]
        lat_i = lat_i[keep] + shift_i_lat_strength[np.newaxis, :]
        lat_j = lat_j[keep]
        lat_j_shifted = lat_j_shifted[keep]
        mps_j = self.lat.lat2mps_idx(np.concatenate([lat_j, [[u2]] * len(lat_j)], axis=1))
        if self.lat.bc_MPS == 'infinite':
            # shift j by whole MPS unit cells for couplings along the infinite direction
            mps_j_shift = lat_j_shifted[:, 0] - np.mod(lat_j_shifted[:, 0], Ls[0])
            mps_j_shift *= (N_sites // Ls[0])
            mps_j += mps_j_shift
            # finally, ensure 0 <= min(i, j) < N_sites.
            mps_ij_shift = np.where(mps_j_shift < 0, -mps_j_shift, 0)
            mps_i += mps_ij_shift
            mps_j += mps_ij_shift

        # loop to perform the sum over {x_0, x_1, ...}
        for i, i_lat, j in zip(mps_i, lat_i, mps_j):
            current_strength = strength[tuple(i_lat)]
            if current_strength == 0.:
                continue
            o1, o2 = op1, op2
            swap = (j < i)  # ensure i <= j
            if swap:
                if raise_op2_left:
                    raise ValueError("Op2 is left")
                i, o1, j, o2 = j, op2, i, op1  # swap OP1 <-> OP2
            # now we have always i < j and 0 <= i < N_sites
            # j >= N_sites indicates couplings between unit_cells of the infinite MPS.
            # o1 is the "left" operator; o2 is the "right" operator
            if str_on_first and op_string != 'Id':
                if swap:
                    o1 = ' '.join([op_string, o1])  # o1==op2 should act first
                else:
                    o1 = ' '.join([o1, op_string])  # o1==op2 should act first
            d1 = self.coupling_terms.setdefault(i, dict())
            # form of d1: ``{('opname_i', 'opname_string'): {j: {'opname_j': current_strength}}}``
            d2 = d1.setdefault((o1, op_string), dict())
            d3 = d2.setdefault(j, dict())
            d3[o2] = d3.get(o2, 0) + current_strength

    def calc_H_onsite(self, tol_zero=1.e-15):
        """Calculate `H_onsite` from `self.onsite_terms`.

        Parameters
        ----------
        tol_zero : float
            prefactors with ``abs(strength) < tol_zero`` are considered to be zero.

        Returns
        -------
        H_onsite : list of npc.Array
            onsite terms of the Hamiltonian.
        """
        self._remove_onsite_terms_zeros(tol_zero)
        res = []
        for i, terms in enumerate(self.onsite_terms):
            s = self.lat.site(i)
            H = npc.zeros([s.leg, s.leg.conj()])
            for opname, strength in terms.items():
                H = H + strength * s.get_op(opname)  # (can't use ``+=``: may change dtype)
            res.append(H)
        return res

    def calc_H_bond(self, tol_zero=1.e-15):
        """calculate `H_bond` from `self.coupling_terms` and `self.H_onsite`.

        If ``self.H_onsite is None``, it is calculated with :meth:`self.calc_H_onsite`.

        Parameters
        ----------
        tol_zero : float
            prefactors with ``abs(strength) < tol_zero`` are considered to be zero.

        Returns
        -------
        H_bond : list of :class:`~tenpy.linalg.np_conserved.Array`
            Bond terms as required by the constructor of :class:`NearestNeighborModel`.
            Legs are ``['p0', 'p0*', 'p1', 'p1*']``

        Raises
        ------
        ValueError : if the Hamiltonian contains longer-range terms.
        """
        self._remove_coupling_terms_zeros(tol_zero)
        if self.H_onsite is None:
            self.H_onsite = self.calc_H_onsite(tol_zero)
        finite = (self.lat.bc_MPS != 'infinite')
        N_sites = self.lat.N_sites
        res = [None] * self.lat.N_sites
        for i, d1 in self.coupling_terms.items():
            j = (i + 1) % self.lat.N_sites
            d1 = self.coupling_terms[i]
            site_i = self.lat.site(i)
            site_j = self.lat.site(j)
            strength_i = 1. if finite and i == 0 else 0.5
            strength_j = 1. if finite and j == self.lat.N_sites - 1 else 0.5
            if finite and j == 0:  # over the boundary
                strength_i, strength_j = 0., 0.  # just to make the assert below happy
            H = npc.outer(strength_i * self.H_onsite[i], site_j.Id)
            H = H + npc.outer(site_i.Id, strength_j * self.H_onsite[j])
            for (op1, op_str), d2 in d1.items():
                for j2, d3 in d2.items():
                    if isinstance(j2, tuple):
                        # This should only happen in a MultiSiteCoupling model
                        raise ValueError("multi-site coupling: can't generate H_bond")
                    # i, j in coupling_terms are defined such that we expect j2 = i + 1
                    if j2 != i + 1:
                        msg = "Can't give nearest neighbor H_bond for long-range {i:d}-{j:d}"
                        raise ValueError(msg.format(i=i, j=j2))
                    for op2, strength in d3.items():
                        H = H + strength * npc.outer(site_i.get_op(op1), site_j.get_op(op2))
            H.iset_leg_labels(['p0', 'p0*', 'p1', 'p1*'])
            res[j] = H
        if finite and 0 in res:
            assert (res[0].norm(np.inf) <= tol_zero)
            res[0] = None
        return res

    def calc_H_MPO(self, tol_zero=1.e-15):
        """Calculate MPO representation of the Hamiltonian.

        Uses :attr:`onsite_terms` and :attr:`coupling_terms` to build an MPO graph
        (and then an MPO).

        Parameters
        ----------
        tol_zero : float
            prefactors with ``abs(strength) < tol_zero`` are considered to be zero.

        Returns
        -------
        H_MPO : :class:`~tenpy.networks.mpo.MPO`
            MPO representation of the Hamiltonian.
        """
        graph = mpo.MPOGraph(self.lat.mps_sites(), self.lat.bc_MPS)
        # onsite terms
        self._remove_onsite_terms_zeros(tol_zero)
        for i, terms in enumerate(self.onsite_terms):
            for opname, strength in terms.items():
                graph.add(i, 'IdL', 'IdR', opname, strength)
        # coupling terms
        self._remove_coupling_terms_zeros(tol_zero)
        self._graph_add_coupling_terms(graph)
        # add 'IdL' and 'IdR' and convert the graph to an MPO
        graph.add_missing_IdL_IdR()
        self.H_MPO_graph = graph
        H_MPO = graph.build_MPO()
        return H_MPO

    def _test_coupling_terms(self):
        """Check the format of self.coupling_terms"""
        sites = self.lat.mps_sites()
        N_sites = len(sites)
        for i, d1 in self.coupling_terms.items():
            site_i = sites[i]
            for (op_i, opstring), d2 in d1.items():
                if not site_i.valid_opname(op_i):
                    raise ValueError("Operator {op!r} not in site".format(op=op_i))
                for j, d3 in d2.items():
                    for op_j in d3.keys():
                        if not sites[j % N_sites].valid_opname(op_j):
                            raise ValueError("Operator {op!r} not in site".format(op=op_j))
        # done

    def _remove_onsite_terms_zeros(self, tol_zero=1.e-15):
        """remove entries of strength `0` from ``self.onsite_terms``."""
        for term in self.onsite_terms:
            for op in list(term.keys()):
                if abs(term[op]) < tol_zero:
                    del term[op]
        # done

    def _remove_coupling_terms_zeros(self, tol_zero=1.e-15):
        """remove entries of strength `0` from ``self.coupling_terms``."""
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

    def _coupling_shape(self, dx):
        """calculate correct shape of the strengths for each coupling."""
        shape = [La - abs(dxa) * int(bca)
                 for La, dxa, bca in zip(self.lat.Ls, dx, self.bc_coupling)]
        shift_strength = [min(0, dxa) for dxa in dx]
        return np.array(shift_strength), tuple(shape)

    def _graph_add_coupling_terms(self, graph):
        # structure of coupling terms:
        # {i: {('opname_i', 'opname_string'): {j: {'opname_j': strength}}}}
        for i, d1 in self.coupling_terms.items():
            for (opname_i, op_string), d2 in d1.items():
                label = (i, opname_i, op_string)
                graph.add(i, 'IdL', label, opname_i, 1.)
                for j, d3 in d2.items():
                    label_j = graph.add_string(i, j, label, op_string)
                    for opname_j, strength in d3.items():
                        graph.add(j, label_j, 'IdR', opname_j, strength)
        # done


class MultiCouplingModel(CouplingModel):
    """Generalizes :class:`CouplingModel` to allow couplings involving more than two sites.

    Attributes
    ----------
    coupling_terms : dict of dict
        Generalization of the coupling_terms of a :class:`CouplingModel` for M-site couplings.
        Filled by :meth:`add_coupling` or :meth:`add_multi_coupling`.
        Nested dictionaries of the following form::

            {i: {('opname_i', 'opname_string_ij'):
                 {j: {('opname_j', 'opname_string_jk'):
                      {k: {('opname_k', 'opname_string_kl'):
                           ...
                           {l: {'opname_l': strength
                           }   }
                      }   }
                 }   }
            }   }

        For a M-site coupling, this involves a nesting depth of 2*M dictionaries.
        Note that always ``i < j < k < ... < l``, but entries with ``j,k,l >= lat.N_sites``
        are allowed for ``lat.bc_MPS == 'infinite'``, in which case they indicate couplings
        between different iMPS unit cells.
    """

    def add_multi_coupling(self, strength, u0, op0, other_ops, op_string=None):
        r"""Add multi-site coupling terms to the Hamiltonian.

        Represents couplings of the form
        :math:`sum_{x_0, ..., x_{dim-1}} strength[loc(\vec{x})] * OP0 * OP1 * ... * OPM`,
        where ``OP_0 := lat.unit_cell[u0].get_op(op0)`` acts on the site
        ``(x_0, ..., x_{dim-1}, u0)``,
        and ``OP_m := lat.unit_cell[other_u[m]].get_op(other_op[m])``, m=1...M, acts on the site
        ``(x_0+other_dx[m][0], ..., x_{dim-1}+other_dx[m][dim-1], other_u[m])``.
        For periodic boundary conditions along direction `a` (``bc_coupling[a] == False``)
        the index ``x_a`` is taken modulo ``lat.Ls[a]`` and runs through ``range(lat.Ls[a])``.
        For open boundary conditions, ``x_a`` is limited to ``0 <= x_a < Ls[a]`` and
        ``0 <= x_a+other_dx[m,a] < lat.Ls[a]``.
        The coupling `strength` may vary spatially, :math:`loc(\vec{x})` indicates the lower left
        corner of the hypercube containing all the involved sites
        :math:`\vec{x}, \vec{x}+\vec{other_dx[m, :]}`.

        The necessary terms are just added to :attr:`coupling_terms`; doesn't rebuild the MPO.

        Parameters
        ----------
        strength : scalar | array
            Prefactor of the coupling. May vary spatially and is tiled to the required shape.
        u0 : int
            Picks the site ``lat.unit_cell[u0]`` for OP0.
        op0 : str
            Valid operator name of an onsite operator in ``lat.unit_cell[u0]`` for OP0.
        other_ops : list of ``(u, op_m, dx)``
            One tuple for each of the other operators ``OP1, OP2, ... OPM`` involved.
            `u` picks the site ``lat.unit_cell[u]``, `op_name` is a valid operator acting on that
            site, and `dx` gives the translation vector between ``OP0`` and the specified operator.
        op_string : str | None
            Name of an operator to be used inbetween the operators, excluding the sites on which
            the operators act. This operator should be defined on all sites in the unit cell.

            Special case: If ``None``, auto-determine whether a Jordan-Wigner string is needed
            (using :meth:`~tenpy.networks.site.Site.op_needs_JW`), for each of the segments
            inbetween the operators and also on the sites of the left operators.
            Note that in this case the ordering of the operators *is* important and handled in the
            usual convention that ``OPM`` acts first and ``OP0`` last on a physical state.

            .. warning :
                ``None`` figures out for each segment between the operators, whether a
                Jordan-Wigner string is needed.
                This is different from a plain ``'JW'``, which just applies a string on
                each segment!
        """
        other_ops = list(other_ops)
        M = len(other_ops)
        all_us = np.array([u0] + [oop[0] for oop in other_ops], np.intp)
        all_ops = [op0] + [oop[1] for oop in other_ops]
        dx = np.array([oop[2] for oop in other_ops], np.intp).reshape([M, self.lat.dim])
        shift_i_lat_strength, coupling_shape = self._coupling_shape(dx)
        strength = to_array(strength, coupling_shape)  # tile to correct shape
        if not np.any(strength != 0.):
            return  # nothing to do: can even accept non-defined onsite operators
        need_JW = np.array([self.lat.unit_cell[u].op_needs_JW(op)
                            for u, op in zip(all_us, all_ops)], dtype=np.bool_)
        if op_string is None and not any(need_JW):
            op_string = 'Id'
        for u, op, _ in [(u0, op0, None)] + other_ops :
            if not self.lat.unit_cell[u].valid_opname(op):
                raise ValueError("unknown onsite operator {0!r} for u={1:d}\n"
                                 "{2:!r}".format(op, u, self.lat.unit_cell[u]))
        if op_string is not None:
            if not np.sum(need_JW) % 2 == 0:
                raise ValueError("Invalid coupling: would need 'JW' string on the very left")
            for u in range(len(self.lat.unit_cell)):
                if not self.lat.unit_cell[u].valid_opname(op_string):
                    raise ValueError("unknown onsite operator {0!r} for u={1:d}\n"
                                     "{2:!r}".format(op_string, u, self.lat.unit_cell[u]))
        if np.all(dx == 0) and np.all(u0 == all_us):
            # note: we DO allow couplings with some onsite terms, but not all of them
            raise ValueError("Coupling shouldn't be purely onsite!")

        # prepare: figure out the necessary mps indices
        Ls = np.array(self.lat.Ls)
        N_sites = self.lat.N_sites
        mps_i, lat_i = self.lat.mps_lat_idx_fix_u(u0)
        lat_jkl_shifted = lat_i[:, np.newaxis, :] + dx[np.newaxis, :, :]
        # lat_jkl* has 3 axes "initial site", "other_op", "spatial directions"
        lat_jkl = np.mod(lat_jkl_shifted, Ls) # assuming PBC
        if self.bc_shift is not None:
            lat_jkl[:, :, 0] += np.sum(((lat_jkl_shifted - lat_jkl) // Ls)[:, :, 1:] *
                                       self.bc_shift, axis=2)
        keep = np.all(
            np.logical_or(
                lat_jkl_shifted == lat_jkl,  # not accross the boundary
                np.logical_not(self.bc_coupling)),  # direction has PBC
            axis=(1, 2))
        mps_i = mps_i[keep]
        lat_i = lat_i[keep, :] + shift_i_lat_strength[np.newaxis, :]
        lat_jkl = lat_jkl[keep, :, :]
        lat_jkl_shifted = lat_jkl_shifted[keep, :, :]
        latu_jkl = np.concatenate((lat_jkl, np.array([all_us[1:]]*len(lat_jkl))[:, :, np.newaxis]),
                                  axis=2)
        mps_jkl = self.lat.lat2mps_idx(latu_jkl)
        if self.lat.bc_MPS == 'infinite':
            # shift by whole MPS unit cells for couplings along the infinite direction
            mps_jkl_shift = lat_jkl_shifted[:, :, 0] - np.mod(lat_jkl_shifted[:, :, 0], Ls[0])
            mps_jkl += mps_jkl_shift * (N_sites // Ls[0])
        mps_ijkl = np.concatenate((mps_i[:, np.newaxis], mps_jkl), axis=1)

        # loop to perform the sum over {x_0, x_1, ...}
        for ijkl, i_lat in zip(mps_ijkl, lat_i):
            current_strength = strength[tuple(i_lat)]
            if current_strength == 0.:
                continue
            ijkl, ops, op_str = _multi_coupling_group_handle_JW(
                ijkl, all_ops, need_JW, op_string, N_sites)
            # create the nested structure
            # {ijkl[0]: {(ops[0], op_str[0]):
            #            {ijkl[1]: {(ops[1], op_str[1]):
            #                       ...
            #                           {ijkl[-1]: {ops[-1]: current_strength}
            #            }         }
            # }         }
            d0 = self.coupling_terms
            for x in range(len(ijkl)-1):
                d1 = d0.setdefault(ijkl[x], dict())
                d0 = d1.setdefault((ops[x], op_str[x]), dict())
            d1 = d0.setdefault(ijkl[-1], dict())
            op = ops[-1]
            d1[op] = d1.get(op, 0) + current_strength
        # done

    def _test_coupling_terms(self, d0=None):
        sites = self.lat.mps_sites()
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
                    self._test_coupling_terms(d2)  # recursive!
                else:  # last term of the coupling
                    op_i = key
                    if not site_i.valid_opname(op_i):
                        raise ValueError("Operator {op!r} not in site".format(op=op_i))
        # done

    def _remove_coupling_terms_zeros(self, tol_zero=1.e-15, d0=None):
        """remove entries of strength `0` from ``self.coupling_terms``."""
        if d0 is None:
            d0 = self.coupling_terms
        # d0 = ``{i: {('opname_i', 'opname_string_ij'): ... {j: {'opname_j': strength}}}``
        for i, d1 in list(d0.items()):
            for key, d2 in list(d1.items()):
                if isinstance(key, tuple):
                    self._remove_coupling_terms_zeros(tol_zero, d2)  # recursive!
                    if len(d2) == 0:
                        del d1[key]
                else:
                    # key is opname_j, d2 is strength
                    if abs(d2) < tol_zero:
                        del d1[key]
            if len(d1) == 0:
                del d0[i]
        # done

    def _coupling_shape(self, dx):
        """calculate correct shape of the strengths for each coupling."""
        if dx.ndim == 1:
            return super()._coupling_shape(dx)
        Ls = self.lat.Ls
        shape = [None]*len(Ls)
        shift_strength = [None]*len(Ls)
        for a in range(len(Ls)):
            max_dx, min_dx = np.max(dx[:, a]), np.min(dx[:, a])
            box_dx = max(max_dx, 0) - min(min_dx, 0)
            shape[a] = Ls[a] - box_dx * int(self.bc_coupling[a])
            shift_strength[a] = min(0, min_dx)
        return np.array(shift_strength), tuple(shape)

    def _graph_add_coupling_terms(self, graph, i=None, d1=None, label_left=None):
        # nested structure of coupling_terms:
        # d0 = {i: {('opname_i', 'opname_string_ij'): ... {l: {'opname_l': strength}}}
        if d1 is None:  # beginning of recursion
            for i, d1 in self.coupling_terms.items():
                self._graph_add_coupling_terms(graph, i, d1, 'IdL')
        else:
            for key, d2 in d1.items():
                if isinstance(key, tuple): # further nesting
                    op_i, op_string_ij = key
                    if isinstance(label_left, str) and label_left == 'IdL':
                        label = (i, op_i, op_string_ij)
                    else:
                        label = label_left + (i, op_i, op_string_ij)
                    graph.add(i, label_left, label, op_i, 1.)
                    for j, d3 in d2.items():
                        label_j = graph.add_string(i, j, label, op_string_ij)
                        self._graph_add_coupling_terms(graph, j, d3, label_j)
                else:  # maximal nesting reached: exit recursion
                    # i is actually the `l`
                    op_i, strength = key, d2
                    graph.add(i, label_left, 'IdR', op_i, strength)
        # done


class NearestNeighborModel(object):
    """Base class for a model of nearest neigbor interactions (w.r.t. the MPS index).

    Suitable for TEBD.

    Parameters
    ----------
    lattice : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the geometry and the local Hilbert space(s).
    H_bond : list of :class:`~tenpy.linalg.np_conserved.Array`
        The Hamiltonian rewritten as ``sum_i H_bond[i]`` for MPS indices ``i``.
        ``H_bond[i]`` acts on sites ``(i-1, i)``; we require ``len(H_bond) == lat.N_sites``.

    Attributes
    ----------
    lat : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the geometry and the local Hilbert space(s).
    H_bond : list of :class:`npc.Array`
        The Hamiltonian rewritten as ``sum_i H_bond[i]`` for MPS indices ``i``.
        ``H_bond[i]`` acts on sites ``(i-1, i)``.
    """

    def __init__(self, lat, H_bond):
        self.lat = lat
        self.H_bond = list(H_bond)
        if self.lat.bc_MPS != 'infinite':
            self.H_bond[0] = None
        NearestNeighborModel.test_sanity(self)
        # like self.test_sanity(), but use the version defined below even for derived class

    def test_sanity(self):
        if len(self.H_bond) != self.lat.N_sites:
            raise ValueError("wrong len of H_bond")

    def trivial_like_NNModel(self):
        """Return a NearestNeighborModel with same lattice, but trivial (H=0) bonds."""
        triv_H = [H.zeros_like() if H is not None else None for H in self.H_bond]
        return NearestNeighborModel(self.lat, triv_H)

    def bond_energies(self, psi):
        """Calculate bond energies <psi|H_bond|psi>.

        Parameters
        ----------
        psi : :class:`~tenpy.networks.mps.MPS`
            The MPS for which the bond energies should be calculated.

        Returns
        -------
        E_bond : 1D ndarray
            List of bond energies: for finite bc, ``E_Bond[i]`` is the energy of bond ``i, i+1``.
            (i.e. we omit bond 0 between sites L-1 and 0);
            for infinite bc ``E_bond[i]`` is the energy of bond ``i-1, i``.
        """
        if self.lat.bc_MPS == 'infinite':
            return psi.expectation_value(self.H_bond, axes=(['p0', 'p1'], ['p0*', 'p1*']))
        # else
        return psi.expectation_value(self.H_bond[1:], axes=(['p0', 'p1'], ['p0*', 'p1*']))


class MPOModel(object):
    """Base class for a model with an MPO representation of the Hamiltonian.

    Suitable for MPO-based algorithms, e.g. DMRG and MPO time evolution.

    .. todo ::
        implement: provide (function to calculate) the MPO for time evolution.
        Also, provide function to get H_MPO from H_bond

    Parameters
    ----------
    lattice : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the geometry and the local Hilbert space(s).
    H_MPO : :class:`~tenpy.networks.mpo.MPO`
        The Hamiltonian rewritten as an MPO.

    Attributes
    ----------
    lat : :class:`tenpy.model.lattice.Lattice`
        The lattice defining the geometry and the local Hilbert space(s).
    H_MPO : :class:`tenpy.tn.mpo.MPO`
        MPO representation of the Hamiltonian.
    """

    def __init__(self, lat, H_MPO):
        self.lat = lat
        self.H_MPO = H_MPO
        MPOModel.test_sanity(self)
        # like self.test_sanity(), but use the version defined below even for derived class

    def test_sanity(self):
        if self.H_MPO.sites != self.lat.mps_sites():
            raise ValueError("lattice incompatible with H_MPO.sites")


def _multi_coupling_group_handle_JW(ijkl, ops, ops_need_JW, op_string, N_sites):
    """Helping function for MultiCouplingModel.add_multi_coupling.

    Sort and groups the operators by sites `ijkl` they act on, such that the returned `new_ijkl`
    is strictly ascending, i.e. has entries `i < j < k < l`.
    Also, handle/figure out Jordan-Wigner strings if needed.
    """
    number_ops = len(ijkl)
    reorder = np.argsort(ijkl, kind='mergesort')  # need stable kind!!!
    if not 0 <= ijkl[reorder[0]] < N_sites:  # ensure this condition with a shift
        ijkl += ijkl[reorder[0]] % N_sites - ijkl[reorder[0]]
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
            right = [z for gr in grouped_reorder[a+1:] for z in gr]
            onsite_ops = []
            need_JW_right = False
            JW_max = -1
            for x in group + [number_ops]:
                JW_min, JW_max = JW_max, x
                need_JW = (np.sum([ops_need_JW[z] for z in right if JW_min < z < JW_max]) % 2 == 1)
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
