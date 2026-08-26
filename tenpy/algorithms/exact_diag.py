"""Full diagonalization (ED) of the Hamiltonian.

The full diagonalization of a small system is a simple approach to test other algorithms.
In case you need the full spectrum, a full diagonalization is often the only way.
This module provides functionality to quickly diagonalize the Hamiltonian of a given model.
This might be used to obtain the spectrum, the ground state or highly excited states.

.. note ::
    Good use of symmetries is crucial to increase the treatable system size.
    While we can simply use the defined `LegCharge` of a model, we don't make use of any other
    symmetries like translation symmetry, SU(2) symmetry or inversion symmetries.
    In other words, this code does not aim to provide state-of-the-art exact diagonalization,
    but just the ability to diagonalize the defined models for small system sizes
    without additional extra work.
"""
# Copyright (C) TeNPy Developers, Apache license

import warnings

import numpy as np
from cyten.symmetries import SymmetryError
from cyten.symmetries.spaces import AbelianLegPipe, TensorProduct
from cyten.tensors import HermitianNumpyArrayLinearOperator, slice_leg
from cyten.tensors import eigh as cyten_eigh

from ..models.model import CouplingModel
from ..networks.mps import MPS
from ..tools.misc import inverse_permutation

__all__ = ['ExactDiag', 'get_full_wavefunction', 'get_numpy_Hamiltonian', 'get_scipy_sparse_Hamiltonian']


class ExactDiag:
    """(Full) exact diagonalization of the Hamiltonian.

    Parameters
    ----------
    model : :class:`~tenpy.models.MPOmodel` | :class:`~tenpy.models.CouplingModel`
        The model which is to be diagonalized.
    charge_sector : ``None`` | charges
        If not ``None``, restrict :meth:`groundstate`/:meth:`sparse_diag` to the given charge
        sector. Note that (unlike the old `np_conserved` version) this does *not* reduce the size
        of `full_H` itself -- :func:`~cyten.tensors.eigh` is already block-diagonal in the coupled
        sector, so there is nothing to gain from projecting beforehand; :meth:`sparse_diag`, via
        :class:`~cyten.tensors.HermitianNumpyArrayLinearOperator`, is what actually restricts the
        work to one sector.
    max_size : int
        The `build_H_*` functions will do nothing (but emit a warning) if the total size of the
        Hamiltonian would be larger than this.

    Attributes
    ----------
    model : :class:`~tenpy.models.MPOmodel` | :class:`~tenpy.models.CouplingModel` | ``None``
        The model which is to be diagonalized, if constructed via :meth:`__init__`.
        ``None`` if constructed via :meth:`from_hamiltonian`.
    symmetry : :class:`~cyten.symmetries.Symmetry`
        The symmetry of the sites (which is the same for all sites).
    charge_sector : ``None`` | charges
        If not ``None``, we restrict to the given charge sector, see above.
    max_size : int
        The ``build_H_*`` functions will do nothing (but emit a warning) if the total size of the
        Hamiltonian would be larger than this.
    full_H : :class:`~cyten.tensors.SymmetricTensor` | ``None``
        The full Hamiltonian to be diagonalized, physical legs ``p0, ..., p{L-1}`` in the
        codomain and their duals in the domain. ``None`` if the ``build_H_*`` functions haven't
        been called yet, or if `max_size` would have been exceeded.
    E : ndarray | ``None``
        1D array of eigenvalues, index-aligned with the ``'eig'`` leg of `V`.
    V : :class:`~cyten.tensors.SymmetricTensor` | ``None``
        Eigenvectors, as returned by :func:`~cyten.tensors.eigh`. Physical legs as in `full_H`,
        plus a new leg ``'eig'`` corresponding to the eigenvalues.
    _sites : list of :class:`~cyten.models.sites.Site`
        The sites in the given order.
    _labels_p : list or str
        The labels use for the physical legs; just ``['p0', 'p1', ...., 'p{L-1}']``.
    _labels_pconj : list or str
        Just each of `_labels_p` with an ``*``.
    _pipe : :class:`~cyten.symmetries.spaces.TensorProduct`
        The (uncombined) tensor product of the physical legs.
    _pipe_conj : :class:`~cyten.symmetries.spaces.TensorProduct`
        Just ``_pipe.dual``.
    _mask : 1D bool ndarray | ``None``
        Only set for an abelian `symmetry` with a fixed `charge_sector`: bool mask, which of the
        (Kronecker-product) basis states of the combined physical legs are in `charge_sector`.
        For non-abelian symmetries a per-basis-state charge does not exist, so this stays ``None``
        even when `charge_sector` is fixed.

    """

    def __init__(self, model, charge_sector=None, max_size=2e6):
        if model.lat.bc_MPS != 'finite':
            raise ValueError('Full diagonalization works only on finite systems')
        self.model = model
        self._init_from_sites(model.lat.mps_sites(), charge_sector, max_size)

    def _init_from_sites(self, sites, charge_sector, max_size):
        self.full_H = None
        self.E = None
        self.V = None
        self.max_size = max_size
        self._sites = list(sites)
        self._labels_p = ['p' + str(i) for i in range(len(self._sites))]
        self._labels_pconj = [l + '*' for l in self._labels_p]
        self.symmetry = self._sites[0].leg.symmetry
        if not self.symmetry.can_be_dropped:
            raise SymmetryError(
                f'ExactDiag needs a symmetry with a dense basis; {self.symmetry} has none. '
                'Anyonic symmetries are not supported.'
            )
        self._pipe = TensorProduct([s.leg for s in self._sites])
        self._pipe_conj = self._pipe.dual
        if charge_sector is not None:
            self.charge_sector = np.asarray(charge_sector)
            sectors = self.possible_charge_sectors()
            if not np.any(np.all(sectors == self.charge_sector[np.newaxis, :], axis=1)):
                raise ValueError('The chosen charge sector is empty.')
            self._mask = None
            if self.symmetry.is_abelian:
                pipe = AbelianLegPipe([s.leg for s in self._sites])
                sectors_of_basis = np.asarray(pipe.sectors_of_basis)
                self._mask = np.all(sectors_of_basis == self.charge_sector[np.newaxis, :], axis=1)
        else:
            self.charge_sector = None
            self._mask = None

    def possible_charge_sectors(self):
        return np.asarray(self._pipe.sector_decomposition)

    @classmethod
    def from_hamiltonian(cls, H, sites, charge_sector=None, max_size=2e6):
        """Initialize directly from a cyten Hamiltonian tensor and its sites.

        Entry point while ``tenpy.models`` is not ported to cyten yet, so :meth:`__init__` cannot
        be exercised without a model.

        Parameters
        ----------
        H : :class:`~cyten.tensors.SymmetricTensor`
            The Hamiltonian, e.g. from ``Coupling.to_tensor()``, with the physical legs
            ``p0, ..., p{L-1}`` in the codomain and their duals in the domain.
        sites : list of :class:`~cyten.models.sites.Site`
            The sites, in MPS order.
        charge_sector, max_size :
            As for :meth:`__init__`.

        """
        res = cls.__new__(cls)
        res.model = None
        res._init_from_sites(sites, charge_sector, max_size)
        if not res._exceeds_max_size():
            res._set_full_H(H)
        return res

    @classmethod
    def from_infinite_model(cls, model, first=0, last=None, enlarge=None, **kwargs):
        """Initialize by extracting a finite segment from a ``bc_MPS=infinite'`` model.

        This method calls :meth:`~tenpy.models.model.Model.extract_segment` on the model and sets
        the boundary conditions to 'finite'. For the ExactDiag, this little hack is equivalent
        to extracting all the coupling terms fitting within the segment specified by
        `first`, `last` and `None`, and generating a finite MPOModel from it.

        Note that it drops the `H_bond` if existent, since :meth:`build_full_H_from_bonds` would
        not include the correct, full onsite-terms at the boundaries if just drop the H_bond going
        outside the segment. Hence you can only use the :meth:`build_full_H_from_mpo` method
        when initializing the ExactDiag with this method.

        Parameters
        ----------
        model : :class:`tenpy.models.model.Model`
            Model with infinite bc and MPO.

        """
        model_segment = model.extract_segment(first, last, enlarge)
        model_segment.lat.bc_MPS = 'finite'
        model_segment.H_MPO.bc = 'finite'
        if hasattr(model_segment, 'H_bond'):
            del model_segment.H_bond  # invalid since it wouldn't terminate onsite terms correctly
        return cls(model_segment, **kwargs)

    @classmethod
    def from_H_mpo(cls, H_MPO, *args, **kwargs):
        """Wrapper taking directly an MPO instead of a Model.

        Parameters
        ----------
        H_MPO : :class:`~tenpy.networks.mpo.MPO`
            The MPO representing the Hamiltonian.
        *args :
            Further keyword arguments as for the ``__init__`` of the class.
        **kwargs :
            Further keyword arguments as for the ``__init__`` of the class.

        """
        from ..models.lattice import TrivialLattice
        from ..models.model import MPOModel

        assert H_MPO.bc == 'finite'
        M = MPOModel(TrivialLattice(H_MPO.sites), H_MPO)
        return cls(M, *args, **kwargs)

    def build_full_H_from_mpo(self):
        """Calculate self.full_H from the MPO (``H_MPO``) of the model."""
        if self._exceeds_max_size():
            return
        mpo = self.model.H_MPO
        full_H = mpo.get_W(0).take_slice(mpo.get_IdL(0), 'wL')
        full_H.ireplace_labels(['p', 'p*'], [self._labels_p[0], self._labels_pconj[0]])
        for i in range(1, mpo.L):
            W = mpo.get_W(i, copy=True)
            W.ireplace_labels(['p', 'p*'], [self._labels_p[i], self._labels_pconj[i]])
            if i == mpo.L - 1:
                W = W.take_slice(mpo.get_IdR(mpo.L - 1), 'wR')
            full_H = npc.tensordot(full_H, W, axes=['wR', 'wL'])
        full_H = full_H.combine_legs(
            [self._labels_p, self._labels_pconj], new_axes=[0, 1], pipes=[self._pipe, self._pipe_conj]
        )
        if mpo.explicit_plus_hc:
            full_H = full_H + full_H.conj().itranspose(full_H.get_leg_labels())
        self._set_full_H(full_H)

    def build_full_H_from_bonds(self):
        """Calculate self.full_H from bond terms (``H_bond``) of the model."""
        if self._exceeds_max_size():
            return
        sites = self.model.lat.mps_sites()
        H_bond = self.model.H_bond
        L = len(sites)
        Ids = [
            s.Id.replace_labels(['p', 'p*'], [self._labels_p[i], self._labels_pconj[i]]) for i, s in enumerate(sites)
        ]
        Ids_L = [Ids[0]]  # Ids_L[j] has identity up to (including) site j
        Ids_R = [Ids[-1]]  # Ids_R[j] is identity starting from (including) site L-1-j
        for j in range(1, L - 2):
            Ids_L.append(npc.outer(Ids_L[-1], Ids[j]))
            Ids_R.append(npc.outer(Ids[L - j - 1], Ids_R[-1]))
        full_H = None
        for i in range(1, L):
            # H_bond[i] lives on sites (i-1, i)
            lL, lLc = self._labels_p[i - 1], self._labels_pconj[i - 1]
            lR, lRc = self._labels_p[i], self._labels_pconj[i]
            Hb = H_bond[i]
            if Hb is None:
                continue
            Hb = Hb.replace_labels(['p0', 'p0*', 'p1', 'p1*'], [lL, lLc, lR, lRc])
            if i > 1:
                Hb = npc.outer(Ids_L[i - 2], Hb)  # need i-2 == j
            if i < L - 1:
                Hb = npc.outer(Hb, Ids_R[L - 2 - i])  # need i+1 == L-1-j   =>   j = L-2-i
            Hb = Hb.combine_legs(
                [self._labels_p, self._labels_pconj], new_axes=[0, 1], pipes=[self._pipe, self._pipe_conj]
            )
            if full_H is None:
                full_H = Hb
            else:
                full_H += Hb
        self._set_full_H(full_H)

    def full_diagonalization(self, **kwargs):
        """Full diagonalization to obtain all eigenvalues and eigenvectors.

        Sets :attr:`V` and :attr:`E`. Keyword arguments are given to :func:`~cyten.tensors.eigh`
        (in addition to ``new_labels='eig', new_leg_dual=False``, which cannot be overridden).
        """
        if self.full_H is None:
            raise ValueError('You need to call one of `build_full_H_*` first!')
        kwargs.setdefault('new_labels', 'eig')
        kwargs['new_leg_dual'] = False
        W, V = cyten_eigh(self.full_H, **kwargs)
        self.E = W.diagonal_as_numpy()
        self.V = V

    def groundstate(self, charge_sector=None):
        """Pick the ground state energy and ground state from ``self.V``.

        Parameters
        ----------
        charge_sector : None | 1D ndarray
            By default (``None``), consider all charge sectors, or -- if a `charge_sector` was
            fixed at construction -- that one. Alternatively, explicitly give the sector which the
            returned state should have; requires that no `charge_sector` was fixed at construction.

        Returns
        -------
        E0 : float
            Ground state energy (possibly in the given sector).
        psi0 : :class:`~cyten.tensors.ChargedTensor`
            Ground state (possibly in the given sector), with ``charged_state=None``, i.e. it
            carries the definite charge of its :attr:`~cyten.tensors.ChargedTensor.charge_leg`
            rather than being trivial. Physical legs ``p0, ..., p{L-1}``.

        """
        if self.E is None or self.V is None:
            raise ValueError('You need to call `full_diagonalization` first!')
        if charge_sector is None:
            charge_sector = self.charge_sector  # may still be None -> global minimum
        elif self.charge_sector is not None:
            raise ValueError('``self.charge_sector`` was specified before.')
        if charge_sector is None:
            i0 = np.argmin(self.E)
        else:
            charge_sector = np.asarray(charge_sector)
            sectors_of_basis = np.asarray(self.V.domain.factors[0].sectors_of_basis)
            mask = np.all(sectors_of_basis == charge_sector[np.newaxis, :], axis=1)
            if np.sum(mask) == 0:
                raise ValueError('The chosen charge sector is empty.')
            i0 = np.argmin(np.where(mask, self.E, np.max(self.E) + 1.0))
        i0 = int(i0)
        return self.E[i0], slice_leg(self.V, 'eig', i0)

    def exp_H(self, dt):
        """Return ``U(dt) := exp(-i H dt)``."""
        if self.E is None or self.V is None:
            raise ValueError('You need to call `full_diagonalization` first!')
        return npc.tensordot(self.V.scale_axis(np.exp(-1.0j * dt * self.E), 'ps*'), self.V.conj(), axes=['ps*', 'ps'])

    def mps_to_full(self, mps):
        """Contract an MPS along the virtual bonds and combine its legs.

        Parameters
        ----------
        mps : :class:`~tenpy.networks.mps.MPS`
            The MPS to be contracted.

        Returns
        -------
        psi : :class:`~tenpy.linalg.np_conserved.Array`
            The MPS contracted along the virtual bonds.

        """
        if mps.bc != 'finite':
            raise ValueError('Full diagonalization works only on finite systems')
        psi = mps.get_theta(0, mps.L)  # does exactly what we need
        psi = psi.take_slice([0, 0], ['vL', 'vR'])
        psi = psi.combine_legs(range(mps.L))
        if self.charge_sector is not None:
            psi.legs[0] = psi.legs[0].to_LegCharge()
            psi = psi[self._mask]
        return psi

    def full_to_mps(self, psi, canonical_form='B'):
        """Convert a full state (with a single leg) to an MPS.

        Parameters
        ----------
        psi : :class:`~tenpy.linalg.np_conserved.Array`
            The state (with a single leg) which should be splitted into an MPS.
        canonical_from : :class:`~tenpy.linalg.np_conserved.Array`
            The form in which the MPS will be afterwards.

        Returns
        -------
        mps : :class:`~tenpy.networks.mps.MPS`
            An normalized MPS representation in canonical form.

        """
        if not isinstance(psi.legs[0], npc.LegPipe):
            # projected onto charge_sector: need to restore the LegPipe.
            full_psi = npc.zeros([self._pipe], psi.dtype, psi.qtotal)
            full_psi[self._mask] = psi
            psi = full_psi
        psi.iset_leg_labels(['(' + '.'.join(self._labels_p) + ')'])
        psi = psi.split_legs([0])  # split the combined leg into the physical legs of the sites
        return MPS.from_full(self._sites, psi, form=canonical_form, unit_cell_width=self.model.lat.mps_unit_cell_width)

    def sparse_diag(self, k, *args, **kwargs):
        """The `k` lowest eigenvalues and eigenvectors, via :mod:`cyten.tensors.sparse`.

        Wraps :class:`~cyten.tensors.HermitianNumpyArrayLinearOperator`, which restricts to
        :attr:`charge_sector` (all sectors if it is ``None``) and calls
        :func:`scipy.sparse.linalg.eigsh` under the hood. Note that cyten does not yet support
        restricting to a single, higher-dimensional sector (e.g. non-trivial SU(2) sectors) this
        way; ``charge_sector=None`` and one-dimensional sectors work.

        Parameters
        ----------
        k : int
            The number of eigenvalues/eigenvectors to compute.
        *args, **kwargs :
            Further arguments given to
            :meth:`~cyten.tensors.HermitianNumpyArrayLinearOperator.eigenvectors`.

        Returns
        -------
        E : 1D ndarray
            The `k` energies, ascending.
        psi : list of :class:`~cyten.tensors.ChargedTensor`
            The corresponding eigenstates, physical legs ``p0, ..., p{L-1}``.

        """
        if self.full_H is None:
            raise ValueError('You need to call one of `build_full_H_*` first!')
        op = HermitianNumpyArrayLinearOperator.from_Tensor(
            self.full_H, legs1=self._labels_pconj, legs2=self._labels_p, charge_sector=self.charge_sector
        )
        kwargs.setdefault('which', 'SA')
        return op.eigenvectors(k, *args, **kwargs)

    def _set_full_H(self, full_H):
        if self.full_H is not None:
            warnings.warn('full_H calculated multiple times!?', stacklevel=2)
        # The np_conserved version projected onto `charge_sector` here. cyten cannot: Mask and
        # DiagonalTensor reject LegPipes, so `full_H` would first have to be flattened onto a single
        # abstract leg -- and then the eigenvectors could no longer be split back into the sites.
        # `charge_sector` therefore restricts `groundstate` instead; `eigh` is block-wise anyway.
        self.full_H = full_H

    def _exceeds_max_size(self):
        size = np.prod([float(s.dim) for s in self._sites]) ** 2  # use float to avoid overflow!
        if size > self.max_size:
            msg = f'size {size:.2e} exceeds max_size {self.max_size:.2e}'
            warnings.warn(msg, stacklevel=2)
            return True
        return False


def get_full_wavefunction(psi: MPS, undo_sort_charge: bool = True):
    """Get the full wavefunction of a finite MPS as a 1D numpy array.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        The input MPS. Must have ``psi.bc == 'finite'``.
    undo_sort_charge : bool
        If we should undo the basis permutation induced by
        :meth:`~tenpy.networks.site.Site.sort_charge`.

    Returns
    -------
    theta : 1D array
        The wavefunction. Basis order is like for a Kronecker product :func:`numpy.kron` of the
        local basis order, see `undo_sort_charge`.

    """
    if psi.bc != 'finite':
        raise ValueError('psi must have finite boundary conditions')
    if len(psi._p_label) != 1:
        # hard-coding standard MPS with a single leg here.
        raise NotImplementedError
    p = psi._p_label[0]
    theta = psi.get_theta(0, psi.L)
    theta = theta.itranspose(['vL'] + [f'{p}{n}' for n in range(psi.L)] + ['vR'])
    theta = theta.to_ndarray()
    theta = np.squeeze(theta, (0, -1))  # squeeze vL, vR
    if undo_sort_charge:
        perms = [inverse_permutation(site.perm) for site in psi.sites]
        theta = theta[np.ix_(*perms)]
    return np.reshape(theta, -1)


def get_numpy_Hamiltonian(model, from_mpo: bool = True, undo_sort_charge: bool = True):
    """Get the Hamiltonian as a matrix (2D numpy array).

    Parameters
    ----------
    model
        The model that defines the Hamiltonian. The lattice should be finite.
    from_mpo : bool
        If we should prioritize using the MPO over ``H_bond`` to build the Hamiltonian.
    undo_sort_charge : bool
        If we should undo the basis permutation induced by
        :meth:`~tenpy.networks.site.Site.sort_charge`.

    Returns
    -------
    H : 2D array
        The Hamiltonian as a matrix. Basis order is like for a Kronecker product :func:`numpy.kron`
        of the local basis order, see `undo_sort_charge`.

    """
    if model.lat.bc_MPS != 'finite':
        raise ValueError('Model must be defined on a finite lattice.')
    if isinstance(model, CouplingModel):
        return _get_Hamiltonian_from_couplings(model, sparse=False, undo_sort_charge=undo_sort_charge)
    return _get_numpy_Hamiltonian_ExactDiag_full_H(model, from_mpo=from_mpo, undo_sort_charge=undo_sort_charge)


def get_scipy_sparse_Hamiltonian(model, undo_sort_charge: bool = True):
    """Get the Hamiltonian as a sparse scipy matrix.

    Parameters
    ----------
    model
        The model that defines the Hamiltonian. The lattice should be finite.
    undo_sort_charge : bool
        If we should undo the basis permutation induced by
        :meth:`~tenpy.networks.site.Site.sort_charge`.

    Returns
    -------
    H : CSR matrix
        The Hamiltonian as a scipy CSR sparse matrix. Basis order is like for a Kronecker product
        :func:`numpy.kron` of the local basis order, see `undo_sort_charge`.

    """
    if model.lat.bc_MPS != 'finite':
        raise ValueError('Model must be defined on a finite lattice.')

    if isinstance(model, CouplingModel):
        return _get_Hamiltonian_from_couplings(model=model, sparse=True, undo_sort_charge=undo_sort_charge)
    else:
        raise NotImplementedError


def _get_numpy_Hamiltonian_ExactDiag_full_H(model, from_mpo: bool, undo_sort_charge: bool):
    ed = ExactDiag(model)
    if from_mpo and hasattr(model, 'H_MPO'):
        ed.build_full_H_from_mpo()
    else:
        ed.build_full_H_from_bonds()
    res = ed.full_H.split_legs()
    n_sites = model.lat.N_sites
    assert res.rank == 2 * n_sites
    # [p0, p1, ..., p0*, p1*, ...]
    res = res.itranspose([f'p{n}{star}' for star in ['', '*'] for n in range(n_sites)])
    res = res.to_ndarray()
    if undo_sort_charge:
        perms = [inverse_permutation(site.perm) for site in model.lat.mps_sites()] * 2
        res = res[np.ix_(*perms)]
    dim = np.prod(res.shape[:n_sites])
    return np.reshape(res, (dim, dim))


def _get_Hamiltonian_from_couplings(model, sparse: bool, undo_sort_charge: bool):
    """Helper to get either dense numpy or sparse scipy matrix of the Hamiltonian."""
    if not isinstance(model, CouplingModel):
        raise ValueError('Must be a coupling model.')

    ot = model.all_onsite_terms()
    ot.remove_zeros()
    ct = model.all_coupling_terms()
    ct.remove_zeros()
    edt = model.exp_decaying_terms
    term_list = ot.to_TermList() + ct.to_TermList() + edt.to_TermList()

    sites = model.lat.mps_sites()
    dims = [s.leg.ind_len for s in sites]
    if sparse:
        import scipy.sparse as spsp

        H = spsp.csr_matrix((np.prod(dims),) * 2, dtype=float)
        kron = spsp.kron
        eye_0 = spsp.eye(1)  # identity on zero sites. starting point for doing kron.
    else:
        H = np.zeros((np.prod(dims),) * 2, dtype=float)
        kron = np.kron
        eye_0 = np.eye(1)  # identity on zero sites. starting point for doing kron.

    for s, terms in zip(term_list.strength, term_list.terms):
        last_site = -1
        t = eye_0
        for op, i in terms:
            sites_since_last_op = range(last_site + 1, i)
            if len(sites_since_last_op) > 0:
                t = kron(t, np.eye(np.prod([dims[n] for n in sites_since_last_op])))
            op = sites[i].get_op(op).to_ndarray()
            if undo_sort_charge:
                perm = inverse_permutation(sites[i].perm)
                op = op[np.ix_(perm, perm)]
            t = kron(t, op)
            last_site = i
        sites_since_last_op = range(last_site + 1, len(sites))
        if len(sites_since_last_op) > 0:
            t = kron(t, np.eye(np.prod([dims[n] for n in sites_since_last_op])))
        H = H + s * t
    return H
