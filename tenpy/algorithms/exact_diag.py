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
    without addional extra work.
"""
# Copyright 2018 TeNPy Developers

import numpy as np
import warnings

from ..linalg import np_conserved as npc
from ..networks.mps import MPS


class ExactDiag:
    """(Full) exact diagonalization of the Hamiltonian.

    Parameters
    ----------
    model : :class:`~tenpy.models.MPOmodel` | :class:`~tenpy.models.CouplingModel`
        The model which is to be diagonalized.
    charge_sector : ``None`` | charges
        If not ``None``, project onto the given charge sector.
    sparse : bool
        If ``True``, don't sort/bunch the LegPipe used to combine the physical legs.
        This results in array `blocks` with just one entry, requires much more charge data,
        and is not what `np_conserved` was designed for, so it's not recommended.

    Attributes
    ----------
    model : :class:`~tenpy.models.MPOmodel` | :class:`~tenpy.models.CouplingModel`
        The model which is to be diagonalized.
    chinfo : :class:`~tenpy.linalg.charges.ChargeInfo`
        The nature of the charge (which is the same for all sites).
    charge_sector : ``None`` | charges
        If not ``None``, we project onto the given charge sector.
    full_H : :class:`~tenpy.linalg.np_conserved.Array` | ``None``
        The full Hamiltonian to be diagonalized
        with legs ``'(p0.p1....)', '(p0*,p1*...)'`` (in that order)
    E : ndarray | ``None``
        1D array of eigenvalues.
    V : :class:`~tenpy.linalg.np_conserved.Array` | ``None``
        Eigenvectors. First leg 'ps' are physical legs,
        the second leg ``'ps*'`` corresponds to the eigenvalues.
    _sites : list of :class:`~tenpy.networks.site.Site`
        The sites in the given order.
    _labels_p : list or str
        The labels use for the physical legs; just ``['p0', 'p1', ...., 'p{L-1}']``.
    _labels_pconj : list or str
        Just each of `_labels_p` with an ``*``.
    _pipe : :class:`~tenpy.linalg.charges.LegPipe`
        The pipe from the single physical legs to the full combined leg.
    _pipe_conj : :class:`~tenpy.linalg.charges.LegPipe`
        Just ``_pipe.conj()``.
    _mask : 1D bool ndarray | ``None``
        Bool mask, which of the indices of the pipe are in the desired `charge_sector`.
    """

    def __init__(self, model, charge_sector=None, sparse=False):
        if model.lat.bc_MPS != 'finite':
            raise ValueError("Full diagonalization works only on finite systems")
        self.model = model
        self.chinfo = model.lat.unit_cell[0].leg.chinfo
        self.full_H = None
        self.E = None
        self.V = None
        self._labels_p = ['p' + str(i) for i in range(model.lat.N_sites)]
        self._labels_pconj = [l + '*' for l in self._labels_p]
        self._sites = model.lat.mps_sites()
        legs = [s.leg for s in self._sites]
        self._pipe = npc.LegPipe(legs, qconj=1, sort=(not sparse), bunch=(not sparse))
        self._pipe_conj = self._pipe.conj()
        if charge_sector is not None:
            self.charge_sector = self.chinfo.make_valid(charge_sector)
            self._mask = np.all(self._pipe.to_qflat() == self.charge_sector[np.newaxis, :], axis=1)
            if np.sum(self._mask) == 0:
                raise ValueError("The chosen charge sector is empty.")
        else:
            self.charge_sector = None
            self._mask = None

    def build_full_H_from_mpo(self):
        """calculate self.full_H from self.mpo"""
        mpo = self.model.H_MPO
        full_H = mpo.get_W(0).take_slice(mpo.get_IdL(0), 'wL')
        full_H.ireplace_labels(['p', 'p*'], [self._labels_p[0], self._labels_pconj[0]])
        for i in range(1, mpo.L):
            W = mpo.get_W(i, copy=True)
            W.ireplace_labels(['p', 'p*'], [self._labels_p[i], self._labels_pconj[i]])
            if i == mpo.L - 1:
                W = W.take_slice(mpo.get_IdR(mpo.L - 1), 'wR')
            full_H = npc.tensordot(full_H, W, axes=['wR', 'wL'])
        full_H = full_H.combine_legs([self._labels_p, self._labels_pconj],
                                     new_axes=[0, 1],
                                     pipes=[self._pipe, self._pipe_conj])
        self._set_full_H(full_H)

    def build_full_H_from_bonds(self):
        """calculate self.full_H from self.mpo"""
        sites = self.model.lat.mps_sites()
        H_bond = self.model.H_bond
        L = len(sites)
        Ids = [
            s.Id.replace_labels(['p', 'p*'], [self._labels_p[i], self._labels_pconj[i]])
            for i, s in enumerate(sites)
        ]
        Ids_L = [Ids[0]]  # Ids_L[j] has identity up to (including) site j
        Ids_R = [Ids[-1]]  # Ids_R[j] is identity starting from (including) site L-1-j
        for j in range(1, L - 2):
            Ids_L.append(npc.outer(Ids_L[-1], Ids[j]))
            Ids_R.append(npc.outer(Ids[L - j - 1], Ids_R[-1]))
        full_H = None
        for i in range(1, L):
            # H_bond[i] lifes on sites (i-1, i)
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
            Hb = Hb.combine_legs([self._labels_p, self._labels_pconj],
                                 new_axes=[0, 1],
                                 pipes=[self._pipe, self._pipe_conj])
            if full_H is None:
                full_H = Hb
            else:
                full_H += Hb
        self._set_full_H(full_H)

    def full_diagonalization(self, *args, **kwargs):
        """Full diagonalization to obtain all eigenvalues and eigenvectors.

        Arguments are given to :class:`~tenpy.linalg.np_conserved.eigh`."""
        if self.full_H is None:
            raise ValueError("You need to call one of `build_full_H_*` first!")
        E, V = npc.eigh(self.full_H, *args, **kwargs)
        V.iset_leg_labels(['ps', 'ps*'])
        self.E = E
        self.V = V

    def groundstate(self):
        """Pick the ground state from self.V"""
        if self.E is None or self.V is None:
            raise ValueError("You need to call `full_diagonalization` first!")
        i0 = np.argmin(self.E)
        return self.V.take_slice(i0, axes='ps*')

    def exp_H(self, dt):
        """Return ``U(dt) := exp(-i H dt)``."""
        if self.E is None or self.V is None:
            raise ValueError("You need to call `full_diagonalization` first!")
        return npc.tensordot(self.V.scale_axis(np.exp(-1.j * dt * self.E), 'ps*'),
                             self.V.conj(),
                             axes=['ps*', 'ps'])

    def mps_to_full(self, mps):
        """Contract an MPS along the virtual bonds and combine its legs.

        Parameters
        ----------
        mps : :class:`~tenpy.networks.mps.MPS`
            The MPS to be contracted.

        Returns
        -------
        psi : :class:`~tenpy.linalg.np_conserved.Array`
            The MPO contracted along the virtual bonds.
        """
        if mps.bc != 'finite':
            raise ValueError("Full diagonalization works only on finite systems")
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
        return MPS.from_full(self._sites, psi, form=canonical_form)

    def matvec(self, psi):
        """Allow to use `self` as LinearOperator for lanczos.

        Just applies `full_H` to (the first axis of) the given `psi`."""
        return npc.tensordot(self.full_H, psi, axes=1)

    def sparse_diag(self, k, *args, **kwargs):
        """Call :func:`~tenpy.linalg.np_conserved.speigs`."""
        return npc.speigs(self.full_H, self.charge_sector, k, *args, **kwargs)

    def _set_full_H(self, full_H):
        if self.full_H is not None:
            warnings.warn("full_H calculated multiple times!?", stacklevel=2)
        if self.charge_sector is not None:
            full_H.legs = [l.to_LegCharge() for l in full_H.legs]  # avoids warnings of project
            full_H = full_H[self._mask, self._mask]
        self.full_H = full_H
