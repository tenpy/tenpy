[latest]
========

Release Notes
-------------
The default (stable) git branch was renamed from ``master`` to ``main``.


Changelog
---------

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Drop official support for Python 3.5
- :meth:`tenpy.linalg.np_conserved.from_ndarray`: raise `ValueError` instead of just a warning in case of the wrong
  non-zero blocks. This behaviour can be switched back with the new argument `raise_wrong_sector`.
- Argument `v0` of :meth:`tenpy.networks.mps.MPS.TransferMatrix.eigenvectors` is renamed to `v0_npc`; `v0` now serves for non-np_conserved guess.


Added
^^^^^
- :meth:`~tenpy.networks.mps.MPS.entanglement_entropy_segment2`
- :meth:`tenpy.linalg.sparse.FlatLinearOperator.eigenvectors` and :meth:`~tenpy.linalg.sparse.FlatHermitianOperator.eigenvectors` to unify
  code from :meth:`tenpy.networks.mps.TransferMatrix.eigenvectors` and :meth:`tenpy.linalg.lanczos.lanczos_arpack`.
- :meth:`tenpy.tools.misc.group_by_degeneracy`
- :meth:`tenpy.tools.fit.entropy_profile_from_CFT` and :meth:`tenpy.tools.fit.central_charge_from_S_profile`
- :meth:`tenpy.networks.site.Site.multiply_operators` as a variant of :meth:`~tenpy.networks.site.Site.multiply_op_names` accepting both string and npc arrays.
- :meth:`tenpy.tools.events.EventHandler` to simplify call-backs e.g. for measurement codes during an algorithms.
- :attr:`tenpy.models.lattice.Lattice.Lu` as a class attribute.

Changed
^^^^^^^
- For finite DMRG, :cfg:option:`DMRGEngine.N_sweeps_check` now defaults to 1 instead of 10 (which is still the default for infinite MPS).
- Merge :meth:`tenpy.linalg.sparse.FlatLinearOperator.npc_to_flat_all_sectors` into :meth:`~tenpy.linalg.sparse.FlatLinearOperator.npc_to_flat`,
  merge :meth:`tenpy.linalg.sparse.FlatLinearOperator.flat_to_npc_all_sectors` into :meth:`~tenpy.linalg.sparse.FlatLinearOperator.flat_to_npc`.
- Change the ``chinfo.names`` of the specific :class:`~tenpy.networks.site.Site` classes to be more consistent and clear.
- Add the more powerful :meth:`tenpy.networks.site.set_common_charges` to replace :meth:`tenpy.networks.site.multi_sites_combine_charges`.

Fixed
^^^^^
- The form of the eigenvectors returned by :meth:`tenpy.networks.mps.TransferMatrix.eigenvectors` 
  was dependent on the `charge_sector` given in the initialization; we try to avoid this now (if possible).
- The charge conserved by ``SpinHalfFermionSite(cons_Sz='parity')`` was wired.
- Allow to pass npc Arrays as Arguments to :meth:`~tenpy.networks.mps.MPS.expectation_value_multi_sites` and
  other correlation functions (:issue:`116`).
- :mod:`tenpy.tools.hdf5_io` did not work with h5py version >= (3,0) due to a change in string encoding (:issue:`117`).
