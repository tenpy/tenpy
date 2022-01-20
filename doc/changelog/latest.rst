[latest]
========

Release Notes
-------------
TODO: Summarize the most important changes

Changelog
---------

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Add more fine grained sweep convergence checks for the :class:`~tenpy.algorithms.mps_common.VariationalCompression` (used when applying an MPO to an MPS!).
  In this context, we renamed the parameter `N_sweeps` to :cfg:option:`VariationalCompression.max_sweeps`.
  Further, we added the parameter :cfg:option:`VariationalCompression.min_sweeps` and :cfg:option:`VariationalCompression.tol_theta_diff`
- Adjusted default paramters of :meth:`tenpy.networks.mps.InitialStateBuilder.randomized` to be as documented with better ``chi_max``.
- No longer return `ov` from :func:`tenpy.linalg.lanczos.gram_schmidt`.

Added
^^^^^
- Wrappers for the helical and irregular lattice (removing sites) in :meth:`~tenpy.models.model.CouplingMPOModel.init_lattice`.
- Options `pos_diag_r`, `qtotal_Q` and `qconj_inner` for :func:`~tenpy.linalg.np_conserved.qr`.
- :class:`tenpy.linalg.lanczos.Arnoldi` (and common base class :class:`~tenpy.linalg.lanczos.KrylovBased` with :class:`~tenpy.linalg.lanczos.LanczosGroundState`).

Changed
^^^^^^^
- Renamed ``tenpy.networks.mpo.MPOGraph.add_string`` to :meth:`~tenpy.networks.mpo.MPOGraph.add_string_left_to_right`
  as part of the fix for :issue:`148`. Added similar :meth:`~tenpy.networks.mpo.MPOGraph.add_string_left_to_right`.
- Automatically shift terms in :meth:`~tenpy.networks.mps.MPS.expectation_value_terms_sum` to start in the MPS unit cell for infinite MPS.
- Possible ordering='folded' for the :class:`~tenpy.models.lattice.Ladder`.

Fixed
^^^^^
- :issue:`145` that :func:`~tenpy.networks.mpo.make_W_II` failed for MPOs with trivial virtual bonds.
- Make :func:`~tenpy.linalg.np_conserved.detect_qtotal` more stable: use the maximal entry instead of the first non-zero one.
- :issue:`148` that generating MPOs with long-range couplings over multiple MPS unit cells and multi-couplings raised errors.
- The :func:`~tenpy.linalg.np_conserved.qr` decomposition with ``mode='complete'`` sometimes returned wrong charges.
- Adjust default `trunc_params` of :func:`~tenpy.networks.mps.MPS.compute_K` and :func:`~tenpy.networks.mps.MPS.permute_sites` to avoid too severe truncation.
- (!) Non-trivial `start_time` parameter caused wrong evolution in :class:`~tenpy.algorithms.mpo_evolution.TimeDependentExpMPOEvolution`.
