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
- Allow to pass and merge multiple parameter files to ``tenpy-run`` from the command line.
- Greatly expanded userguide on :doc:`/intro/simulations` and added more parameter examples.
- Option `preserve_norm` for :class:`~tenpy.algorithms.mpo_evolution.ExpMPOEvolution`.
- Allow non-trivial :attr:`~tenpy.models.lattice.Lattice.position_disorder` for lattices.
- Option `fix_u` for :func:`~tenpy.simulations.measurement.onsite_expectation_value`.
- Lattice :attr:`~tenpy.models.lattice.Lattice.cylinder_axis`.
- Random number generator :attr:`~tenpy.models.model.Model.rng` for models.
- :meth:`~tenpy.models.aklt.AKLTChain.psi_AKLT` for the exact MPS ground state of (spin-1/2) AKLT chain.
- :func:`~tenpy.simulations.simulation.init_simulation` and :func:`~tenpy.simulations.simulation.init_simulation_from_checkpoint` for debugging or post-simulation measurement.
- :func:`~tenpy.linalg.np_conserved.orthogonal_columns` constructing orthogonal columns to a given (rectangular) matrix.
- :meth:`~tenpy.networks.mps.MPS.enlarge_chi` for artificially enlarging the bond dimension.

Changed
^^^^^^^
- Renamed ``tenpy.networks.mpo.MPOGraph.add_string`` to :meth:`~tenpy.networks.mpo.MPOGraph.add_string_left_to_right`
  as part of the fix for :issue:`148`. Added similar :meth:`~tenpy.networks.mpo.MPOGraph.add_string_left_to_right`.
- Automatically shift terms in :meth:`~tenpy.networks.mps.MPS.expectation_value_terms_sum` to start in the MPS unit cell for infinite MPS.
- Possible ordering='folded' for the :class:`~tenpy.models.lattice.Ladder`.
- Enhanced implementation of :meth:`~tenpy.networks.mps.MPS.canonical_form_infinite2` to replace :meth:`~tenpy.networks.mps.MPS.canonical_form_infinite`.
- Split up :meth:`tenpy.networks.mpo.MPO.expectation_value` into :meth:`~tenpy.networks.mpo.MPO.expectation_value_finite`
  and :meth:`~tenpy.networks.mpo.MPO.expectation_value_power` and add :meth:`tenpy.networks.mpo.MPO.expectation_value_TM`

Fixed
^^^^^
- :issue:`145` that :func:`~tenpy.networks.mpo.make_W_II` failed for MPOs with trivial virtual bonds.
- Make :func:`~tenpy.linalg.np_conserved.detect_qtotal` more stable: use the maximal entry instead of the first non-zero one.
- :issue:`148` that generating MPOs with long-range couplings over multiple MPS unit cells and multi-couplings raised errors.
- The :func:`~tenpy.linalg.np_conserved.qr` decomposition with ``mode='complete'`` sometimes returned wrong charges.
  Moreover, it sometimes gave zero columns in Q if the R part was completely zero for that charge block.
- Adjust default `trunc_params` of :func:`~tenpy.networks.mps.MPS.compute_K` and :func:`~tenpy.networks.mps.MPS.permute_sites` to avoid too severe truncation.
- (!) Non-trivial `start_time` parameter caused wrong evolution in :class:`~tenpy.algorithms.mpo_evolution.TimeDependentExpMPOEvolution`.
- Make sure that :meth:`~tenpy.models.lattice.lat2mps_idx` doesn't modify arguments in place.
- The power-method :meth:`tenpy.networks.mpo.MPO.expectation_value` did not work correctly for ``H.L != psi.L``.
- :meth:`~tenpy.models.model.CouplingModel.add_local_term` did not work with `plus_hc=True`.
- :meth:`tenpy.linalg.sparse.FlatLinearOperator.eigenvectors` did not always return orthogonal eigenvectors with well-defined charges.
- Fix :class:`tenpy.linalg.sparse.FlatLinearOperator` to not use the full flat array, but just the block with nonzero entries (which can be much smaller for a few charges).
  This is enabled over a new option `compact_flat` that defaults to True if the vector leg is blocked by charge (and charge_sector is not None).
- Make ``cons_Sz='parity'`` for the :class:`~tenpy.networks.site.SpinHalfSite` non-trivial.
- The first, initial measurements for time-dependent Hamiltonians might have used wrong time for sequential/resume run.
