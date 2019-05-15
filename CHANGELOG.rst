CHANGELOG
=========

All notable changes to the project will be documented in this file.
The project adheres `semantic versioning <http://semver.org/spec/v2.0.0.html>`_

[Unreleased]
------------

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- nothing yet

Changed
^^^^^^^
- nothing yet

Added
^^^^^
- `max_range` attribute in :class:`~tenpy.networks.mpo.MPO` and :class:`~tenpy.networks.mpo.MPOGraph`.
- :meth:`~tenpy.networks.mpo.MPO.is_hermitian`
- nearest-Neighbor interaction in :class:`~tenpy.models.bose_hubbard.BoseHubbardModel`

Fixed
^^^^^
- Missing a factor 0.5 in :func:`~tenpy.linalg.random_matrix.GUE`.


[0.4.0] - 2019-04-28
--------------------

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- The argument order of :class:`tenpy.models.lattice.Lattice` could be a tuple ``(priority, snake_winding)`` before. 
  This is no longer valid and needs to be replaced by ``("standard", snake_winding, priority)``.
- Moved the boundary conditions `bc_coupling` from the :class:`tenpy.models.model.CouplingModel` into the :class:`tenpy.models.lattice.Lattice` (as `bc`).
  Using the parameter `bc_coupling` will raise a FutureWarning, one should set the boundary conditions directly in the lattice.
- Added parameter `permute` (True by default) in :meth:`tenpy.networks.mps.MPS.from_product_state` and :meth:`tenpy.networks.mps.MPS.from_Bflat`.
  The resulting state will therefore be independent of the "conserve" parameter of the Sites - unlike before, 
  where the meaning of the p_state argument might have changed.
- Generalize and rename  ``tenpy.networks.site.DoubleSite`` to :class:`tenpy.networks.site.GroupedSite`,
  to allow for an arbitrary number of sites to be grouped. 
  Arguments ``site0, site1, label0, label1`` of the __init__ can be replaced with ``[site0, site1], [label0, label1]``
  and ``op0, op1`` of the `kronecker_product` with ``[op0, op1]``; this will recover the functionality of the `DoubleSite`.
- Restructured callstructure of Mixer in DMRG, allowing an implementation of other mixers.
  To enable the mixer, set the DMRG parameter ``"mixer"`` to ``True`` or ``'DensityMatrixMixer'``
  instead of just ``'Mixer'``.
- The interaction parameter in the :class:`tenpy.models.bose_hubbbard_chain.BoseHubbardModel` 
  (and :class:`tenpy.models.bose_hubbbard_chain.BoseHubbardChain`) did not correspond to :math:`U/2 N (N-1)` 
  as claimed in the Hamiltonian, but to :math:`U N^2`.
  The correcting factor 1/2 and change in the chemical potential have been fixed.
- Major restructuring of :mod:`tenpy.linalg.np_conserved` and :mod:`tenpy.linalg.charges`.
  This should not break backwards-compatibility, but if you compiled the cython files, you **need** to remove the 
  old binaries in the source directory. Using ``bash cleanup.sh`` might be helpful to do that, but also remove other files within the repository, so be careful and make a backup beforehand to be on the save side.
  Afterwards recompile with ``bash compile.sh``.
- Changed structure of :attr:`tenpy.models.model.CouplingModel.onsite_terms` and :attr:`tenpy.models.model.CouplingModel.coupling_terms`:
  Each of them is now a dictionary with category strings as keys and the newly introduced
  :class:`tenpy.networks.terms.OnsiteTerms` and :class:`tenpy.networks.terms.CouplingTerms` as values.
- :meth:`tenpy.models.model.CouplingModel.calc_H_onsite` is deprecated in favor of new methods.
- Argument `raise_op2_left` of :meth:`tenpy.models.model.CouplingModel.add_coupling` is deprecated.


Added
^^^^^
- :meth:`tenpy.networks.mps.MPS.canonical_form_infinite`.
- :meth:`tenpy.networks.mps.MPS.expectation_value_term`, :meth:`tenpy.networks.mps.MPS.expectation_value_terms_sum` and
  :meth:`tenpy.networks.mps.MPS.expectation_value_multi_sites` for expectation values of terms.
- :meth:`tenpy.networks.mpo.MPO.expectation_value` for an MPO.
- :meth:`tenpy.linalg.np_conserved.Array.extend` and :meth:`tenpy.linalg.charges.LegCharge.extend`,
  allowing to extend an Array with zeros.
- DMRG parameter ``'orthogonal_to'`` allows to calculate excited states for finite systems.
- possibility to change the number of charges after creating LegCharges/Arrays.
- more general way to specify the order of sites in a :class:`tenpy.models.lattice.Lattice`.
- new :class:`tenpy.models.lattice.Triangular`, :class:`tenpy.models.lattice.Honeycomb` and :class:`tenpy.models.lattice.Kagome` lattice
- a way to specify nearest neighbor couplings in a :class:`~tenpy.models.lattice.Lattice`, 
  along with methods to count the number of nearest neighbors for sites in the bulk, and
  a way to plot them (:meth:`~tenpy.models.lattice.Lattice.plot_coupling` and friends)
- :meth:`tenpy.networks.mpo.MPO.from_grids` to generate the MPO from a grid.
- :class:`tenpy.models.model.MultiCouplingModel` for couplings involving more than 2 sites.
- request #8: Allow shift in boundary conditions of :class:`~tenpy.models.model.CouplingModel`.
- Allow to use state labels in :meth:`tenpy.networks.mps.MPS.from_product_state`.
- :class:`tenpy.models.model.CouplingMPOModel` structuring the default initialization of most models.
- Allow to force periodic boundary conditions for finite MPS in the :class:`~tenpy.modles.model.CouplingMPOModel`.
  This is not recommended, though.
- :meth:`tenpy.models.model.NearestNeighborModel.calc_H_MPO_from_bond` and
  :meth:`tenpy.models.model.MPOModel.calc_H_bond_from_MPO` for conversion of H_bond into H_MPO and vice
  versa.
- :class:`tenpy.algorithms.tebd.RandomUnitaryEvolution` for random unitary circuits
- Allow documentation links to github issues, arXiv, papers by doi and the forum with 
  e.g. ``:issue:`5`, :arxiv:`1805.00055`, :doi:`10.21468/SciPostPhysLectNotes.5`, :forum:`3```
- :meth:`tenpy.models.model.CouplingModel.coupling_strength_add_ext_flux` for adding hoppings with external flux.
- :meth:`tenpy.models.model.CouplingModel.plot_coupling_terms` to visualize the added coupling terms.
- :class:`tenpy.networks.terms.OnsiteTerms`, :class:`tenpy.networks.terms.CouplingTerms`, :class:`tenpy.networks.terms.MultiCouplingTerm` 
  containing the of terms for the :class:`~tenpy.models.model.CouplingModel` and :class:`~tenpy.models.model.MultiCouplingModel`.
  This allowed to add the `category` argument to :class:`~tenpy.models.model.CouplingModel.add_onsite`, :class:`~tenpy.models.model.CouplingModel.add_coupling` and :class:`~tenpy.models.model.MultiCouplingModel.add_multi_coupling`.
- :class:`tenpy.networks.terms.TermList` as another (more human readable) representation of terms with conversion from
  and to the other ``*Term`` classes.
- :meth:`tenpy.networks.mps.MPS.init_LP` and :meth:`tenpy.networks.mps.MPS.init_RP` to initialize left and right parts
  of an Environment.
- :meth:`tenpy.networks.mpo.MPOGraph.from_terms` and :meth:`tenpy.networks.mpo.MPOGraph.from_term_list`.
- argument `charge_sector` in :meth:`tenpy.networks.mps.MPS.correlation_length`.


Changed
^^^^^^^
- moved toycodes from the folder ``examples/`` to a new folder ``toycodes/`` to separate them clearly.
- major remodelling of the internals of :class:`tenpy.linalg.np_conserved` and :class:`tenpy.linalg.charges`.
    - Introduced the new module ``tenpy/linalg/_npc_helper.pyx`` which contains all the Cython code, and gets imported by
    - :class:`~tenpy.linalg.np_conserved.Array` now rejects addition/subtraction with other types
    - :class:`~tenpy.linalg.np_conserved.Array` now rejects multiplication/division  with non-scalar types
    - By default, make deep copies of npc Arrays.
- Restructured lanczos into a class, added time evolution calculating ``exp(A*dt)|psi0>``
- Warning for poorly conditioned Lanczos; to overcome this enable the new parameter `reortho`.
- Simplified call strucutre of :meth:`~tenpy.linalg.np_conserved.Array.extend`, and
  :meth:`~tenpy.linalg.charges.LegCharge.extend`.
- Restructured :mod:`tenpy.algorithms.dmrg`:

  - :func:`~tenpy.algorithms.dmrg.run` is now just a wrapper around the new 
    :meth:`~tenpy.algorithms.dmrg.Engine.run`, ``run(psi, model, pars)`` is roughly equivalent to
    ``eng = EngineCombine(psi, model, pars); eng.run()``.
  - Added :meth:`~tenpy.algorithms.dmrg.Engine.init_env` and :meth:`~tenpy.algorithms.dmrg.Engine.reset_stats`
    to allow a simple restart of DMRG with slightly different parameters, e.g. for tuning Hamiltonian parameters.
  - Call :meth:`~tenpy.networks.mps.MPS.canonical_form` for infinite systems if the final state is not in canonical form.

- Changed **default values** for some parameters:

  - set ``trunc_params['chi_max'] = 100``. Not setting a `chi_max` at all will lead to memory problems.
    Disable ``DMRG_params['chi_list'] = None`` by default to avoid conflicting settings.
  - reduce to ``mixer_params['amplitude'] = 1.e-5``. A too strong mixer screws DMRG up pretty bad.
  - increase ``Lanczos_params['N_cache'] = N_max`` (i.e., keep all states)
  - set ``DMRG_params['P_tol_to_trunc'] = 0.05`` and provide reasonable ..._min and ..._max values.
  - increased (default) DMRG accuracy by setting
    ``DMRG_params['max_E_err'] = 1.e-8`` and ``DMRG_params['max_S_err'] = 1.e-5``.
  - don't check the (absolute) energy for convergence in Lanczos.
  - set ``DMRG_params['norm_tol'] = 1.e-5`` to check whether the final state is in canonical form.

- Verbosity of :func:`~tenpy.tools.params.get_parameter` reduced: Print parameters only for verbosity >=1.
  and default values only for verbosity >= 2.
- Don't print the energy during real-time TEBD evolution - it's preserved up to truncation errors.
- Renamed the `SquareLattice` class to :class:`tenpy.models.lattice.Square` for better consistency.
- auto-determine whether Jordan-Wigner strings are necessary in
  :meth:`~tenpy.models.model.CouplingModel.add_coupling`.
- The way the labels of npc Arrays are stored internally changed to a simple list with None entries.
  There is a deprecated propery setter yielding a dictionary with the labels.
- renamed `first_LP` and `last_RP` arguments of :class:`~tenpy.networks.mps.MPSEnvironment` and :class:`~tenpy.networks.mpo.MPOEnvironment` to `init_LP` and `init_RP`.
- Testing: insetad of the (outdated) `nose <https://nose.readthedocs.io/en/latest/>`_, we now use `pytest <https://pytest.org>` for testing.

Fixed
^^^^^
- :issue:`22`: **Serious bug** in :func:`tenpy.linalg.np_conserved.inner`: if ``do_conj=True`` is used with non-zero
  ``qtotal``, it returned 0. instead of non-zero values.
- avoid error in :meth:`tenpy.networks.mps.MPS.apply_local_op`
- Don't carry around total charge when using DMRG with a mixer
- Corrected couplings of the FermionicHubbardChain
- :issue:`2`: memory leak in cython parts when using intelpython/anaconda
- :issue:`4`: incompatible data types.
- :issue:`6`: the CouplingModel generated wrong Couplings in some cases
- :issue:`19`: Convergence of energy was slow for infinite systems with ``N_sweeps_check=1``
- more reasonable traceback in case of wrong labels
- wrong dtype of npc.Array when adding/subtracting/... arrays of different data types
- could get wrong H_bond for completely decoupled chains.
- SVD could return outer indices with different axes
- :meth:`tenpy.networks.mps.MPS.overlap` works now for MPS with different total charge
  (e.g. after ``psi.apply_local_op(i, 'Sp')``).
- skip existing graph edges in MPOGraph.add() when building up terms without the strength part.

Removed
^^^^^^^
- Attribute `chinfo` of :class:`~tenpy.models.lattice.Lattice`.

[0.3.0] - 2018-02-19
--------------------
This is the first version published on github.

Added
^^^^^
- Cython modules for np_conserved and charges, which can optionally be compiled for speed-ups
- tools.optimization for dynamical optimization
- Various models.
- More predefined lattice sites.
- Example toy-codes.
- Network contractor for general networks

Changed
^^^^^^^
- Switch to python3

Removed
^^^^^^^
- Python 2 support.


[0.2.0] - 2017-02-24
--------------------
- Compatible with python2 and python3 (using the 2to3 tool).
- Development version.
- Includes TEBD and DMRG.


Changes compared to previous TeNPy
----------------------------------
This library is based on a previous (closed source) version developed mainly by
Frank Pollmann, Michael P. Zaletel and Roger S. K. Mong.
While allmost all files are completely rewritten and not backwards compatible, the overall structure is similar.
In the following, we list only the most important changes.

Global Changes
^^^^^^^^^^^^^^
- syntax style based on PEP8. Use ``$>yapf -r -i ./`` to ensure consitent formatting over the whole project.
  Special comments ``# yapf: disable`` and ``# yapf: enable`` can be used for manual formatting of some regions in code.
- Following PEP8, we distinguish between 'private' functions, 
  indicated by names starting with an underscore and to be used only within the library, and the public API. 
  The puplic API should be backwards-compatible with different releases, while private functions might change at any time.
- all modules are in the folder ``tenpy`` to avoid name conflicts with other libraries.
- withing the library, relative imports are used, e.g., ``from ..tools.math import (toiterable, tonparray)``
  Exception: the files in `tests/` and `examples/` run as ``__main__`` and can't use relative imports

  Files outside of the library (and in `tests/`, `examples/`) should use
  absolute imports, e.g. ``import tenpy.algorithms.tebd``
- renamed `tenpy/mps/` to `tenpy/networks`, since it containes various tensor networks.
- added :class:`~tenpy.networks.site.Site` describing the local physical sites by providing the physical LegCharge and
  onsite operators.

np_conserved
^^^^^^^^^^^^
- pure python, no need to compile!
- in module :mod:`tenpy.linalg` instead of ``algorithms/linalg``.
- moved functionality for charges to :mod:`~tenpy.linalg.charges`
- Introduced the classes :class:`~tenpy.linalg.charges.ChargeInfo` (basically the old ``q_number``, and ``mod_q``)
  and :class:`~tenpy.linalg.charges.LegCharge` (the old ``qind, qconj``).
- Introduced the class :class:`~tenpy.linalg.charges.LegPipe` to replace the old ``leg_pipe``.
  It is derived from ``LegCharge`` and used as a leg in the `array` class. Thus any inherited array (after
  ``tensordot`` etc still has all the necessary information to split the legs.
  (The legs are shared between different arrays, so it's saved only once in memory)
- Enhanced indexing of the array class to support slices and 1D index arrays along certain axes
- more functions, e.g. :func:`~tenpy.linalg.np_conserved.grid_outer`

TEBD
^^^^
- Introduced TruncationError for easy handling of total truncation error.
- some truncation parameters are renamed and may have a different meaning, e.g. `svd_max` -> `svd_min` 
  has no 'log' in the definition.

DMRG
^^^^
- separate Lanczos module in `tenpy/linalg/`. Strangely, the old version orthoganalized
  against the complex conjugates of `orthogonal_to` (contrary to it's doc string!)
  (and thus calculated 'theta_o' as bra, not ket).
- cleaned up, provide prototypes for DMRG engine and mixer.

Tools
^^^^^
- added :mod:`tenpy.tools.misc`, which contains 'random stuff' from old ``tools.math``
  like ``to_iterable`` and ``to_array`` (renamed to follow PEP8, documented)
- moved stuff for fitting to :mod:`tenpy.tools.fit`
- enhanced :func:`tenpy.tools.string.vert_join` for nice formatting
- moved (parts of) old `cluster/omp.py` to :mod:`tenpy.tools.process`
- added :mod:`tenpy.tools.params` for a simplified handling of parameter/arguments for models and/or algorithms.
  Similar as the old `models.model.set_var`, but use it also for algorithms. Also, it may modify the given dictionary.
