[latest]
========

Release Notes
-------------
TODO: Summarize the most important changes

Changelog
---------

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- For infinite MPS (:class:`~tenpy.networks.mps.MPS` with ``bs='infinite'``), we now only store
  ``L`` singular values instead of ``L + 1`` in the :attr:``~tenpy.networks.mps.MPS._S`` attribute.
  The ``_S[L]`` entry always was equal to ``_S[0]`` anyway. With the incorporation of shift-symmetry,
  i.e. symmetries that depend on spatial position, we need to distinguish the concept of
  the singular values to the right of the last site from the singular values left of the first site,
  since they differ by a spatial translation.
  Singular values should only be accessed via the methods :meth:`~tenpy.networks.mps.MPS.get_SL`,
  :meth:`~tenpy.networks.mps.MPS.get_SR`, :meth:`~tenpy.networks.mps.MPS.set_SL`,
  and :meth:`~tenpy.networks.mps.MPS.set_SR` which account for these spatial translations.

Added
^^^^^
- nothing yet

Changed
^^^^^^^
- nothing yet

Fixed
^^^^^
- nothing yet
