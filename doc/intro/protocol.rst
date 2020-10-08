Protocol for using (i)DMRG
==========================

While this documentation contains extensive guidance on how to interact with the tenpy, it is often unclear how to approach a physics question using these methods.
This page is an attempt to provide such guidance, describing a protocol on how to go from a model implementation to an answered question.

The basic workflow for an (i)DMRG project is as follows, with individual steps expanded on later where necessary.

1. Confirm the correctness of the model implementation.
2. Run some low-effort tests to see whether the question seems answerable.
3. If the tests are successful, run production-quality simulations.
   This will be entirely particular to the project you're working on.
4. Confirm that your results are converged.

Confirming the model is correct
-------------------------------

Although TeNPy makes model implementation much easier than constructing the MPO by hand, one should still ensure that the MPO represents the intended model faithfully.
There are several possible ways to do this. Firstly, for sufficiently small system sizes, one can contract the entire MPO into a matrix, and inspect the matrix elements. In TeNPy, this can be done using :meth:`~tenpy.networks.mpo.MPO.get_full_hamiltonian`. These should reproduce the analytical Hamiltonian up to machine precision, or any other necessary cut-off (e.g., long-range interactions may be truncated at some finite distance).

Secondly, if the model basis allows it, one can construct (product state) MPSs for known eigenstates of the model and evaluate whether these reproduce the correct eigenvalues upon contraction with the MPO.

Finally, one can sometimes construct a basis of single- or even two-particle MPSs in some basis, and evaluate the MPO on this basis to get a representation of the single- and two-particle Hamiltonian.
If the model contains only single- and two-body terms, this latter approach should reproduce all terms in the Hamiltonian.

Low-effort tests
----------------
As not every state can be accurately represented by an MPS, some results are outside the reach of (i)DMRG. 
To prevent wasting considerable numerical resources on a fruitless project, it is recommended to run some low-effort trials first, and see whether any indication of the desired result can be found.
If so, one can then go on to more computationally expensive simulations.
If not, one should evaluate:

1. Whether there is a mistake in the model or simulation set-up, 
2. Whether a slightly more computationally expensive test would potentially yield a result, or
3. Whether your approach is unfortunately out of reach of (i)DMRG.

To set up low-effort trials, one should limit system size, bond dimension and the range of interactions, as well as (if possible) target a non-critical region of phase space. 
All these measures reduce the size of and/or entanglement entropy needing to be captured by the MPS, which yields both memory and run time advantages.
Of course, one introduces a trade-off between computational cost and accuracy, which is why one should be careful to not put too much faith into results obtained at this stage.

Detecting convergence issues
----------------------------

Ensuring that the results of an (i)DMRG simulation are well-converged and thus reliable is a hugely important part of any (i)DMRG study.
Possible indications that there might be a convergence issue include:

1. The simulation shows a non-monotonous decrease of energy, and/or a non-monotonous increase of entanglement entropy. An increase of energy or decrease of entanglement entropy on subsequent steps within a sweep, or between subsequent sweeps, are particularly suspicious.
2. The simulation does not halt because it reached a convergence criterion, but because it reached its maximum number of sweeps.
3. Results vary wildly under small changes of parameters. In particular, if a small change in bond dimension yields a big change in results, one should be suspicious of the data.

Combating convergence issues
----------------------------

To combat convergence issues of the (i)DMRG algorithm, several strategies (short of switching to a different method) can be attempted:

1. Ensure that there are no errors in the model (see above) or the simulation set-up.
2. Increase the maximum bond dimension.
3. Ramp up the maximum bond dimension during simulation, rather than starting at the highest value. I.e., define a schedule wherein the first :math:`N_{\mathrm{sweeps}}` sweeps run at some :math:`\chi_1 < \chi_\mathrm{max}`, the next :math:`N_{\mathrm{sweeps}}` at :math:`\chi_1 < \chi_2 < \chi_{\mathrm{max}}`, etc. 
   This can be done through the ``chi_list`` option of the :class:`~tenpy.algorithms.dmrg.DMRGEngine`.
   You should also make sure that the ``max_hours`` option is set to sufficiently long runtimes.
4. Increase the maximum number of sweeps the algorithm is allowed to make, through the ``max_sweeps`` option of the :class:`~tenpy.algorithms.dmrg.DMRGEngine`.
5. Change the :class:`~tenpy.algorithms.dmrg.Mixer` settings to in- or decrease the effects of the mixer.
6. Change convergence criteria. This will not overcome convergence issues in itself, but can help fine tune the (i)DMRG simulation if it takes a long time to converge (relax the convergence constraints), or if the simulation finishes too soon (tighten the constraints).
   Criteria to consider are ``max_E_err`` and ``max_S_err``, in :class:`~tenpy.algorithms.dmrg.DMRGEngine`.
7. Increase the minimum number of sweeps taken by the algorithm. Again, this will not resolve issues due to bad convergence, but might prevent bad results due to premature convergence.
   This can be done through the ``min_sweeps`` option of the :class:`~tenpy.algorithms.dmrg.DMRGEngine`.
8. Change the size and shape of the MPS unit cell (where possible), in case an artificially enforced translational invariance prevents the algorithm from finding a true ground state which is incommensurate with this periodicity.
   For example, a chain system which has a true ground state that is periodic in three sites, will not be accurately represented by a two-site MPS unit cell, as the latter enforces two-site periodicity.
	

In some instances, it is essentially unavoidable to encounter convergence issues.
In particular, a simulation of a critical state can cause problems with (i)DMRG convergence, as these states violate the area law underlying an accurate MPS approximation.
In these cases, one should acknowledge the difficulties imposed by the method and take care to be very careful in interpreting the data.