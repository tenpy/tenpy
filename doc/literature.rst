Literature
==========

This is a (by far non-exhaustive) list of some references for the various ideas behind the code.
They can be cited from the python doc-strings using the format ``[Author####]_``.
Within each category, we sort the references by year and author.

.. comment
    When you add something, please also add a reference to it, i.e., give a short comment in the top of the subsection.

TeNPy related sources
---------------------
[TeNPyNotes]_ are lecture notes, meant as an introduction to tensor networks (focusing on MPS), and introduced TeNPy to
the scientific community by giving examples how to call the algorithms in TeNPy.
[TeNPySource]_ is the location of the source code, and the place where you can report bugs.
[TeNPyDoc]_ is where the location is hosted online.
[TeNPyForum]_ is the place where you can ask questions and look for help, when you are stuck with implementing something.

.. [TeNPyNotes] 
    "Efficient numerical simulations with Tensor Networks: Tensor Network Python (TeNPy)"
    J. Hauschild, F. Pollmann, SciPost Phys. Lect. Notes 5 (2018), :arxiv:`1805.00055`, :doi:`10.21468/SciPostPhysLectNotes.5`
.. [TeNPySource] 
    https://github.com/tenpy/tenpy
.. [TeNPyDoc] 
    Online documentation, https://tenpy.github.io/
.. [TeNPyForum] 
    Community forum for discussions, FAQ and announcements, https://tenpy.johannes-hauschild.de


General reading
---------------
[Schollwoeck2011]_ is an extensive introduction to MPS, DMRG and TEBD with lots of details on the implementations, and a classic read, although a bit lengthy.
Our [TeNPyNotes]_ are a shorter summary of the important concepts, similar as [Orus2014]_.
[Hubig2019]_ is a very good, recent review focusing on time evolution with MPS.
The lecture notes of [Eisert2013]_ explain the area law as motivation for tensor networks very well.
PEPS are for example reviewed in [Verstraete2009]_, [Eisert2013]_ and [Orus2014]_.
[Stoudenmire2011]_ reviews the use of DMRG for 2D systems.
[Cirac2009]_ discusses the different groups of tensor network states.

.. [Cirac2009]
    "Renormalization and tensor product states in spin chains and lattices"
    J. I. Cirac and F. Verstraete, Journal of Physics A: Mathematical and Theoretical, 42, 50 (2009) :arxiv:`0910.1130` :doi:`10.1088/1751-8113/42/50/504004`
.. [Verstraete2009]
    "Matrix Product States, Projected Entangled Pair States, and variational renormalization group methods for quantum spin systems"
    F. Verstraete  and  V. Murg  and  J.I. Cirac, Advances in Physics 57 2, 143-224 (2009) :arxiv:`0907.2796` :doi:`10.1080/14789940801912366`
.. [Schollwoeck2011]
    "The density-matrix renormalization group in the age of matrix product states"
    U. Schollwoeck, Annals of Physics 326, 96 (2011), :arxiv:`1008.3477` :doi:`10.1016/j.aop.2010.09.012`
.. [Stoudenmire2011]
    "Studying Two Dimensional Systems With the Density Matrix Renormalization Group"
    E.M. Stoudenmire, Steven R. White, Ann. Rev. of Cond. Mat. Physics, 3: 111-128 (2012), :arxiv:`1105.1374` :doi:`10.1146/annurev-conmatphys-020911-125018`
.. [Eisert2013]
    "Entanglement and tensor network states"
    J. Eisert, Modeling and Simulation 3, 520 (2013) :arxiv:`1308.3318`
.. [Orus2014]
    "A Practical Introduction to Tensor Networks: Matrix Product States and Projected Entangled Pair States"
    R. Orus, Annals of Physics 349, 117-158 (2014) :arxiv:`1306.2164` :doi:`10.1016/j.aop.2014.06.013`
.. [Hubig2019]
    "Time-evolution methods for matrix-product states"
    S. Paeckel, T. Köhler, A. Swoboda, S. R. Manmana, U. Schollwöck, C. Hubig, :arxiv:`1901.05824`

Algorithm developments
----------------------
[White1992]_ is the invention of DMRG, which started everything.
[Vidal2004]_ introduced TEBD.
[White2005]_ and [Hubig2015]_ solved problems for single-site DMRG.
[McCulloch2008]_ was a huge step forward to solve convergence problems for infinite DMRG.
[Singh2009]_, [Singh2010]_ explain how to incorporate Symmetries.
[Haegeman2011]_ introduced TDVP, again explained more accessible in [Haegeman2016]_.
[Karrasch2013]_ gives some tricks to do finite-temperature simulations (DMRG), which is a bit extended in [Hauschild2018]_.
[Vidal2007]_ introduced MERA.


.. [White1992]
    "Density matrix formulation for quantum renormalization groups"
    S. White, Phys. Rev. Lett. 69, 2863 (1992) :doi:`10.1103/PhysRevLett.69.2863`,
    S. White, Phys. Rev. B 84, 10345 (1992) :doi:`10.1103/PhysRevB.48.10345`
.. [Vidal2004]
    "Efficient Simulation of One-Dimensional Quantum Many-Body Systems"
    G. Vidal, Phys. Rev. Lett. 93, 040502 (2004), :arxiv:`quant-ph/0310089` :doi:`10.1103/PhysRevLett.93.040502`
.. [White2005]
    "Density matrix renormalization group algorithms with a single center site"
    S. White, Phys. Rev. B 72, 180403(R) (2005), :arxiv:`cond-mat/0508709` :doi:`10.1103/PhysRevB.72.180403`
.. [Vidal2007]
    "Entanglement Renormalization"
    G. Vidal, Phys. Rev. Lett. 99, 220405 (2007), :arxiv:`cond-mat/0512165`, :doi:`10.1103/PhysRevLett.99.220405`
.. [McCulloch2008]
    "Infinite size density matrix renormalization group, revisited"
    I. P. McCulloch, :arxiv:`0804.2509`
.. [Singh2009]
    "Tensor network decompositions in the presence of a global symmetry"
    S. Singh, R. Pfeifer, G. Vidal, Phys. Rev. A 82, 050301(R), :arxiv:`0907.2994` :doi:`10.1103/PhysRevA.82.050301`
.. [Singh2010]
    "Tensor network states and algorithms in the presence of a global U(1) symmetry"
    S. Singh, R. Pfeifer, G. Vidal, Phys. Rev. B 83, 115125, :arxiv:`1008.4774` :doi:`10.1103/PhysRevB.83.115125`
.. [Haegeman2011]
    "Time-Dependent Variational Principle for Quantum Lattices"
    J. Haegeman, J. I. Cirac, T. J. Osborne, I. Pizorn, H. Verschelde, F. Verstraete, Phys. Rev. Lett. 107, 070601 (2011), :arxiv:`1103.0936` :doi:`10.1103/PhysRevLett.107.070601`
.. [Karrasch2013]
    "Reducing the numerical effort of finite-temperature density matrix renormalization group calculations"
    C. Karrasch, J. H. Bardarson, J. E. Moore, New J. Phys. 15, 083031 (2013), :arxiv:`1303.3942` :doi:`10.1088/1367-2630/15/8/083031`
.. [Hubig2015]
    "Strictly single-site DMRG algorithm with subspace expansion"
    C. Hubig, I. P. McCulloch, U. Schollwoeck, F. A. Wolf, Phys. Rev. B 91, 155115 (2015), :arxiv:`1501.05504` :doi:`10.1103/PhysRevB.91.155115`
.. [Haegeman2016]
    "Unifying time evolution and optimization with matrix product states"
    J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete, Phys. Rev. B 94, 165116 (2016), :arxiv:`1408.5056` :doi:`10.1103/PhysRevB.94.165116`
.. [Hauschild2018] 
    "Finding purifications with minimal entanglement"
    J. Hauschild, E. Leviatan, J. H. Bardarson, E. Altman, M. P. Zaletel, F. Pollmann, Phys. Rev. B 98, 235163 (2018), :arxiv:`1711.01288` :doi:`10.1103/PhysRevB.98.235163`

Related theory
--------------
The following are referenced from somewhere in the algorithms.

.. [Resta1997]
    "Quantum-Mechanical Position Operator in Extended Systems"
    R. Resta, Phys. Rev. Lett. 80, 1800 (1997) :doi:`10.1103/PhysRevLett.80.1800`
.. [Neupert2011]
    "Fractional quantum Hall states at zero magnetic field"
    Titus Neupert, Luiz Santos, Claudio Chamon, and Christopher Mudry, Phys. Rev. Lett. 106, 236804 (2011), :arxiv:`1012.4723` :doi:`10.1103/PhysRevLett.106.236804`
.. [Yang2012]
    "Topological flat band models with arbitrary Chern numbers"
    Shuo Yang, Zheng-Cheng Gu, Kai Sun, and S. Das Sarma, Phys. Rev. B 86, 241112(R) (2012), :arxiv:`1205.5792`, :doi:`10.1103/PhysRevB.86.241112`
.. [CincioVidal2013]
    "Characterizing Topological Order by Studying the Ground States on an Infinite Cylinder"
    L. Cincio, G. Vidal, Phys. Rev. Lett. 110, 067208 (2013), :arxiv:`1208.2623` :doi:`10.1103/PhysRevLett.110.067208`
.. [Schuch2013]
    "Condensed Matter Applications of Entanglement Theory"
    N. Schuch, Quantum Information Processing. Lecture Notes of the 44th IFF Spring School (2013) :arxiv:`1306.5551`
.. [PollmannTurner2012]
    "Detection of symmetry-protected topological phases in one dimension"
    F. Pollmann, A. Turner, Phys. Rev. B 86, 125441 (2012), :arxiv:`1204.0704` :doi:`10.1103/PhysRevB.86.125441`
.. [Grushin2015]
    "Characterization and stability of a fermionic ν=1/3 fractional Chern insulator"
    Adolfo G. Grushin, Johannes Motruk, Michael P. Zaletel, and Frank Pollmann, Phys. Rev. B 91, 035136 (2015), :arxiv:`1407.6985` :doi:`10.1103/PhysRevB.91.035136`
