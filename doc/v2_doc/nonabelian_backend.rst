Implementation details of the nonabelian backend
================================================

It turns out that the machinery we choose to represent tensors that are symmetric under
general non-abelian symmetries, can be extended to represent fermionic or anyonic grading.



Definition of topological data of the symmetry
----------------------------------------------

- Direct sum and sectors

- Fusion
  - meaning of \otimes (tensor product / monoidal product)
  - N symbol
  - fusion maps X and splitting maps Y
  - orthogonality and completeness relations
  - F symbol (associativity of fusion)
  - constraints from pentagon and triangle coherence equations

- Duality
  - Distinction between dual of a simple object and its equivalence class
  - Z isomorphism
  - Frobenius Schur indicator
  - trace, quantum dimension
  - bending lines : A symbol and B symbol

- Braiding
  - R symbol
  - hexagon coherence equation


- Example: U(1)

- Example: SU(2)

- Example: fibonacci anyons


Fusion Trees
------------

- definition

- can decompose as pairwise fusion, i.e. successive application of X
  - need to choose bracketing order.
  - choose to represent in canonical order

- example tree

- data :
  - uncoupled sectors : [a1, ..., aN]
  - presence of Z isomorphisms : [yes, no, ...] carefully define meaning of uncoupled!
  - coupled sector
  - (N - 2) sectors on the internal edges
  - (N - 1) multiplicity indices on the vertices

- manipulations
  - generally : "manipulation" gives a map from a set of uncoupled sectors to a coupled sector,
  which can be decomposed in terms of canonical fusion trees (since they are a basis)
  - combining trees (insert)
  - braiding (general braid decomposes into elementary artin braids)

- bending lines for a pair of splitting and fusion trees

- Example: U(1)

- Example: SU(2)

- Example: fibonacci anyons


Data of a symmetric tensor
--------------------------

- What is a tensor in the above language

- Leg order and bipartition

- Formulation of Schurs Lemma using fusion trees

- Organizing the free parameters into blocks

- Explicit conversion (tensor -> block) and (blocks -> tensor)

- Special cases and examples:
  - 2-leg tensor, i.e. matrix -> fusion trees are trivial
  - abelian symmetry group -> fusion trees only give charge rule
  - SU(2) symmetric 2-body operator on spin-1/2 sites -> (identity and Heisenberg coupling)
  - 2-body operator for Fibonacci anyons?


Operations on tensors
---------------------

- conversion to/from dense

- contraction (tdot)
  - might need to braid / bend first

- decomposition (svd)
  - might need to braid / bend first

