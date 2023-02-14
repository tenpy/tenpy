from __future__ import annotations

from . import NoSymmetryNumpyBackend, AbstractBackend
from ..symmetries import Symmetry, no_symmetry, AbelianGroup

_backend_lookup = dict(
    no_symmetry=dict(
        numpy=(NoSymmetryNumpyBackend, {}),
        torch=None,  # TODO
        tensorflow=None,  # TODO
        jax=None,  # TODO
        cpu=(NoSymmetryNumpyBackend, {}),
        gpu=None,  # TODO
        tpu=None,  # TODO
    ),
    #
    abelian=dict(
        numpy=None,  # TODO: pure-python npc
        torch=None,  # TODO: quick-and-dirty npc + torch
        tensorflow=None,  # FUTURE
        jax=None,  # FUTURE
        cpu=None,  # TODO: npc
        gpu=None,  # FUTURE
        tpu=None,  # FUTURE
    ),
    #
    non_abelian=dict(
        numpy=None,  # FUTURE
        torch=None,  # FUTURE
        tensorflow=None,  # FUTURE
        jax=None,  # FUTURE
        cpu=(NoSymmetryNumpyBackend, {}),
        gpu=None,  # FUTURE
        tpu=None,  # FUTURE
    ),
)


def get_backend(symmetry: AbstractSymmetry = no_symmetry, block_backend: str = 'numpy',
                symmetry_backend: str = None) -> AbstractBackend:
    """
    Parameters
    ----------
    symmetry : AbstractSymmetry
    block_backend : {'numpy', 'torch', 'tensorflow', 'jax', 'cpu', 'gpu', 'tpu'}
    symmetry_backend : {None, 'no_symmetry', 'abelian', 'nonabelian'}
        None means select based on the symmetry.
        It is possible though, to request the non-abelian backend even though the symmetry is actually abelian
    """
    # TODO cache these instances, make sure there is only ever one.
    #  -> need hash for AbstractSymmetry instances
    assert block_backend in ['numpy', 'torch', 'tensorflow', 'jax', 'cpu', 'gpu', 'tpu']
    if symmetry_backend is None:
        if symmetry == no_symmetry:
            symmetry_backend = 'no_symmetry'
        elif isinstance(symmetry, AbelianGroup):
            symmetry_backend = 'abelian'
        else:
            symmetry_backend = 'nonabelian'

    if symmetry_backend == 'no_symmetry':
        assert symmetry == no_symmetry
        if block_backend in ['numpy', 'cpu']:
            return NoSymmetryNumpyBackend()
        if block_backend == 'torch':
            raise NotImplementedError  # TODO
        if block_backend == 'tensorflow':
            raise NotImplementedError  # TODO
        if block_backend == 'jax':
            raise NotImplementedError  # TODO
        if block_backend == 'gpu':
            raise NotImplementedError  # TODO, torch with device arg ..?
        if block_backend == 'tpu':
            raise NotImplementedError  # TODO, torch with device arg ..?

    if symmetry_backend == 'abelian':
        assert isinstance(symmetry, AbelianGroup)
        if block_backend in ['numpy', 'torch', 'tensorflow', 'jax']:
            # TODO for these cases, could do a pure-python version of AbstractAbelianBackend
            #  then we should issue a warning regarding performance
            raise NotImplementedError
        if block_backend == 'cpu':
            # TODO: quick-and-dirty: np_conserved, long-term: redo it as a C extension
            raise NotImplementedError
        if block_backend in ['gpu', 'tpu']:
            # FUTURE: extension in C/C++ that calls CUDA or whatever
            raise NotImplementedError

    if symmetry_backend == 'nonabelian':
        assert isinstance(symmetry, AbelianGroup)
        if block_backend in ['numpy', 'torch', 'tensorflow', 'jax']:
            # TODO for these cases, could do a pure-python version of AbstractNonabelianBackend
            #  then we should issue a warning regarding performance
            raise NotImplementedError
        if block_backend == 'cpu':
            # FUTURE: extension in C/C++ that calls BLAS or whatever
            raise NotImplementedError
        if block_backend in ['gpu', 'tpu']:
            # FUTURE: extension in C/C++ that calls CUDA or whatever
            raise NotImplementedError

    raise RuntimeError  # if none of the above if clauses applied, there is an error in the code.
