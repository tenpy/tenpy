"""Definitions of tensor networks like MPS and MPO.

Here, 'tensor network' refers just to the (parital) contraction of tensors.
For example an MPS represents the contraction of it's `Bs` along the 'virtual' dimensions.
"""

from . import mps, mpo

__all__ = ['mps']
