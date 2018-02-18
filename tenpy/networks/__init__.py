"""Definitions of tensor networks like MPS and MPO.

Here, 'tensor network' refers just to the (parital) contraction of tensors.
For example an MPS represents the contraction of it's `Bs` along the 'virtual' dimensions.
"""
# Copyright 2018 TeNPy Developers

from . import site, mps, mpo, purification_mps

__all__ = ['site', 'mps', 'mpo', 'purification_mps']
