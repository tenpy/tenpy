"""Block-backends implement matrix and array algebra on dense blocks, similar to e.g. numpy"""
# Copyright (C) TeNPy Developers, Apache license

from ._block_backend import Block, BlockBackend
from .array_api import ArrayApiBlockBackend
from .dtypes import Dtype
from .numpy import NumpyBlockBackend
from .torch import TorchBlockBackend
