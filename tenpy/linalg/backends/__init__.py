"""TODO write docs"""
# Copyright (C) TeNPy Developers, GNU GPLv3

# hide the folder-structure and expose everyhting as if everything was implemented
# directly in tenpy.linalg.backends
from . import abelian, abstract_backend, backend_factory, no_symmetry, fusion_tree_backend, numpy, torch
from .abelian import *
from .abstract_backend import *
from .array_api import *
from .backend_factory import *
from .no_symmetry import *
from .fusion_tree_backend import *
from .numpy import *
from .torch import *
__all__ = [
    *abelian.__all__, *abstract_backend.__all__, *backend_factory.__all__, *no_symmetry.__all__,
    *fusion_tree_backend.__all__, *numpy.__all__, *torch.__all__
]
