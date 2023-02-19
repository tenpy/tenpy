# Copyright 2018-2023 TeNPy Developers, GNU GPLv3
import warnings
from .old.charges import *

__all__ = ['ChargeInfo', 'LegCharge', 'LegPipe', 'QTYPE']

warnings.warn("Import of tenpy.linalg.charges: new tenpy.linalg.tensors interface!")

