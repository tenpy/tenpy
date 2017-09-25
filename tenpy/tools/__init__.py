r"""A collection of tools: mostly short yet quite useful functions.

Some functions are explicitly imported in other parts of the library,
others might just be useful when using the libary.
Common to all tools is that they are not just useful for a single algorithm but fairly general.
"""

from . import fit, math, misc, params, process, string

__all__ = ['fit', 'math', 'misc', 'params', 'process', 'string']
