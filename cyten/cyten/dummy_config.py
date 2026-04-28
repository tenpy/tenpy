"""Temporary solution for global config options."""
# Copyright (C) TeNPy Developers, Apache license


class printoptions:
    """A collection of global config options. The class is used as a namespace"""

    linewidth: int = 100
    indent: int = 2
    precision: int = 8  # #digits
    maxlines_spaces: int = 15
    maxlines_tensors: int = 30
    skip_data: bool = False  # skip Data section in Tensor prints
    summarize_blocks: bool = False  # True -> always summarize (show only shape, not entries)


class config:
    """A collection of global config options. The class is used as a namespace"""

    printoptions = printoptions
    do_fusion_input_checks = True  # If the Symmetry methods should check their inputs are valid
    default_symmetry_backend = 'abelian'
    default_block_backend = 'numpy'
