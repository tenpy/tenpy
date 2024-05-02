# Copyright (C) TeNPy Developers, GNU GPLv3

__all__ = ['config', 'printoptions']

# TODO this whole module is a dummy

class printoptions:
    linewidth: int = 75
    indent: int = 2
    precision: int = 8  # #digits
    maxlines_spaces: int = 15
    maxlines_tensors: int = 30
    skip_data: bool = False  # skip Data section in Tensor prints
    summarize_blocks: bool = False  # True -> always summarize (show only shape, not entries)


class config:
    strict_labels = True
    printoptions = printoptions
