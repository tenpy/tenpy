"""Tools for handling strings."""

# Copyright 2018 TeNPy Developers


def is_non_string_iterable(x):
    """Check if x is a non-string iterable, (e.g., list, tuple, dictionary, np.ndarray)"""
    if isinstance(x, str):
        return False
    try:
        iter(x)
        return True
    except TypeError:
        return False


def vert_join(strlist, valign='t', halign='l', delim=' '):
    r"""Join strings with multilines vertically such that they appear next to each other.

    Parameters
    ----------
    strlist : list of str
        the strings to be joined vertically
    valing : ``'t', 'c', 'b'``
        vertical alignment of the strings: top, center, or bottom
    halign : ``'l', 'c', 'r'``
        horizontal alignment of the strings: left, center, or right
    delim : str
        field separator between the strings

    Returns
    -------
    joined : str
        a string where the strings of strlist are aligned vertically

    Examples
    --------
    >>> print vert_join(['a\nsample\nmultiline\nstring', str(np.arange(9).reshape(3, 3))],
    ...                 delim=' | ')
    a         | [[0 1 2]
    sample    |  [3 4 5]
    multiline |  [6 7 8]]
    string
    """
    # expand tabs, split to newlines
    strlist = [str(s).expandtabs().split('\n') for s in strlist]
    numstrings = len(strlist)
    # number of lines in each string
    numlines = [len(lines) for lines in strlist]
    # maximum number of lines = total number of lines in the resulting string
    totallines = max([0] + numlines)
    # width for each of thestrings
    widths = [max([len(l) for l in lines]) for lines in strlist]
    # translate halign to string format mini language
    halign = {'l': '<', 'c': '^', 'r': '>'}[halign]
    fstr = ['{0: ' + halign + str(w) + 's}' for w in widths]

    # create a 2d table
    res = [[' ' * widths[j] for j in range(numstrings)] for i in range(totallines)]

    for j, lines in enumerate(strlist):
        if valign == 't':
            voffset = 0
        elif valign == 'b':
            voffset = totallines - len(lines)
        elif valign == 'c':
            voffset = (totallines - len(lines)) // 2  # rounds to int
        else:
            raise ValueError('invalid valign ' + str(valign))

        for i, l in enumerate(lines):
            res[i + voffset][j] = fstr[j].format(l)  # format to fixed widths[j]

    # convert the created table to a single string
    res = '\n'.join([delim.join(lines) for lines in res])
    return res


def to_mathematica_lists(a):
    """convert nested `a` to string readable by mathematica using curly brackets '{...}'"""
    if isinstance(a, str):
        return '"' + str(a) + '"'
    try:
        iter(a)
        s = "{" + ", ".join([to_mathematica_lists(suba) for suba in a]) + "}"
        return s
    except TypeError:
        if isinstance(a, float) or isinstance(a, complex):
            return str(a).replace('e', '*^').replace('j', ' I')
        return str(a)
