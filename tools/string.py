import numpy as np


def uni_to_str(d):
    """Given a nested structure (though assuming all iterables are either tuples, lists, or dicts, recursively converts unicode objects to python strings. This is useful when importing from json format (as it imports text as unicode).
    """
    if type(d) == unicode:
        return str(d)
    elif type(d) == dict:
        dn = {}
        for k, v in d.items():
            if type(k) == unicode:
                k = str(k)
            dn[k] = uni_to_str(v)
        return dn
    elif type(d) == list:
        return [uni_to_str(v) for v in d]
    elif type(d) == tuple:
        return tuple([uni_to_str(v) for v in d])
    else:
        return d


def is_non_string_iterable(x):
    """Check if x is a non-string iterable.  (E.g., list, tuple, dictionary, np.ndarray) """
    if isinstance(x, str):
        return False
    try:
        iter(x)
        return True
    except TypeError:
        return False


def joinstr(strlist, valign='c', delim=''):
    # TODO, does not handle tabs properly
    # TODO, vary the vertical justification
    """ Join strings with multilines
            no newline at the end of everything

            with tabs, it tries its best to guess where it is (if any of the strings has more than one line)
            """
    # below are all lists, an item for each one in strlist
    numstr = len(strlist)
    slist = []		# list of string in strlist[i]
    numlines = []  # number of lines in strlist[i]
    # strwidth = []		# max width of strlist[i]
    # a string with only spaces and tabs, with strwidth number of characters
    # (use for padding)
    empty_str = []
    for s in strlist:
        if isinstance(s, str):
            list_of_lines = s.split('\n')
        else:
            list_of_lines = str(s).split('\n')		# convert to string
        slist.append(list_of_lines)
        numlines.append(len(list_of_lines))
        list_of_str_lengths = list(len(l) for l in list_of_lines)
        the_longest_line = list_of_lines[np.argmax(list_of_str_lengths)]
        empty_maxlen_liststr = [' '] * len(the_longest_line)
        for i in range(len(the_longest_line)):
            if the_longest_line[i] == '\t':
                empty_maxlen_liststr[i] = '\t'
        empty_str.append("".join(empty_maxlen_liststr))
    maxlines = max(numlines)
    s = ""
    for i in range(maxlines):
        for t in range(numstr):
            if i < int((maxlines - numlines[t]) / 2) or i >= int((maxlines - numlines[t]) / 2) + numlines[t]:
                s += empty_str[t]
            else:
                print_str = slist[t][i - int((maxlines - numlines[t]) / 2)]
                s += print_str + empty_str[t][len(print_str):]
            if t < numstr - 1:
                s += delim
        if i < maxlines - 1:
            s += '\n'
    return s


def to_mathematica_lists(a):
    """ curly brackets """
#	if not isinstance(a, np.ndarray): raise ValueError
    if isinstance(a, str):
        return '"' + str(a) + '"'
    try:
        iter(a)
        s = "{"
        for i, suba in enumerate(a):
            if i > 0:
                s += ", "
            s += to_mathematica_lists(suba)
        s += "}"
        return s
    except TypeError:
        if isinstance(a, float) or isinstance(a, complex):
            return str(a).replace('e', '*^').replace('j', ' I')
        return str(a)
