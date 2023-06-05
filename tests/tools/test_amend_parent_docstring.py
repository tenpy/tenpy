"""A collection of tests to check the functionality of `tenpy.tools.docs`"""
# Copyright 2023 TeNPy Developers, GNU GPLv3
from tenpy.tools.docs import amend_parent_docstring


def foo(x):
    """A function that adds one.

    Parameters
    ==========
    x
        The number to add to
    """
    return x + 1


@amend_parent_docstring(parent=foo)
def bar(x):
    """If `x` is a string, we append ``" + 1"`` instead.

    """
    if isinstance(x, str):
        return x + ' + 1'
    else:
        return x + 1


def bar_explicit(x):
    """A function that adds one.

    If `x` is a string, we append ``" + 1"`` instead.

    Parameters
    ==========
    x
        The number to add to
    """
    return bar(x)


class Parent:
    def do_stuff(self, x):
        """Prints ``x``.

        Parameters
        ==========
        x
            The value to print
        """
        print(x)


class Child:
    def __init__(self):
        self.memory = []

    @amend_parent_docstring(parent=Parent.do_stuff)
    def do_stuff(self, x):
        """Unlike in the superclass ``Parent``, `x` is also stored in :attr:`memory`.

        """
        self.memory.append(x)
        print(x)

    def do_stuff_explicit(self, x):
        """Prints ``x``.

        Unlike in the superclass ``Parent``, `x` is also stored in :attr:`memory`.

        Parameters
        ==========
        x
            The value to print
        """
        self.do_stuff()


# useful for debugging
def show_diff(msg1, msg2):
    import difflib
    for i, s in enumerate(difflib.ndiff(msg1, msg2)):
        if s[0] == ' ':
            continue
        if s[0] == '-':
            print(f'Delete "{s[-1]}" from position {i}')
        elif s[0] == '+':
            print(f'Add "{s[-1]}" to posisiton {i}')
        else:
            raise RuntimeError


def test_amend_parent_docstring():
    print('test function')
    assert bar.__doc__ == bar_explicit.__doc__
    print('test method')
    assert Child.do_stuff.__doc__ == Child.do_stuff_explicit.__doc__
    
