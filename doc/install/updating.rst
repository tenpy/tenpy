Updating to a new version
=========================

**Before** you update, take a look at the :doc:`/releases`, which lists the changes, fixes, and new stuff. 
Most importantly, it has a section on *backwards incompatible changes* (i.e., changes which may break your
existing code) along with information how to fix it. Of course, we try to avoid introducing such incompatible changes,
but sometimes, there's no way around them. If you skip some intermediate version(s) for the update, read also the release
notes of those!

How to update depends a little bit on the way you installed TeNPy. Of course, you have always the option to just remove
the tenpy files and download the newest version, following the instructions above.

Alternatively, if you used ``git clone ...`` to download the repository, you can update to the newest version using `Git`.
First, briefly check that you didn't change anything you need to keep with ``git status``.
Then, do a ``git pull`` to download (and possibly merge) the newest commit from the repository.

.. note ::
    
    If some Cython file (ending in ``.pyx``) got renamed/removed (e.g., when updating from v0.3.0 to v0.4.0), 
    you first need to remove the corresponding binary files. 
    You can do so with the command ``bash cleanup.sh``.
    
    Furthermore, whenever one of the cython files (ending in ``.pyx``) changed, you need to re-compile it.
    To do that, simply call the command ``bash ./compile`` again.
    If you are unsure whether a cython file changed, compiling again doesn't hurt.

To summarize, you need to execute the following bash commands in the repository::

    # 0) make a backup of the whole folder
    git status   # check the output whether you modified some files
    git pull
    bash ./cleanup.sh  # (confirm with 'y')
    bash ./compile.sh

