#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2019-2020 TeNPy Developers, GNU GPLv3
#
import sys
import os
import inspect
import sphinx_rtd_theme

# ensure parent folder is in sys.path to allow import of tenpy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'sphinx_ext')))

if not sys.version_info >= (3, 5):
    print("ERROR: old python version, called by python version\n" + sys.version)
    sys.exit(1)

# don't use compiled version to avoid problems with doc-strings of compiled functions
os.environ['TENPY_OPTIMIZE'] = "0"
try:
    import tenpy.version
except:
    print("ERROR: can't import tenpy.")
    sys.exit(1)

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = '3.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.linkcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram',
    'sphinx_rtd_theme',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['sphinx_templates']

source_suffix = '.rst'  # could add markdown, but that makes it only more complicated
master_doc = 'index'  # The master toctree document.
language = None  # no translations
pygments_style = 'sphinx'  # syntax highlighting style

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False

# General information about the project.
project = 'TeNPy'
copyright = '2016-2020, TeNPy Developers'
author = 'TeNPy Developers'
version = tenpy.__version__  # The short X.Y version.
release = tenpy.__full_version__  # The full version, including alpha/beta/rc tags.

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['sphinx_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output ----------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_logo = "images/logo.png"
html_favicon = "images/logo.ico"
html_static_path = ['_static']
#  html_extra_path = []
html_last_updated_fmt = '%b %d, %Y'

# used by tenpy.readthedocs.io
html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "tenpy",  # Username
    "github_repo": "tenpy",  # Repo name
    "github_version": "master",  # Version
    "conf_py_path": "/doc/",  # Path in the checkout to the docs root
    "css_files": ["_static/custom.css"],  # to highlight targets
}

html_theme_options = {
    'collapse_navigation': False,
    'style_external_links': True,
}

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}
html_sidebars = {
    '**': ['localtoc.html', 'relations.html', 'searchbox.html', 'globaltoc.html'],
}

# == Options for extensions ===============================================

# -- sphinx.ext.autodoc ---------------------------------------------------

autodoc_default_options = {}
autodoc_member_order = 'bysource'
# some options are included in the templates under
# sphinx_templates/autosummary/class.rst
# for example :inherited-members: and :show-inheritance:
autosummary_generate = True

# -- sphinx.ext.todo ------------------------------------------------------

todo_include_todos = True  # show todo-boxes in output

# -- sphinx.ext.napoleon --------------------------------------------------
# numpy-like doc strings

napoleon_use_admonition_for_examples = True
napoleon_use_ivar = False  # otherwise :attr:`...` doesn't work anymore

# -- sphinx.ext.inheritance_diagram ---------------------------------------

inheritance_graph_attrs = {
    'rankdir': "TB",  # top-to-bottom
    'fontsize': 14,
    'ratio': 'compress',
}

# -- sphinx.ext.intersphinx -----------------------------------------------
# cross links to other sphinx documentations
# this makes  e.g. :class:`numpy.ndarray` work
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('https://matplotlib.org', None),
    'h5py': ('https://docs.h5py.org/en/stable/', None),
}

# -- sphinx.ext.extlinks --------------------------------------------------
# allows to use, e.g., :arxiv:`1805.00055`
extlinks = {
    'arxiv': ('https://arxiv.org/abs/%s', 'arXiv:'),
    'doi': ('https://dx.doi.org/%s', 'doi:'),
    'issue': ('https://github.com/tenpy/tenpy/issues/%s', 'issue #'),
    'forum': ('https://tenpy.johannes-hauschild.de/viewtopic.php?t=%s', 'Community forum topic ')
}


# -- sphinx.ext.linkcode --------------------------------------------------
# linkcode to put links to the github repository from the documentation
def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object."""
    # based on the corresponding linkcode_resolve in the `conf.py` of the numpy repository.
    if domain != 'py':
        return None
    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None
    obj = inspect.unwrap(obj)  # strip decorators

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        return None

    # find out the lines
    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""
    fn = os.path.relpath(fn, start=os.path.dirname(tenpy.__file__))
    if fn.startswith('..'):
        return None

    if tenpy.version.released:
        return "https://github.com/tenpy/tenpy/blob/v%s/tenpy/%s%s" % (tenpy.__version__, fn,
                                                                       linespec)
    else:
        return "https://github.com/tenpy/tenpy/blob/master/tenpy/%s%s" % (fn, linespec)
