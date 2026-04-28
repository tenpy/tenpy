"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""

import inspect
import os
import sys

import cyten

project = 'Cyten'
copyright = '2024, Cyten developer team'
author = 'Cyten developer team'
release = '0.1'

GITHUBBASE = 'https://github.com/tenpy/cyten'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'breathe',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.linkcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
]

templates_path = ['sphinx/templates']
exclude_patterns = ['build_docs', 'Thumbs.db', '.DS_Store']

source_suffix = {'.rst': 'restructuredtext'}  # can add markdown if needed
master_doc = 'index'  # The master toctree document.
language = 'en'  # no translations
pygments_style = 'sphinx'  # syntax highlighting style


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_logo = 'images/cyten_logo.png'
#  html_favicon = "images/logo.ico"
html_static_path = ['sphinx/static']
html_last_updated_fmt = '%b %d, %Y'

html_css_files = [
    'custom.css',  # to highlight targets
]

html_context = {
    'display_github': True,  # Integrate GitHub
    'github_user': 'tenpy',  # Username
    'github_repo': 'cyten',  # Repo name
    'github_version': 'main',  # Version
    'conf_py_path': '/docs/',  # Path in the checkout to the docs root
}

# -- breathe (including doxygen docs) -------------------------------------

breathe_projects = {'cyten': 'build_docs/doxy_xml'}

breathe_default_project = 'cyten'

# -- sphinx.ext.autodoc ---------------------------------------------------

autodoc_default_options = {}
autodoc_member_order = 'bysource'
# some options are included in the templates under
# sphinx_templates/autosummary/class.rst
# for example :inherited-members: and :show-inheritance:
autosummary_generate = True

# -- sphinx.ext.todo ------------------------------------------------------

todo_include_todos = True  # show todo-boxes in output

# -- sphinx.ext.doctest ---------------------------------------------------

doctest_global_setup = """
import numpy as np
import scipy
import scipy.linalg
import cyten
np.set_printoptions(suppress=True)
"""

trim_doctest_flag = True

# -- sphinx.ext.napoleon --------------------------------------------------
# numpy-like doc strings

napoleon_use_admonition_for_examples = True
napoleon_use_ivar = False  # otherwise :attr:`...` doesn't work anymore
napoleon_custom_sections = ['Options']

# -- sphinx.ext.inheritance_diagram ---------------------------------------

inheritance_graph_attrs = {
    'rankdir': 'TB',  # top-to-bottom
    'fontsize': 14,
    'ratio': 'compress',
}

# -- sphinx.ext.intersphinx -----------------------------------------------
# cross links to other sphinx documentations
# this makes  e.g. :class:`numpy.ndarray` work
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'tenpy': ('https://tenpy.readthedocs.org/en/stable', None),
    'matplotlib': ('https://matplotlib.org', None),
    'h5py': ('https://docs.h5py.org/en/stable/', None),
}

# -- sphinx.ext.extlinks --------------------------------------------------
# allows to use, e.g., :arxiv:`1805.00055`
extlinks = {
    'arxiv': ('https://arxiv.org/abs/%s', 'arXiv:%s'),
    'doi': ('https://dx.doi.org/%s', 'doi:%s'),
    'issue': (GITHUBBASE + '/issues/%s', 'issue #%s'),
    'pull': (GITHUBBASE + '/pulls/%s', 'PR #%s'),
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
        linespec = f'#L{lineno}-L{lineno + len(source) - 1}'
    else:
        linespec = ''
    fn = os.path.relpath(fn, start=os.path.dirname(cyten.__file__))
    if fn.startswith('..'):
        return None

    if cyten.__version__ == cyten.__full_version__:
        return f'{GITHUBBASE}/blob/v{cyten.__version__}/cyten/{fn}{linespec}'
    else:
        return f'{GITHUBBASE}/blob/main/cyten/{fn}{linespec}'
