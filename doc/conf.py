#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2019-2021 TeNPy Developers, GNU GPLv3
#
import sys
import os
import inspect
import sphinx_rtd_theme
import io

# ensure parent folder is in sys.path to allow import of tenpy
REPO_PREFIX = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_PREFIX)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'sphinx_ext')))
GITHUBBASE = "https://github.com/tenpy/tenpy"

if not sys.version_info >= (3, 5):
    print("ERROR: old python version, called by python version\n" + sys.version)
    sys.exit(1)

# don't use compiled version to avoid problems with doc-strings of compiled functions
os.environ["TENPY_NO_CYTHON"] = "true"
try:
    import tenpy
except:
    print("ERROR: can't import tenpy.")
    sys.exit(1)

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = '3.2'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
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
    'nbsphinx',
    'sphinx_copybutton',
    'sphinxcontrib.bibtex',
    'sphinx_cfg_options',
    'matplotlib.sphinxext.plot_directive',
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

language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = [
    'sphinx_build', 'Thumbs.db', '.DS_Store', 'notebooks/README.rst', 'notebooks/_template.ipynb'
]

# -- example stubs  -=-----------------------------------------------------


def create_example_stubs():
    """create stub files for examples and toycodes to include them in the documentation."""
    folders = [
        (['examples'], '.py', []),
        (['examples'], '.yml', []),
        (['examples', 'advanced'], '.py', []),
        (['examples', 'chern_insulators'], '.py', []),
        (['toycodes'], '.py', []),
        (['examples', 'yaml'], '.yml', []),
    ]
    for subfolders, extension, excludes in folders:
        outdir = os.path.join(os.path.dirname(__file__), *subfolders)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        files = os.listdir(os.path.join(REPO_PREFIX, *subfolders))
        files = sorted([fn for fn in files if fn.endswith(extension) and fn not in excludes])
        for fn in files:
            outfile = os.path.join(outdir, os.path.splitext(fn)[0] + '.rst')
            if os.path.exists(outfile):
                continue
            dirs = '/'.join(subfolders)
            sentence = ("`on github <{base}/blob/main/{dirs!s}/{fn!s}>`_ "
                        "(`download <{base}/raw/main/{dirs!s}/{fn!s}>`_).")
            sentence = sentence.format(dirs=dirs, fn=fn, base=GITHUBBASE)
            include = '.. literalinclude:: /../{dirs!s}/{fn!s}'.format(dirs=dirs, fn=fn)
            text = '\n'.join([fn, '=' * len(fn), '', sentence, '', include, ''])
            with open(outfile, 'w') as f:
                f.write(text)
    # done


create_example_stubs()

# -- include output of command line help ----------------------------------


def include_command_line_help():
    parser = tenpy._setup_arg_parser(width=98)
    parser.prog = 'tenpy-run'
    help_text = parser.format_help()
    # help_text = '\n'.join(['    ' + l for l in help_text.splitlines()])
    fn = 'commandline-help.txt'
    with open(fn, 'w') as f:
        f.write(help_text)
    tenpy.console_main.__doc__ = tenpy.console_main.__doc__ + '\n' '.. literalinclude:: /' + fn


include_command_line_help()

# -- Options for HTML output ----------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_logo = "images/logo.png"
html_favicon = "images/logo.ico"
html_static_path = ['_static']
html_last_updated_fmt = '%b %d, %Y'

html_css_files = [
    "custom.css",  # to highlight targets
]

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "tenpy",  # Username
    "github_repo": "tenpy",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/doc/",  # Path in the checkout to the docs root
}

html_theme_options = {
    'collapse_navigation': False,
    'style_external_links': True,
}


# == Options for extensions ===============================================

# -- nbsphinx -------------------------------------------------------------

nbsphinx_execute = 'never'

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=False)[10:] %}

.. raw :: html

    <div class="admonition note">
      This page was generated from
      <a class="reference external" href="https://github.com/tenpy/tenpy_notebooks/blob/main/{{ docname|e }}">{{ docname|e }}</a>
      (<a class="reference external" href="https://github.com/tenpy/tenpy_notebooks/raw/main/{{ docname|e }}" download="{{docname | e}}">download</a>).
    </div>
"""

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
import tenpy.linalg.np_conserved as npc
import tenpy
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
    'rankdir': "TB",  # top-to-bottom
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
    'matplotlib': ('https://matplotlib.org', None),
    'h5py': ('https://docs.h5py.org/en/stable/', None),
}

# -- sphinx.ext.extlinks --------------------------------------------------
# allows to use, e.g., :arxiv:`1805.00055`
extlinks = {
    'arxiv': ('https://arxiv.org/abs/%s', 'arXiv:%s'),
    'doi': ('https://dx.doi.org/%s', 'doi:%s'),
    'issue': (GITHUBBASE + '/issues/%s', 'issue #%s'),
    'forum': ('https://tenpy.johannes-hauschild.de/viewtopic.php?t=%s',
              'community forum (topic %s)')
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
        return "%s/blob/v%s/tenpy/%s%s" % (GITHUBBASE, tenpy.__version__, fn, linespec)
    else:
        return "%s/blob/main/tenpy/%s%s" % (GITHUBBASE, fn, linespec)


# -- sphinx_cfg_options ---------------------------------------------------

cfg_options_default_in_summary_table = False
cfg_options_parse_comma_sep_names = True
cfg_options_always_include = ["Config"]

# -- sphinxcontrib.bibtex -------------------------------------------------

bibtex_bibfiles = ['literature.bib', 'papers_using_tenpy.bib']

from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.labels import BaseLabelStyle
from pybtex.style.sorting.author_year_title import SortingStyle
from pybtex.plugin import register_plugin


class CustomBibtexStyle1(UnsrtStyle):
    default_sorting_style = 'key'
    default_label_style = 'key'


class CustomBibtexStyle2(UnsrtStyle):
    default_sorting_style = 'year_author_title'
    default_label_style = 'key'


class KeyLabelStyle(BaseLabelStyle):
    def format_labels(self, sorted_entries):
        return [entry.key for entry in sorted_entries]


class YearAuthorTitleSort(SortingStyle):
    def sorting_key(self, entry):
        author_key, year, title = super().sorting_key(entry)
        return (year, author_key, title)


class KeySort(SortingStyle):
    def sorting_key(self, entry):
        return entry.key


register_plugin('pybtex.style.formatting', 'custom1', CustomBibtexStyle1)
register_plugin('pybtex.style.formatting', 'custom2', CustomBibtexStyle2)
register_plugin('pybtex.style.labels', 'key', KeyLabelStyle)
register_plugin('pybtex.style.sorting', 'key', KeySort)
register_plugin('pybtex.style.sorting', 'year_author_title', YearAuthorTitleSort)
