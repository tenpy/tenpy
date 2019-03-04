# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAPIDOC  = sphinx-apidoc
SPHINXPROJ    = tenpy
SOURCEDIR     = .
BUILDDIR      = sphinx_build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile src2rst

src2rst:
	# note: there seems to be a bug in sphinx-apidoc such that it ignores following 'excludedirs'
	rm reference/tenpy.*.rst
	$(SPHINXAPIDOC) -d 3 -e -f -o reference ../tenpy/
	rm reference/modules.rst  # this file does not contain useful information
	# thus, I didn't include it to the :toctree:
	# but we have to remove it because it's existence triggers a warning.
	git checkout reference/tenpy.rst # instead we changed the title in this file.

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	export TENPY_OPTIMIZE=0  # disable import of cython modules
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


