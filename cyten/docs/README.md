# Cyten documentation

C++ functions are documented with [doxygen](https://doxygen.nl),
python parts with [sphinx](https://www.sphinx-doc.org).

## Building the docs

- install cyten, doxygen, sphinx and the required packages as detailed in the `docs/environments.yml`
- run doxygen from the `docs/` folder (or build the C++ code with `BUILD_DOC=ON`).
- run sphinx by calling `make html` or `sphinx-build -M html . build_docs` from the `docs/` folder.

You can then find the sphinx docs (including the doxygen C++ API reference) in `docs/html/`.
