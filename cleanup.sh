#!/bin/bash
# clean up all the compiled code to allow a clean re-compilation.

test -f tenpy/linalg/*.c && rm tenpy/linalg/*.c 
test -f tenpy/linalg/*.cpp && rm tenpy/linalg/*.cpp 
test -f tenpy/linalg/*.so && rm tenpy/linalg/*.so 
test -d build && rm -r build

# clean-up generated documentation
test -d doc/sphinx_build && rm -r doc/sphinx_build
