#!/bin/bash

# clean up
find tenpy/linalg -name "*.c" -delete
find tenpy/linalg -name "*.so" -delete

# re-compile
python3 setup.py build_ext --inplace
