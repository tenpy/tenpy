#!/bin/bash
cd tenpy/algorithms/linalg
rm npc_helper.so
#rm tokyo.so
export ARCHFLAGS="-arch x86_64"
python ./npc_setup.py build_ext --inplace
