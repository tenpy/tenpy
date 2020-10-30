#!/bin/bash
# Copyright 2018 TeNPy Developers
set -e
DIR="$(dirname ${BASH_SOURCE[0]})"

if [ -n "$1" ]
then
    MOD="$1"
else
    MOD="tensordot"
fi

common_args="-t 0.1"
extra_args=(
    "-l 1 -q "
    "-l 1 -q 1 -s 5"
    "-l 1 -q 1 -s 20"
    "-l 2 -q "
    "-l 2 -q 1 -s 5"
    "-l 2 -q 1 -s 20"
    "-l 3 -q "
    "-l 3 -q 1 -s 5"
    "-l 3 -q 1 -s 20"
    # "-l 1 -q 2 2 -s 4"
    # "-l 1 -q 1 1 -s 20"
    # "-l 2 -q 2 2 -s 4"
    # "-l 2 -q 1 1 -s 20"
    # "-l 3 -q 2 2 -s 4"
    # "-l 3 -q 1 1 -s 20"
)

for extra in "${extra_args[@]}"
do
    echo "========================================"
    echo "python $DIR/benchmark.py -m ${MOD}_npc $common_args $extra"
    python $DIR/benchmark.py -m ${MOD}_npc $common_args $extra
    echo "========================================"
    echo "python $DIR/benchmark.py -m ${MOD}_numpy $common_args $extra"
    python $DIR/benchmark.py -m ${MOD}_numpy $common_args $extra
done
# plot, if we have an X-server (otherwise matplotlib fails.)
test -n "$DISPLAY" && python $DIR/benchmark.py -p ${MOD}_*.txt
