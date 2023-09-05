#!/bin/bash
extra_args=(
    # 1 leg
    # no symmetry
    "-l 1 -q no_symmetry"
    # U(1) x U(1) x Z2
    "-l 1 -q u1_symmetry u1_symmetry z2_symmetry -s 2"
    "-l 1 -q u1_symmetry u1_symmetry z2_symmetry -s 5"
    "-l 1 -q u1_symmetry u1_symmetry z2_symmetry -s 20"
    # U(1)
    "-l 1 -q u1_symmetry -s 2"
    "-l 1 -q u1_symmetry -s 5"
    "-l 1 -q u1_symmetry -s 20"
    # Z2
    "-l 1 -q z2_symmetry -s 2"
    "-l 1 -q z2_symmetry -s 5"
    "-l 1 -q z2_symmetry -s 20"
    # 2 legs
    # no symmetry
    "-l 2 -q no_symmetry"
    # U(1) x U(1) x Z2
    "-l 2 -q u1_symmetry u1_symmetry z2_symmetry -s 2"
    "-l 2 -q u1_symmetry u1_symmetry z2_symmetry -s 5"
    "-l 2 -q u1_symmetry u1_symmetry z2_symmetry -s 20"
    # U(1)
    "-l 2 -q u1_symmetry -s 2"
    "-l 2 -q u1_symmetry -s 5"
    "-l 2 -q u1_symmetry -s 20"
    # Z2
    "-l 2 -q z2_symmetry -s 2"
    "-l 2 -q z2_symmetry -s 5"
    "-l 2 -q z2_symmetry -s 20"
)
for extra in "${extra_args[@]}"
do
	echo $extra
done
