#!/bin/bash
 
eval "$(conda shell.bash hook)"
conda activate l2o
 
#make -f makefile-pybind.mk clean
make -f makefile-pybind.mk
