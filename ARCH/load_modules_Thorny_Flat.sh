#!/bin/bash

module purge

if [ "$1" == "intel22" ]
then
    module load lang/gcc/9.3.0 lang/python/cpython_3.10.5_gcc93 compiler/2023.1.0 mpi/2021.9.0 libs/hdf5/1.12.2_intel22_impi22 
elif [ "$1" == "gcc93" ]
then
    module load lang/gcc/9.3.0 libs/hdf5/1.12.2_gcc93 lang/python/cpython_3.10.5_gcc93 parallel/openmpi/4.1.4_gcc93 
elif [ "$1" == "gcc122" ]
then
    module load lang/gcc/12.2.0 libs/hdf5/1.14.0_gcc122 lang/python/cpython_3.11.3_gcc122 parallel/openmpi/4.1.5_gcc122 
elif [ "$1" == "nvhpc" ]
then
    module load lang/gcc/9.3.0 lang/python/cpython_3.10.5_gcc93 libs/hdf5/1.12.2_gcc93 lang/nvidia/nvhpc
fi

