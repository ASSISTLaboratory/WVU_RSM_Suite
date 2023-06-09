#!/bin/bash

if [ "$1" == "intel22" ]
then
    module purge
    module load lang/gcc/9.3.0 lang/python/cpython_3.10.5_gcc93 compiler/2023.1.0 mpi/2021.9.0 libs/hdf5/1.12.2_intel22_impi22 
elif [ "$1" == "gcc93" ]
then
    module purge
    module load lang/gcc/9.3.0 libs/hdf5/1.12.2_gcc93 lang/python/cpython_3.10.5_gcc93 parallel/openmpi/4.1.4_gcc93 
elif [ "$1" == "gcc122" ]
then
    module purge
    module load lang/gcc/12.2.0 libs/hdf5/1.14.0_gcc122 lang/python/cpython_3.11.3_gcc122 parallel/openmpi/4.1.5_gcc122 
elif [ "$1" == "nvhpc" ]
then
    module purge
    module load lang/gcc/9.3.0 lang/python/cpython_3.10.5_gcc93 libs/hdf5/1.12.2_gcc93 lang/nvidia/nvhpc/23.3 
else
    echo ""
    echo "Missing argument for choosing the compiler to be used."
    echo "Select one of the following options:"
    echo ""
    echo " * intel22 : Intel Compilers and Intel MPI"
    echo " * gcc93   : GCC Compilers 9.3 and OpenMPI 4.1.4"
    echo " * gcc122  : GCC Compilers 12.2 and OpenMPI 4.1.5"
    echo " * nvhpc   : NVIDIA HPC 23.3 and OpenMPI 3.1.5"
    echo ""
fi

