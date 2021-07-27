#!/bin/bash

module purge

if  [ "$1" == 'intel18' ]
then
    module load lang/gcc/9.3.0  lang/python/intelpython3_2019.5 lang/intel/2018 libs/hdf5/1.10.5_intel18
elif [ "$1" == "intel19" ]
then
    module load lang/gcc/9.3.0  lang/python/intelpython3_2019.5 lang/intel/2019 libs/hdf5/1.12.0_intel19
elif [ "$1" == "gcc82" ]
then
    module load lang/gcc/8.2.0 libs/hdf5/1.12.0_gcc82 lang/python/cpython_3.9.5_gcc82 parallel/openmpi/4.0.5_gcc82
elif [ "$1" == "gcc93" ]
then
    module load lang/gcc/9.3.0 libs/hdf5/1.12.0_gcc93 lang/python/cpython_3.9.5_gcc93 parallel/openmpi/4.0.5_gcc93
elif [ "$1" == "gcc111" ]
then
    module load lang/gcc/11.1.0 libs/hdf5/1.12.0_gcc111 lang/python/cpython_3.9.5_gcc111 parallel/openmpi/4.0.5_gcc111
fi

