TEST PARTICLE MODEL
===================

Calculates the drag coefficient of a given object (from a .stl mesh file) using the test particle method

INPUT
-----

 * GS_MODEL: Specifies gas-surface interaction model
 	0 = Maxwell's Model
 	1 = Diffuse Reflection with Incomplete Accommodation (DRIA)
 	2 = Cercignani-Lampis-Lord (CLL)

 * Ux, Uy, Uz = satellite speed relative to atmosphere [m/s]

 * Ts = satellite surface temperature [K]

 * Ta = atmospheric translational temperature [K]

 * sigmat = tangential momentum accommodation coefficient [unitless] (ONLY USED IN CLL)

 * alphan = normal energy accommodation coefficient [unitless] (ONLY USED IN CLL)

 * alpha = energy accommodation coefficient (ONLY USED IN DRIA)

 * epsilon = fraction of particles specularly reflected (ONLY USED IN MAXWELL'S MODEL)

 * X = species mole fractions list in order [O, O2, N, N2, He, H]

 * f = directory path for the mesh file to be used e.g "./Mesh Files/sphere_res05_ascii.stl"

OUTPUT
------

 *  Cd = Drag Coefficient [unitless]

PREREQUISITES
-------------
 
The Test Particle Model (TPM) is written in C and uses MPI and some extra libraries. 
 
To compile TPM several packages are needed: 
 
  * C compiler 
  * GSL libraries 
  * HDF5 libraries (Serial version is enough) 
  * MPI implementation and libraries 
  * Autoconf needed if installing from the Github repository 
  * Git to clone the repository 
 
To install these dependencies you can use the packages provided by your Linux distribution 

On Ubuntu the dependencies above can be satisfied executing the command: 
 
``` 
$> sudo apt-get install build-essential git autoconf libgsl-dev libhdf5-dev libmpich-dev 
``` 

On RedHat systems (RHEL) including derivatives such as RockyLinux, CentOS, Fedora and Almalinux the dependencies can be satisfied with:

```
$> sudo dnf install autoconf automake gcc mpich-devel openmpi-devel hdf5-devel gsl-devel 
```

Checking for GSL
~~~~~~~~~~~~~~~~

You can check for the presence of GSL executing this command:

```
$> pkg-config gsl --libs
```

You should get the location of the libraries and linking flags. For example:

```
-L/shared/software/lang/gcc/9.3.0/lib -lgsl -lgslcblas -lm
```

Checking for HDF5
~~~~~~~~~~~~~~~~~

Checking for the availability of HDF5 can be done indirectly by looking at the compiler wrappers

```
$> h5cc --version
gcc (GCC) 9.3.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

Checking for MPI
~~~~~~~~~~~~~~~~

There are several implementations of MPI. The most popular are OpenMPI, MPICH, MVAPICH and Intel MPI (A derivative from MPICH).
Several implementations can be installed on a system. 
Checking the presence of MPI can be done by looking at the MPI C wrapper:

```
$ mpicc --version

nvc 23.3-0 64-bit target on x86-64 Linux -tp skylake-avx512 
NVIDIA Compilers and Tools
Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
```

During runtime, the execution is usually managed by the commands `mpirun` or `mpiexec`
It is very important to use during runtime, the same implementation that was used for compilation.


COMPILATION
-----------

Go to the folder `WVU_RSM_Suite/RSM_TPMC/tpm`.
This is the folder where this README.md is located:

```
cd WVU_RSM_Suite/RSM_TPMC/tpm
```

If your clone the repository from Github, the script configure is still not created.
Create the configure using the script `autogen.sh`

```
$> ./autogen.sh
```

This will create a script called `configure`.
The script will try to identify the C compiler, HDF5 and GSL libraries and will give you the ability to activate MPI and enable flags for debugging and profiling.

In the simplest form, you can just execute:

```
$> ./configure
```

This will search for the default C compiler (Usually `gcc`) and for the presence of the GSL and HDF5 libraries.
One important option is to enable MPI so the code can work on multiple CPU cores or even distributed across multiple compute nodes.

```
./configure --enable-mpi
```

Other options availble include adding debugging and/or profiling flags. Enabling this features is useful for developers but will create a slower executable

```
./configure --enable-mpi --enable-debug --enable-gprof
```

You can also create a separate folder for compilation.
This is a good practice specially for developers who want to compile the code under different settings.
As example, consider creating a folder called `build` where the code will be compiled.

```
$> mkdir build
$> cd build
```
Now the configure script is located one folder up and must be called with `../configure`

```
$> ../configure
```

In what follows we will assume the compilation is done on the base folder for tpm, ie, the same folder where `configure` is located.

If the default C compiler or MPI C wrapper is not what you intend to use you can declare it by using the variables CC and MPICC.
For example, if you want to compile serialy (No MPI) using the Intel Compilers use:

```
$> CC=icc ./configure
```

To use the NVidia compilers would be

```
$> CC=nvc ./configure
```

Enabling MPI and using the Intel MPI on top of Intel Compilers

```
$> CC=icc MPICC=mpiicc ./configure --enable-mpi
```

The configure will figure out the right values for HDF5 and GSL libraries, otherwise you can also explicictly insert those like this:

```
$> CC=mpiicc LIBS="-lhdf5 -lm -lgsl -lgslcblas" ./configure
```

Execute make to build the code:

```
$> make
```

After compilation the executable will be located  with sources at `src` with the name `tpm`

You can use the binary from that location or you can install it. 
For installing the code you can decide a good place for installation with:

```
$> ./configure --prefix=$HOME/.local
```

and after make, install the code with:

```
$> make install
```

