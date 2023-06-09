# Response Surface Model + Test Particle Model (RSM+TPM)

## Response Surface Model

The Response Surface Model (RSM) is written in Python and uses this Python packages:

  * Python version 3+ (Tested on Python 3.8+)
  * Numpy
  * Matplotlib
  * Pandas
  * sklearn
  * h5py

You can install these dependencies using the packages on the Linux distribution.
On Ubuntu focal you can install most dependencies executing the command:

```
sudo apt-get install python3 python3-numpy python3-matplotlib python3-pandas python3-sklearn python3-pip
```

An exception  here is h5py. The package provided by Ubuntu depends on OpenMPI.
To install the serial version use pip:

```
sudo python3 -m pip install h5py
```

## Cloning the repository

Clone the repository from Github using the git command:

```
git clone https://github.com/ASSISTLaboratory/WVU_RSM_Suite.git
```

## Compiling TPM

See the file `RSM_TPMC/tpm/README.md` for instructions on how to compile tpm before using RSM  

## Execute RSM

The main script for RSM is `rsm_run_script.py`.
You can execute this script with two options `--mpiexec` and `--tpm`. This options will select which will be the MPI execution command and the location of the tpm executable.
The values for default are equivalent to use this command:

```
./rsm_run_script.py --mpiexec=mpiexec --tpm=tpm/src/tpm
```

# Running RSM+TPMC on an HPC Cluster

The main development of RSM+TPMC is done on a HPC cluster at West Virginia University.
The cluster is called Thorny Flat.

The folder `ARCH` contains scripts to load the correspoding modules and definition files for conteinarize the code using Singularity.

## Loading Environment Modules

The script `load_modules_Thorny_Flat.sh` load a selection of modules used for development and testing of the code.
To activate the modules in the current shell session the script must be source, otherwise the modules will not be loaded in the current shell.

```
$> source ./ARCH/load_modules_Thorny_Flat.sh 

Missing argument for choosing the compiler to be used.
Select one of the following options:

 * intel22 : Intel Compilers and Intel MPI
 * gcc93   : GCC Compilers 9.3 and OpenMPI 4.1.4
 * gcc122  : GCC Compilers 12.2 and OpenMPI 4.1.5
 * nvhpc   : NVIDIA HPC 23.3 and OpenMPI 3.1.5
```

For example, if you want to use RSM+TPMC with the GCC 12.2 compiler, execute:

```
$> source ./ARCH/load_modules_Thorny_Flat.sh gcc122
```

This will load the following modules

```
Loading gcc version 12.2.0 : lang/gcc/12.2.0
Loading hdf5 version 1.14.0_gcc122 : libs/hdf5/1.14.0_gcc122
Loading python version cpython_3.11.3_gcc122 : lang/python/cpython_3.11.3_gcc122
Loading gcc version 12.2.0 : lang/gcc/12.2.0
Loading openmpi version 4.1.5_gcc122 : parallel/openmpi/4.1.5_gcc122
```

This modules will provide all the libraries, compilers and Python installed with the necessary packages for running the code.

## Singularity containers

On the folder ARCH there are 3 examples of Singularity definition files used to create containers for RSM+TPM


 * `RSM+TPM_RockyLinux_9.def`   
 * `RSM+TPM_Ubuntu_focal.def`  
 * `RSM+TPM_Ubuntu_jammy.def`

To create the singularity image install Singularity on your Linux machine and as superuser execute the following command:

```
singularity build RSM+TPM_RockyLinux_9.sif RSM+TPM_RockyLinux_9.def
```

