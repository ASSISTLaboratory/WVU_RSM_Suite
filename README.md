# Response Surface Model + Test Particle Model (RSM+TPM)

## Test Particle Model

The Test Particle Model (TPM) is written in C and uses MPI and some extra libraries.

To compile TPM several packages are needed:

  * C compiler
  * GSL libraries
  * HDF5 libraries (Serial version is enough)
  * MPI implementation and libraries
  * Autoconf needed if installing from the Github repository
  * Git to clone the repository

To install these dependencies you can use the packages provided by your Linux distribution
For example on Ubuntu focal the dependencies above can be satisfied executing the command:

```
sudo apt-get install build-essential git autoconf libgsl-dev libhdf5-dev libmpich-dev
```

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
git clone https://github.com/WVUResearchComputing/RSM_TPMC.git
```

or

```
git clone https://github.com/ASSISTLaboratory/WVU_RSM_Suite.git
```

## Compile TPM

Go to the folder `TSM_TPMC/tpm` and prepare the sources for compilation:

```
cd TSM_TPMC/tpm
./autogen.sh
```

Execute the script `./configure`. One important option is to enable MPI so the code can work on multiple CPU cores or even distributed across multiple compute nodes.

```
./configure --enable-mpi
```

Other options availble include adding debugging and/or profiling flags. Enabling this features is useful for developers but will create a slower executable

```
./configure --enable-mpi --enable-debug --enable-gprof
```

If everything is correct at this point, the configure will have detected the location for GSL, HDF5 and the proper way of adding MPI support to the code.
Compile the code with:

```
make
```

## Execute RSM

The main script for RSM is `rsm_run_script.py`.
You can execute this script with two options `--mpiexec` and `--tpm`. This options will select which will be the MPI execution command and the location of the tpm executable.
The values for default are equivalent to use this command:

```
./rsm_run_script.py --mpiexec=mpiexec --tpm=tpm/src/tpm
```


# RSM+TPM on a Singularity container

The following recipe can be use to create a Singularity image for RSM+TPM.
Create a Singularity definition file such as `RSM+TPM.def` with the following contents:

```
Bootstrap: docker
From: ubuntu:focal

%files
RSM+TPM.def

%environment
SHELL=/bin/bash
export SHELL

%post
export DEBIAN_FRONTEND=noninteractive
touch /etc/localtime
apt-get -y update && apt-get -y upgrade && apt-get -y install apt-utils locales dialog && \
locale-gen en_US.UTF-8 && \
update-locale

ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
apt-get install -y tzdata && \
dpkg-reconfigure --frontend noninteractive tzdata

apt-get -y upgrade && \
apt-get -y install wget vim less build-essential git autoconf libgsl-dev libhdf5-dev libmpich-dev \
python3 python3-numpy python3-matplotlib python3-pandas python3-sklearn python3-pip && \
apt-get -y autoclean

python3 -m pip install h5py

cd /opt && \
git clone https://github.com/WVUResearchComputing/RSM_TPM.git && \
cd RSM_TPM/tpm && \
./autogen.sh && \
./configure --enable-mpi && \
make && \
cd .. && \
./rsm_run_script.py --mpiexec=mpiexec --tpm=tpm/src/tpm --help

echo "Sorting some env variables..."
echo 'LANGUAGE="en_US:en"' >> $SINGULARITY_ENVIRONMENT
echo 'LC_ALL="en_US.UTF-8"' >> $SINGULARITY_ENVIRONMENT
echo 'LC_CTYPE="UTF-8"' >> $SINGULARITY_ENVIRONMENT
echo 'LANG="en_US.UTF-8"' >>  $SINGULARITY_ENVIRONMENT
```

To create the singularity image install Singularity on your Linux machine and as superuser execute the following command:

```
singularity build RSM+TPM.sif RSM+TPM.def
```


