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


HOW TO COMPILE
--------------

These are the instructions to compile tpm assuming that your current working directory is the place where this README is located.

1. Create a folder for compilation, it is not mandatory, but it is a good practice to separate the compiled code from the sources.

	$> mkdir build
	$> cd build

2. Execute the configure with appropiated flags, the code needs HDF5 and GSL, this could be one example:

	$> CC=mpiicc ../configure

The configure will figure out the right values for HDF5 and GSL libraries, otherwise you can also explicictly insert those like this:

	$> CC=mpiicc LIBS="-lhdf5 -lm -lgsl -lgslcblas" ../configure
	
3. Execute make to build the code

	$> make

4. After compilation the executable will be located at: build/src with the name "tpm"


You can use the binary directly or you can install it. For installing the code you can decide a good place for installation with:

	$> CC=mpiicc LIBS="-lhdf5 -lm -lgsl -lgslcblas" ../configure --prefix=$HOME/.local

and after make, install the code with:

	$> make install

