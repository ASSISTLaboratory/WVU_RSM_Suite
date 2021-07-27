AC_INIT([tpm], [1.0], [gufranco@mail.wvu.edu]) 

m4_include([m4/ax_lib_hdf5.m4])
m4_include([m4/ax_mpi.m4])

AM_INIT_AUTOMAKE([foreign -Wall -Werror]) 
AC_PROG_CC

# MPI Macro
AX_MPI()

# GSL Macro
AC_CHECK_LIB([m],[cos])
AC_CHECK_LIB([gslcblas],[cblas_dgemm])
AC_CHECK_LIB([gsl],[gsl_blas_dgemm])

# HDF5 Macros
AX_LIB_HDF5()
AC_MSG_CHECKING(for HDF5 include)
AC_MSG_RESULT($HDF5_CPPFLAGS)
AC_MSG_CHECKING(for HDF5 libs)
AC_MSG_RESULT([$HDF5_LDFLAGS $HDF5_LIBS])
LIBS="$LIBS $HDF5_LIBS"

AC_CONFIG_HEADERS([config.h]) 
AC_CONFIG_FILES([Makefile src/Makefile]) 
AC_OUTPUT

