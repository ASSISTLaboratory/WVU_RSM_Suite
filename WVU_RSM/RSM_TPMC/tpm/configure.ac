# ------------------------------------------------------------------------------
# Package setup: template configure.ac for TPM project
# ------------------------------------------------------------------------------

AC_INIT([tpm],[1.0],[gufranco@mail.wvu.edu])

m4_include([m4/ax_lib_hdf5.m4])
m4_include([m4/ax_mpi.m4])

AM_INIT_AUTOMAKE([-Wall -Werror foreign -Wno-portability])

# ------------------------------------------------------------------------------
# Default flags reinitialization
# ------------------------------------------------------------------------------
if test -z "$CFLAGS"
then
   CFLAGS="-Wall"
fi

#if test -z "$CXXFLAGS"
#then
#   CXXFLAGS="-Wall "
#fi

# ------------------------------------------------------------------------------
# Dummy headline formater
# ------------------------------------------------------------------------------
AC_DEFUN([CONFIGURE_HEADLINE],
[
        echo; echo "+++ $1 +++"
])

# ------------------------------------------------------------------------------
# Checks for tool support
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ C compiler ])
AC_PROG_CC
#CONFIGURE_HEADLINE([ C++ compiler ])
#AC_PROG_CXX
#CONFIGURE_HEADLINE([ Libtool ])
#AC_PROG_LIBTOOL

# ------------------------------------------------------------------------------
# Checks for C/C++ support
# ------------------------------------------------------------------------------
AC_LANG([C])
#AC_LANG([C++])

# ------------------------------------------------------------------------------
# Provides debug and release compilation modes
# ------------------------------------------------------------------------------
AC_ARG_ENABLE([debug],
   [  --enable-debug          Turn on debugging],
   [case "${enableval}" in
      yes) DEBUG=true ;;
      no)  DEBUG=false ;;
      *)   AC_MSG_ERROR([bad value ${enableval} for --enable-debug]) ;;
   esac],
   [DEBUG=false]
)
if test x$DEBUG = xtrue
then
    CFLAGS="-g3 -O0 -D_GLIBCXX_DEBUG ${CFLAGS}"
else
    CFLAGS="-O3 ${CFLAGS}"
fi

# ------------------------------------------------------------------------------
# Turn on gprof flags
# ------------------------------------------------------------------------------
AC_ARG_ENABLE([gprof],
	[  --enable-gprof          Turn on profiling with gprof (pass gnu or intel as argument)],
	[case "${enableval}" in
	      gnu | yes)
	      	   gprof_enabled=true
		   gprof_compiler="gnu compiler"
		   gprof_flags="-pg"
		   ;;
	      intel)
		   gprof_enabled=true
		   gprof_compiler="intel compiler"
		   gprof_flags="-p"
		   ;;
	      no)  gprof_enabled=false
	      	   gprof_compiler="none"
		   gprof_flags=""
		   ;;
	      *) AC_MSG_ERROR([bad value ${enableval} for --enable-gprof (only yes, gnu or intel are possible)]) ;;
	 esac],
	 [gprof_enabled=false
	  gprof_compiler="none"
	  gprof_flags=""
	 ]
)
if test x$gprof_enabled = xtrue
then
	CXXFLAGS="${CXXFLAGS} ${gprof_flags}"
	CPPFLAGS="${CPPFLAGS} ${gprof_flags}"
	LDFLAGS="${LDFLAGS} ${gprof_flags}"
fi

# ------------------------------------------------------------------------------
#    Check for lib MPI (OpenMPI or MPICH)
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ MPI compiler ])
AC_ARG_ENABLE(mpi, [AS_HELP_STRING([--enable-mpi],[MPI library required])], enable_mpi=$enableval, enable_mpi=no)

if test "$enable_mpi" = "yes"; then
   ACX_MPI([],[AC_MSG_ERROR([could not find mpi library for --enable-mpi])])
   AC_CHECK_PROG(MPIRUN, mpirun, mpirun)
   AC_SUBST(MPIRUN)
else
   AC_MSG_NOTICE([MPI is currently disabled; use option --enable-mpi to enable])
fi
AM_CONDITIONAL(USE_MPI, test "$enable_mpi" = "yes")

# try to find if we are using OpenMPI / MPICH by looking inside mpi.h
AS_IF([test "$enable_mpi" = "yes"],
	[
		sav_CC=$CC
		sav_CXX=$CXX
		sav_CXXFLAGS=$CXXFLAGS
		CC=$MPICC
		CXX=$MPICXX
		CXXFLAGS="$CXXFLAGS"
		AC_CHECK_DECL([OPEN_MPI], [mpi_vendor="OpenMPI"],
		[], [#include "mpi.h"])
		AC_CHECK_DECL([MPICH_VERSION], [mpi_vendor="MPICH"],
		[], [#include "mpi.h"])
		CC=$sav_CC
		CXX=$sav_CXX
		CXXFLAGS=$sav_CXXFLAGS
	])

#
# try to set MPI_CXXFLAGS and MPI_LDFLAGS
#
MPI_CXXFLAGS=
MPI_LDFLAGS=
if test "$enable_mpi" = "yes" -a "$mpi_vendor" = "OpenMPI"
then
     MPI_CXXFLAGS=`$MPICXX --showme:compile`
     MPI_LDFLAGS=`$MPICXX --showme:link`
     AC_MSG_NOTICE([OpenMPI found])
     AC_MSG_NOTICE([MPI_CXXFLAGS=$MPI_CXXFLAGS])
     AC_MSG_NOTICE([MPI_LDFLAGS=$MPI_LDFLAGS])
elif test "$enable_mpi" = "yes" -a "$mpi_vendor" = "MPICH"
then
     # first grab CXXFLAGS (awk script is used to remove CXX at front)
     tmp=`$MPICXX -compile-info | awk '{$1=""; print $0 }'`
     MPI_CXXFLAGS=
     for i in $tmp
     do
     	 case $i in
     	      -[[DIUbi]]*)
     		MPI_CXXFLAGS="$MPI_CXXFLAGS $i"
             	;;
     	 esac
     done
     # second grab LDFLAGS (awk script is used to remove CXX at front)
     tmp=`$MPICXX -link-info | awk '{$1=""; print $0 }'`
     for i in $tmp
     do
     	case $i in
     	      [[\\/]]*.a | ?:[[\\/]]*.a | -[[lLRu]]* | -Wl* )
     	      	MPI_LDFLAGS="$MPI_LDFLAGS $i"
     	       	;;
     	esac
     done
     AC_MSG_NOTICE([MPICH found])
     AC_MSG_NOTICE([MPI_CXXFLAGS=$MPI_CXXFLAGS])
     AC_MSG_NOTICE([MPI_LDFLAGS=$MPI_LDFLAGS])
else
     AC_MSG_WARN([Neither OpenMPI and MPICH have been recognized...])
fi
AC_SUBST(MPI_CXXFLAGS)
AC_SUBST(MPI_LDFLAGS)

# MPI Macro
#AX_MPI()
if test "$enable_mpi" = "yes"
then
     CC=$MPICC
fi


# ------------------------------------------------------------------------------
#    Check for GSL support
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ GSL support ])
AC_CHECK_LIB([m],[cos])
AC_CHECK_LIB([gslcblas],[cblas_dgemm])
AC_CHECK_LIB([gsl],[gsl_blas_dgemm])

# ------------------------------------------------------------------------------
#    Check for HDF5 support
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ HDF5 support ])
AX_LIB_HDF5()
AC_MSG_CHECKING(for HDF5 include)
AC_MSG_RESULT($HDF5_CPPFLAGS)
AC_MSG_CHECKING(for HDF5 libs)
AC_MSG_RESULT([$HDF5_LDFLAGS $HDF5_LIBS])
LIBS="$LIBS $HDF5_LIBS"
LDFLAGS="$LDFLAGS $HDF5_LDFLAGS"
CPPFLAGS="$CPPFLAGS $HDF5_CPPFLAGS"

# ------------------------------------------------------------------------------
# Declares output files
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ Creating config.h and Makefiles ])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([
	Makefile
	src/Makefile
	tests/Makefile
	])
AC_OUTPUT

AC_MSG_RESULT([
	--------------------------------------------------
	Configuration summary for $PACKAGE_NAME ($VERSION)
        --------------------------------------------------

	* Installation prefix    : $prefix
	* Build debug            : $DEBUG
	* gprof profiler enabled : $gprof_enabled ($gprof_compiler)
	* HDF5 LDFLAGS           : $HDF5_LDFLAGS
	* HDF5 LIBS              : $HDF5_LIBS
	* MPI enabled            : $enable_mpi ($mpi_vendor)

	* LDFLAGS                : $LDFLAGS
	* LIBS                   : $LIBS

	* CC                     : $CC
	* CFLAGS                 : $CFLAGS
])

