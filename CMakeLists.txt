# CMake project file for HelFEM
cmake_minimum_required (VERSION 2.6)
project (helfem CXX Fortran)

option(USE_OPENMP "Compile OpenMP enabled version (for parallel calculations)?" ON)
# Find OpenMP support
if(USE_OPENMP)
 find_package( OpenMP )
 if(OPENMP_FOUND)
  # Add flags to CXX flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
 endif()
endif()

INCLUDE_DIRECTORIES("/home/work/armadillo/include/")
INCLUDE_DIRECTORIES("/home/work/libxc/trunk/install/include/")
LINK_DIRECTORIES("/home/work/libxc/trunk/install/lib/")
LINK_DIRECTORIES("/opt/intel/compilers_and_libraries_2018.0.128/linux/mkl/lib/intel64_lin")
# Large matrix support
add_definitions(-DARMA_64BIT_WORD -DARMA_BLAS_LONG)
add_definitions(-DMKL_ILP64)

add_library(helfem general/polynomial.cpp general/gaunt.cpp
general/chebyshev.cpp general/diis.cpp general/lbfgs.cpp
general/utils.cpp general/spherical_harmonics.cpp general/timer.cpp)

add_library(legendre
  legendre/accuracy.f90
  legendre/input_output.f90
  legendre/itoc.f90
  legendre/Matrix_Print.f90
  legendre/Data_Module.f90
  legendre/Special_Functions.f90
  legendre/Auxilliary_Subroutines.f90
  legendre/Prolate_Functions.f90
  legendre/Lentz_Thompson.f90
  legendre/Associated_Legendre_Functions.f90
  legendre/Ass_Leg_Poly.f90
  legendre/Legendre_Wrapper.f90)

add_executable(gaunt_test general/gaunt_test.cpp)
target_link_libraries(gaunt_test helfem)
target_link_libraries(gaunt_test gsl mkl_intel_ilp64 mkl_gnu_thread mkl_core gomp pthread m dl)

add_executable(sphtest general/sphtest.cpp)
target_link_libraries(sphtest helfem)
target_link_libraries(sphtest gsl mkl_intel_ilp64 mkl_gnu_thread mkl_core gomp pthread m dl)

add_executable(harmonic harmonic/main.cpp harmonic/quadrature.cpp)
target_link_libraries(harmonic helfem)
target_link_libraries(harmonic gsl mkl_intel_ilp64 mkl_gnu_thread mkl_core gomp pthread m dl)

add_executable(atomic atomic/main.cpp atomic/basis.cpp atomic/quadrature.cpp atomic/dftgrid.cpp general/dftfuncs.cpp)
target_link_libraries(atomic helfem)
target_link_libraries(atomic gsl mkl_intel_ilp64 mkl_gnu_thread mkl_core gomp pthread m dl xc)

add_executable(atomic_qtest atomic/qtest.cpp atomic/basis.cpp atomic/quadrature.cpp)
target_link_libraries(atomic_qtest helfem)
target_link_libraries(atomic_qtest gsl mkl_intel_ilp64 mkl_gnu_thread mkl_core gomp pthread m dl)

add_executable(atomic_itest atomic/inttest.cpp atomic/basis.cpp atomic/quadrature.cpp)
target_link_libraries(atomic_itest helfem)
target_link_libraries(atomic_itest gsl mkl_intel_ilp64 mkl_gnu_thread mkl_core gomp pthread m dl)

add_executable(diatomic diatomic/main.cpp diatomic/basis.cpp diatomic/quadrature.cpp)
target_link_libraries(diatomic helfem legendre)
target_link_libraries(diatomic gsl mkl_intel_ilp64 mkl_gnu_thread mkl_core gomp pthread m dl)

add_executable(diatomic_itest diatomic/inttest.cpp diatomic/basis.cpp diatomic/quadrature.cpp)
target_link_libraries(diatomic_itest helfem legendre)
target_link_libraries(diatomic_itest gsl mkl_intel_ilp64 mkl_gnu_thread mkl_core gomp pthread m dl)
