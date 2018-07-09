# CMake project file for HelFEM
cmake_minimum_required (VERSION 2.6)
project (helfem CXX)

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
# Large matrix support
add_definitions(-DARMA_64BIT_WORD -DARMA_BLAS_LONG)

add_library(helfem general/polynomial.cpp general/gaunt.cpp
general/chebyshev.cpp general/diis.cpp general/lbfgs.cpp
general/utils.cpp general/spherical_harmonics.cpp general/timer.cpp)

add_library(legendre legendre/legendre.cpp
legendre/legendre_p_normal.cpp legendre/legendre_pq.cpp
legendre/legendre_q_prolate.cpp legendre/legendre_p_prolate.cpp
legendre/legendre_q_normal.cpp)

add_executable(gaunt_test general/gaunt_test.cpp)
target_link_libraries(gaunt_test helfem)
target_link_libraries(gaunt_test gsl openblaso64)

add_executable(sphtest general/sphtest.cpp)
target_link_libraries(sphtest helfem)
target_link_libraries(sphtest gsl openblaso64)

add_executable(harmonic harmonic/main.cpp harmonic/quadrature.cpp)
target_link_libraries(harmonic helfem)
target_link_libraries(harmonic gsl openblaso64)

add_executable(legendre_test legendre/legendre_test.cpp)
target_link_libraries(legendre_test helfem legendre)
target_link_libraries(legendre_test gsl openblaso64)

add_executable(atomic atomic/main.cpp atomic/basis.cpp atomic/quadrature.cpp atomic/dftgrid.cpp general/dftfuncs.cpp)
target_link_libraries(atomic helfem)
target_link_libraries(atomic gsl openblaso64 xc)

add_executable(atomic_qtest atomic/qtest.cpp atomic/basis.cpp atomic/quadrature.cpp)
target_link_libraries(atomic_qtest helfem)
target_link_libraries(atomic_qtest gsl openblaso64)

add_executable(atomic_itest atomic/inttest.cpp atomic/basis.cpp atomic/quadrature.cpp)
target_link_libraries(atomic_itest helfem)
target_link_libraries(atomic_itest gsl openblaso64)

add_executable(diatomic diatomic/main.cpp diatomic/basis.cpp diatomic/quadrature.cpp)
target_link_libraries(diatomic helfem legendre)
target_link_libraries(diatomic gsl openblaso64)

add_executable(diatomic_itest diatomic/inttest.cpp diatomic/basis.cpp diatomic/quadrature.cpp)
target_link_libraries(diatomic_itest helfem legendre)
target_link_libraries(diatomic_itest gsl openblaso64)
