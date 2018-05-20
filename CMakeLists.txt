# CMake project file for HelFEM
cmake_minimum_required (VERSION 2.6)
project (helfem CXX)

INCLUDE_DIRECTORIES("/home/work/armadillo/include/")
INCLUDE_DIRECTORIES("../arpack-arma/include/")
# Large matrix support
add_definitions(-DARMA_64BIT_WORD -DARMA_BLAS_LONG)

add_library(helfem general/polynomial.cpp general/gaunt.cpp
general/chebyshev.cpp general/diis.cpp general/lbfgs.cpp
general/utils.cpp)

add_library(legendre legendre/legendre.cpp
legendre/legendre_p_normal.cpp legendre/legendre_pq.cpp
legendre/legendre_q_prolate.cpp legendre/legendre_p_prolate.cpp
legendre/legendre_q_normal.cpp)


add_executable(harmonic harmonic/main.cpp harmonic/quadrature.cpp)
target_link_libraries(harmonic helfem)
target_link_libraries(harmonic gsl openblas64)

add_executable(atomic atomic/main.cpp atomic/basis.cpp atomic/quadrature.cpp)
target_link_libraries(atomic helfem)
target_link_libraries(atomic gsl openblas64 boost_timer)

add_executable(atomic_qtest atomic/qtest.cpp atomic/basis.cpp atomic/quadrature.cpp)
target_link_libraries(atomic_qtest helfem)
target_link_libraries(atomic_qtest gsl openblas64)

add_executable(atomic_itest atomic/inttest.cpp atomic/basis.cpp atomic/quadrature.cpp)
target_link_libraries(atomic_itest helfem)
target_link_libraries(atomic_itest gsl openblas64)

add_executable(diatomic diatomic/main.cpp diatomic/basis.cpp diatomic/quadrature.cpp)
target_link_libraries(diatomic helfem legendre)
target_link_libraries(diatomic gsl openblas boost_timer)

add_executable(diatomic_itest diatomic/inttest.cpp diatomic/basis.cpp diatomic/quadrature.cpp)
target_link_libraries(diatomic_itest helfem legendre)
target_link_libraries(diatomic_itest gsl openblas)
