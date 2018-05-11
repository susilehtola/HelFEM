# CMake project file for HelFEM
cmake_minimum_required (VERSION 2.6)
project (helfem CXX)

INCLUDE_DIRECTORIES("/home/work/armadillo/include/")

add_library(helfem general/polynomial.cpp general/gaunt.cpp general/chebyshev.cpp)

add_executable(harmonic harmonic/main.cpp harmonic/quadrature.cpp)
target_link_libraries(harmonic helfem)
target_link_libraries(harmonic gsl openblas)

add_executable(atomic atomic/main.cpp atomic/basis.cpp atomic/quadrature.cpp)
target_link_libraries(atomic helfem)
target_link_libraries(atomic gsl openblas boost_timer)

add_executable(atomic_itest atomic/inttest.cpp atomic/basis.cpp atomic/quadrature.cpp)
target_link_libraries(atomic_itest helfem)
target_link_libraries(atomic_itest gsl openblas)

#add_executable(diatomic diatomic/main.cpp diatomic/basis.cpp diatomic/quadrature.cpp)
#target_link_libraries(diatomic helfem)
#target_link_libraries(diatomic gsl openblas)
