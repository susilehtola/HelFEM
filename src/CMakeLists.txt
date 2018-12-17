add_library(helfem general/polynomial.cpp general/polynomial_basis.cpp
general/gaunt.cpp general/chebyshev.cpp general/diis.cpp
general/lbfgs.cpp general/utils.cpp general/spherical_harmonics.cpp
general/timer.cpp general/elements.cpp general/lobatto.cpp
general/angular.cpp general/scf_helpers.cpp general/lcao.cpp
general/gsz.cpp general/dftfuncs.cpp general/checkpoint.cpp
atomic/basis.cpp atomic/quadrature.cpp atomic/dftgrid.cpp
sadatom/basis.cpp sadatom/dftgrid.cpp sadatom/configurations.cpp general/dftfuncs.cpp
diatomic/basis.cpp diatomic/quadrature.cpp diatomic/dftgrid.cpp diatomic/twodquadrature.cpp)

add_executable(sadtest sadatom/main.cpp)
target_link_libraries(sadtest helfem)
target_link_libraries(sadtest gsl ${LAPACK_LIBRARY} rt hdf5 z xc)

add_library(legendre legendre/accuracy.f90 legendre/input_output.f90
  legendre/itoc.f90 legendre/Matrix_Print.f90 legendre/Data_Module.f90
  legendre/Special_Functions.f90 legendre/Auxilliary_Subroutines.f90
  legendre/Prolate_Functions.f90 legendre/Lentz_Thompson.f90
  legendre/Associated_Legendre_Functions.f90 legendre/Ass_Leg_Poly.f90
  legendre/Legendre_Wrapper.f90 general/legendretable.cpp)

add_executable(legendre_test legendre/legendre_test.cpp)
target_link_libraries(legendre_test helfem legendre)
target_link_libraries(legendre_test gsl ${LAPACK_LIBRARY} rt hdf5 z)

add_executable(gaunt_test general/gaunt_test.cpp)
target_link_libraries(gaunt_test helfem legendre)
target_link_libraries(gaunt_test gsl ${LAPACK_LIBRARY} rt hdf5 z)

add_executable(sphtest general/sphtest.cpp)
target_link_libraries(sphtest helfem legendre)
target_link_libraries(sphtest gsl ${LAPACK_LIBRARY} rt hdf5 z)

add_executable(harmonic harmonic/main.cpp harmonic/quadrature.cpp)
target_link_libraries(harmonic helfem legendre)
target_link_libraries(harmonic gsl ${LAPACK_LIBRARY} rt hdf5 z)

add_executable(atomic atomic/main.cpp)
target_link_libraries(atomic helfem legendre)
target_link_libraries(atomic gsl ${LAPACK_LIBRARY} rt hdf5 z xc)

add_executable(atomic_gszopt atomic/gszopt.cpp)
target_link_libraries(atomic_gszopt helfem legendre)
target_link_libraries(atomic_gszopt gsl ${LAPACK_LIBRARY} rt hdf5 z xc)

add_executable(atomic_itest atomic/inttest.cpp)
target_link_libraries(atomic_itest helfem legendre)
target_link_libraries(atomic_itest gsl ${LAPACK_LIBRARY} rt hdf5 z)

add_executable(diatomic diatomic/main.cpp)
target_link_libraries(diatomic helfem legendre)
target_link_libraries(diatomic gsl ${LAPACK_LIBRARY} rt hdf5 z xc)

add_executable(diatomic_itest diatomic/inttest.cpp)
target_link_libraries(diatomic_itest helfem legendre)
target_link_libraries(diatomic_itest gsl ${LAPACK_LIBRARY} rt hdf5 z)

add_executable(diatomic_cbasis diatomic/corebasis.cpp)
target_link_libraries(diatomic_cbasis helfem legendre)
target_link_libraries(diatomic_cbasis gsl ${LAPACK_LIBRARY} rt hdf5 z)

add_executable(diatomic_cpl diatomic/completeness.cpp)
target_link_libraries(diatomic_cpl helfem legendre)
target_link_libraries(diatomic_cpl gsl ${LAPACK_LIBRARY} rt hdf5 z)

add_executable(diatomic_dline diatomic/density_line.cpp)
target_link_libraries(diatomic_dline helfem legendre)
target_link_libraries(diatomic_dline gsl ${LAPACK_LIBRARY} rt hdf5 z)

# Install libraries and main executables
install (TARGETS helfem legendre atomic diatomic diatomic_cbasis DESTINATION bin)
