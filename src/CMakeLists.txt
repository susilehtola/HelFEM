add_library(helfem-common
general/gaunt.cpp general/diis.cpp
general/lbfgs.cpp general/utils.cpp general/spherical_harmonics.cpp
general/timer.cpp general/elements.cpp
general/angular.cpp general/scf_helpers.cpp general/lcao.cpp
general/gsz.cpp general/sap.cpp general/dftfuncs.cpp
general/checkpoint.cpp atomic/basis.cpp atomic/quadrature.cpp
atomic/RadialBasis.cpp atomic/TwoDBasis.cpp
atomic/dftgrid.cpp atomic/erfc_expn.cpp sadatom/basis.cpp
sadatom/dftgrid.cpp sadatom/solver.cpp sadatom/configurations.cpp
general/dftfuncs.cpp diatomic/basis.cpp diatomic/quadrature.cpp
diatomic/dftgrid.cpp diatomic/twodquadrature.cpp
general/model_potential.cpp
general/PointNucleus.cpp
general/GaussianNucleus.cpp
general/SphericalNucleus.cpp
general/HollowNucleus.cpp
)
target_link_libraries(helfem-common PUBLIC helfem)
target_include_directories(helfem-common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../libhelfem/src/")

add_executable(gensap sadatom/main.cpp)
target_link_libraries(gensap helfem-common)
target_link_libraries(gensap gsl ${LAPACK_LIBRARY} rt hdf5 z xc)

add_library(legendre
  legendre/accuracy.f90 legendre/input_output.f90
  legendre/itoc.f90 legendre/Matrix_Print.f90 legendre/Data_Module.f90
  legendre/Special_Functions.f90 legendre/Auxilliary_Subroutines.f90
  legendre/Prolate_Functions.f90 legendre/Lentz_Thompson.f90
  legendre/Associated_Legendre_Functions.f90 legendre/Ass_Leg_Poly.f90
  legendre/Legendre_Wrapper.f90 general/legendretable.cpp)

add_executable(legendre_test legendre/legendre_test.cpp)
target_link_libraries(legendre_test helfem-common legendre)
target_link_libraries(legendre_test gsl ${LAPACK_LIBRARY} rt hdf5 z)

add_executable(gaunt_test general/gaunt_test.cpp)
target_link_libraries(gaunt_test helfem-common legendre)
target_link_libraries(gaunt_test gsl ${LAPACK_LIBRARY} rt hdf5 z)

add_executable(sphtest general/sphtest.cpp)
target_link_libraries(sphtest helfem-common legendre)
target_link_libraries(sphtest gsl ${LAPACK_LIBRARY} rt hdf5 z)

add_executable(harmonic harmonic/main.cpp harmonic/quadrature.cpp)
target_link_libraries(harmonic helfem-common legendre)
target_link_libraries(harmonic gsl ${LAPACK_LIBRARY} rt hdf5 z)

add_executable(atomic atomic/main.cpp)
target_link_libraries(atomic helfem-common legendre)
target_link_libraries(atomic gsl ${LAPACK_LIBRARY} rt hdf5 z xc)

add_executable(atomic_itest atomic/inttest.cpp)
target_link_libraries(atomic_itest helfem-common legendre)
target_link_libraries(atomic_itest gsl ${LAPACK_LIBRARY} rt hdf5 z)

add_executable(diatomic diatomic/main.cpp)
target_link_libraries(diatomic helfem-common legendre)
target_link_libraries(diatomic gsl ${LAPACK_LIBRARY} rt hdf5 z xc)

add_executable(diatomic_itest diatomic/inttest.cpp)
target_link_libraries(diatomic_itest helfem-common legendre)
target_link_libraries(diatomic_itest gsl ${LAPACK_LIBRARY} rt hdf5 z)

add_executable(diatomic_cbasis diatomic/corebasis.cpp)
target_link_libraries(diatomic_cbasis helfem-common legendre)
target_link_libraries(diatomic_cbasis gsl ${LAPACK_LIBRARY} rt hdf5 z)

add_executable(diatomic_cpl diatomic/completeness.cpp)
target_link_libraries(diatomic_cpl helfem-common legendre)
target_link_libraries(diatomic_cpl gsl ${LAPACK_LIBRARY} rt hdf5 z)

add_executable(diatomic_dline diatomic/density_line.cpp)
target_link_libraries(diatomic_dline helfem-common legendre)
target_link_libraries(diatomic_dline gsl ${LAPACK_LIBRARY} rt hdf5 z)

add_executable(diatomic_dgrid diatomic/density_grid.cpp)
target_link_libraries(diatomic_dgrid helfem-common legendre)
target_link_libraries(diatomic_dgrid gsl ${LAPACK_LIBRARY} rt hdf5 z)

# Install libraries and main executables
install (TARGETS helfem-common legendre atomic diatomic diatomic_cbasis
diatomic_cpl gensap DESTINATION bin OPTIONAL)
