add_library(helfem-common
general/gaunt.cpp general/diis.cpp
general/lbfgs.cpp general/spherical_harmonics.cpp
general/timer.cpp general/elements.cpp
general/angular.cpp general/scf_helpers.cpp general/lcao.cpp
general/gsz.cpp general/sap.cpp general/dftfuncs.cpp
general/checkpoint.cpp atomic/basis.cpp
atomic/TwoDBasis.cpp
atomic/dftgrid.cpp sadatom/basis.cpp
sadatom/dftgrid.cpp sadatom/solver.cpp sadatom/configurations.cpp
general/dftfuncs.cpp diatomic/basis.cpp diatomic/quadrature.cpp
diatomic/dftgrid.cpp diatomic/twodquadrature.cpp
general/model_potential.cpp
)
target_link_libraries(helfem-common PUBLIC helfem)
target_include_directories(helfem-common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../libhelfem/src/")

add_executable(gensap sadatom/main.cpp)
target_link_libraries(gensap helfem-common legendre)

add_library(legendre
  legendre/accuracy.f90 legendre/input_output.f90
  legendre/itoc.f90 legendre/Matrix_Print.f90 legendre/Data_Module.f90
  legendre/Special_Functions.f90 legendre/Auxilliary_Subroutines.f90
  legendre/Prolate_Functions.f90 legendre/Lentz_Thompson.f90
  legendre/Associated_Legendre_Functions.f90 legendre/Ass_Leg_Poly.f90
  legendre/Legendre_Wrapper.f90 general/legendretable.cpp)

add_executable(legendre_test legendre/legendre_test.cpp)
target_link_libraries(legendre_test helfem-common legendre)

add_executable(gaunt_test general/gaunt_test.cpp)
target_link_libraries(gaunt_test helfem-common legendre)

add_executable(sphtest general/sphtest.cpp)
target_link_libraries(sphtest helfem-common legendre)

add_executable(harmonic harmonic/main.cpp)
target_link_libraries(harmonic helfem-common legendre)

add_executable(atomic atomic/main.cpp)
target_link_libraries(atomic helfem-common legendre)

add_executable(atomic_itest atomic/inttest.cpp)
target_link_libraries(atomic_itest helfem-common legendre)

add_executable(diatomic diatomic/main.cpp)
target_link_libraries(diatomic helfem-common legendre)

add_executable(diatomic_cbasis diatomic/corebasis.cpp)
target_link_libraries(diatomic_cbasis helfem-common legendre)

add_executable(diatomic_1e diatomic/1e.cpp)
target_link_libraries(diatomic_1e helfem-common legendre)

add_executable(diatomic_cpl diatomic/completeness.cpp)
target_link_libraries(diatomic_cpl helfem-common legendre)

add_executable(diatomic_dline diatomic/density_line.cpp)
target_link_libraries(diatomic_dline helfem-common legendre)

add_executable(diatomic_dgrid diatomic/density_grid.cpp)
target_link_libraries(diatomic_dgrid helfem-common legendre)

# Install libraries and main executables
install (TARGETS atomic diatomic diatomic_cbasis diatomic_cpl gensap DESTINATION bin OPTIONAL)
install (TARGETS helfem-common legendre DESTINATION lib OPTIONAL)
