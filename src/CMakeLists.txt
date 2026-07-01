add_library(helfem-common
general/gaunt.cpp general/diis.cpp
general/lbfgs.cpp general/spherical_harmonics.cpp
general/timer.cpp general/elements.cpp
general/angular.cpp general/scf_helpers.cpp general/lcao.cpp
general/dftgrid_common.cpp
general/gsz.cpp general/sap.cpp general/dftfuncs.cpp
general/checkpoint.cpp atomic/basis.cpp
atomic/TwoDBasis.cpp
atomic/dftgrid.cpp sadatom/basis.cpp
sadatom/dftgrid.cpp sadatom/solver.cpp sadatom/configurations.cpp
general/dftfuncs.cpp diatomic/basis.cpp diatomic/quadrature.cpp
diatomic/dftgrid.cpp diatomic/twodquadrature.cpp
general/model_potential.cpp
)
target_link_libraries(helfem-common PUBLIC helfem wignernj::wignernj)
target_include_directories(helfem-common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../libhelfem/src/")
target_compile_features(helfem-common PUBLIC cxx_std_17)

add_executable(gensap sadatom/main.cpp)
target_link_libraries(gensap helfem-common legendre)

# Sadatom driver using OpenOrbitalOptimizer for SCF convergence.
# Proves OOO integration on the smallest of the three SCF-driver
# suites; the atomic / diatomic OOO variants follow the same pattern.
add_executable(sadatom_ooo sadatom/ooo.cpp)
target_link_libraries(sadatom_ooo helfem-common legendre)
if(TARGET OpenOrbitalOptimizer::OpenOrbitalOptimizer)
    target_link_libraries(sadatom_ooo OpenOrbitalOptimizer::OpenOrbitalOptimizer)
elseif(TARGET openorbitaloptimizer)
    target_link_libraries(sadatom_ooo openorbitaloptimizer)
endif()

add_executable(1e_atom sadatom/1e.cpp)
target_link_libraries(1e_atom helfem-common legendre)

add_library(legendre
  legendre/Legendre.cpp general/legendretable.cpp)
target_compile_features(legendre PUBLIC cxx_std_17)

add_executable(legendre_test legendre/legendre_test.cpp)
target_link_libraries(legendre_test helfem-common legendre)

add_executable(legendre_unit_test legendre/legendre_unit_test.cpp)
target_link_libraries(legendre_unit_test legendre)
# If GCC's __float128 + libquadmath is available, run the unit test at quad
# precision too. The compiler check looks for both the type and the math
# library.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES quadmath)
check_cxx_source_compiles("
#include <quadmath.h>
int main(){__float128 q = 1.0Q; return (int)logq(q + 1);}"
  HELFEM_HAVE_FLOAT128)
unset(CMAKE_REQUIRED_LIBRARIES)
if(HELFEM_HAVE_FLOAT128)
  target_compile_definitions(legendre_unit_test PRIVATE HELFEM_LEGENDRE_TEST_FLOAT128)
  target_link_libraries(legendre_unit_test quadmath)
endif()

add_executable(gaunt_test general/gaunt_test.cpp)
target_link_libraries(gaunt_test helfem-common legendre)

add_executable(sphtest general/sphtest.cpp)
target_link_libraries(sphtest helfem-common legendre)

add_executable(eigen_foundation_test general/eigen_foundation_test.cpp)
target_link_libraries(eigen_foundation_test helfem)

add_executable(harmonic harmonic/main.cpp)
target_link_libraries(harmonic helfem-common legendre)

add_executable(softcoulomb harmonic/softcoulomb.cpp)
target_link_libraries(softcoulomb helfem-common legendre)

add_executable(atomic atomic/main.cpp)
target_link_libraries(atomic helfem-common legendre)

# Atomic driver using OpenOrbitalOptimizer for SCF convergence.
# Same OOO pattern as sadatom_ooo, extended to the full 2D angular +
# radial basis. DFT only in this first cut.
add_executable(atomic_ooo atomic/ooo.cpp)
target_link_libraries(atomic_ooo helfem-common legendre)
if(TARGET OpenOrbitalOptimizer::OpenOrbitalOptimizer)
    target_link_libraries(atomic_ooo OpenOrbitalOptimizer::OpenOrbitalOptimizer)
elseif(TARGET openorbitaloptimizer)
    target_link_libraries(atomic_ooo openorbitaloptimizer)
endif()

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
