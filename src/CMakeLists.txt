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
general/tei_utils.cpp
)
target_link_libraries(helfem-common PUBLIC helfem wignernj::wignernj)
# Propagate OpenMP to consumers of helfem-common. Without this, an
# external consumer linking helfem::helfem misses -fopenmp / libgomp
# and fails with undefined references like GOMP_parallel.
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(helfem-common PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(helfem-common
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../libhelfem/src/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        # `chebyshev.h` and `lobatto.h` are thin arma::vec shims over
        # lib1dfem's Eigen-typed Gauss-Chebyshev / Gauss-Lobatto
        # quadrature. They live next to their arma-consuming callers
        # in src/general/ so libhelfem itself ships no arma-typed
        # code. Callers include them as `#include "chebyshev.h"` /
        # `#include "lobatto.h"`, which resolve through this path.
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/general>
        # Installed headers live at include/helfem/{atomic,general}/;
        # consumers include as <helfem/atomic/TwoDBasis.h>.
        $<INSTALL_INTERFACE:include>
)
target_compile_features(helfem-common PUBLIC cxx_std_17)

# Sadatom driver -- the OpenOrbitalOptimizer-based implementation
# replaces the bespoke SCF loop that used to live here (retired in
# favour of feature-complete OOO parity).
add_executable(gensap sadatom/main.cpp)
target_link_libraries(gensap helfem-common legendre)
if(TARGET OpenOrbitalOptimizer::OpenOrbitalOptimizer)
    target_link_libraries(gensap OpenOrbitalOptimizer::OpenOrbitalOptimizer)
elseif(TARGET openorbitaloptimizer)
    target_link_libraries(gensap openorbitaloptimizer)
endif()

add_executable(1e_atom sadatom/1e.cpp)
target_link_libraries(1e_atom helfem-common legendre)

add_library(legendre
  legendre/Legendre.cpp general/legendretable.cpp)
target_compile_features(legendre PUBLIC cxx_std_17)

# helfem-common uses LegendreTable symbols internally, so a consumer
# linking only helfem-common needs legendre transitively. Declared
# after `legendre` is defined so the target reference resolves.
target_link_libraries(helfem-common PUBLIC legendre)

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

# Atomic driver -- OpenOrbitalOptimizer-based (bespoke SCF retired).
add_executable(atomic atomic/main.cpp)
target_link_libraries(atomic helfem-common legendre)
if(TARGET OpenOrbitalOptimizer::OpenOrbitalOptimizer)
    target_link_libraries(atomic OpenOrbitalOptimizer::OpenOrbitalOptimizer)
elseif(TARGET openorbitaloptimizer)
    target_link_libraries(atomic openorbitaloptimizer)
endif()

add_executable(atomic_itest atomic/inttest.cpp)
target_link_libraries(atomic_itest helfem-common legendre)

# Diatomic driver -- OpenOrbitalOptimizer-based (bespoke SCF retired).
add_executable(diatomic diatomic/main.cpp)
target_link_libraries(diatomic helfem-common legendre)
if(TARGET OpenOrbitalOptimizer::OpenOrbitalOptimizer)
    target_link_libraries(diatomic OpenOrbitalOptimizer::OpenOrbitalOptimizer)
elseif(TARGET openorbitaloptimizer)
    target_link_libraries(diatomic openorbitaloptimizer)
endif()

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

# Install libraries and main executables.
# helfem-common + legendre are exported as part of the helfemTargets
# set so external consumers doing find_package(helfem) pick up the
# transitive linkage automatically.
install (TARGETS atomic diatomic diatomic_cbasis diatomic_cpl gensap DESTINATION bin OPTIONAL)
install (TARGETS helfem-common legendre
    EXPORT helfemTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include)

# Install atomic public headers into ${PREFIX}/include/helfem/atomic/
# so a consumer can #include <helfem/atomic/TwoDBasis.h>. This is the
# consumer-visible surface for the atomic FE basis backend that
# libatomscf integrates.
#
# TwoDBasis.h has \"../general/model_potential.h\" and \"../general/sap.h\"
# relative includes; keep the same directory layout under helfem/atomic/
# and helfem/general/ so the relative-include chain still resolves
# against the installed headers.
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/atomic/TwoDBasis.h"
    DESTINATION include/helfem/atomic)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/general/model_potential.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/general/RadialPotential.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/general/sap.h"
    DESTINATION include/helfem/general)
