add_library(helfem-common
general/gaunt.cpp
general/spherical_harmonics.cpp
general/timer.cpp general/elements.cpp
general/angular.cpp general/scf_helpers.cpp general/lcao.cpp
general/dftgrid_common.cpp
general/gsz.cpp general/sap.cpp general/dftfuncs.cpp
general/checkpoint.cpp atomic/basis.cpp
atomic/TwoDBasis.cpp
atomic/dftgrid.cpp sadatom/basis.cpp
sadatom/dftgrid.cpp sadatom/scf.cpp
general/dftfuncs.cpp diatomic/basis.cpp diatomic/quadrature.cpp
diatomic/dftgrid.cpp diatomic/dftgrid_purem.cpp diatomic/twodquadrature.cpp
general/model_potential.cpp
)
target_link_libraries(helfem-common PUBLIC helfem wignernj::wignernj)
# HDF5 (checkpoint.cpp) and libxc (dftfuncs / dftgrid) are consumed here.
# helfem::hdf5 and PkgConfig::LIBXC are defined by the root CMakeLists
# under HELFEM_BINARIES, which is the only configuration that builds src/.
target_link_libraries(helfem-common PUBLIC helfem::hdf5 PkgConfig::LIBXC)
# sadatom/scf.cpp uses OOO's header-only SCFSolver both from the
# gensap driver and from diatomic/twodquadrature.cpp's atomic-orbital
# guess. Propagate the header include path via the PUBLIC link.
if(TARGET OpenOrbitalOptimizer::OpenOrbitalOptimizer)
    target_link_libraries(helfem-common PUBLIC OpenOrbitalOptimizer::OpenOrbitalOptimizer)
elseif(TARGET openorbitaloptimizer)
    target_link_libraries(helfem-common PUBLIC openorbitaloptimizer)
endif()
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
        # `chebyshev.h` and `lobatto.h` are thin double-only shims over
        # lib1dfem's Eigen-typed Gauss-Chebyshev / Gauss-Lobatto
        # quadrature, spelling their outputs as helfem::Vector. They
        # live in src/general/ next to their callers; callers include
        # them as `#include "chebyshev.h"` / `#include "lobatto.h"`,
        # which resolve through this path.
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

# Validates the pure-m (analytic-phi) DFT grid against the general 3D grid
# on a single Fock build with an identical density, and finite-difference-
# checks TwoDBasis::eval_lf.
add_executable(diatomic_purem_test diatomic/purem_test.cpp)
target_link_libraries(diatomic_purem_test helfem-common legendre)

# Measures how compressible the in-element two-electron integrals are, i.e.
# what a thresholded Cholesky / low-rank factorization would buy.
# Arbitrary-precision demonstration: the same FEM code at several scalar types.
add_executable(precision harmonic/precision.cpp)
target_link_libraries(precision helfem-common legendre)

# The same, one layer up: the whole atomic Hartree-Fock path (TwoDBasisT<T>,
# GauntT<T>, the 2e integrals, Coulomb, exchange, orthonormalisation) at
# double / long double / _Float128, on helium, whose HF basis-set limit is
# known from the literature.
add_executable(atomic_precision atomic/precision.cpp)
target_link_libraries(atomic_precision helfem-common legendre)

add_executable(diatomic_teirank diatomic/teirank.cpp)
target_link_libraries(diatomic_teirank helfem-common legendre)

# Same measurement for the atomic in-element integrals.
add_executable(atomic_teirank atomic/teirank.cpp)
target_link_libraries(atomic_teirank helfem-common legendre)

add_library(legendre
  legendre/Legendre.cpp general/legendretable.cpp)
target_compile_features(legendre PUBLIC cxx_std_17)
# legendretable.{h,cpp} store the Plm/Qlm tables as helfem::Matrix, so
# legendre needs the libhelfem (Matrix.h) and Eigen include paths that
# the `helfem` target carries.
target_link_libraries(legendre PUBLIC helfem)

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

# Checks that psi(r) = B_u(r)/r is handled correctly as r -> 0 by EVERY basis.
add_executable(over_r_test general/over_r_test.cpp)
target_link_libraries(over_r_test helfem-common legendre)

add_executable(gaunt_test general/gaunt_test.cpp)
target_link_libraries(gaunt_test helfem-common legendre)

# Round-trips the templated HDF5 checkpoint float I/O at double / long double
# / _Float128 and asserts bit-identical reloads (proves the per-T memory type,
# not a double-capped path).
add_executable(checkpoint_precision_test general/checkpoint_precision_test.cpp)
target_link_libraries(checkpoint_precision_test helfem-common legendre)

add_executable(sphtest general/sphtest.cpp)
target_link_libraries(sphtest helfem-common legendre)

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
