# lib1dfem -- generic 1D finite-element method library
#
# Placeholder for the v2 refactor (Phase 1). Populated incrementally:
# files that are strictly 1D-FEM-generic (no quantum-chemistry assumptions)
# are migrated here from libhelfem/, templated on the scalar type T.
#
# The target is exposed as `helfem::lib1dfem` and is currently an empty
# INTERFACE library -- it adds no sources or include directories until the
# first migration lands. Linking against it is a no-op today; once Phase 1
# starts, downstream targets (libhelfem, helfem-common) will rebase onto it.

add_library(lib1dfem INTERFACE)
add_library(helfem::lib1dfem ALIAS lib1dfem)

target_include_directories(lib1dfem INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)

# When Phase 1 starts, this is the point where the migrated source files
# (PolynomialBasis, LIPBasis, HIPBasis, FiniteElementBasis, lobatto,
# chebyshev, quadrature, grid, math half of utils) will be added as a
# templated header-only or compiled implementation.
