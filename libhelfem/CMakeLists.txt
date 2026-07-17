set(LIBHELFEM_VERSION "v0.0.1-alpha")

# Version information. Landed at include/ so the header ships with the
# rest of the public API when we install(DIRECTORY .../include/).
#
# Older configurations of this file placed the generated helfem.h at
# CMAKE_CURRENT_BINARY_DIR/helfem.h (no `include/` prefix). Existing
# build trees still have that stale copy sitting there, and the
# include search path picks it up before the freshly configured file
# under include/, so a stale signature there can shadow the freshly
# configured one and break the build. Delete the stale copy at
# configure time so incremental rebuilds into pre-existing build
# directories just work.
file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/helfem.h")
configure_file("include/helfem.source.h" "include/helfem.h")
list(APPEND LIBHELFEM_PUBLIC_HEADERS
    "ModelPotential.h"
    "PolynomialBasis.h"
    "FiniteElementBasis.h"
    "RadialBasis.h"
    "Matrix.h"
    # Templated FEM primitives folded in from the former lib1dfem
    # header-only library (Phase 4): quadrature, grid, math helpers, and
    # the primitive polynomial bases (templated base + LIP/HIP/HIP2/HIP3/
    # Legendre subclasses and their generated evaluation tables).
    "types.h"
    "chebyshev.h"
    "lobatto.h"
    "grid.h"
    "math.h"
    "legendre_poly.h"
    "PolynomialBasisT.h"
    "LIPBasis.h"
    "LIPBasis_eval.h"
    "HIPBasis.h"
    "HIPBasis_eval.h"
    "HIPBasis_over_r.h"
    "HIP2Basis.h"
    "HIP2Basis_eval.h"
    "HIP2Basis_over_r.h"
    "HIP3Basis.h"
    "HIP3Basis_eval.h"
    "HIP3Basis_over_r.h"
    "LegendreBasis.h"
)
foreach(header IN LISTS LIBHELFEM_PUBLIC_HEADERS)
    configure_file("include/${header}" "include/${header}" COPYONLY)
endforeach()

add_library(helfem
    src/helfem.cpp
    src/grid.cpp
    src/utils.cpp
    src/erfc_expn.cpp
    src/PolynomialBasis.cpp
    src/quadrature.cpp
    src/ModelPotential.cpp
    src/RadialPotential.cpp
    src/FiniteElementBasis.cpp
    src/RadialBasis.cpp
)
target_include_directories(helfem
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(helfem PUBLIC Eigen3::Eigen)
target_compile_features(helfem PUBLIC cxx_std_17)

# Quad precision: raise the standard (only) where it is needed, and switch on
# the guarded _Float128 explicit instantiations. The sources themselves stay
# C++17 -- this changes what they are compiled AT, not what they require.
if(HELFEM_FLOAT128)
  target_compile_features(helfem PUBLIC cxx_std_23)
  target_compile_definitions(helfem PUBLIC HELFEM_HAVE_FLOAT128)
  message(STATUS "Quad precision (_Float128) enabled: helfem compiled as C++23")
else()
  message(STATUS "Quad precision (_Float128) disabled (needs C++23); double and long double only")
endif()
# helfem::blas carries an external LP64 BLAS/LAPACK on the link line when
# HELFEM_EIGEN_BLAS is on (to back EIGEN_USE_BLAS), and is empty otherwise
# -- the linear algebra is Eigen's own header-only kernels. PUBLIC so
# downstream consumers inherit the same (possibly empty) linkage.
target_link_libraries(helfem PUBLIC helfem::blas)

# In-tree alias so downstream targets in HelFEM's own tree can link
# helfem::helfem consistently with the exported name.
add_library(helfem::helfem ALIAS helfem)

install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/" DESTINATION include)
install(TARGETS helfem
    EXPORT helfemTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include)
