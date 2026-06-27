# lib1dfem -- generic 1D finite-element method library
#
# v2 refactor destination for strictly-generic 1D-FEM code migrated from
# libhelfem/, templated on the scalar type T. Header-only INTERFACE
# library; downstream consumers depend on it via `helfem::lib1dfem`.
#
# Currently populated with:
#   - chebyshev: Gauss-Chebyshev quadrature of the second kind
#                (templated on T, header-only).
#   - lobatto:   Gauss-Lobatto quadrature (templated on T, header-only;
#                Newton iteration on Legendre polynomial derivatives,
#                replaces the hardcoded double-precision tables that
#                lived in libhelfem prior to v2).
#   - grid:      Radial element-boundary grid generator (linear / quadratic /
#                generalised polynomial / generalised exponential / geometric;
#                templated on T, header-only).
#   - math:      Pure-math helpers (arcosh, arsinh, modified Bessel i_L / k_L)
#                used inside the FEM machinery. Templated on T; Bessel paths
#                use std::cyl_bessel_i/k for the standard float/double/long
#                double overloads, plus an explicit Taylor series at small |x|
#                that works at any precision.
#   - PolynomialBasis: abstract templated base for primitive polynomial bases
#                on the reference element [-1, 1]. Concrete subclasses
#                (LIPBasis, HIPBasis, GeneralHIPBasis) still live in libhelfem
#                during the migration; they derive from
#                lib1dfem::polynomial_basis::PolynomialBasis<double>.
#   - legendre_poly: Templated Legendre polynomial evaluators (value, first,
#                second derivatives) via the standard recurrence. Replaces the
#                LGPL oomph-lib Orthpoly that was bundled in libhelfem.
#   - LegendreBasis: Templated spectral-element shape-function basis using the
#                Flores/Clementi/Sonnad 1989 orthogonal combinations of
#                Legendre polynomials.

add_library(lib1dfem INTERFACE)
add_library(helfem::lib1dfem ALIAS lib1dfem)

target_include_directories(lib1dfem INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)
target_compile_features(lib1dfem INTERFACE cxx_std_17)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/lib1dfem"
        DESTINATION include)
