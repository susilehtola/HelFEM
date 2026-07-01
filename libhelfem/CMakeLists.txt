set(LIBHELFEM_VERSION "v0.0.1-alpha")

# Version information. Landed at include/ so the header ships with the
# rest of the public API when we install(DIRECTORY .../include/).
configure_file("include/helfem.source.h" "include/helfem.h")
list(APPEND LIBHELFEM_PUBLIC_HEADERS
    "ModelPotential.h"
    "GaussianNucleus.h"
    "HollowNucleus.h"
    "ModelPotential.h"
    "PointNucleus.h"
    "PolynomialBasis.h"
    "FiniteElementBasis.h"
    "RadialBasis.h"
    "SphericalNucleus.h"
    "RegularizedNucleus.h"
    "Matrix.h"
    "ArmaEigen.h"
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
    src/GaussianNucleus.cpp
    src/HollowNucleus.cpp
    src/PointNucleus.cpp
    src/RegularizedNucleus.cpp
    src/SphericalNucleus.cpp
    src/FiniteElementBasis.cpp
    src/RadialBasis.cpp
)
target_include_directories(helfem
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(helfem PUBLIC helfem::lib1dfem Eigen3::Eigen)
target_compile_features(helfem PUBLIC cxx_std_17)
# ARMA_64BIT_WORD / ARMA_BLAS_LONG must be seen by any consumer that
# includes libhelfem headers -- some public headers still transitively
# include <armadillo> (e.g., helfem.source.h -> utils::invh), and the
# macros affect arma's ABI. Propagate as INTERFACE.
target_compile_definitions(helfem PUBLIC ARMA_64BIT_WORD ARMA_BLAS_LONG)

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
