set(LIBHELFEM_VERSION "v0.0.1-alpha")

# Version information
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
)
foreach(header IN LISTS LIBHELFEM_PUBLIC_HEADERS)
    configure_file("include/${header}" "include/helfem/${header}" COPYONLY)
endforeach()

add_library(helfem
    src/helfem.cpp
    src/grid.cpp
    src/chebyshev.cpp
    src/lobatto.cpp
    src/utils.cpp
    src/erfc_expn.cpp
    src/PolynomialBasis.cpp
    src/LIPBasis.cpp
    src/HIPBasis.cpp
    src/LegendreBasis.cpp
    src/quadrature.cpp
    src/ModelPotential.cpp
    src/RadialPotential.cpp
    src/GaussianNucleus.cpp
    src/HollowNucleus.cpp
    src/PointNucleus.cpp
    src/SphericalNucleus.cpp
    src/FiniteElementBasis.cpp
    src/RadialBasis.cpp
)
target_include_directories(helfem PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/include/")

install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/" DESTINATION include)
install(TARGETS helfem DESTINATION lib)
