set(LIBHELFEM_VERSION "v0.0.1-alpha")

# Version information
configure_file("include/helfem.source.h" "include/helfem.h")
configure_file("include/ModelPotential.h" "include/helfem/ModelPotential.h" COPYONLY)
configure_file("include/PolynomialBasis.h" "include/helfem/PolynomialBasis.h" COPYONLY)
configure_file("include/RadialBasis.h" "include/helfem/RadialBasis.h" COPYONLY)

add_library(helfem STATIC
    src/helfem.cpp
    src/grid.cpp
    src/chebyshev.cpp
    src/lobatto.cpp
    src/utils.cpp
    src/erfc_expn.cpp
    src/polynomial.cpp
    src/polynomial_basis.cpp
    src/quadrature.cpp
    src/ModelPotential.cpp
    src/RadialPotential.cpp
    src/RadialBasis.cpp
)
target_include_directories(helfem PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/include/" "${ARMADILLO_INCLUDE_DIRS}")
target_include_directories(helfem PRIVATE "${GSL_INCLUDE_DIRS}" "${CMAKE_CURRENT_BINARY_DIR}/src/")
target_link_libraries(helfem PUBLIC ${GSL_LIBRARIES} ${ARMADILLO_LIBRARIES})

message("include: ${CMAKE_CURRENT_BINARY_DIR}/include/")

install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/" DESTINATION include)
install(TARGETS helfem DESTINATION lib)
