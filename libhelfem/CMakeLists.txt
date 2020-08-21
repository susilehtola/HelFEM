set(LIBHELFEM_VERSION "v0.0.1-alpha")

# Version information
configure_file("include/helfem.source.h" "include/helfem.h")
configure_file("include/ModelPotential.h" "include/helfem/ModelPotential.h" COPYONLY)
configure_file("include/PolynomialBasis.h" "include/helfem/PolynomialBasis.h" COPYONLY)
configure_file("include/RadialBasis.h" "include/helfem/RadialBasis.h" COPYONLY)

add_library(helfem STATIC
    src/helfem.cc
    src/grid.cc
    src/chebyshev.cc
    src/lobatto.cc
    src/utils.cc
    src/erfc_expn.cc
    src/polynomial.cc
    src/polynomial_basis.cc
    src/quadrature.cc
    src/ModelPotential.cc
    src/RadialPotential.cc
    src/RadialBasis.cc
)
target_include_directories(helfem PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/include/" "${ARMADILLO_INCLUDE_DIRS}")
target_include_directories(helfem PRIVATE "${GSL_INCLUDE_DIRS}" "${CMAKE_CURRENT_BINARY_DIR}/src/")
target_link_libraries(helfem PUBLIC ${GSL_LIBRARIES} ${ARMADILLO_LIBRARIES})

message("include: ${CMAKE_CURRENT_BINARY_DIR}/include/")

install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/" DESTINATION include)
install(TARGETS helfem DESTINATION lib)
