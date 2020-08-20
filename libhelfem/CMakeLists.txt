set(LIBHELFEM_VERSION "v0.0.1-alpha")

# Version information
configure_file("include/helfem.source.hh" "include/helfem")

add_library(helfem STATIC
    src/helfem.cc
    src/utils.cc
    src/chebyshev.cc
)
target_include_directories(helfem PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/include/" "${ARMADILLO_INCLUDE_DIRS}")
target_include_directories(helfem PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/src/")
target_link_libraries(helfem PUBLIC ${ARMADILLO_LIBRARIES})

message("include: ${CMAKE_CURRENT_BINARY_DIR}/include/")

install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/" DESTINATION include)
install(TARGETS helfem DESTINATION lib)
