# CMake project file for HelFEM
cmake_minimum_required (VERSION 2.6)
project (helfem CXX Fortran)

# Try to find the Armadillo C++ library
#
# If the library is installed in a non-standard location, this prefix path can be passed via
# CMAKE_PREFIX_PATH variable, e.g.
#
#     cmake .. -DCMAKE_PREFIX_PATH=$HOME/local/usr
#
find_package(Armadillo REQUIRED)
message("Found Armadillo ${ARMADILLO_VERSION_NAME} (v${ARMADILLO_VERSION_MAJOR}.${ARMADILLO_VERSION_MINOR}.${ARMADILLO_VERSION_PATCH})")
message("ARMADILLO_VERSION_STRING: ${ARMADILLO_VERSION_STRING}")
message("ARMADILLO_INCLUDE_DIRS:   ${ARMADILLO_INCLUDE_DIRS}")
message("ARMADILLO_LIBRARIES:      ${ARMADILLO_LIBRARIES}")
if(NOT (${ARMADILLO_VERSION_MAJOR} EQUAL "9"))
    message(FATAL_ERROR "Invalid Armadillo version, expected 9.")
endif()

option(USE_OPENMP "Compile OpenMP enabled version (for parallel calculations)?" ON)
# Find OpenMP support
if(USE_OPENMP)
 find_package( OpenMP )
 if(OPENMP_FOUND)
  # Add flags to CXX flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
 endif()
endif()

# System specific definitions
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/CMake.system")
    message("Including user-provided CMake.system")
    include("${CMAKE_CURRENT_SOURCE_DIR}/CMake.system")
endif()

# Compile code
add_subdirectory(src)
